"""Enchaînement d'un run : lecture de la fenêtre, transformation, publication.

Ce fichier teste ce que les étages ne peuvent pas tester seuls — la fenêtre
lue, ce qui repart en base, et l'idempotence de la partition publiée.

La base est remplacée par un moteur qui rend un lot préparé à la lecture et
mémorise les instructions à l'écriture. Ce qui compte n'est pas que PostgreSQL
accepte les instructions — c'est son métier, et le schéma figé le garantit —
mais que l'ETL lise la bonne fenêtre et n'en repose que la bonne journée.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from conftest import FakeEngine

from etl.__main__ import (
    DEFAULT_DAYS,
    feature_spec,
    of_day,
    parse_args,
    publish,
    run,
    transform,
)
from etl.extract import window
from predict_common import io
from predict_common.config import Config
from predict_common.paths import features_partition
from predict_common.schemas import TIMESTAMP_COLUMN

DAY = date(2026, 9, 10)
SPEC_LAGS = [1, 24]


class ReadingEngine(FakeEngine):
    """Moteur factice qui rend un lot fixe à la lecture.

    `pd.read_sql_query` passe par `connect()` ; l'écriture passe par `begin()`.
    Séparer les deux permet de vérifier ce qui est relu autant que ce qui est
    réécrit, sans jamais joindre une base.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__()
        self.frame = frame
        self.queried: list[dict] = []

    def connect(self) -> ReadingEngine:
        return self

    def __enter__(self) -> ReadingEngine:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def config_for(root: Path) -> Config:
    """Configuration minimale d'un run, pointée sur un stockage jetable."""
    return Config(
        values={
            "storage": {"root": str(root)},
            "database": {
                "url": "postgresql+psycopg://u:p@localhost/db",
                "batch_size": 1000,
            },
            "etl": {
                "feature_version": "v1",
                "resample_rule": "1h",
                "lag_hours": SPEC_LAGS,
                "rolling_window_h": 2,
            },
        }
    )


def measures(make_raw, make_reading, days: int = 5, **overrides) -> pd.DataFrame:
    """Lot tel qu'une lecture de `mesure` le rendrait, sur `days` journées."""
    readings = [
        make_reading(
            f"{(DAY - timedelta(days=offset)).isoformat()}T{hour:02d}:00:00Z",
            **overrides,
        )
        for offset in range(days)
        for hour in range(24)
    ]
    return make_raw(readings)


@pytest.fixture
def patched_read(monkeypatch):
    """Remplace la lecture SQL par le lot que porte le moteur factice."""

    def install(engine: ReadingEngine) -> None:
        def read_sql_query(query, connection, params=None, **kwargs):
            connection.queried.append(dict(params or {}))
            return engine.frame.copy()

        monkeypatch.setattr("etl.extract.pd.read_sql_query", read_sql_query)

    return install


def test_feature_spec_is_read_from_the_configuration(tmp_path: Path) -> None:
    spec = feature_spec(config_for(tmp_path))
    assert spec.version == "v1"
    assert spec.lag_hours == (1, 24)


def test_the_command_line_overrides_the_configured_version(tmp_path: Path) -> None:
    assert feature_spec(config_for(tmp_path), "v2").version == "v2"


class TestWindow:
    """La fenêtre lue déborde sur les journées que les décalages réclament."""

    def test_it_ends_at_midnight_after_the_produced_day(self) -> None:
        # Borne haute exclue : une mesure de minuit pile appartient au
        # lendemain et à lui seul, sinon deux runs voisins la liraient.
        _, end = window(DAY, 3)
        assert end == datetime(2026, 9, 11, tzinfo=UTC)

    def test_it_covers_the_requested_lookback(self) -> None:
        start, end = window(DAY, 3)
        assert (end - start).days == 3
        assert start == datetime(2026, 9, 8, tzinfo=UTC)

    def test_an_empty_lookback_is_refused(self) -> None:
        with pytest.raises(ValueError):
            window(DAY, 0)


def test_run_publishes_the_features_partition(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    rows = run(config_for(tmp_path), engine, DAY, version=None)
    partition = features_partition(str(tmp_path), "v1", DAY)
    assert rows == 24
    assert len(io.read_frames([partition])) == 24


def test_run_reads_the_window_the_lags_need(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    # Sans les journées précédentes, lag_24h n'aurait rien à désigner et la
    # journée sortirait presque entièrement écartée.
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    run(config_for(tmp_path), engine, DAY, version=None)
    spec = feature_spec(config_for(tmp_path))
    queried = engine.queried[0]
    assert (queried["end_time"] - queried["start_time"]).days == spec.lookback_days


def test_run_reposes_only_the_produced_day(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    # La fenêtre lue ne sert qu'aux décalages : la reposer entière ferait
    # réécrire cinq journées pour en produire une.
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    run(config_for(tmp_path), engine, DAY, version=None)
    written = engine.executed[0].compile().params
    stamps = {
        value.date() for key, value in written.items() if key.startswith("ts_")
    }
    assert stamps == {DAY}


def test_a_replay_reproduces_the_partition(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    # Le rejeu après incident est le mode d'exploitation normal.
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    config = config_for(tmp_path)
    run(config, engine, DAY, version=None)
    run(config, engine, DAY, version=None)
    partition = features_partition(str(tmp_path), "v1", DAY)
    assert len(io.read_frames([partition])) == 24


def test_two_versions_do_not_overwrite_each_other(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    # C'est ce qui permet à un modèle entraîné sur v1 de rester reproductible
    # après la sortie de v2.
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    config = config_for(tmp_path)
    run(config, engine, DAY, version="v1")
    run(config, engine, DAY, version="v2")
    assert io.exists(features_partition(str(tmp_path), "v1", DAY))
    assert io.exists(features_partition(str(tmp_path), "v2", DAY))


def test_the_published_partition_carries_its_provenance(
    tmp_path: Path, make_raw, make_reading
) -> None:
    import pyarrow.parquet

    spec = feature_spec(config_for(tmp_path))
    _, features = transform(measures(make_raw, make_reading), spec, DAY)
    partition = publish(features, str(tmp_path), spec, DAY)
    written = next(Path(partition).glob("*.parquet"))
    metadata = pyarrow.parquet.read_schema(written).metadata
    assert metadata[b"feature_version"] == b"v1"
    assert metadata[b"lag_hours"] == b"1,24"


def test_transform_returns_the_whole_window_of_measures(
    tmp_path: Path, make_raw, make_reading
) -> None:
    spec = feature_spec(config_for(tmp_path))
    enriched, _ = transform(measures(make_raw, make_reading), spec, DAY)
    assert "consumption_kw_imputed" in enriched.columns
    # Toute la fenêtre : c'est elle que les variables consomment. C'est
    # `of_day` qui restreint ensuite ce qui repart en base.
    assert len(enriched) == 24 * 5


def test_of_day_keeps_only_the_produced_day(
    tmp_path: Path, make_raw, make_reading
) -> None:
    spec = feature_spec(config_for(tmp_path))
    enriched, _ = transform(measures(make_raw, make_reading), spec, DAY)
    assert set(of_day(enriched, DAY)[TIMESTAMP_COLUMN].dt.date) == {DAY}


def test_a_day_of_outages_publishes_an_empty_partition(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    # Une journée sans mesure exploitable existe et elle est vide : c'est une
    # information, pas une absence de partition.
    engine = ReadingEngine(
        measures(
            make_raw,
            make_reading,
            consumption_kw=None,
            null_reasons=["sensor_failure"],
        )
    )
    patched_read(engine)
    rows = run(config_for(tmp_path), engine, DAY, version=None)
    assert rows == 0
    assert io.exists(features_partition(str(tmp_path), "v1", DAY))


def test_the_features_carry_no_excluded_measure(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    run(config_for(tmp_path), engine, DAY, version=None)
    features = io.read_frames([features_partition(str(tmp_path), "v1", DAY)])
    assert features["consumption_kw"].notna().all()
    assert pd.api.types.is_float_dtype(features["consumption_kw"])


class TestParseArgs:
    """La ligne de commande dit ce que le run produit."""

    def test_the_date_defaults_to_today(self) -> None:
        # La boucle du conteneur appelle `python -m etl` sans argument : une
        # date obligatoire la faisait sortir en erreur à chaque cycle, donc
        # ne publiait jamais rien.
        assert parse_args([]).date is None

    def test_a_single_day_is_produced_by_default(self) -> None:
        assert parse_args([]).days == DEFAULT_DAYS

    def test_the_window_can_be_widened(self) -> None:
        assert parse_args(["--days", "2"]).days == 2

    def test_the_version_can_be_forced(self) -> None:
        args = parse_args(["--date", "2026-09-10", "--feature-version", "v2"])
        assert args.feature_version == "v2"

    def test_the_side_output_flag_is_gone(self) -> None:
        # La base n'est plus une sortie annexe : elle est la couche brute, et
        # l'écriture de retour n'est plus optionnelle.
        with pytest.raises(SystemExit):
            parse_args(["--date", "2026-09-10", "--load-db"])
