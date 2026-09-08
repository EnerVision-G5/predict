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
from etl_fakes import FakeEngine
from sqlalchemy.exc import OperationalError

from etl import __main__ as main_module
from etl.__main__ import (
    DEFAULT_DAYS,
    EXIT_FAILED,
    db_error_line,
    feature_spec,
    is_db_warming_up,
    main,
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
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    run(config_for(tmp_path), engine, DAY, version=None)
    spec = feature_spec(config_for(tmp_path))
    queried = engine.queried[0]
    assert (queried["end_time"] - queried["start_time"]).days == spec.lookback_days


def test_run_reposes_only_the_produced_day(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
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


def test_an_empty_run_keeps_the_partition_it_would_have_erased(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    assert run(config_for(tmp_path), engine, DAY, version=None) == 24
    partition = features_partition(str(tmp_path), "v1", DAY)

    vide = ReadingEngine(
        measures(
            make_raw,
            make_reading,
            consumption_kw=None,
            null_reasons=["sensor_failure"],
        )
    )
    patched_read(vide)
    assert run(config_for(tmp_path), vide, DAY, version=None) == 0
    assert len(io.read_frames([partition])) == 24


def test_publish_reports_that_nothing_was_written(
    tmp_path: Path, make_raw, make_reading, patched_read
) -> None:
    engine = ReadingEngine(measures(make_raw, make_reading))
    patched_read(engine)
    run(config_for(tmp_path), engine, DAY, version=None)
    spec = feature_spec(config_for(tmp_path))
    assert publish(pd.DataFrame(), str(tmp_path), spec, DAY) is None


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
        assert parse_args([]).date is None

    def test_a_single_day_is_produced_by_default(self) -> None:
        assert parse_args([]).days == DEFAULT_DAYS

    def test_the_window_can_be_widened(self) -> None:
        assert parse_args(["--days", "2"]).days == 2

    def test_the_version_can_be_forced(self) -> None:
        args = parse_args(["--date", "2026-09-10", "--feature-version", "v2"])
        assert args.feature_version == "v2"

    def test_the_side_output_flag_is_gone(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--date", "2026-09-10", "--load-db"])


class TestADatabaseStillStartingUp:
    """Le conteneur repart avant que TimescaleDB ait rejoué son WAL.

    Au redémarrage du poste, le démon Docker relance tous les conteneurs
    ensemble sans lire les `depends_on` du compose — ceux-ci n'ordonnent que
    `compose up`. Le cycle suivant passera ; la trace du driver à cet endroit
    faisait chercher un bug là où il n'y a qu'un ordre de démarrage.
    """

    def failing_open(self, monkeypatch, tmp_path: Path, error: Exception) -> None:
        """Fait échouer l'ouverture du moteur sur l'erreur donnée."""

        def refuse(*args, **kwargs):
            raise error

        monkeypatch.setattr(main_module, "load_config", lambda: config_for(tmp_path))
        monkeypatch.setattr(main_module, "open_engine", refuse)

    def test_it_is_one_information_line_without_a_trace(
        self, monkeypatch, tmp_path: Path, caplog
    ) -> None:
        origin = Exception(
            'connection failed: connection to server at "172.24.0.4", port'
            " 5432 failed: FATAL:  the database system is starting up"
        )
        self.failing_open(monkeypatch, tmp_path, OperationalError("", {}, origin))

        with caplog.at_level("INFO", logger="etl.__main__"):
            code = main([])

        assert code == EXIT_FAILED
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "INFO"
        assert caplog.records[0].exc_info is None
        assert "base en attente" in caplog.text

    def test_an_unreachable_database_stays_an_error_on_one_line(
        self, monkeypatch, tmp_path: Path, caplog
    ) -> None:
        origin = Exception("connection refused\nseconde ligne du driver")
        self.failing_open(monkeypatch, tmp_path, OperationalError("", {}, origin))

        with caplog.at_level("INFO", logger="etl.__main__"):
            code = main([])

        assert code == EXIT_FAILED
        assert caplog.records[0].levelname == "ERROR"
        assert "base injoignable : connection refused" in caplog.text
        assert "sqlalche.me" not in caplog.text
        assert "seconde ligne" not in caplog.text


class TestDatabaseErrorReading:
    """Les deux prédicats qui décident du ton du journal."""

    def test_the_startup_states_of_postgres_are_recognised(self) -> None:
        assert is_db_warming_up(Exception("FATAL: the database system is starting up"))
        assert is_db_warming_up(Exception("The Database System Is Shutting Down"))

    def test_a_refused_connection_is_not_a_startup(self) -> None:
        assert not is_db_warming_up(Exception("connection refused"))

    def test_the_driver_reason_fits_on_one_line(self) -> None:
        origin = Exception("connection failed\nseconde ligne")
        assert db_error_line(OperationalError("", {}, origin)) == "connection failed"

    def test_an_error_without_a_message_is_named_by_its_type(self) -> None:
        assert db_error_line(TimeoutError()) == "TimeoutError"
