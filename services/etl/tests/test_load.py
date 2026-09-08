"""Écriture de retour dans `mesure` : ce que l'ETL a le droit de réécrire.

Aucun test ne joint PostgreSQL. Trois garanties comptent ici.

Le `DO UPDATE` ne porte que sur les colonnes déduites. C'est ce qui empêche
l'étage dont le métier est de décrire la panne d'effacer la panne elle-même :
même si le lot soumis portait une consommation différente de celle en base, la
base garderait celle de la source.

Il repose, il n'ignore pas. Contrairement au collecteur, l'ETL a quelque chose
de nouveau à dire sur une ligne déjà présente : sans `DO UPDATE`, corriger une
règle de qualification n'aurait aucun effet sur l'historique déjà traité.

Les manquants pandas sortent en NULL et non en NaN flottant, qu'une colonne
NUMERIC accepterait en polluant silencieusement les agrégats.
"""

from __future__ import annotations

import pandas as pd
import pytest
from etl_fakes import FakeEngine
from sqlalchemy.dialects import postgresql

from etl.clean import to_measures
from etl.exclude import to_exclusions
from etl.impute import impute_frame
from etl.load import (
    LoadError,
    build_exclusion_upsert,
    build_upsert,
    load,
    to_records,
)
from predict_common.db import DERIVED_COLUMNS, write_batches


def to_loadable(make_raw, readings):
    """Fait traverser au lot les étages qui précèdent le chargement."""
    return impute_frame(to_measures(make_raw(readings)))


def test_the_upsert_reposes_the_derived_columns() -> None:
    # Sans DO UPDATE, corriger une règle de qualification n'aurait aucun effet
    # sur l'historique déjà traité.
    from sqlalchemy.dialects import postgresql

    records = [
        {
            "ts": pd.Timestamp("2026-09-02T08:00:00Z").to_pydatetime(),
            "site_id": "SITE001",
            "consumption_kw": 10.0,
            "consumption_kwh": None,
            "voltage_v": None,
            "current_a": None,
            "power_factor": None,
            "temperature_celsius": None,
            "humidity_percent": None,
            "null_reasons": [],
            "data_quality": "good",
            "consumption_kw_imputed": 10.0,
            "imputation_method": "none",
        }
    ]
    compiled = str(build_upsert(records).compile(dialect=postgresql.dialect()))
    assert "DO UPDATE" in compiled
    for column in DERIVED_COLUMNS:
        assert f"{column} = excluded.{column}" in compiled


def test_the_upsert_never_touches_the_source_columns() -> None:
    # La panne capteur ne peut pas être effacée par l'étage qui a justement
    # pour métier de la décrire.
    from sqlalchemy.dialects import postgresql

    records = [
        {name: None for name in ("consumption_kwh", "voltage_v", "current_a")}
        | {
            "ts": pd.Timestamp("2026-09-02T08:00:00Z").to_pydatetime(),
            "site_id": "SITE001",
            "consumption_kw": 10.0,
            "power_factor": None,
            "temperature_celsius": None,
            "humidity_percent": None,
            "null_reasons": [],
            "data_quality": "good",
            "consumption_kw_imputed": 10.0,
            "imputation_method": "none",
        }
    ]
    updated = str(build_upsert(records).compile(dialect=postgresql.dialect()))
    assert "consumption_kw = excluded.consumption_kw" not in updated
    assert "voltage_v = excluded.voltage_v" not in updated


def test_to_records_converts_missing_values_to_none(make_raw, make_reading) -> None:
    frame = to_loadable(
        make_raw, [make_reading("2026-09-02T08:00:00Z", consumption_kw=None)]
    )
    record = to_records(frame)[0]
    assert record["consumption_kw"] is None
    assert record["null_reasons"] == ["consumption_kw:undeclared"]
    assert record["site_id"] == "SITE001"


def test_to_records_carries_the_imputation_columns(make_raw, make_reading) -> None:
    frame = to_loadable(make_raw, [make_reading("2026-09-02T08:00:00Z")])
    record = to_records(frame)[0]
    assert record["consumption_kw"] == 87.34
    assert record["consumption_kw_imputed"] == 87.34
    assert record["imputation_method"] == "none"


def test_to_records_refuses_a_batch_that_skipped_imputation(
    make_raw, make_reading
) -> None:
    # `imputation_method` est NOT NULL en base : une colonne silencieusement
    # absente ferait échouer l'insertion sans dire pourquoi.
    with pytest.raises(LoadError):
        to_records(to_measures(make_raw([make_reading("2026-09-02T08:00:00Z")])))


def test_build_upsert_targets_the_natural_key(make_raw, make_reading) -> None:
    frame = to_loadable(make_raw, [make_reading("2026-09-02T08:00:00Z")])
    compiled = build_upsert(to_records(frame)).compile(dialect=postgresql.dialect())
    assert "ON CONFLICT (site_id, ts) DO UPDATE" in str(compiled)


def test_build_exclusion_upsert_ignores_an_exclusion_already_filed() -> None:
    records = [
        {
            "site_id": "SITE001",
            "ts": pd.Timestamp("2026-09-02T08:00:00Z").to_pydatetime(),
            "raison": "sensor_failure",
        }
    ]
    compiled = build_exclusion_upsert(records).compile(dialect=postgresql.dialect())
    assert "ON CONFLICT (site_id, ts) DO NOTHING" in str(compiled)


def test_write_batches_writes_nothing_for_an_empty_batch() -> None:
    engine = FakeEngine()
    assert write_batches(engine, [], 10, build_upsert) == 0
    assert engine.executed == []


def test_write_batches_splits_the_batch(make_raw, make_reading) -> None:
    frame = to_loadable(
        make_raw, [make_reading(f"2026-09-02T0{hour}:00:00Z") for hour in range(5)]
    )
    engine = FakeEngine()
    assert write_batches(engine, to_records(frame), 2, build_upsert) == 5
    # 5 lignes par lots de 2 : trois instructions, la dernière incomplète.
    assert len(engine.executed) == 3


def test_load_writes_the_measures_before_their_exclusions(
    make_raw, make_reading
) -> None:
    # `mesure_exclu` porte une clé étrangère vers `mesure` : une exclusion
    # insérée avant sa mesure serait rejetée par la base.
    frame = to_loadable(
        make_raw,
        [
            make_reading(
                "2026-09-02T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
            )
        ],
    )
    engine = FakeEngine()
    rows, excluded = load(engine, frame, batch_size=10)
    assert (rows, excluded) == (1, 1)
    tables = [str(statement.table.name) for statement in engine.executed]
    assert tables == ["mesure", "mesure_exclu"]


def test_load_writes_no_exclusion_when_nothing_is_excluded(
    make_raw, make_reading
) -> None:
    frame = to_loadable(make_raw, [make_reading("2026-09-02T08:00:00Z")])
    engine = FakeEngine()
    assert load(engine, frame, batch_size=10) == (1, 0)
    assert len(engine.executed) == 1


def test_load_reports_submitted_rows_not_inserted_ones(
    make_raw, make_reading
) -> None:
    # ON CONFLICT DO NOTHING ne remonte pas les doublons ignorés : prétendre
    # compter les insertions fausserait le suivi d'ingestion.
    frame = to_loadable(make_raw, [make_reading("2026-09-02T08:00:00Z")])
    engine = FakeEngine()
    assert load(engine, frame, batch_size=10)[0] == len(frame)


def test_the_exclusions_carry_the_source_cause(make_raw, make_reading) -> None:
    frame = to_loadable(
        make_raw,
        [
            make_reading(
                "2026-09-02T08:00:00Z",
                consumption_kw=None,
                null_reasons=["network_loss"],
            )
        ],
    )
    assert to_exclusions(frame)[0]["raison"] == "network_loss"
