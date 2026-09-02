"""Chargement idempotent des mesures.

Aucun test ne joint PostgreSQL : la garantie qui compte ici est que
l'instruction produite porte bien le ON CONFLICT sur la clé primaire composite,
et que les manquants pandas sortent en NULL.

Les lots passent par l'imputation avant le chargement, comme dans le pipeline
et le poller : ce sont les colonnes imputées, autant que les brutes, que la
table attend.
"""

import pytest
from sqlalchemy.dialects import postgresql

from conftest import FakeEngine
from etl.impute import impute_frame
from etl.load import LoadError, build_upsert, load_frame, to_records
from etl.transform import to_frame


def to_loadable(readings):
    """Fait traverser au lot les étages qui précèdent le chargement."""
    return impute_frame(to_frame(readings))


def test_to_records_converts_missing_values_to_none(make_reading) -> None:
    frame = to_loadable(
        [make_reading("2026-01-15T08:00:00Z", consumption_kw=None)]
    )
    record = to_records(frame)[0]
    assert record["consumption_kw"] is None
    assert record["null_reasons"] == ["consumption_kw:undeclared"]
    assert record["site_id"] == "SITE001"


def test_to_records_carries_the_imputation_columns(make_reading) -> None:
    frame = to_loadable([make_reading("2026-01-15T08:00:00Z")])
    record = to_records(frame)[0]
    assert record["consumption_kw"] == 87.34
    assert record["consumption_kw_imputed"] == 87.34
    assert record["imputation_method"] == "none"


def test_to_records_refuses_a_batch_that_skipped_imputation(
    make_reading,
) -> None:
    with pytest.raises(LoadError):
        to_records(to_frame([make_reading("2026-01-15T08:00:00Z")]))


def test_build_upsert_ignores_rows_already_loaded(make_reading) -> None:
    frame = to_loadable([make_reading("2026-01-15T08:00:00Z")])
    compiled = build_upsert(to_records(frame)).compile(
        dialect=postgresql.dialect()
    )
    assert "ON CONFLICT (site_id, ts) DO NOTHING" in str(compiled)


def test_load_frame_writes_nothing_for_an_empty_frame() -> None:
    engine = FakeEngine()
    assert load_frame(engine, to_loadable([]), batch_size=10) == 0
    assert engine.executed == []


def test_load_frame_splits_the_batch(make_reading) -> None:
    frame = to_loadable(
        [
            make_reading(f"2026-01-15T0{hour}:00:00Z")
            for hour in range(5)
        ]
    )
    engine = FakeEngine()
    assert load_frame(engine, frame, batch_size=2) == 5
    # 5 lignes par lots de 2 : trois instructions, la dernière incomplète.
    assert len(engine.executed) == 3


def test_load_frame_reports_submitted_rows(make_reading) -> None:
    frame = to_loadable([make_reading("2026-01-15T08:00:00Z")])
    engine = FakeEngine()
    assert load_frame(engine, frame, batch_size=10) == len(frame)
    assert len(engine.executed) == 1
