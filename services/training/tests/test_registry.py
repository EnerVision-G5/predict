from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.dialects import postgresql
from training_fakes import FakeEngine

from training.registry import (
    build_demotion,
    build_upsert,
    publish_champion,
    to_record,
)

MODEL = "enervision_xgboost"
RUN_ID = "0123456789abcdef0123456789abcdef"
TRAINED_AT = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)


def compile_statement(statement: object) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def compiled_upsert(version: str = "3") -> str:
    record = to_record(MODEL, version, RUN_ID, TRAINED_AT)
    return compile_statement(build_upsert(record))


def test_the_promoted_version_is_recorded_active() -> None:
    record = to_record(MODEL, "3", RUN_ID, TRAINED_AT)
    assert record["nom"] == MODEL
    assert record["version"] == "3"
    assert record["mlflow_run_id"] == RUN_ID
    assert record["actif"] is True


def test_the_recorded_date_is_the_run_not_the_promotion() -> None:
    record = to_record(MODEL, "3", RUN_ID, TRAINED_AT)
    assert record["date_entrainement"] == TRAINED_AT


def test_a_registry_without_run_id_records_null_not_empty_text() -> None:
    assert to_record(MODEL, "3", "", TRAINED_AT)["mlflow_run_id"] is None


def test_a_version_already_known_is_reposed_not_duplicated() -> None:
    compiled = compiled_upsert()
    assert "ON CONFLICT (nom, version) DO UPDATE" in compiled
    assert "actif = excluded.actif" in compiled
    assert "mlflow_run_id = excluded.mlflow_run_id" in compiled
    assert "date_entrainement = excluded.date_entrainement" in compiled


def test_the_upsert_never_rewrites_the_key_columns() -> None:
    compiled = compiled_upsert()
    assert "nom = excluded.nom" not in compiled
    assert "version = excluded.version" not in compiled


def test_the_demotion_spares_the_promoted_version() -> None:
    compiled = compile_statement(build_demotion(MODEL, "3"))
    assert "UPDATE modele SET actif" in compiled
    assert "modele.version !=" in compiled
    assert "modele.nom =" in compiled


def test_the_demotion_stays_within_one_model() -> None:
    compiled = compile_statement(build_demotion(MODEL, "3"))
    assert compiled.count("modele.nom =") == 1


def test_publishing_sends_both_statements_in_one_transaction() -> None:
    engine = FakeEngine()
    publish_champion(engine, MODEL, "3", RUN_ID, TRAINED_AT)
    assert engine.transactions == 1
    assert len(engine.executed) == 2


def test_publishing_records_before_it_demotes() -> None:
    engine = FakeEngine()
    publish_champion(engine, MODEL, "3", RUN_ID, TRAINED_AT)
    inserted, demoted = (compile_statement(item) for item in engine.executed)
    assert "INSERT INTO modele" in inserted
    assert "UPDATE modele" in demoted


def test_publishing_carries_the_version_as_text() -> None:
    assert to_record(MODEL, 3, RUN_ID, TRAINED_AT)["version"] == "3"


def test_promoting_twice_writes_the_same_thing() -> None:
    first, second = FakeEngine(), FakeEngine()
    publish_champion(first, MODEL, "3", RUN_ID, TRAINED_AT)
    publish_champion(second, MODEL, "3", RUN_ID, TRAINED_AT)
    assert [compile_statement(item) for item in first.executed] == [
        compile_statement(item) for item in second.executed
    ]
