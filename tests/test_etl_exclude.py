"""Rangement des mesures inexploitables dans `mesure_exclu`.

Aucun test ne joint PostgreSQL. Ce qui compte ici est le tri — quelles mesures
sont écartées, lesquelles ne le sont pas — et le fait que le rejeu d'une même
fenêtre ne réécrive pas une exclusion déjà posée.
"""

import pandas as pd
from sqlalchemy.dialects import postgresql

from conftest import FakeEngine
from etl.exclude import (
    DEFAULT_REASON,
    build_exclusion_upsert,
    exclusion_reason,
    load_exclusions,
    to_exclusions,
)
from etl.impute import impute_frame
from etl.transform import deduplicate, to_frame


def to_excludable(readings):
    """Fait traverser au lot les étages qui précèdent l'exclusion."""
    return impute_frame(deduplicate(to_frame(readings)))


def test_to_exclusions_files_a_null_no_imputation_could_restore(
    make_reading,
) -> None:
    frame = to_excludable(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
            )
        ]
    )
    exclusions = to_exclusions(frame)
    assert len(exclusions) == 1
    assert exclusions[0]["site_id"] == "SITE001"
    assert exclusions[0]["raison"] == "sensor_failure"


def test_to_exclusions_spares_a_measure_the_batch_could_rebuild(
    make_reading,
) -> None:
    frame = to_excludable(
        [
            make_reading("2026-01-15T08:00:00Z", consumption_kw=10.0),
            make_reading("2026-01-15T09:00:00Z", consumption_kw=None),
            make_reading("2026-01-15T10:00:00Z", consumption_kw=30.0),
        ]
    )
    assert to_exclusions(frame) == []


def test_to_exclusions_spares_a_complete_measure(make_reading) -> None:
    frame = to_excludable([make_reading("2026-01-15T08:00:00Z")])
    assert to_exclusions(frame) == []


def test_to_exclusions_joins_every_motive_of_the_source(make_reading) -> None:
    frame = to_excludable(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure", "network_loss"],
            )
        ]
    )
    assert to_exclusions(frame)[0]["raison"] == "sensor_failure, network_loss"


def test_to_exclusions_falls_back_on_its_own_observation(make_reading) -> None:
    frame = to_excludable(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                consumption_kw=None,
                null_reasons=[],
            )
        ]
    )
    # La normalisation a nommé le capteur muet, l'exclusion en hérite.
    assert to_exclusions(frame)[0]["raison"] == "consumption_kw:undeclared"


def test_to_exclusions_returns_nothing_for_an_empty_batch() -> None:
    assert to_exclusions(impute_frame(to_frame([]))) == []


def test_exclusion_reason_survives_a_batch_without_motives() -> None:
    assert exclusion_reason([]) == DEFAULT_REASON
    assert exclusion_reason(None) == DEFAULT_REASON


def test_build_exclusion_upsert_ignores_an_exclusion_already_filed() -> None:
    records = [
        {
            "site_id": "SITE001",
            "ts": pd.Timestamp("2026-01-15T08:00:00Z").to_pydatetime(),
            "raison": "sensor_failure",
        }
    ]
    compiled = build_exclusion_upsert(records).compile(
        dialect=postgresql.dialect()
    )
    assert "ON CONFLICT (site_id, ts) DO NOTHING" in str(compiled)


def test_load_exclusions_writes_nothing_when_nothing_is_excluded(
    make_reading,
) -> None:
    engine = FakeEngine()
    frame = to_excludable([make_reading("2026-01-15T08:00:00Z")])
    assert load_exclusions(engine, frame, batch_size=10) == 0
    assert engine.executed == []


def test_load_exclusions_splits_the_batch(make_reading) -> None:
    frame = to_excludable(
        [
            make_reading(f"2026-01-15T0{hour}:00:00Z", consumption_kw=None)
            for hour in range(5)
        ]
    )
    engine = FakeEngine()
    assert load_exclusions(engine, frame, batch_size=2) == 5
    assert len(engine.executed) == 3
