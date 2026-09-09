from __future__ import annotations

from etl.clean import deduplicate, to_measures
from etl.exclude import (
    DEFAULT_REASON,
    exclusion_reason,
    keep_usable,
    to_exclusions,
)
from etl.impute import impute_frame


def ingest(make_raw, readings):
    return impute_frame(deduplicate(to_measures(make_raw(readings))))


def test_to_exclusions_files_a_null_no_imputation_could_restore(
    make_raw, make_reading
) -> None:
    frame = ingest(
        make_raw,
        [
            make_reading(
                "2026-09-02T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
            )
        ],
    )
    exclusions = to_exclusions(frame)
    assert len(exclusions) == 1
    assert exclusions[0]["site_id"] == "SITE001"
    assert exclusions[0]["raison"] == "sensor_failure"


def test_to_exclusions_spares_a_measure_the_batch_could_rebuild(
    make_raw, make_reading
) -> None:
    frame = ingest(
        make_raw,
        [
            make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
            make_reading("2026-09-02T09:00:00Z", consumption_kw=None),
            make_reading("2026-09-02T10:00:00Z", consumption_kw=30.0),
        ],
    )
    assert to_exclusions(frame) == []


def test_to_exclusions_spares_a_complete_measure(make_raw, make_reading) -> None:
    frame = ingest(make_raw, [make_reading("2026-09-02T08:00:00Z")])
    assert to_exclusions(frame) == []


def test_to_exclusions_joins_every_motive_of_the_source(
    make_raw, make_reading
) -> None:
    frame = ingest(
        make_raw,
        [
            make_reading(
                "2026-09-02T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure", "network_loss"],
            )
        ],
    )
    assert to_exclusions(frame)[0]["raison"] == "sensor_failure, network_loss"


def test_to_exclusions_falls_back_on_its_own_observation(
    make_raw, make_reading
) -> None:
    frame = ingest(
        make_raw,
        [make_reading("2026-09-02T08:00:00Z", consumption_kw=None, null_reasons=[])],
    )
    assert to_exclusions(frame)[0]["raison"] == "consumption_kw:undeclared"


def test_to_exclusions_returns_nothing_for_an_empty_batch(make_raw) -> None:
    assert to_exclusions(impute_frame(to_measures(make_raw([])))) == []


def test_exclusion_reason_survives_a_batch_without_motives() -> None:
    assert exclusion_reason([]) == DEFAULT_REASON
    assert exclusion_reason(None) == DEFAULT_REASON


def test_exclusion_reason_reads_the_array_parquet_returns() -> None:
    import numpy

    assert exclusion_reason(numpy.array(["sensor_failure"], dtype=object)) == (
        "sensor_failure"
    )


def test_keep_usable_removes_what_no_aggregate_could_use(
    make_raw, make_reading
) -> None:
    frame = ingest(
        make_raw,
        [
            make_reading("2026-09-02T08:00:00Z", consumption_kw=None),
            make_reading("2026-09-02T09:00:00Z", consumption_kw=30.0),
        ],
    )
    usable = keep_usable(frame)
    assert len(usable) == 1
    assert usable.loc[0, "consumption_kw"] == 30.0


def test_keep_usable_keeps_a_reconstructed_measure(make_raw, make_reading) -> None:
    frame = ingest(
        make_raw,
        [
            make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
            make_reading("2026-09-02T09:00:00Z", consumption_kw=None),
            make_reading("2026-09-02T10:00:00Z", consumption_kw=30.0),
        ],
    )
    assert len(keep_usable(frame)) == 3


def test_keep_usable_accepts_an_empty_batch(make_raw) -> None:
    assert keep_usable(impute_frame(to_measures(make_raw([])))).empty
