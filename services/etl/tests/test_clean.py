"""Normalisation de la couche brute en tableau de mesures typé."""

from __future__ import annotations

import pandas as pd
import pytest

from etl.clean import MEASURE_COLUMNS, CleanError, deduplicate, to_measures


def test_to_measures_returns_the_measure_columns_on_an_empty_batch(
    make_raw,
) -> None:
    frame = to_measures(make_raw([]))
    assert list(frame.columns) == list(MEASURE_COLUMNS)
    assert frame.empty


def test_to_measures_renames_timestamp_to_ts(make_raw, make_reading) -> None:
    # Une seule frontière de renommage dans toute la chaîne, et c'est celle-ci.
    frame = to_measures(make_raw([make_reading("2026-09-02T08:00:00Z")]))
    assert "ts" in frame.columns
    assert "timestamp" not in frame.columns


def test_to_measures_keeps_the_timestamp_in_utc(make_raw, make_reading) -> None:
    frame = to_measures(make_raw([make_reading("2026-09-02T08:30:00+02:00")]))
    assert str(frame.loc[0, "ts"].tz) == "UTC"
    assert frame.loc[0, "ts"].hour == 6


def test_to_measures_keeps_missing_values(make_raw, make_reading) -> None:
    frame = to_measures(
        make_raw(
            [
                make_reading(
                    "2026-09-02T08:00:00Z",
                    consumption_kw=None,
                    null_reasons=["sensor_failure"],
                    data_quality="critical",
                )
            ]
        )
    )
    assert pd.isna(frame.loc[0, "consumption_kw"])
    assert frame.loc[0, "null_reasons"] == ["sensor_failure"]
    assert frame.loc[0, "data_quality"] == "critical"


def test_to_measures_normalizes_null_reasons_to_a_list(
    make_raw, make_reading
) -> None:
    raw = make_raw([make_reading("2026-09-02T08:00:00Z", null_reasons=None)])
    frame = to_measures(raw)
    assert frame.loc[0, "null_reasons"] == []


def test_to_measures_accepts_the_array_parquet_returns(
    make_raw, make_reading
) -> None:
    # Une colonne list<string> relue par pyarrow est un tableau numpy, pas une
    # liste : la refuser refuserait ce que le collecteur a écrit valide.
    import numpy

    raw = make_raw([make_reading("2026-09-02T08:00:00Z")])
    raw["null_reasons"] = [numpy.array(["network_loss"], dtype=object)]
    frame = to_measures(raw)
    assert frame.loc[0, "null_reasons"] == ["network_loss"]


def test_to_measures_rejects_a_partition_without_the_key(make_raw) -> None:
    with pytest.raises(CleanError):
        to_measures(pd.DataFrame([{"consumption_kw": 12.0}]))


def test_deduplicate_keeps_the_last_row_of_a_duplicated_key(
    make_raw, make_reading
) -> None:
    # Le rattrapage et le poller écrivent la même journée : la même minute
    # peut donc arriver deux fois, et /readings fait autorité.
    frame = to_measures(
        make_raw(
            [
                make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
                make_reading("2026-09-02T08:00:00Z", consumption_kw=20.0),
                make_reading("2026-09-02T09:00:00Z", consumption_kw=30.0),
            ]
        )
    )
    deduplicated = deduplicate(frame)
    assert len(deduplicated) == 2
    assert deduplicated.loc[0, "consumption_kw"] == 20.0


def test_deduplicate_separates_two_sites_at_the_same_instant(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw(
            [
                make_reading("2026-09-02T08:00:00Z", site_id="SITE001"),
                make_reading("2026-09-02T08:00:00Z", site_id="SITE002"),
            ]
        )
    )
    assert len(deduplicate(frame)) == 2


def test_deduplicate_drops_rows_without_a_usable_timestamp(
    make_raw, make_reading
) -> None:
    frame = to_measures(make_raw([make_reading("pas une date")]))
    assert deduplicate(frame).empty
