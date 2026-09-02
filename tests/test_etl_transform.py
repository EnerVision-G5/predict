"""Normalisation des mesures brutes en tableau prêt pour la base."""

import pandas as pd
import pytest

from etl.transform import (
    MEASURE_COLUMNS,
    TransformError,
    deduplicate,
    to_frame,
)


def test_to_frame_returns_measure_columns_on_empty_batch() -> None:
    frame = to_frame([])
    assert list(frame.columns) == list(MEASURE_COLUMNS)
    assert frame.empty


def test_to_frame_maps_timestamp_to_ts_in_utc(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:30:00+02:00")])
    assert str(frame.loc[0, "ts"].tz) == "UTC"
    assert frame.loc[0, "ts"].hour == 6


def test_to_frame_keeps_missing_values(make_reading) -> None:
    frame = to_frame(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
                data_quality="critical",
            )
        ]
    )
    assert pd.isna(frame.loc[0, "consumption_kw"])
    assert frame.loc[0, "null_reasons"] == ["sensor_failure"]
    assert frame.loc[0, "data_quality"] == "critical"


def test_to_frame_normalizes_null_reasons_to_a_list(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z", null_reasons=None)])
    assert frame.loc[0, "null_reasons"] == []


def test_to_frame_rejects_a_record_without_the_primary_key() -> None:
    with pytest.raises(TransformError):
        to_frame([{"consumption_kw": 12.0}])


def test_deduplicate_keeps_the_last_row_of_a_duplicated_key(make_reading) -> None:
    frame = to_frame(
        [
            make_reading("2026-01-15T08:00:00Z", consumption_kw=10.0),
            make_reading("2026-01-15T08:00:00Z", consumption_kw=20.0),
            make_reading("2026-01-15T09:00:00Z", consumption_kw=30.0),
        ]
    )
    deduplicated = deduplicate(frame)
    assert len(deduplicated) == 2
    assert deduplicated.loc[0, "consumption_kw"] == 20.0


def test_deduplicate_drops_rows_without_a_usable_timestamp(make_reading) -> None:
    frame = to_frame([make_reading("pas une date")])
    assert deduplicate(frame).empty
