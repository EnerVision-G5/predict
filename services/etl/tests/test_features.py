from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from etl.features import FeatureError, FeatureSpec, build, resample
from predict_common.schemas import (
    METHOD_INTERPOLATION,
    METHOD_NONE,
    QUALITY_CRITICAL,
    QUALITY_GOOD,
)

DAY = date(2026, 9, 8)
SPEC = FeatureSpec(
    version="v1", resample_rule="1h", lag_hours=(1, 24), rolling_window_h=3
)


def measures(
    hours: int,
    start: str = "2026-09-01T00:00:00Z",
    **columns,
) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": stamps,
            "site_id": "SITE001",
            "consumption_kw": [50.0 + index % 24 for index in range(hours)],
            "temperature_celsius": 20.0,
            "data_quality": QUALITY_GOOD,
            "consumption_kw_imputed": [50.0 + index % 24 for index in range(hours)],
            "imputation_method": METHOD_NONE,
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


class TestFeatureSpec:
    def test_the_lookback_follows_the_deepest_lag(self) -> None:
        spec = FeatureSpec("v1", "1h", (1, 24, 168), 24)
        assert spec.lookback_days == 9

    def test_the_grid_step_converts_hours_into_periods(self) -> None:
        assert FeatureSpec("v1", "1h", (1,), 24).periods_per_hour == 1
        assert FeatureSpec("v1", "15min", (1,), 24).periods_per_hour == 4

    def test_a_step_that_does_not_divide_the_hour_is_refused(self) -> None:
        spec = FeatureSpec("v1", "7min", (1,), 24)
        with pytest.raises(FeatureError):
            assert spec.periods_per_hour

    def test_the_columns_follow_the_shared_order(self) -> None:
        assert SPEC.columns == (
            "hour",
            "day_of_week",
            "is_weekend",
            "temperature_celsius",
            "lag_1h",
            "lag_24h",
            "roll_mean_3h",
        )


class TestResample:
    def test_minutes_are_averaged_into_an_hour(self) -> None:
        stamps = pd.date_range(
            "2026-09-08T00:00:00Z", periods=4, freq="15min", tz="UTC"
        )
        frame = pd.DataFrame(
            {
                "ts": stamps,
                "site_id": "SITE001",
                "consumption_kw": [10.0, 20.0, 30.0, 40.0],
                "consumption_kw_imputed": [10.0, 20.0, 30.0, 40.0],
                "temperature_celsius": 20.0,
                "data_quality": QUALITY_GOOD,
                "imputation_method": METHOD_NONE,
            }
        )
        hourly = resample(frame, SPEC)
        assert len(hourly) == 1
        assert hourly.loc[0, "consumption_kw"] == 25.0

    def test_the_target_comes_from_the_imputed_column(self) -> None:
        frame = measures(1)
        frame.loc[0, "consumption_kw"] = None
        frame.loc[0, "consumption_kw_imputed"] = 42.0
        frame.loc[0, "imputation_method"] = METHOD_INTERPOLATION
        assert resample(frame, SPEC).loc[0, "consumption_kw"] == 42.0

    def test_an_hour_keeps_the_worst_quality_of_its_minutes(self) -> None:
        stamps = pd.date_range(
            "2026-09-08T00:00:00Z", periods=2, freq="30min", tz="UTC"
        )
        frame = pd.DataFrame(
            {
                "ts": stamps,
                "site_id": "SITE001",
                "consumption_kw": [10.0, 20.0],
                "consumption_kw_imputed": [10.0, 20.0],
                "temperature_celsius": 20.0,
                "data_quality": [QUALITY_GOOD, QUALITY_CRITICAL],
                "imputation_method": METHOD_NONE,
            }
        )
        assert resample(frame, SPEC).loc[0, "data_quality"] == QUALITY_CRITICAL

    def test_the_imputed_ratio_is_the_share_of_rebuilt_minutes(self) -> None:
        stamps = pd.date_range(
            "2026-09-08T00:00:00Z", periods=4, freq="15min", tz="UTC"
        )
        frame = pd.DataFrame(
            {
                "ts": stamps,
                "site_id": "SITE001",
                "consumption_kw": [10.0, 20.0, 30.0, 40.0],
                "consumption_kw_imputed": [10.0, 20.0, 30.0, 40.0],
                "temperature_celsius": 20.0,
                "data_quality": QUALITY_GOOD,
                "imputation_method": [
                    METHOD_NONE,
                    METHOD_INTERPOLATION,
                    METHOD_NONE,
                    METHOD_NONE,
                ],
            }
        )
        assert resample(frame, SPEC).loc[0, "imputed_ratio"] == 0.25

    def test_sites_are_resampled_apart(self) -> None:
        first = measures(2)
        second = measures(2)
        second["site_id"] = "SITE002"
        hourly = resample(pd.concat([first, second], ignore_index=True), SPEC)
        assert set(hourly["site_id"]) == {"SITE001", "SITE002"}
        assert len(hourly) == 4

    def test_an_empty_batch_keeps_its_columns(self) -> None:
        assert "site_id" in resample(pd.DataFrame(), SPEC).columns


class TestBuild:
    def test_only_the_requested_day_is_returned(self) -> None:
        features = build(measures(24 * 9), SPEC, DAY)
        assert set(features["ts"].dt.date) == {DAY}

    def test_the_lags_point_at_the_right_hour(self) -> None:
        features = build(measures(24 * 9), SPEC, DAY)
        row = features.iloc[5]
        earlier = features.iloc[4]
        assert row["lag_1h"] == earlier["consumption_kw"]

    def test_a_gap_in_the_series_does_not_shift_the_lags(self) -> None:
        frame = measures(24 * 9)
        gap = frame["ts"].between("2026-09-05T00:00:00Z", "2026-09-05T05:00:00Z")
        with_gap = frame[~gap].reset_index(drop=True)
        features = build(with_gap, SPEC, DAY)
        row = features.iloc[10]
        expected = with_gap[with_gap["ts"] == row["ts"] - pd.Timedelta(hours=1)]
        assert row["lag_1h"] == pytest.approx(expected["consumption_kw"].iloc[0])

    def test_the_rolling_mean_excludes_the_hour_it_describes(self) -> None:
        features = build(measures(24 * 9), SPEC, DAY)
        row = features.iloc[5]
        previous = features.iloc[2:5]["consumption_kw"]
        assert row["roll_mean_3h"] == pytest.approx(previous.mean())

    def test_an_hour_without_enough_history_is_dropped(self) -> None:
        features = build(measures(24 * 2, start="2026-09-07T00:00:00Z"), SPEC, DAY)
        assert features["lag_24h"].notna().all()

    def test_a_site_without_a_thermometer_keeps_its_hours(self) -> None:
        frame = measures(24 * 9)
        frame["temperature_celsius"] = None
        assert not build(frame, SPEC, DAY).empty

    def test_an_empty_batch_returns_the_declared_columns(self) -> None:
        features = build(pd.DataFrame(), SPEC, DAY)
        assert set(SPEC.columns) <= set(features.columns)
        assert features.empty

    def test_the_calendar_columns_describe_the_hour(self) -> None:
        features = build(measures(24 * 9), SPEC, DAY)
        row = features.iloc[8]
        assert row["hour"] == row["ts"].hour
        assert row["day_of_week"] == row["ts"].dayofweek
        assert row["is_weekend"] in (0, 1)

    def test_two_sites_are_derived_apart(self) -> None:
        first = measures(24 * 9)
        second = measures(24 * 9)
        second["site_id"] = "SITE002"
        features = build(pd.concat([first, second], ignore_index=True), SPEC, DAY)
        assert set(features["site_id"]) == {"SITE001", "SITE002"}
        assert not features.duplicated(subset=["site_id", "ts"]).any()


def test_build_says_which_lag_emptied_the_day(caplog) -> None:
    spec = FeatureSpec(version="v1", resample_rule="1h", lag_hours=(1, 168),
                       rolling_window_h=24)
    jeune = measures(72, start="2026-09-06T00:00:00Z")
    with caplog.at_level("WARNING"):
        assert build(jeune, spec, DAY).empty
    assert "lag_168h absent sur" in caplog.text

