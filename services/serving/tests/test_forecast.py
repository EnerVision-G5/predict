from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from predict_common import io
from predict_common.paths import features_partition
from predict_common.schemas import (
    SITE_COLUMN,
    feature_columns,
    features_arrow_schema,
)
from serving import forecast
from serving.forecast import (
    CONFIDENCE_Z,
    ForecastSpec,
    NoHistory,
    build_row,
    confidence_band,
    horizon_stamps,
    predict_series,
    read_history,
    site_history,
)

LAGS = (1, 24)
WINDOW = 2
COLUMNS = feature_columns(LAGS, WINDOW)
TODAY = date(2026, 9, 10)


def spec_for(root: Path) -> ForecastSpec:
    return ForecastSpec(
        root=str(root),
        feature_version="v1",
        lag_hours=LAGS,
        rolling_window_h=WINDOW,
        lookback_days=3,
    )


def features(day: date, site_id: str = "SITE001") -> pd.DataFrame:
    stamps = pd.date_range(
        f"{day.isoformat()}T00:00:00Z", periods=24, freq="h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "ts": stamps,
            "site_id": site_id,
            "consumption_kw": [50.0 + hour for hour in range(24)],
            "hour": stamps.hour,
            "day_of_week": stamps.dayofweek,
            "is_weekend": (stamps.dayofweek >= 5).astype(int),
            "temperature_celsius": 20.0,
            "lag_1h": 50.0,
            "lag_24h": 50.0,
            "roll_mean_2h": 50.0,
            "data_quality": "good",
            "imputed_ratio": 0.0,
        }
    )


def seed(root: Path, days: int = 3, site_id: str = "SITE001") -> None:
    for offset in range(days):
        day = TODAY - timedelta(days=offset)
        io.write_frame(
            features(day, site_id=site_id),
            features_partition(str(root), "v1", day),
            schema=features_arrow_schema(LAGS, WINDOW),
        )


def series(hours: int = 48) -> pd.Series:
    index = pd.date_range("2026-09-08T00:00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.Series([50.0 + index_ % 24 for index_ in range(hours)], index=index)


def test_read_history_gathers_the_recent_partitions(tmp_path: Path) -> None:
    seed(tmp_path)
    assert len(read_history(spec_for(tmp_path), today=TODAY)) == 72


def test_read_history_tolerates_a_missing_night(tmp_path: Path) -> None:
    seed(tmp_path, days=1)
    assert len(read_history(spec_for(tmp_path), today=TODAY)) == 24


def test_read_history_returns_nothing_when_no_partition_exists(
    tmp_path: Path,
) -> None:
    assert read_history(spec_for(tmp_path), today=TODAY).empty


def test_site_history_indexes_the_series_on_time(tmp_path: Path) -> None:
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    history = site_history(frame, "SITE001", spec_for(tmp_path))
    assert isinstance(history.index, pd.DatetimeIndex)
    assert len(history) == 72


def test_site_history_refuses_an_unknown_site(tmp_path: Path) -> None:
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    with pytest.raises(NoHistory):
        site_history(frame, "SITE404", spec_for(tmp_path))


def test_site_history_refuses_an_empty_read(tmp_path: Path) -> None:
    with pytest.raises(NoHistory):
        site_history(pd.DataFrame(), "SITE001", spec_for(tmp_path))


def test_site_history_leaves_a_gap_as_a_gap(tmp_path: Path) -> None:
    seed(tmp_path)
    frame = read_history(spec_for(tmp_path), today=TODAY)
    without = frame[frame["ts"].dt.hour != 5]
    history = site_history(without, "SITE001", spec_for(tmp_path))
    assert history.isna().any()


def test_horizon_stamps_follow_the_last_observed_hour() -> None:
    stamps = horizon_stamps(series(), 3)
    assert stamps[0] == series().index.max() + pd.Timedelta(hours=1)
    assert len(stamps) == 3


class TestBuildRow:
    def test_the_calendar_columns_are_integers(self, tmp_path: Path) -> None:
        row = build_row(series(), horizon_stamps(series(), 1)[0], spec_for(tmp_path))
        assert isinstance(row["hour"], int)
        assert isinstance(row["day_of_week"], int)

    def test_the_lags_are_read_from_the_history(self, tmp_path: Path) -> None:
        history = series()
        stamp = horizon_stamps(history, 1)[0]
        row = build_row(history, stamp, spec_for(tmp_path))
        assert row["lag_1h"] == history.loc[stamp - pd.Timedelta(hours=1)]
        assert row["lag_24h"] == history.loc[stamp - pd.Timedelta(hours=24)]

    def test_the_future_temperature_is_not_presented(self, tmp_path: Path) -> None:
        row = build_row(series(), horizon_stamps(series(), 1)[0], spec_for(tmp_path))
        assert "temperature_celsius" not in row

    def test_a_missing_lag_gives_no_row(self, tmp_path: Path) -> None:
        short = series(hours=2)
        assert build_row(short, horizon_stamps(short, 1)[0], spec_for(tmp_path)) is None

    def test_the_rolling_window_stops_before_the_predicted_hour(
        self, tmp_path: Path
    ) -> None:
        history = series()
        stamp = horizon_stamps(history, 1)[0]
        row = build_row(history, stamp, spec_for(tmp_path))
        expected = history.iloc[-WINDOW:].mean()
        assert row["roll_mean_2h"] == pytest.approx(expected)


class TestPredictSeries:
    def test_the_horizon_is_served_in_full(self, tmp_path: Path) -> None:
        points = predict_series(
            lambda frame: [42.0], series(), 6, spec_for(tmp_path), COLUMNS
        )
        assert len(points) == 6

    def test_a_prediction_becomes_the_next_lag(self, tmp_path: Path) -> None:
        seen: list[float] = []

        def predict(frame: pd.DataFrame) -> list[float]:
            seen.append(float(frame["lag_1h"].iloc[0]))
            return [99.0]

        predict_series(predict, series(), 3, spec_for(tmp_path), COLUMNS)
        assert seen[1] == 99.0
        assert seen[2] == 99.0

    def test_the_columns_are_presented_in_the_signature_order(
        self, tmp_path: Path
    ) -> None:
        seen: list[list[str]] = []

        def predict(frame: pd.DataFrame) -> list[float]:
            seen.append(list(frame.columns))
            return [42.0]

        predict_series(predict, series(), 1, spec_for(tmp_path), COLUMNS)
        assert seen[0] == list(COLUMNS)

    def test_the_horizon_stops_when_history_runs_out(self, tmp_path: Path) -> None:
        points = predict_series(
            lambda frame: [42.0], series(hours=2), 6, spec_for(tmp_path), COLUMNS
        )
        assert points == []


class TestConfidenceBand:
    def test_no_spread_gives_no_bounds(self) -> None:
        assert confidence_band(50.0, 1, None) == (None, None)

    def test_the_band_is_centred_on_the_prediction(self) -> None:
        lower, upper = confidence_band(50.0, 1, 2.0)
        assert (lower + upper) / 2 == pytest.approx(50.0)

    def test_the_first_step_is_the_plain_quantile(self) -> None:
        lower, upper = confidence_band(50.0, 1, 2.0)
        assert upper - 50.0 == pytest.approx(CONFIDENCE_Z * 2.0)

    def test_the_band_grows_as_the_square_root_of_the_step(self) -> None:
        first = confidence_band(50.0, 1, 2.0)
        fourth = confidence_band(50.0, 4, 2.0)
        assert (fourth[1] - fourth[0]) == pytest.approx(2 * (first[1] - first[0]))

    def test_the_lower_bound_is_not_clipped_at_zero(self) -> None:
        lower, _ = confidence_band(1.0, 1, 10.0)
        assert lower < 0.0


def test_la_fenetre_n_est_lue_qu_une_fois_dans_le_ttl(monkeypatch) -> None:
    forecast.reset_history_cache()
    reads = []

    def counting(spec, today=None):
        reads.append(today)
        return pd.DataFrame({SITE_COLUMN: ["SITE001"]})

    monkeypatch.setattr(forecast, "read_history", counting)
    spec = ForecastSpec(
        root="data", feature_version="v1", lag_hours=(1,),
        rolling_window_h=24, lookback_days=10,
    )

    first = forecast.cached_history(spec, ttl_s=300.0, today=date(2026, 9, 4))
    second = forecast.cached_history(spec, ttl_s=300.0, today=date(2026, 9, 4))

    assert len(reads) == 1
    assert first is second
    forecast.reset_history_cache()


def test_un_ttl_nul_relit_a_chaque_fois(monkeypatch) -> None:
    forecast.reset_history_cache()
    reads = []

    def counting(spec, today=None):
        reads.append(today)
        return pd.DataFrame({SITE_COLUMN: ["SITE001"]})

    monkeypatch.setattr(forecast, "read_history", counting)
    spec = ForecastSpec(
        root="data", feature_version="v1", lag_hours=(1,),
        rolling_window_h=24, lookback_days=10,
    )

    forecast.cached_history(spec, ttl_s=0.0, today=date(2026, 9, 4))
    forecast.cached_history(spec, ttl_s=0.0, today=date(2026, 9, 4))

    assert len(reads) == 2


def test_un_changement_de_journee_invalide_le_cache(monkeypatch) -> None:
    forecast.reset_history_cache()
    reads = []

    def counting(spec, today=None):
        reads.append(today)
        return pd.DataFrame({SITE_COLUMN: ["SITE001"]})

    monkeypatch.setattr(forecast, "read_history", counting)
    spec = ForecastSpec(
        root="data", feature_version="v1", lag_hours=(1,),
        rolling_window_h=24, lookback_days=10,
    )

    forecast.cached_history(spec, ttl_s=300.0, today=date(2026, 9, 4))
    forecast.cached_history(spec, ttl_s=300.0, today=date(2026, 9, 5))

    assert reads == [date(2026, 9, 4), date(2026, 9, 5)]
    forecast.reset_history_cache()


def spec_with_fallback(root: Path, years: int = 2) -> ForecastSpec:
    return ForecastSpec(
        root=str(root),
        feature_version="v1",
        lag_hours=LAGS,
        rolling_window_h=WINDOW,
        lookback_days=3,
        reference_years=years,
    )


def seed_reference(root: Path, years: int, days: int = 3) -> date:
    origin = date.today() - timedelta(
        days=forecast.REFERENCE_SHIFT_DAYS * years
    )
    for offset in range(days):
        day = origin - timedelta(days=offset)
        io.write_frame(
            features(day),
            features_partition(str(root), "v1", day),
            schema=features_arrow_schema(LAGS, WINDOW),
        )
    return origin


def test_le_repli_decale_de_semaines_entieres(tmp_path) -> None:
    assert forecast.REFERENCE_SHIFT_DAYS % 7 == 0
    origin = seed_reference(tmp_path, years=1)
    history, returned = forecast.reference_history(
        spec_with_fallback(tmp_path), "SITE001"
    )
    assert returned == origin
    assert not history.empty
    assert history.index.max().date() >= date.today() - timedelta(days=1)


def test_le_repli_ne_deborde_pas_dans_le_futur(tmp_path) -> None:
    seed_reference(tmp_path, years=1)
    history, _ = forecast.reference_history(
        spec_with_fallback(tmp_path), "SITE001"
    )
    now = pd.Timestamp.now(tz="UTC").floor("h")
    assert history.index.max() <= now


def test_le_repli_essaie_les_annees_dans_l_ordre(tmp_path) -> None:
    origin = seed_reference(tmp_path, years=2)
    _, returned = forecast.reference_history(
        spec_with_fallback(tmp_path, years=2), "SITE001"
    )
    assert returned == origin


def test_le_repli_desactive_refuse(tmp_path) -> None:
    seed_reference(tmp_path, years=1)
    with pytest.raises(NoHistory):
        forecast.reference_history(spec_with_fallback(tmp_path, years=0), "SITE001")


def test_le_repli_ignore_un_site_absent_de_la_reference(tmp_path) -> None:
    seed_reference(tmp_path, years=1)
    with pytest.raises(NoHistory):
        forecast.reference_history(spec_with_fallback(tmp_path), "SITE404")
