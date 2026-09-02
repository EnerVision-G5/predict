"""Construction du jeu d'apprentissage."""

import pandas as pd
import pytest

from training.dataset import (
    FEATURE_COLUMNS,
    LAG_HOURS,
    TARGET_COLUMN,
    build_features,
    split_train_test,
)

HOURS = max(LAG_HOURS) + 48


def hourly_measures(hours: int = HOURS) -> pd.DataFrame:
    """Série horaire régulière, suffisante pour renseigner tous les décalages."""
    timestamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": timestamps,
            "site_id": "SITE001",
            TARGET_COLUMN: [50.0 + index % 24 for index in range(hours)],
            "temperature_celsius": [15.0] * hours,
        }
    )


def test_build_features_adds_calendar_and_lag_columns() -> None:
    features = build_features(hourly_measures())
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert features["is_weekend"].isin([0, 1]).all()


def test_build_features_drops_rows_without_enough_history() -> None:
    features = build_features(hourly_measures())
    assert len(features) == HOURS - max(LAG_HOURS)
    assert not features[list(FEATURE_COLUMNS)].isna().to_numpy().any()


def test_build_features_returns_empty_when_history_is_too_short() -> None:
    assert build_features(hourly_measures(hours=max(LAG_HOURS))).empty


def test_split_train_test_keeps_the_chronological_order() -> None:
    features = build_features(hourly_measures())
    train, test = split_train_test(features, test_ratio=0.25)
    assert len(train) + len(test) == len(features)
    assert train["ts"].max() <= test["ts"].min()


def test_split_train_test_rejects_a_ratio_out_of_range() -> None:
    features = build_features(hourly_measures())
    with pytest.raises(ValueError):
        split_train_test(features, test_ratio=1.0)
