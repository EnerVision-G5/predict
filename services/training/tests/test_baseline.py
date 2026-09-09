from __future__ import annotations

import pandas as pd
import pytest

from training.baseline import BaselineError, Persistence, naive_baselines


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lag_1h": [10.0, 11.0, 12.0],
            "lag_24h": [20.0, 21.0, 22.0],
            "consumption_kw": [30.0, 31.0, 32.0],
        }
    )


class TestPersistence:
    def test_the_name_carries_the_lag(self) -> None:
        assert Persistence(hours=24).name == "persistance-24h"

    def test_the_column_is_the_one_the_etl_publishes(self) -> None:
        assert Persistence(hours=168).column == "lag_168h"

    def test_it_predicts_the_observed_value(self) -> None:
        predicted = Persistence(hours=24).predict(frame())
        assert list(predicted) == [20.0, 21.0, 22.0]

    def test_a_missing_column_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="lag_168h"):
            Persistence(hours=168).predict(frame())


def test_one_baseline_per_published_lag() -> None:
    baselines = naive_baselines([1, 24, 168])
    assert [baseline.hours for baseline in baselines] == [1, 24, 168]


def test_no_lag_gives_no_baseline() -> None:
    assert naive_baselines([]) == ()
