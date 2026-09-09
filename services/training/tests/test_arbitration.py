from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from predict_common.config import Config
from predict_common.schemas import feature_columns
from training.arbitration import (
    BENCH_WINDOW_PARAM,
    NAIVE_METRIC,
    ArbitrationError,
    Bench,
    BenchResult,
    bench_metrics,
    naive_reference,
    read_bench,
    read_naive,
    resolve_bench,
    score,
)
from training.baseline import Persistence

LAGS = (1, 24)
COLUMNS = feature_columns(LAGS, 2)
END = date(2026, 9, 10)
BENCH = Bench(start=date(2026, 9, 1), end=date(2026, 9, 3), pinned=False)


def config_for(start: str = "", end: str = "", days: int = 14) -> Config:
    return Config(
        values={
            "training": {
                "arbitration": {"start": start, "end": end, "days": days}
            }
        }
    )


def bench_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: [1.0, 2.0, 3.0, 4.0] for column in COLUMNS}
    )
    frame["lag_24h"] = [10.0, 20.0, 30.0, 40.0]
    frame["lag_1h"] = [11.0, 21.0, 31.0, 41.0]
    frame["consumption_kw"] = [10.0, 20.0, 30.0, 40.0]
    return frame


class TestResolveBench:
    def test_the_sliding_bench_ends_on_the_requested_day(self) -> None:
        bench = resolve_bench(config_for(days=3), END)
        assert (bench.start, bench.end) == (date(2026, 9, 8), END)
        assert not bench.pinned

    def test_the_pinned_bench_comes_from_the_configuration(self) -> None:
        bench = resolve_bench(config_for("2026-08-01", "2026-08-14"), END)
        assert (bench.start, bench.end) == (date(2026, 8, 1), date(2026, 8, 14))
        assert bench.pinned

    def test_a_half_written_bench_is_refused(self) -> None:
        with pytest.raises(ArbitrationError, match="ensemble"):
            resolve_bench(config_for(start="2026-08-01"), END)

    def test_an_inverted_bench_is_refused(self) -> None:
        with pytest.raises(ArbitrationError, match="précède"):
            resolve_bench(config_for("2026-08-14", "2026-08-01"), END)

    def test_an_empty_sliding_bench_is_refused(self) -> None:
        with pytest.raises(ArbitrationError, match="au moins un jour"):
            resolve_bench(config_for(days=0), END)


class TestBench:
    def test_the_label_is_what_travels_in_the_run(self) -> None:
        assert BENCH.label == "2026-09-01/2026-09-03"

    def test_the_bounds_are_included_in_the_count(self) -> None:
        assert BENCH.days == 3


def test_score_measures_any_predictor() -> None:
    result = score(Persistence(hours=24), bench_frame(), COLUMNS, "p24", BENCH)
    assert result.window == BENCH.label
    assert result.metrics["mae"] == pytest.approx(0.0)


def test_the_naive_reference_is_the_hardest_to_beat() -> None:
    baselines = (Persistence(hours=1), Persistence(hours=24))
    reference = naive_reference(baselines, bench_frame(), COLUMNS, BENCH)
    assert reference.name == "persistance-24h"


def test_no_baseline_at_all_is_refused() -> None:
    with pytest.raises(ArbitrationError, match="ADR-010"):
        naive_reference((), bench_frame(), COLUMNS, BENCH)


def test_bench_metrics_do_not_collide_with_the_test_block() -> None:
    candidate = BenchResult(name="xgboost", window=BENCH.label, metrics={"mae": 3.0})
    naive = BenchResult(
        name="persistance-24h", window=BENCH.label, metrics={"mae": 5.0}
    )
    metrics = bench_metrics(candidate, naive)
    assert metrics["arbitrage_mae"] == 3.0
    assert metrics[NAIVE_METRIC] == 5.0
    assert "mae" not in metrics


class TestReadBack:
    def test_a_run_carrying_a_bench_is_read_back(self) -> None:
        result = read_bench(
            {"arbitrage_mae": 3.0, "mae": 9.0},
            {BENCH_WINDOW_PARAM: BENCH.label},
            "champion v4",
        )
        assert result is not None
        assert result.error == 3.0
        assert result.window == BENCH.label

    def test_a_run_without_bench_metrics_reads_back_as_nothing(self) -> None:
        assert read_bench({"mae": 9.0}, {BENCH_WINDOW_PARAM: "x"}, "v1") is None

    def test_a_run_without_bench_window_reads_back_as_nothing(self) -> None:
        assert read_bench({"arbitrage_mae": 3.0}, {}, "v1") is None

    def test_the_naive_reference_is_read_back_with_its_window(self) -> None:
        naive = read_naive(
            {NAIVE_METRIC: 5.0}, {BENCH_WINDOW_PARAM: BENCH.label}
        )
        assert naive is not None
        assert naive.error == 5.0

    def test_a_run_without_naive_metric_reads_back_as_nothing(self) -> None:
        assert read_naive({}, {BENCH_WINDOW_PARAM: BENCH.label}) is None
