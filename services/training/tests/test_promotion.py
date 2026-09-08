from __future__ import annotations

from training.arbitration import BenchResult
from training.promotion import decide

WINDOW = "2026-08-01/2026-08-14"
OTHER_WINDOW = "2026-08-08/2026-08-21"


def result(error: float, window: str = WINDOW, name: str = "xgboost") -> BenchResult:
    return BenchResult(name=name, window=window, metrics={"mae": error})


NAIVE = result(10.0, name="persistance-24h")


def test_a_model_that_loses_to_persistence_is_refused() -> None:
    verdict = decide(result(11.0), champion=result(12.0), naive=NAIVE)
    assert not verdict.accepted
    assert "ADR-010" in verdict.reason


def test_persistence_is_checked_before_the_champion() -> None:
    verdict = decide(result(11.0), champion=result(50.0), naive=NAIVE)
    assert not verdict.accepted
    assert "persistance" in verdict.reason


def test_the_first_promotion_has_nothing_to_degrade() -> None:
    verdict = decide(result(6.0), champion=None, naive=NAIVE)
    assert verdict.accepted
    assert "première" in verdict.reason


def test_two_different_benches_are_not_compared() -> None:
    verdict = decide(
        result(6.0), champion=result(7.0, window=OTHER_WINDOW), naive=NAIVE
    )
    assert not verdict.accepted
    assert "training.arbitration" in verdict.reason


def test_a_degradation_is_refused() -> None:
    verdict = decide(result(8.0), champion=result(7.0), naive=NAIVE)
    assert not verdict.accepted
    assert "dégradation" in verdict.reason


def test_an_equal_candidate_passes_at_zero_margin() -> None:
    assert decide(result(7.0), champion=result(7.0), naive=NAIVE).accepted


def test_the_margin_is_what_tolerates_the_noise() -> None:
    assert decide(result(7.5), champion=result(7.0), naive=NAIVE, margin=0.1).accepted
    assert not decide(result(7.5), champion=result(7.0), naive=NAIVE).accepted


def test_a_better_candidate_is_accepted() -> None:
    verdict = decide(result(5.0), champion=result(7.0), naive=NAIVE)
    assert verdict.accepted
    assert "5.00" in verdict.reason


def test_a_champion_without_bench_measure_is_not_a_comparison() -> None:
    champion = BenchResult(name="champion v3", window=WINDOW, metrics={})
    verdict = decide(result(5.0), champion=champion, naive=NAIVE)
    assert not verdict.accepted
    assert "ne comparer à rien" in verdict.reason


def test_a_candidate_without_bench_measure_is_refused() -> None:
    candidate = BenchResult(name="xgboost", window=WINDOW, metrics={})
    verdict = decide(candidate, champion=None, naive=NAIVE)
    assert not verdict.accepted
    assert "pari" in verdict.reason
