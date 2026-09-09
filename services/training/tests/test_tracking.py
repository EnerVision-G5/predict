from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from training import tracking

METRICS = {"mae": 3.0, "arbitrage_mae": 2.5}
PARAMS = {"arbitrage_window": "2026-08-01/2026-08-14"}


@dataclass
class FakeRun:
    data: Any


@dataclass
class FakeData:
    metrics: dict[str, float]
    params: dict[str, str]


@dataclass
class FakeVersion:
    version: str
    run_id: str


@dataclass
class FakeClient:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeVersion:
        self.calls.append(("alias", alias))
        return FakeVersion(version="7", run_id="run-7")

    def get_model_version(self, name: str, version: str) -> FakeVersion:
        self.calls.append(("version", version))
        return FakeVersion(version=version, run_id=f"run-{version}")

    def get_run(self, run_id: str) -> FakeRun:
        self.calls.append(("run", run_id))
        return FakeRun(data=FakeData(metrics=dict(METRICS), params=dict(PARAMS)))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    double = FakeClient()
    monkeypatch.setattr(tracking.mlflow, "MlflowClient", lambda: double)
    return double


class TestAliasSnapshot:
    def test_both_halves_of_the_run_are_read(self, client: FakeClient) -> None:
        snapshot = tracking.alias_snapshot("enervision_xgboost", "champion")
        assert snapshot.metrics == METRICS
        assert snapshot.params == PARAMS

    def test_the_served_version_comes_back_with_them(
        self, client: FakeClient
    ) -> None:
        assert tracking.alias_snapshot("enervision_xgboost", "champion").version == "7"

    def test_the_alias_is_resolved_before_its_run(self, client: FakeClient) -> None:
        tracking.alias_snapshot("enervision_xgboost", "champion")
        assert client.calls == [("alias", "champion"), ("run", "run-7")]


def test_baseline_metrics_keeps_only_the_metrics(client: FakeClient) -> None:
    assert tracking.baseline_metrics("enervision_xgboost", "champion") == METRICS


def test_a_numbered_version_is_read_like_an_aliased_one(
    client: FakeClient,
) -> None:
    snapshot = tracking.version_snapshot("enervision_xgboost", "4")
    assert snapshot.version == "4"
    assert snapshot.params == PARAMS
    assert client.calls == [("version", "4"), ("run", "run-4")]


def test_tags_travel_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: dict[str, Any] = {}
    monkeypatch.setattr(tracking.mlflow, "set_tags", posted.update)
    tracking.set_tags({"challenge": True, "famille": "naive"})
    assert posted == {"challenge": "True", "famille": "naive"}
