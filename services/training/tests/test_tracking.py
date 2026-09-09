from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
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


class TestLogCandidateModel:
    """Le modèle d'un candidat, et ce qui arrive quand il ne part pas."""

    @staticmethod
    def features() -> Any:
        import pandas as pd

        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    def test_the_safe_format_is_used_with_its_declared_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_log_model(model: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", fake_log_model)
        monkeypatch.setattr(tracking, "infer_signature", lambda *_: None)
        tracking.log_candidate_model(
            object(), self.features(), [1.0, 2.0], name="v1-challenge-ridge"
        )
        assert seen["serialization_format"] == "skops"
        assert "xgboost.sklearn.XGBRegressor" in seen["skops_trusted_types"]
        assert seen["name"] == "v1-challenge-ridge"

    def test_an_attached_model_reports_its_logged_identifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tracking.mlflow.sklearn,
            "log_model",
            lambda *a, **k: SimpleNamespace(model_id="m-42"),
        )
        monkeypatch.setattr(tracking, "infer_signature", lambda *_: None)
        assert tracking.log_candidate_model(
            object(), self.features(), []
        ) == "m-42"

    def test_a_model_that_does_not_leave_is_reported_and_not_raised(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def refuse(*_: Any, **__: Any) -> None:
            raise RuntimeError("untrusted types")

        monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", refuse)
        monkeypatch.setattr(tracking, "infer_signature", lambda *_: None)
        with caplog.at_level(logging.WARNING, logger="training.tracking"):
            logged = tracking.log_candidate_model(object(), self.features(), [])
        assert logged == ""
        assert "non attaché" in caplog.text
