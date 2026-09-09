from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predict_common import io
from predict_common.paths import features_partition
from predict_common.schemas import feature_columns, features_arrow_schema
from serving import api
from serving.forecast import ForecastSpec
from serving.loader import ModelRegistry

LAGS = (1, 24)
WINDOW = 2
COLUMNS = feature_columns(LAGS, WINDOW)
TODAY = date.today()


class StubModel:
    def __init__(
        self,
        version: str = "3",
        residual_std: float | None = None,
    ) -> None:
        self.version = version
        self.columns = COLUMNS
        self.residual_std = residual_std

    def predict(self, frame: pd.DataFrame):
        return [42.0] * len(frame)


class StubRegistry(ModelRegistry):
    def __init__(self, model: StubModel | None) -> None:
        super().__init__("http://mlflow.invalid", "models:/enervision_xgboost@champion")
        self._loaded = model

    def load(self):
        return self._loaded

    def input_columns(self) -> list[str]:
        return list(COLUMNS)


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


@pytest.fixture
def serving_root(tmp_path: Path) -> Path:
    for offset in range(3):
        day = TODAY - timedelta(days=offset)
        io.write_frame(
            features(day),
            features_partition(str(tmp_path), "v1", day),
            schema=features_arrow_schema(LAGS, WINDOW),
        )
    return tmp_path


@pytest.fixture
def client(serving_root: Path, monkeypatch):
    def build(served: bool = True, model: StubModel | None = None):
        def configure() -> None:
            resolved = (model or StubModel()) if served else None
            api.state["registry"] = StubRegistry(resolved)
            api.state["spec"] = ForecastSpec(
                root=str(serving_root),
                feature_version="v1",
                lag_hours=LAGS,
                rolling_window_h=WINDOW,
                lookback_days=3,
            )
            api.state["source"] = None

        monkeypatch.setattr(api, "configure", configure)
        return TestClient(api.app)

    return build


def test_health_answers_without_any_dependency(client) -> None:
    with client() as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_the_requested_horizon(client) -> None:
    with client() as http:
        response = http.post(
            "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 6}
        )
    assert response.status_code == 200
    assert len(response.json()["points"]) == 6


def test_the_answer_names_the_model_that_produced_it(client) -> None:
    with client() as http:
        response = http.post("/api/v1/predict", json={"site_id": "SITE001"})
    assert response.json()["model_version"] == "3"


def test_the_points_follow_one_another_in_time(client) -> None:
    with client() as http:
        response = http.post(
            "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 4}
        )
    stamps = [point["timestamp"] for point in response.json()["points"]]
    assert stamps == sorted(stamps)


def test_a_version_without_a_spread_serves_null_bounds(client) -> None:
    with client() as http:
        response = http.post("/api/v1/predict", json={"site_id": "SITE001"})
    point = response.json()["points"][0]
    assert point["lower_bound_kw"] is None
    assert point["upper_bound_kw"] is None


def test_the_bounds_frame_the_prediction(client) -> None:
    with client(model=StubModel(residual_std=2.0)) as http:
        response = http.post("/api/v1/predict", json={"site_id": "SITE001"})
    point = response.json()["points"][0]
    assert point["lower_bound_kw"] < point["predicted_consumption_kw"]
    assert point["upper_bound_kw"] > point["predicted_consumption_kw"]


def test_the_bounds_widen_with_the_horizon(client) -> None:
    with client(model=StubModel(residual_std=2.0)) as http:
        response = http.post(
            "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 6}
        )
    points = response.json()["points"]
    widths = [
        point["upper_bound_kw"] - point["lower_bound_kw"] for point in points
    ]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


def test_an_unknown_site_is_a_404(client) -> None:
    with client() as http:
        response = http.post("/api/v1/predict", json={"site_id": "SITE404"})
    assert response.status_code == 404
    assert "SITE404" in response.json()["detail"]


def test_an_empty_registry_is_a_503(client) -> None:
    with client(served=False) as http:
        response = http.post("/api/v1/predict", json={"site_id": "SITE001"})
    assert response.status_code == 503
    assert "models:/" in response.json()["detail"]


def test_a_horizon_beyond_the_contract_is_a_422(client) -> None:
    with client() as http:
        response = http.post(
            "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 99}
        )
    assert response.status_code == 422
    assert "horizon_hours" in response.json()["detail"]


def test_a_request_without_a_site_is_a_422(client) -> None:
    with client() as http:
        response = http.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_a_validation_error_uses_the_shared_model(client) -> None:
    with client() as http:
        response = http.post("/api/v1/predict", json={})
    assert isinstance(response.json()["detail"], str)


class TestSurQuoiLaPrevisionSAppuie:
    def test_the_answer_says_which_hour_it_starts_from(self, client) -> None:
        with client() as http:
            body = http.post(
                "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 2}
            ).json()
        assert body["history_end"] < body["points"][0]["timestamp"]

    def test_the_feature_lag_is_the_gap_to_the_answer(self, client) -> None:
        with client() as http:
            body = http.post(
                "/api/v1/predict", json={"site_id": "SITE001", "horizon_hours": 1}
            ).json()
        generated = datetime.fromisoformat(body["generated_at"])
        history_end = datetime.fromisoformat(body["history_end"])
        expected = (generated - history_end).total_seconds() / 3600
        assert body["feature_lag_hours"] == pytest.approx(expected)

    def test_a_clock_ahead_of_the_features_is_not_hidden(self) -> None:
        generated = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
        history_end = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)
        assert api.feature_lag_hours(generated, history_end) == pytest.approx(-2.0)


class TestReadiness:
    def test_a_served_model_and_features_are_ready(self, client) -> None:
        with client() as http:
            body = http.get("/ready").json()
        assert body["ready"] is True
        assert body["model_resolved"] is True
        assert body["features_available"] is True
        assert body["detail"] == ""

    def test_an_empty_registry_is_named(self, client) -> None:
        with client(served=False) as http:
            response = http.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is False
        assert body["model_resolved"] is False
        assert body["model_version"] is None
        assert "Modèle" in body["detail"]

    def test_an_unconfigured_service_says_so(self, monkeypatch) -> None:
        monkeypatch.setattr(api, "configure", lambda: None)
        api.state["registry"] = None
        api.state["spec"] = None
        with TestClient(api.app) as http:
            body = http.get("/ready").json()
        assert body["ready"] is False
        assert "non configuré" in body["detail"]
