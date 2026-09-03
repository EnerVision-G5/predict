"""Contrat HTTP du service : les quatre réponses qu'il sait donner.

Le service est branché ici sur un modèle et un stockage factices, sans MLflow
ni réseau. Ce qui est testé est la correspondance entre une situation
d'exploitation et le code de statut renvoyé : un registre vide donne 503, un
site inconnu donne 404, une requête hors bornes donne 422, et le reste donne
une série. Un service qui confondrait 503 et 404 enverrait les exploitants
chercher la panne du mauvais côté.
"""

from __future__ import annotations

from datetime import date, timedelta
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
    """Modèle chargé qui prédit une constante, sans MLflow derrière."""

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
    """Registre déjà résolu, ou volontairement vide."""

    def __init__(self, model: StubModel | None) -> None:
        super().__init__("http://mlflow.invalid", "models:/enervision_xgboost@champion")
        self._loaded = model

    def load(self):
        return self._loaded

    def input_columns(self) -> list[str]:
        return list(COLUMNS)


def features(day: date, site_id: str = "SITE001") -> pd.DataFrame:
    """Partition de variables d'une journée, telle que l'ETL la publie."""
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
    """Publie trois journées de variables pour un site."""
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
    """Client HTTP dont le démarrage résout un modèle factice."""

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

        monkeypatch.setattr(api, "configure", configure)
        return TestClient(api.app)

    return build


def test_health_answers_without_any_dependency(client) -> None:
    # La lier à MLflow ferait redémarrer un service en parfait état chaque
    # fois que le registre tousse.
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
    # Sans elle, une prévision aberrante ne serait imputable à rien.
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
    # Le contrat les prévoit optionnelles : une version qui ne déclare pas la
    # dispersion de son erreur sert une prévision nue, pas une bande inventée.
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
    # L'erreur s'accumule à chaque pas de la récurrence : une bande constante
    # sur 24 heures annoncerait la 24e aussi sûre que la première.
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
    # 503 et non 404 : la panne est chez MLflow, pas dans la requête. Confondre
    # les deux enverrait les exploitants chercher du côté du référentiel des
    # sites une panne qui est celle du registre.
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
    # `detail` est une chaîne, jamais la liste du HTTPValidationError de
    # FastAPI : c'est ce que le contrat gelé annonce aux consommateurs.
    with client() as http:
        response = http.post("/api/v1/predict", json={})
    assert isinstance(response.json()["detail"], str)
