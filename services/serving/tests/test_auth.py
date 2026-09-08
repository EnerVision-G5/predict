from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serving import api as serving_api
from serving.auth import (
    API_KEY_HEADER,
    ServingAuthError,
    check_api_key,
    is_public,
)

KEY = "cle-de-service-de-test-suffisamment-longue"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setitem(serving_api.state, "api_key", KEY)
    return TestClient(serving_api.app)


@pytest.mark.parametrize(
    "path",
    ["/health", "/openapi.json", "/docs"],
)
def test_les_routes_publiques_restent_servies_sans_cle(
    client: TestClient,
    path: str,
) -> None:
    assert client.get(path).status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/sites"),
        ("get", "/ready"),
        ("post", "/api/v1/predict"),
        ("post", "/api/v1/simulate/spike/SITE001"),
    ],
)
def test_les_routes_du_contrat_exigent_la_cle(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.json()["detail"] == "Clé de service absente ou invalide."


def test_une_cle_fausse_est_refusee_comme_une_cle_absente(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/sites", headers={API_KEY_HEADER: "mauvaise"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Clé de service absente ou invalide."


def test_la_bonne_cle_laisse_passer(client: TestClient) -> None:
    response = client.get("/ready", headers={API_KEY_HEADER: KEY})
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_sans_cle_configuree_le_service_refuse_de_demarrer() -> None:
    with pytest.raises(ServingAuthError, match="SERVING_API_KEY"):
        check_api_key("", auth_enabled=True)


def test_le_mode_ouvert_est_explicite_et_journalise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        assert check_api_key("", auth_enabled=False) == ""
    assert "SERVING_AUTH_ENABLED=false" in caplog.text


def test_un_prefixe_public_ne_couvre_pas_un_chemin_qui_le_prolonge() -> None:
    assert is_public("/health")
    assert is_public("/docs/oauth2-redirect")
    assert not is_public("/healthz")
    assert not is_public("/api/v1/predict")
