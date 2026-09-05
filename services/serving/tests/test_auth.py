"""Contrôle d'accès du service d'inférence.

Le service relaie `POST /api/v1/simulate/spike`, qui écrit sur la source.
L'API métier protège la même opération derrière le rôle `writer` : un service
ouvert rendait ce contrôle contournable, il suffisait de l'appeler
directement. Ces tests fixent ce qui est fermé, ce qui reste ouvert, et le
fait que le service refuse de démarrer sans clé.
"""

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
    """Application montée avec une clé connue, sans lifespan ni registre."""
    monkeypatch.setitem(serving_api.state, "api_key", KEY)
    # Pas de `with` : le lifespan joindrait MLflow et la source.
    return TestClient(serving_api.app)


@pytest.mark.parametrize(
    "path",
    ["/health", "/openapi.json", "/docs"],
)
def test_les_routes_publiques_restent_servies_sans_cle(
    client: TestClient,
    path: str,
) -> None:
    """La sonde et la spécification n'exigent rien.

    La sonde est interrogée par l'orchestrateur, qui n'a pas de secret à
    porter ; la spécification est de toute façon publiée dans le dépôt des
    contrats, et le scan DAST de la CI en part.
    """
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
    """Aucune route du contrat ne répond sans clé, écriture comprise."""
    response = getattr(client, method)(path)
    assert response.status_code == 401
    assert response.json()["detail"] == "Clé de service absente ou invalide."


def test_une_cle_fausse_est_refusee_comme_une_cle_absente(
    client: TestClient,
) -> None:
    """Le message ne dit pas laquelle des deux erreurs a été commise.

    Les distinguer confirmerait à un appelant qu'il a trouvé le bon en-tête,
    ce qui est précisément l'information à ne pas donner.
    """
    response = client.get("/api/v1/sites", headers={API_KEY_HEADER: "mauvaise"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Clé de service absente ou invalide."


def test_la_bonne_cle_laisse_passer(client: TestClient) -> None:
    """Avec la clé, la requête atteint la route.

    `/ready` est choisie parce qu'elle répond toujours 200, y compris sans
    registre ni partition : le test porte sur le passage du contrôle, pas sur
    ce que la route sait servir.
    """
    response = client.get("/ready", headers={API_KEY_HEADER: KEY})
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_sans_cle_configuree_le_service_refuse_de_demarrer() -> None:
    """Démarrer ouvert serait le pire des trois états : ça a l'air sain."""
    with pytest.raises(ServingAuthError, match="SERVING_API_KEY"):
        check_api_key("", auth_enabled=True)


def test_le_mode_ouvert_est_explicite_et_journalise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`auth_enabled=false` ouvre, et le dit dans le journal de démarrage."""
    with caplog.at_level("WARNING"):
        assert check_api_key("", auth_enabled=False) == ""
    assert "SERVING_AUTH_ENABLED=false" in caplog.text


def test_un_prefixe_public_ne_couvre_pas_un_chemin_qui_le_prolonge() -> None:
    """`/healthz` n'est pas `/health` : la comparaison est sur le segment.

    Un `startswith` nu aurait laissé passer `/health-interne` ou, plus
    gênant, n'importe quelle route future commençant par un préfixe public.
    """
    assert is_public("/health")
    assert is_public("/docs/oauth2-redirect")
    assert not is_public("/healthz")
    assert not is_public("/api/v1/predict")
