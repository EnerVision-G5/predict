"""Test de fumée du contrat du service d'inférence.

Il vérifie que l'application se construit et que la spécification OpenAPI
reste générable, ce qui fait de la CI un garde-fou même quand le registre
MLflow est injoignable — cas de la CI, précisément.

Il vérifie aussi que la génération n'a besoin d'aucune dépendance de
démarrage : le module ne doit joindre ni MLflow ni le stockage à l'import,
sinon le job `contract-drift` échouerait sur une panne d'infrastructure et
non sur une dérive de contrat.
"""

from __future__ import annotations

from fastapi.openapi.utils import get_openapi

from serving.api import CONTRACT_VERSION, app


def spec() -> dict:
    """Retourne la spécification telle que le script d'export la produit."""
    return app.openapi()


def test_app_exposes_contract_version() -> None:
    assert app.version == CONTRACT_VERSION


def test_openapi_spec_is_generable() -> None:
    generated = get_openapi(
        title=app.title,
        version=CONTRACT_VERSION,
        description=app.description,
        routes=app.routes,
    )
    assert generated["openapi"].startswith("3.")
    assert "/health" in generated["paths"]
    assert "/api/v1/predict" in generated["paths"]


def test_the_health_probe_is_not_versioned() -> None:
    # C'est la sonde de disponibilité de l'hébergeur, pas une route métier.
    assert "/health" in spec()["paths"]
    assert "/api/v1/health" not in spec()["paths"]


def test_predict_declares_its_error_responses() -> None:
    responses = spec()["paths"]["/api/v1/predict"]["post"]["responses"]
    assert {"200", "404", "422", "503"} <= set(responses)


def test_every_error_response_uses_the_shared_model() -> None:
    # Le gestionnaire renvoie toujours un ErrorResponse : annoncer le
    # HTTPValidationError de FastAPI, dont `detail` est une liste, mentirait
    # aux trois consommateurs du contrat.
    generated = spec()
    responses = generated["paths"]["/api/v1/predict"]["post"]["responses"]
    for code in ("404", "422", "503"):
        schema = responses[code]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ErrorResponse")
    assert "HTTPValidationError" not in generated["components"]["schemas"]


def test_the_dto_are_declared_in_the_contract() -> None:
    schemas = spec()["components"]["schemas"]
    assert {
        "ErrorResponse",
        "HealthOut",
        "PredictionRequest",
        "PredictionPoint",
        "PredictionOut",
    } <= set(schemas)


def test_the_horizon_keeps_its_bounds() -> None:
    # 48 h est la borne du contrat : au-delà, l'erreur de la récurrence
    # dépasse ce que la prévision vaut encore.
    horizon = spec()["components"]["schemas"]["PredictionRequest"]["properties"][
        "horizon_hours"
    ]
    assert (horizon["minimum"], horizon["maximum"]) == (1, 48)
