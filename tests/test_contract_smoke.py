"""Test de fumée du contrat du service d'inférence.

Ne teste pas de logique métier (il n'y en a pas encore) : il vérifie que
l'application se construit et que la spécification OpenAPI reste générable,
ce qui suffit à faire de la CI un garde-fou dès le premier commit.
"""

from fastapi.openapi.utils import get_openapi

from inference.app import CONTRACT_VERSION, app


def test_app_exposes_contract_version() -> None:
    assert app.version == CONTRACT_VERSION


def test_openapi_spec_is_generable() -> None:
    spec = get_openapi(
        title=app.title,
        version=CONTRACT_VERSION,
        description=app.description,
        routes=app.routes,
    )
    assert spec["openapi"].startswith("3.")
    assert "/health" in spec["paths"]
    assert "/api/v1/predict" in spec["paths"]
