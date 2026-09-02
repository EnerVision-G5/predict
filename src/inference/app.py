"""Service FastAPI d'inférence EnerVision.

Ce module ne contient que la déclaration du contrat d'interface. Le chargement
du modèle XGBoost et la prédiction relèvent des tickets EV-20 et suivants.

Source de vérité du contrat. Toute modification exige une PR sur
enervision/docs/contracts et la relecture des trois consommateurs.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from inference.schemas import (
    ErrorResponse,
    HealthOut,
    PredictionOut,
    PredictionRequest,
)

# Version du contrat gelé dans enervision/docs/contracts/openapi-predict.json.
# Incrémentée en semver : patch pour une description, minor pour un champ
# optionnel ajouté, major pour un champ retiré ou renommé.
CONTRACT_VERSION = "1.0.0"

API_PREFIX = "/api/v1"

app = FastAPI(
    title="EnerVision service d'inférence",
    version=CONTRACT_VERSION,
    description=(
        "Contrat du service de prédiction déployé sur Azure. L'endpoint de"
        " prédiction renvoie 501 tant que le modèle XGBoost n'est pas servi."
    ),
)


@app.exception_handler(RequestValidationError)
def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Aligne les erreurs de validation sur le modèle ErrorResponse."""
    detail = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in exc.errors()
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": detail or "Requête invalide."},
    )


@app.get(
    "/health",
    response_model=HealthOut,
    tags=["health"],
    summary="Vérifier la disponibilité du service",
)
def get_health() -> HealthOut:
    """Endpoint trivial, réellement implémenté : aucune dépendance externe."""
    return HealthOut(status="ok", timestamp=datetime.now(UTC))


@app.post(
    f"{API_PREFIX}/predict",
    response_model=PredictionOut,
    tags=["predict"],
    summary="Prédire la consommation d'un site",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Site inconnu du modèle.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Paramètres de requête invalides.",
        },
    },
)
def predict(payload: PredictionRequest) -> PredictionOut:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Contrat EV-06 uniquement. Inférence implémentée par EV-20"
            " et suivants."
        ),
    )


def _normalize_error_responses(spec: dict[str, Any]) -> dict[str, Any]:
    """Aligne toutes les réponses 422 sur le modèle ErrorResponse.

    Sans cette normalisation, la spécification annoncerait le
    HTTPValidationError par défaut de FastAPI, dont le champ detail est une
    liste, alors que le gestionnaire ci-dessus renvoie toujours un
    ErrorResponse.
    """
    error_ref = {"$ref": "#/components/schemas/ErrorResponse"}
    for operations in spec.get("paths", {}).values():
        for operation in operations.values():
            response = operation.get("responses", {}).get("422")
            if response is None:
                continue
            response["description"] = "Paramètres de requête invalides."
            response["content"] = {"application/json": {"schema": error_ref}}
    schemas = spec.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    return spec


def custom_openapi() -> dict[str, Any]:
    """Spécification OpenAPI servant de source unique au contrat gelé."""
    if app.openapi_schema is None:
        app.openapi_schema = _normalize_error_responses(
            get_openapi(
                title=app.title,
                version=CONTRACT_VERSION,
                description=app.description,
                routes=app.routes,
            )
        )
    return app.openapi_schema


app.openapi = custom_openapi
