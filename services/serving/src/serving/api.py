"""Service FastAPI d'inférence EnerVision déployé sur Azure.

Le service ne rejoue aucun calcul de l'amont. Il résout un modèle dans le
registre MLflow, lit la dernière partition de variables publiée par l'ETL, et
enchaîne les deux. Il n'importe le code d'aucun autre service : ses entrées
sont un alias et un chemin.

Il n'écrit rien. L'archivage des prévisions dans `prediction` appartient à
l'API EnerVision, qui sert ce contrat à ses consommateurs et tient sa propre
base ; le référencement du modèle dans `modele` appartient à l'entraînement,
qui est le seul à savoir quelle version il vient de mettre en service. Ce
service calcule et répond ; ce qu'on fait de sa réponse ne le regarde pas.

Source de vérité du contrat. Toute modification exige une PR sur
enervision/docs/contracts et la relecture des trois consommateurs.

Attention : `CONTRACT_VERSION` est passée en 1.2.0. Trois changements
l'imposent, tous additifs — `history_end` et `feature_lag_hours` sur
`PredictionOut`, et la route `/ready`. Les deux champs disent sur quelles
variables la prévision s'appuie, information que personne d'autre ne détient :
le service lit les partitions publiées par l'ETL, et une prévision calculée
sur des variables vieilles de trois jours n'est pas fausse, elle est aveugle.
La route dit pourquoi une prévision manque, là où un 503 nu ne le dit pas.

`/health` reste sans dépendance, et c'est délibéré : voir `get_health`.

Le contrat gelé doit être régénéré et relu :

    python scripts/export_openapi.py ../docs/contracts/openapi-predict.json
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from predict_common.config import load_config
from predict_common.schemas import TIMESTAMP_COLUMN
from serving.forecast import (
    ForecastSpec,
    NoHistory,
    predict_series,
    read_history,
    site_history,
)
from serving.loader import ModelRegistry, ModelUnavailable
from serving.schemas import (
    ErrorResponse,
    HealthOut,
    PredictionOut,
    PredictionPoint,
    PredictionRequest,
    ReadinessOut,
)

# Version du contrat gelé dans enervision/docs/contracts/openapi-predict.json.
# Incrémentée en semver : patch pour une description, minor pour un champ
# optionnel ajouté, major pour un champ retiré ou renommé.
CONTRACT_VERSION = "1.2.0"

API_PREFIX = "/api/v1"

SECONDS_PER_HOUR = 3600.0

logger = logging.getLogger(__name__)

# Ni le modèle ni la configuration ne sont chargés à l'import : un module qui
# joint un registre au moment où on l'importe rend le service intestable et
# fait échouer la génération de la spécification OpenAPI en CI, où aucun
# MLflow ne tourne.
state: dict[str, Any] = {"registry": None, "spec": None}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Résout le modèle au démarrage, sans faire dépendre le démarrage de lui.

    Un registre injoignable ne doit pas empêcher le processus de vivre : le
    conteneur redémarrerait en boucle et la sonde de disponibilité ne
    répondrait jamais, ce que l'hébergeur lirait comme une panne du service
    alors que la panne est chez MLflow.
    """
    configure()
    yield
    state["registry"] = None


def configure() -> None:
    """Lit la configuration et tente un premier chargement du modèle."""
    config = load_config()
    state["registry"] = ModelRegistry(
        tracking_uri=config.get_str("mlflow.tracking_uri"),
        model_uri=config.get_str("serving.model_uri"),
    )
    state["spec"] = ForecastSpec(
        root=config.get_str("storage.root"),
        feature_version=config.get_str("etl.feature_version"),
        lag_hours=tuple(config.get_int_list("etl.lag_hours")),
        rolling_window_h=config.get_int("etl.rolling_window_h"),
        lookback_days=config.get_int("serving.feature_lookback_days"),
    )
    state["registry"].load()


app = FastAPI(
    title="EnerVision service d'inférence",
    version=CONTRACT_VERSION,
    description=(
        "Contrat du service de prédiction déployé sur Azure. La prévision est"
        " servie par le modèle que le registre MLflow désigne, et renvoie 503"
        " tant qu'aucun modèle n'est résolu."
    ),
    lifespan=lifespan,
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
    """Endpoint trivial, réellement implémenté : aucune dépendance externe.

    Il ne consulte volontairement pas le registre. C'est la sonde de vivacité
    de l'hébergeur : la lier à MLflow ferait redémarrer un service en parfait
    état chaque fois que le registre tousse.
    """
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
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Aucun modèle résolu par le registre.",
        },
    },
)
def predict(payload: PredictionRequest) -> PredictionOut:
    """Retourne la série prédite d'un site sur l'horizon demandé."""
    registry, spec = _resources()
    try:
        model = registry.current()
        columns = registry.input_columns()
    except ModelUnavailable as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    try:
        history = site_history(read_history(spec), payload.site_id, spec)
    except NoHistory as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    points = predict_series(
        model.predict,
        history,
        payload.horizon_hours,
        spec,
        columns,
        residual_std=model.residual_std,
    )
    if not points:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                f"Historique insuffisant pour {payload.site_id} : les décalages"
                " demandés par le modèle ne sont pas tous disponibles."
            ),
        )
    generated_at = datetime.now(UTC)
    history_end = history.index.max().to_pydatetime()
    return PredictionOut(
        site_id=payload.site_id,
        model_version=model.version,
        generated_at=generated_at,
        # Le service prédit à partir des partitions publiées par l'ETL, pas de
        # la base : ses variables peuvent dater sans que rien ne le signale.
        # C'est ce couple qui le dit, et il est calculé ici parce que le
        # service est le seul à savoir sur quoi il vient de s'appuyer.
        history_end=history_end,
        feature_lag_hours=feature_lag_hours(generated_at, history_end),
        points=[
            PredictionPoint(
                timestamp=point.stamp.to_pydatetime(),
                predicted_consumption_kw=point.value,
                # Bornes calculées par `forecast.confidence_band` à partir de
                # la dispersion que la version servie déclare. Nulles quand la
                # version ne la déclare pas : le contrat les prévoit
                # optionnelles depuis l'origine, précisément pour ce cas.
                lower_bound_kw=point.lower,
                upper_bound_kw=point.upper,
            )
            for point in points
        ],
    )


def feature_lag_hours(generated_at: datetime, history_end: datetime) -> float:
    """Âge des variables ayant servi la prévision, en heures.

    Un écart négatif n'est pas ramené à zéro : il signale une partition dont
    l'horodatage est en avance sur l'horloge du service, et masquer cela
    ferait passer un problème de fuseau pour une prévision fraîche.
    """
    return (generated_at - history_end).total_seconds() / SECONDS_PER_HOUR


@app.get(
    "/ready",
    response_model=ReadinessOut,
    tags=["health"],
    summary="Dire ce dont le service dispose pour prédire",
)
def get_readiness() -> ReadinessOut:
    """Diagnostic, et non sonde de vivacité : voir ReadinessOut.

    Répond toujours 200, y compris quand rien n'est prêt. Un 503 ici ferait
    de cette route une seconde sonde et l'hébergeur redémarrerait le service
    à chaque hoquet de MLflow, ce que `/health` évite précisément. Son
    consommateur est l'API métier, qui a besoin de dire à ses utilisateurs
    pourquoi la prévision manque — un 503 nu ne le dit pas.
    """
    registry, spec = state.get("registry"), state.get("spec")
    if registry is None or spec is None:
        return ReadinessOut(
            ready=False,
            model_resolved=False,
            model_version=None,
            features_available=False,
            history_end=None,
            detail="Service non configuré : aucune ressource chargée.",
        )

    version, model_detail = _model_state(registry)
    history_end, feature_detail = _feature_state(spec)
    detail = " ".join(part for part in (model_detail, feature_detail) if part)
    return ReadinessOut(
        ready=version is not None and history_end is not None,
        model_resolved=version is not None,
        model_version=version,
        features_available=history_end is not None,
        history_end=history_end,
        detail=detail,
    )


def _model_state(registry: ModelRegistry) -> tuple[str | None, str]:
    """Retourne la version servie, ou la raison pour laquelle il n'y en a pas."""
    try:
        return registry.current().version, ""
    except ModelUnavailable as exc:
        return None, f"Modèle : {exc}"


def _feature_state(spec: ForecastSpec) -> tuple[datetime | None, str]:
    """Retourne la dernière heure disponible, ou la raison de son absence.

    Le filet est large à dessein : cette route existe pour dire ce qui ne va
    pas, et une lecture de partitions qui échoue est exactement ce qu'elle
    doit rapporter plutôt que propager.
    """
    try:
        frame = read_history(spec)
    except Exception as exc:  # noqa: BLE001 - la lecture d'objets lève large
        logger.warning("variables illisibles : %s", exc)
        return None, f"Variables : {type(exc).__name__}: {exc}"
    if frame.empty:
        return None, (
            f"Variables : aucune partition {spec.feature_version} sur les"
            f" {spec.lookback_days} dernière(s) journée(s)."
        )
    return frame[TIMESTAMP_COLUMN].max().to_pydatetime(), ""


def _resources() -> tuple[ModelRegistry, ForecastSpec]:
    """Retourne les ressources du processus, en refusant de servir sans elles."""
    registry, spec = state.get("registry"), state.get("spec")
    if registry is None or spec is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service n'a pas terminé son démarrage.",
        )
    return registry, spec


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
