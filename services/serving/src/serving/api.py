"""Service FastAPI d'inférence EnerVision déployé sur Azure.

Le service ne rejoue aucun calcul de l'amont. Il résout un modèle dans le
registre MLflow, lit la dernière partition de variables publiée par l'ETL, et
enchaîne les deux. Il n'importe le code d'aucun autre service : ses entrées
sont un alias et un chemin.

Il n'écrit dans aucune base. La seule écriture qu'il déclenche est celle du
pic simulé, qui agit sur la source et non sur un stockage. L'archivage des
prévisions dans `prediction` appartient à
l'API EnerVision, qui sert ce contrat à ses consommateurs et tient sa propre
base ; le référencement du modèle dans `modele` appartient à l'entraînement,
qui est le seul à savoir quelle version il vient de mettre en service. Ce
service calcule et répond ; ce qu'on fait de sa réponse ne le regarde pas.

Source de vérité du contrat. Toute modification exige une PR sur
enervision/docs/contracts et la relecture des trois consommateurs.

Attention : `CONTRACT_VERSION` est passée en 1.3.0. Deux routes additives
l'imposent, `GET /api/v1/sites` et `POST /api/v1/simulate/spike/{site_id}`.
Elles relaient la source vers l'API métier, qui ne la connaît pas et ne doit
pas la connaître. Le service continue de ne rien écrire : le pic agit sur la
source, et l'historique des pics appartient à l'API, qui tient la base.

Note précédente : `CONTRACT_VERSION` était passée en 1.2.0. Trois changements
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
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from predict_common.config import load_config
from predict_common.schemas import TIMESTAMP_COLUMN
from predict_common.source import (
    DEFAULT_SPIKE_MINUTES,
    MAX_SPIKE_MINUTES,
    MIN_SPIKE_MINUTES,
    SourceClient,
    SourceError,
    SourceSettings,
)
from serving.auth import build_middleware, check_api_key
from serving.forecast import (
    ForecastSpec,
    NoHistory,
    cached_history,
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
    SourceSiteOut,
    SpikeReadingOut,
    SpikeSimulationOut,
)

# Version du contrat gelé dans enervision/docs/contracts/openapi-predict.json.
# Incrémentée en semver : patch pour une description, minor pour un champ
# optionnel ajouté, major pour un champ retiré ou renommé.
CONTRACT_VERSION = "1.3.0"

API_PREFIX = "/api/v1"

SECONDS_PER_HOUR = 3600.0

# Repli du TTL du cache des variables, pour une application montée sans passer
# par `configure` — c'est le cas de plusieurs tests. La valeur d'exploitation
# vient de `serving.feature_cache_ttl_s`.
DEFAULT_FEATURE_CACHE_TTL_S = 300.0

# Forme admise d'un identifiant de site. Le paramètre voyage jusque dans le
# CHEMIN de l'appel sortant vers la source (`simulate_spike_path.format(...)`),
# et Starlette décode `%2F` avant de remplir le paramètre : sans cette borne,
# un identifiant peut porter des segments de chemin et faire émettre au
# service des requêtes vers des routes de la source qu'il n'expose pas.
#
# Le motif est celui du référentiel — `SITE001` — élargi de ce qu'un
# identifiant technique peut raisonnablement porter, et de rien d'autre : ni
# barre oblique, ni point, ni pourcentage.
#
# La vérification est faite EN CODE et non par `Path(pattern=...)`, qui
# publierait le motif dans la spécification et ferait dériver le contrat gelé.
# Le refus est le même — 422, que le contrat documente déjà sur cette route —
# et déclarer le motif au contrat reste la bonne cible, en patch semver, par
# une PR sur enervision/docs/contracts.
SITE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,20}$")

INVALID_SITE_ID = (
    "site_id ne respecte pas la forme d'un identifiant de site :"
    " 1 à 20 caractères parmi les lettres, les chiffres, le tiret et le"
    " tiret bas."
)


def ensure_site_id(site_id: str) -> str:
    """Refuse un identifiant qui ne peut pas être un site du référentiel."""
    if not SITE_ID_PATTERN.match(site_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=INVALID_SITE_ID,
        )
    return site_id

logger = logging.getLogger(__name__)

# Ni le modèle ni la configuration ne sont chargés à l'import : un module qui
# joint un registre au moment où on l'importe rend le service intestable et
# fait échouer la génération de la spécification OpenAPI en CI, où aucun
# MLflow ne tourne.
state: dict[str, Any] = {
    "registry": None,
    "spec": None,
    "source": None,
    "api_key": "",
    "feature_cache_ttl_s": DEFAULT_FEATURE_CACHE_TTL_S,
}


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
    source = state.get("source")
    if source is not None:
        source.close()
    state["registry"] = None
    state["source"] = None


def configure() -> None:
    """Lit la configuration et tente un premier chargement du modèle.

    La clé de service est validée AVANT tout le reste : un service qui
    démarrerait sans elle servirait la simulation de pic à qui la demande, et
    l'échec doit arriver au démarrage plutôt qu'à la première requête.
    """
    config = load_config()
    state["api_key"] = check_api_key(
        config.get_optional_str("serving.api_key"),
        auth_enabled=config.get_bool("serving.auth_enabled", True),
    )
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
    # Le client de la source est construit ici, pas à chaque requête : il tient
    # sa connexion ouverte, et le relais du référentiel est appelé à chaque
    # démarrage de l'API métier.
    state["source"] = SourceClient(SourceSettings.from_config(config))
    state["feature_cache_ttl_s"] = config.get_float(
        "serving.feature_cache_ttl_s",
        DEFAULT_FEATURE_CACHE_TTL_S,
    )
    state["registry"].load()


def _feature_cache_ttl_s() -> float:
    """Durée de vie du cache des variables, telle que la configuration la pose.

    Lue dans `state` et non capturée : le service peut être reconfiguré, et un
    test qui monte l'application sans passer par `configure` doit trouver un
    défaut plutôt qu'une clé absente.
    """
    return float(state.get("feature_cache_ttl_s", DEFAULT_FEATURE_CACHE_TTL_S))


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

# Monté à l'import et non dans le `lifespan` : Starlette fige la pile de
# middlewares au premier démarrage, et en ajouter un après lève. La clé, elle,
# est relue dans `state` à chaque requête — c'est `configure()` qui l'y pose.
app.middleware("http")(build_middleware(lambda: str(state.get("api_key") or "")))


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
        history = site_history(
            cached_history(spec, _feature_cache_ttl_s()),
            payload.site_id,
            spec,
        )
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


SOURCE_UNAVAILABLE = {
    "model": ErrorResponse,
    "description": "La source n'a pas répondu.",
}


@app.get(
    f"{API_PREFIX}/sites",
    response_model=list[SourceSiteOut],
    tags=["sites"],
    summary="Relayer le référentiel des sites servi par la source",
    responses={
        status.HTTP_502_BAD_GATEWAY: SOURCE_UNAVAILABLE,
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Le service n'a pas terminé son démarrage.",
        },
    },
)
def list_source_sites() -> list[SourceSiteOut]:
    """Rend le référentiel de la source, sans le traduire ni le stocker.

    Le service relaie, il n'entretient rien : c'est l'API métier qui décide
    d'en faire un référentiel en base, parce que c'est elle qui tient la base.
    Le relais existe parce qu'elle ne connaît pas la source, et ne doit pas la
    connaître : une adresse d'API Mock dans sa configuration ferait d'elle un
    second client de la source, avec sa propre façon de la lire.

    502 et non 503 : la panne est en amont du service, pas en lui. Les
    distinguer permet à l'appelant de dire lequel des deux est tombé.
    """
    client = _source()
    try:
        payload = client.fetch_sites()
    # ValueError couvre le corps JSON illisible : `response.json()` lève une
    # JSONDecodeError, qui en dérive et que le client ne traduit pas en
    # SourceError. Sans elle, une source qui répond 200 avec du HTML sortait
    # en 500 — soit « la panne est chez moi », l'inverse de ce que le 502
    # établit, et l'API métier partait chercher l'incident du mauvais côté.
    except (SourceError, ValueError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    # Une entrée sans identifiant n'est pas un site : la relayer ferait
    # échouer la validation et emporterait tout le référentiel avec elle.
    return [
        SourceSiteOut.model_validate(site)
        for site in payload
        if isinstance(site, dict) and site.get("site_id")
    ]


@app.post(
    f"{API_PREFIX}/simulate/spike/{{site_id}}",
    response_model=SpikeSimulationOut,
    tags=["simulate"],
    summary="Déclencher un pic de consommation sur la source",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Paramètres de requête invalides.",
        },
        status.HTTP_502_BAD_GATEWAY: SOURCE_UNAVAILABLE,
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "Le service n'a pas terminé son démarrage.",
        },
    },
)
def simulate_spike(
    site_id: str,
    duration_minutes: Annotated[
        int,
        Query(
            ge=MIN_SPIKE_MINUTES,
            le=MAX_SPIKE_MINUTES,
            description="Durée du pic simulé, en minutes.",
        ),
    ] = DEFAULT_SPIKE_MINUTES,
) -> SpikeSimulationOut:
    """Déclenche un pic sur la source et rend la mesure qui suit.

    Deux appels et non un seul : la source confirme la simulation sans dire ce
    qu'elle sert désormais, et une confirmation nue ne prouve rien à qui
    regarde un dashboard. La relecture de `/current` donne la valeur constatée.

    Son échec ne fait pas échouer la réponse : le pic est déclenché, le dire
    en erreur inviterait à rejouer l'appel et à superposer deux pics.
    """
    # Avant tout appel sortant : l'identifiant part dans le chemin de la
    # requête vers la source, un refus tardif l'aurait déjà émise.
    ensure_site_id(site_id)
    client = _source()
    try:
        payload = client.simulate_spike(site_id, duration_minutes)
    # ValueError pour la même raison qu'au relais du référentiel : un corps
    # illisible est une panne de la source, pas du service.
    except (SourceError, ValueError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return SpikeSimulationOut(
        site_id=str(payload.get("site_id", site_id)),
        status=str(payload.get("status", "simulated")),
        event=str(payload.get("event", "consumption_spike")),
        duration_minutes=int(payload.get("duration_minutes", duration_minutes)),
        message=str(payload.get("message", "")),
        simulated_at=datetime.now(UTC),
        reading=_reading_after_spike(client, site_id),
    )


def _reading_after_spike(
    client: SourceClient,
    site_id: str,
) -> SpikeReadingOut | None:
    """Relit la mesure courante, ou rend None sans faire échouer l'appelant."""
    try:
        readings = client.fetch_current(site_id)
    except (SourceError, ValueError) as exc:
        logger.warning("pic déclenché sur %s, mesure illisible : %s", site_id, exc)
        return None
    if not readings:
        logger.warning(
            "pic déclenché sur %s, la source n'a servi aucune mesure", site_id
        )
        return None
    return SpikeReadingOut.model_validate(readings[0])


def _source() -> SourceClient:
    """Rend le client de la source, en refusant de servir sans lui."""
    client = state.get("source")
    if client is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le service n'a pas terminé son démarrage.",
        )
    return client


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
