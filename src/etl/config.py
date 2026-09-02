"""Configuration du pipeline ETL, lue dans l'environnement.

Aucune valeur sensible n'est écrite ici. Les identifiants de base et l'URL de
l'API source viennent de variables d'environnement, décrites dans
`.env.example`. Le fichier `.env` réel n'est jamais commité.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

NumberT = TypeVar("NumberT", int, float)

DEFAULT_MOCK_API_URL = "http://localhost:8000"
DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"
DEFAULT_MLFLOW_EXPERIMENT = "enervision-consumption"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_REQUEST_TIMEOUT_S = 30.0

# Le mode continu lit la mesure courante d'un site, pas une page d'historique :
# c'est un autre endpoint que celui du rattrapage.
DEFAULT_CURRENT_PATH = "/api/v1/sites/{site_id}/current"

# Une minute : la source produit une mesure par minute et par site. Interroger
# plus vite relirait la même valeur, plus lentement en perdrait.
DEFAULT_POLL_INTERVAL_S = 60.0

# Une seule reprise immédiate. Au-delà, insister mangerait la cadence pour
# rien : la vraie reprise, c'est le tick suivant.
DEFAULT_POLL_RETRIES = 1
DEFAULT_POLL_BACKOFF_S = 2.0

# Plus court que le timeout du rattrapage : un site muet ne doit pas retarder
# les six autres au-delà de leur propre cadence.
DEFAULT_POLL_TIMEOUT_S = 10.0

# Trois minutes, soit trois cadences manquées : en deçà, un simple à-coup de la
# source déclencherait une alerte sans qu'aucune donnée ne soit perdue.
DEFAULT_LAG_WARNING_S = 180.0


class ConfigError(RuntimeError):
    """Une variable d'environnement obligatoire manque ou est invalide."""


@dataclass(frozen=True)
class EtlConfig:
    """Paramètres d'exécution d'un run d'ingestion."""

    database_url: str
    mock_api_url: str = DEFAULT_MOCK_API_URL
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT
    batch_size: int = DEFAULT_BATCH_SIZE
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    current_path: str = DEFAULT_CURRENT_PATH
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    poll_retries: int = DEFAULT_POLL_RETRIES
    poll_backoff_s: float = DEFAULT_POLL_BACKOFF_S
    poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S
    lag_warning_s: float = DEFAULT_LAG_WARNING_S


def read_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Retourne l'entier porté par `name`, ou `default` si la clé est absente.

    Une valeur présente mais illisible est une erreur de configuration, pas un
    cas à rattraper silencieusement par le défaut : le run partirait avec des
    paramètres que personne n'a demandés.
    """
    return _read_number(env, name, default, int)


def read_float(env: Mapping[str, str], name: str, default: float) -> float:
    """Équivalent de `read_int` pour une valeur décimale."""
    return _read_number(env, name, default, float)


def read_text(env: Mapping[str, str], name: str, default: str) -> str:
    """Retourne la chaîne portée par `name`, vide ou absente valant `default`."""
    value = env.get(name, "").strip()
    return value or default


def load_config(env: Mapping[str, str] | None = None) -> EtlConfig:
    """Construit la configuration depuis l'environnement.

    `env` est injectable pour que les tests n'aient pas à écrire dans
    os.environ, qui est un état global partagé entre tests.
    """
    source = os.environ if env is None else env
    database_url = source.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ConfigError(
            "DATABASE_URL est obligatoire. Copier .env.example en .env et le"
            " renseigner."
        )
    return EtlConfig(
        database_url=database_url,
        mock_api_url=read_text(source, "MOCK_API_URL", DEFAULT_MOCK_API_URL),
        mlflow_tracking_uri=read_text(
            source, "MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI
        ),
        mlflow_experiment=read_text(
            source, "MLFLOW_EXPERIMENT", DEFAULT_MLFLOW_EXPERIMENT
        ),
        batch_size=read_int(source, "ETL_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        request_timeout_s=read_float(
            source, "ETL_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S
        ),
        current_path=read_text(
            source, "MOCK_API_CURRENT_PATH", DEFAULT_CURRENT_PATH
        ),
        poll_interval_s=read_float(
            source, "ETL_POLL_INTERVAL_S", DEFAULT_POLL_INTERVAL_S
        ),
        poll_retries=read_int(source, "ETL_POLL_RETRIES", DEFAULT_POLL_RETRIES),
        poll_backoff_s=read_float(
            source, "ETL_POLL_BACKOFF_S", DEFAULT_POLL_BACKOFF_S
        ),
        poll_timeout_s=read_float(
            source, "ETL_POLL_TIMEOUT_S", DEFAULT_POLL_TIMEOUT_S
        ),
        lag_warning_s=read_float(
            source, "ETL_LAG_WARNING_S", DEFAULT_LAG_WARNING_S
        ),
    )


def _read_number(
    env: Mapping[str, str],
    name: str,
    default: NumberT,
    parse: Callable[[str], NumberT],
) -> NumberT:
    """Analyse une valeur numérique, en signalant une saisie illisible."""
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return parse(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{name} doit être un nombre, reçu {raw!r}."
        ) from exc
