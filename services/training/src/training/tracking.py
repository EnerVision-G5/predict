# **********************************************************************
# * Nom     : tracking.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Journalisation des runs, des versions et des alias dans  *
# *   le registre MLflow                                               *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature

# Nom de l'artefact sous lequel le modèle est enregistré.
ARTIFACT_NAME = "model"

# Alias posé sur toute version fraîchement enregistrée.
STAGING_ALIAS = "challenger"

# Alias de la version réellement servie en production.
PRODUCTION_ALIAS = "champion"

# MLflow horodate en millisecondes, Python en secondes.
MILLISECONDS_PER_SECOND = 1000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingSettings:
    """Classe : TrackingSettings
    Description : Coordonnées du registre : son adresse, l'expérience où
      écrire, et le nom sous lequel le modèle est enregistré.
    """
    tracking_uri: str
    experiment: str
    registered_model: str


def connect(settings: TrackingSettings) -> None:
    """Méthode : connect
    Description : Pointe le client MLflow sur le registre configuré.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)


@contextmanager
def run(settings: TrackingSettings, run_name: str):
    """Méthode : run
    Description : Ouvre un run dans l'expérience et le referme à la sortie du
      bloc.
    """
    connect(settings)
    mlflow.set_experiment(settings.experiment)
    with mlflow.start_run(run_name=run_name) as active:
        yield active


def log_params(values: Mapping[str, Any]) -> None:
    """Méthode : log_params
    Description : Journalise les hyperparamètres du run courant.
    """
    mlflow.log_params(dict(values))


def log_metrics(values: Mapping[str, float]) -> None:
    """Méthode : log_metrics
    Description : Journalise les métriques du run courant.
    """
    mlflow.log_metrics(dict(values))


def set_tags(values: Mapping[str, Any]) -> None:
    """Méthode : set_tags
    Description : Pose les tags du run courant, valeurs converties en chaînes.
    """
    mlflow.set_tags({key: str(value) for key, value in values.items()})


def version_snapshot(name: str, version: str) -> AliasSnapshot:
    """Méthode : version_snapshot
    Description : Lit les métriques et paramètres d'une version donnée.
    """
    client = mlflow.MlflowClient()
    run_id = client.get_model_version(name, version).run_id
    data = client.get_run(run_id).data
    return AliasSnapshot(
        version=str(version),
        metrics=dict(data.metrics),
        params=dict(data.params),
    )


def log_model(
    model: Any,
    settings: TrackingSettings,
    features: pd.DataFrame,
    predictions: Any,
    tags: Mapping[str, Any] | None = None,
) -> str:
    """Méthode : log_model
    Description : Enregistre le modèle avec sa signature, le décrit par ses
      tags, et le marque challenger.
    """
    info = mlflow.xgboost.log_model(
        model,
        name=ARTIFACT_NAME,
        signature=infer_signature(features, predictions),
        input_example=features.head(5),
        registered_model_name=settings.registered_model,
    )
    version = _registered_version(info)
    if version:
        set_version_tags(settings.registered_model, version, tags or {})
        set_alias(settings.registered_model, version, STAGING_ALIAS)
    return version


def started_at(active: Any) -> datetime:
    """Méthode : started_at
    Description : Retourne l'instant de démarrage d'un run actif.
    """
    return _as_datetime(active.info.start_time)


def version_identity(name: str, version: str) -> tuple[str, datetime]:
    """Méthode : version_identity
    Description : Retourne le run d'une version et l'instant où il a démarré.
    """
    client = mlflow.MlflowClient()
    run_id = client.get_model_version(name, version).run_id
    return run_id, _as_datetime(client.get_run(run_id).info.start_time)


def _as_datetime(epoch_ms: int) -> datetime:
    """Méthode : _as_datetime
    Description : Convertit un horodatage MLflow en datetime UTC.
    """
    return datetime.fromtimestamp(epoch_ms / MILLISECONDS_PER_SECOND, tz=UTC)


def set_version_tags(
    name: str,
    version: str,
    tags: Mapping[str, Any],
) -> None:
    """Méthode : set_version_tags
    Description : Décrit une version enregistrée par un jeu de tags.
    """
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, str(value))
    if tags:
        logger.info(
            "modèle %s version %s décrit par %d tag(s)", name, version, len(tags)
        )


def set_alias(name: str, version: str, alias: str) -> None:
    """Méthode : set_alias
    Description : Déplace un alias du registre vers la version indiquée.
    """
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, alias, version)
    logger.info("modèle %s version %s marqué %s", name, version, alias)


def _registered_version(info: Any) -> str:
    """Méthode : _registered_version
    Description : Extrait le numéro de version, vide si le serveur n'a pas de
      Model Registry.
    """
    version = getattr(info, "registered_model_version", None)
    if version is None:
        logger.warning(
            "modèle journalisé sans version de registre : le serveur MLflow"
            " n'a pas de Model Registry, le service ne pourra pas le résoudre."
        )
        return ""
    return str(version)


@dataclass(frozen=True)
class AliasSnapshot:
    """Classe : AliasSnapshot
    Description : Ce qu'une version portait au moment où on l'a lue : son
      numéro, ses métriques et ses paramètres.
    """
    version: str
    metrics: dict[str, float]
    params: dict[str, str]


def alias_snapshot(name: str, alias: str) -> AliasSnapshot:
    """Méthode : alias_snapshot
    Description : Lit la version que porte un alias, avec ses mesures.
    """
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    data = client.get_run(version.run_id).data
    return AliasSnapshot(
        version=str(version.version),
        metrics=dict(data.metrics),
        params=dict(data.params),
    )


def baseline_metrics(name: str, alias: str) -> dict[str, float]:
    """Méthode : baseline_metrics
    Description : Retourne les seules métriques de la version aliasée.
    """
    return alias_snapshot(name, alias).metrics


def served_version(name: str, alias: str) -> str:
    """Méthode : served_version
    Description : Retourne le numéro de la version actuellement servie.
    """
    return str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)
