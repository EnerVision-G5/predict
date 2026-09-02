"""Enregistrement du run dans MLflow : paramètres, métriques, modèle, alias.

C'est la seule frontière entre l'entraînement et le service d'inférence, et
elle ne passe pas par un fichier. Un `model.pkl` que le service rechargerait
par `pickle.load` transporterait un objet Python et rien d'autre : ni les
colonnes attendues, ni leur ordre, ni la version des bibliothèques qui l'ont
produit. Le jour où l'entraînement monte de version, le service charge le
pickle sans broncher et prédit faux.

Un modèle enregistré dans MLflow transporte les trois. La signature dit les
colonnes et leurs types, l'environnement dit les versions, le registre dit
lequel est en service. Le service d'inférence résout un alias, pas un chemin :
promouvoir un modèle devient un déplacement d'alias, sans redéploiement, et le
retour arrière est le même geste en sens inverse.

Le run porte enfin de quoi être refait. `feature_version` et la fenêtre
d'apprentissage sont journalisées comme paramètres : deux modèles aux mêmes
hyperparamètres et aux mêmes métriques ne sont pas comparables s'ils n'ont pas
vu la même période ni les mêmes variables.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature

# Nom de l'artefact du modèle à l'intérieur du run. Il entre dans l'URI que le
# service résout : le changer casse les modèles déjà enregistrés.
ARTIFACT_NAME = "model"

# Alias posé sur la version qui vient d'être enregistrée. `challenger` et non
# `champion` : promouvoir est une décision, pas une conséquence automatique de
# la fin d'un entraînement. Le service, lui, résout `champion`.
STAGING_ALIAS = "challenger"

# Alias de la version réellement servie. C'est lui que le service d'inférence
# résout, et lui que la surveillance prend pour référence.
PRODUCTION_ALIAS = "champion"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingSettings:
    """Où le run est écrit, et sous quel nom le modèle est enregistré."""

    tracking_uri: str
    experiment: str
    registered_model: str


@contextmanager
def run(settings: TrackingSettings, run_name: str):
    """Ouvre un run MLflow, en fixant serveur et expérience au préalable.

    Le contexte est explicite pour que l'appelant ne puisse pas écrire de
    métrique hors run : une métrique orpheline part dans l'expérience par
    défaut, où plus personne ne la relie à son modèle.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment)
    with mlflow.start_run(run_name=run_name) as active:
        yield active


def log_params(values: Mapping[str, Any]) -> None:
    """Journalise les paramètres du run, hyperparamètres et provenance."""
    mlflow.log_params(dict(values))


def log_metrics(values: Mapping[str, float]) -> None:
    """Journalise les métriques comparables entre deux runs."""
    mlflow.log_metrics(dict(values))


def log_model(
    model: Any,
    settings: TrackingSettings,
    features: pd.DataFrame,
    predictions: Any,
    tags: Mapping[str, Any] | None = None,
) -> str:
    """Enregistre le modèle avec sa signature et retourne sa version.

    La signature est déduite d'un vrai lot de validation et de vraies
    prédictions : déclarée à la main, elle finirait par décrire ce que le code
    croit produire plutôt que ce qu'il produit.
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


def set_version_tags(
    name: str,
    version: str,
    tags: Mapping[str, Any],
) -> None:
    """Décrit la version dans le registre, à côté de son alias.

    Un tag et un alias ne disent pas la même chose. L'alias désigne un rôle —
    qui est servi aujourd'hui — et se déplace ; le tag décrit la version et ne
    bouge plus. Quelqu'un qui ouvre le registre six mois plus tard voit sur
    quelles variables et sur quelle période cette version a été entraînée sans
    avoir à retrouver son run.

    C'est aussi ce qui rend la surveillance possible : la dérive se mesure par
    rapport à ce que la version affichait à l'entraînement, et il faut savoir
    ce qu'elle était.
    """
    client = mlflow.MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, str(value))
    if tags:
        logger.info(
            "modèle %s version %s décrit par %d tag(s)", name, version, len(tags)
        )


def set_alias(name: str, version: str, alias: str) -> None:
    """Pose un alias sur une version du registre.

    Promouvoir, c'est déplacer un alias — pas reconstruire une image ni
    redéployer un service. Le retour arrière est le même geste en sens
    inverse, ce qui en fait une opération qu'on ose faire.
    """
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, alias, version)
    logger.info("modèle %s version %s marqué %s", name, version, alias)


def _registered_version(info: Any) -> str:
    """Retourne la version que le registre vient d'attribuer, si elle existe.

    Un serveur MLflow sans backend relationnel n'a pas de registre : le run
    est alors journalisé sans version, ce qui est utilisable pour comparer des
    entraînements mais pas pour en servir un. Le cas est signalé, pas masqué.
    """
    version = getattr(info, "registered_model_version", None)
    if version is None:
        logger.warning(
            "modèle journalisé sans version de registre : le serveur MLflow"
            " n'a pas de Model Registry, le service ne pourra pas le résoudre."
        )
        return ""
    return str(version)


def baseline_metrics(name: str, alias: str) -> dict[str, float]:
    """Retourne les métriques du run qui a produit la version aliasée.

    C'est la référence de la surveillance : un écart n'a de sens que rapporté
    à ce que le modèle savait faire quand on l'a accepté. La chercher dans le
    registre plutôt que dans un fichier garantit qu'elle suit le modèle —
    promouvoir une autre version change la référence du même geste.
    """
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    return dict(client.get_run(version.run_id).data.metrics)


def served_version(name: str, alias: str) -> str:
    """Retourne la version du registre que l'alias désigne aujourd'hui."""
    return str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)
