# **********************************************************************
# * Nom     : tracking.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Journalisation des runs, des versions et des alias dans  *
# *   le registre MLflow                                               *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature

# Repli si l'appelant ne nomme pas son modèle. MLflow 3 en fait une entité
# nommée dans l'onglet « Models » : les appelants passent leur nom de run,
# sinon toutes les lignes s'appellent `model`.
ARTIFACT_NAME = "model"

# Chemin de l'artefact où atterrit le journal d'un run.
LOG_ARTIFACT = "logs/run.log"

# Format du journal capturé. Identique à celui de la sortie standard : ce qui
# est relu des mois plus tard dans MLflow doit se lire comme ce que
# l'exploitant avait sous les yeux le jour du run.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Types que skops accepte de relire. Il refuse tout ce qu'il ne connaît pas,
# et c'est ce qui le rend plus sûr que pickle. Sans cette liste, log_model
# échoue et le run garde ses métriques mais pas son modèle.
SKOPS_TRUSTED_TYPES = (
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
)

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
def run(settings: TrackingSettings, run_name: str, nested: bool = False):
    """Méthode : run
    Description : Ouvre un run dans l'expérience et le referme à la sortie du
      bloc.

      `nested` rattache le run à celui qui l'englobe. C'est ce qui donne à
      l'arbitrage une arborescence dans l'interface : un parent qui porte la
      confrontation, un enfant par candidat. Sans lui, les candidats sont des
      runs frères que rien ne relie, et retrouver « quels modèles se sont
      affrontés ce jour-là » demande de filtrer sur un tag à la main.

      Le journal de tout ce qui se dit dans le bloc est attaché au run avant
      sa fermeture. Un run qui porte ses métriques sans porter ce qui les a
      produites oblige à retrouver la sortie console du jour même, quand elle
      existe encore.
    """
    connect(settings)
    mlflow.set_experiment(settings.experiment)
    with (
        mlflow.start_run(run_name=run_name, nested=nested) as active,
        _captured_logs(),
    ):
        yield active


@contextmanager
def _captured_logs(artifact_file: str = LOG_ARTIFACT):
    """Méthode : _captured_logs
    Description : Recopie le journal émis dans le bloc vers un artefact du run.

      La capture est branchée sur le logger racine et non sur celui de ce
      module : ce qui intéresse l'exploitant qui relit un run, ce sont les
      lignes de l'entraînement, du jeu de données et de l'arbitrage, pas les
      nôtres.

      Un échec d'écriture ne fait pas échouer le run : le modèle est appris,
      ses métriques sont posées, et perdre le journal ne vaut pas de perdre
      cela.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.flush()
        text = buffer.getvalue()
        if text:
            try:
                mlflow.log_text(text, artifact_file)
            except Exception as exc:  # noqa: BLE001 - annexe : rien ne remonte
                logger.warning("journal du run non attaché : %s", exc)


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


def log_table(frame: pd.DataFrame, artifact_file: str) -> None:
    """Méthode : log_table
    Description : Attache un tableau au run, affichable tel quel dans
      l'interface.

      Le classement d'un arbitrage n'est pas une métrique : c'est une
      comparaison, et une comparaison se lit en lignes et en colonnes. Posée
      en métriques du run parent, elle se retrouverait éclatée en autant de
      nombres sans lien apparent.
    """
    _attach(lambda: mlflow.log_table(frame, artifact_file), artifact_file)


def log_text(text: str, artifact_file: str) -> None:
    """Méthode : log_text
    Description : Attache un texte au run, tel qu'il se lirait dans un terminal.
    """
    _attach(lambda: mlflow.log_text(text, artifact_file), artifact_file)


def _attach(write: Any, artifact_file: str) -> bool:
    """Méthode : _attach
    Description : Écrit un artefact, et se contente d'un avertissement s'il ne
      part pas. Rend l'issue, pour que l'appelant en tienne compte.

      Une exception ferait perdre l'entraînement qui l'a produit. Mais tous les
      artefacts ne se valent pas : un journal manquant se lit ailleurs, un
      MODÈLE manquant rend inservable la version inscrite depuis lui.
    """
    try:
        write()
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("artefact %s non attaché : %s", artifact_file, exc)
        return False
    return True


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
    name: str = ARTIFACT_NAME,
) -> str:
    """Méthode : log_model
    Description : Enregistre le modèle avec sa signature, le décrit par ses
      tags, et le marque challenger.

      `name` nomme le modèle dans l'onglet « Models » de l'expérience. On y
      passe le nom du run, pour qu'une ligne de cet onglet dise d'elle-même de
      quelle version de variables et de quelle famille elle vient.
    """
    info = mlflow.xgboost.log_model(
        model,
        name=name,
        signature=infer_signature(features, predictions),
        input_example=features.head(5),
        registered_model_name=settings.registered_model,
    )
    version = _registered_version(info)
    if version:
        set_version_tags(settings.registered_model, version, tags or {})
        set_alias(settings.registered_model, version, STAGING_ALIAS)
    return version


def log_candidate_model(
    model: Any,
    features: pd.DataFrame,
    predictions: Any,
    name: str = ARTIFACT_NAME,
) -> str:
    """Méthode : log_candidate_model
    Description : Attache le modèle d'un candidat à son run, sans l'inscrire au
      registre.

      Le registre catalogue ce qui est DÉPLOYABLE ; la comparaison, elle, se
      lit dans l'expérience, où MLflow sait confronter des runs. Y verser une
      version par famille à chaque arbitrage ferait du catalogue un journal
      d'expériences, au milieu duquel il faudrait retrouver ce qui tourne
      réellement.

      Le modèle est attaché quand même, et ce n'est pas contradictoire : sans
      l'objet, un run d'arbitrage ne garde que des nombres, et l'on ne peut ni
      rejouer une prédiction du perdant pour comprendre POURQUOI il a perdu, ni
      l'inscrire le jour où il gagne — c'est de ce modèle journalisé que part
      `register_logged_model`.

      `name` le nomme dans l'onglet « Models » : on y passe le nom du run,
      sinon les six candidats d'un arbitrage y sont six lignes `model`.

      `mlflow.sklearn` et non `mlflow.xgboost` : les trois familles exposent
      l'interface sklearn, et une seule saveur évite de faire dépendre
      l'enregistrement de la famille arbitrée.

      Rend l'identifiant du modèle journalisé, vide s'il n'est pas parti :
      inscrire une version qui pointe vers un modèle absent passerait la règle
      de promotion, puis refuserait de se charger en production.
    """
    logged: list[str] = []

    def write() -> None:
        info = mlflow.sklearn.log_model(
            model,
            name=name,
            signature=infer_signature(features, predictions),
            input_example=features.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
        )
        logged.append(str(info.model_id))

    if not _attach(write, name):
        return ""
    return logged[0] if logged else ""


def register_logged_model(
    settings: TrackingSettings,
    model_id: str,
    tags: Mapping[str, Any] | None = None,
) -> str:
    """Méthode : register_logged_model
    Description : Inscrit au registre un modèle déjà journalisé, sans poser
      d'alias.

      L'inscription part du modèle journalisé et non d'un objet en mémoire :
      la version enregistrée pointe alors vers le run qui l'a produit, et la
      lignée se remonte depuis le catalogue jusqu'aux métriques et au journal
      de l'arbitrage qui l'a désignée.

      `models:/<id>` et non `runs:/<id>/<nom>` : MLflow 3 ne range plus les
      modèles parmi les artefacts du run. Le chemin d'artefact ne tenait que
      par un repli, qui suppose un seul modèle par run.

      Aucun alias n'est posé, et c'est la différence avec `log_model` : gagner
      un banc ne met pas en service. `challenger` désigne ce qui sort d'un
      entraînement ordinaire, `champion` ce qui est servi, et la promotion
      reste une décision qui passe par sa propre règle.
    """
    try:
        version = mlflow.register_model(
            f"models:/{model_id}", settings.registered_model
        )
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("vainqueur non inscrit au registre : %s", exc)
        return ""
    set_version_tags(settings.registered_model, version.version, tags or {})
    return str(version.version)


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
