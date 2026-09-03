"""Résolution du modèle servi, par alias et non par version figée.

Le service ne charge pas un fichier, il résout `models:/nom@champion` dans le
registre MLflow. La différence est une propriété d'exploitation : promouvoir
un modèle devient un déplacement d'alias, sans image à reconstruire ni service
à redéployer, et le retour arrière est le même geste en sens inverse. Un chemin
figé dans la configuration aurait fait de chaque promotion un déploiement.

C'est aussi ici, et nulle part ailleurs, qu'on sait comment parler au modèle
chargé. Sa signature dit les colonnes attendues, leur ordre et leurs types, et
MLflow refuse une conversion qu'il ne peut pas garantir sans perte — un `hour`
en int64 présenté à un modèle qui l'a vu en int32 est rejeté. Le module
présente donc les variables exactement comme la signature les déclare, ce qui
évite au reste du service d'avoir à deviner ces types, et de finir par en
tenir une seconde copie qui divergerait de la première.

Le chargement est tolérant à l'échec, et volontairement. Un registre
injoignable au démarrage ne doit pas empêcher le processus de vivre : le
conteneur redémarrerait en boucle, la sonde de disponibilité ne répondrait
jamais, et l'hébergeur conclurait à une panne du service alors que la panne
est chez MLflow. Le service démarre donc sans modèle, le dit dans son journal,
et refuse les prédictions jusqu'à ce qu'un rechargement aboutisse.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

# Préfixe des URI du registre. Un modèle désigné autrement — un chemin
# d'artefact, un run — n'a pas de version de registre à résoudre.
ALIAS_PREFIX = "models:/"

# Nom du tag posé par `training.tracking` sur chaque version enregistrée. Écrit
# des deux côtés, il est la frontière entre l'entraînement et le service : le
# changer d'un seul côté ferait servir des prévisions sans bornes, en silence.
RESIDUAL_STD_TAG = "residual_std"

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Aucun modèle n'est chargé : le registre n'a rien su résoudre."""


class UnservableModel(ValueError):
    """Le modèle résolu n'a pas de signature, donc rien ne dit quoi lui donner."""


@dataclass(frozen=True)
class LoadedModel:
    """Le modèle servi, sa version, et la forme d'entrée qu'il déclare.

    `version` part dans `PredictionOut.model_version`, que les consommateurs
    du contrat lisent pour rattacher une prévision au modèle qui l'a produite.
    Sans elle, une prévision aberrante ne serait imputable à rien.

    `residual_std` est la dispersion de l'erreur mesurée sur le jeu de test au
    moment de l'entraînement. Elle est portée par un tag de la version, donc
    elle suit le modèle : promouvoir une autre version change la largeur des
    intervalles du même geste. Nulle quand la version ne la déclare pas — un
    modèle enregistré avant cette mesure reste servable, il rend simplement
    une prévision sans bornes plutôt qu'une bande inventée.
    """

    model: Any
    uri: str
    version: str
    columns: tuple[str, ...]
    dtypes: dict[str, Any]
    residual_std: float | None = None

    def predict(self, features: pd.DataFrame) -> Any:
        """Applique le modèle à un lot, présenté comme sa signature l'exige."""
        return self.model.predict(self.conform(features))

    def conform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ordonne et type les colonnes comme la signature les déclare.

        L'appelant fournit des valeurs, pas des types : c'est le modèle qui
        sait en quoi il les attend, et lui seul. Sans cette mise en forme, une
        colonne entière construite en int64 par pandas serait refusée par un
        modèle entraîné sur de l'int32, alors que la valeur est la bonne.
        """
        missing = [name for name in self.columns if name not in features.columns]
        if missing:
            raise UnservableModel(
                f"Variables absentes du lot présenté au modèle : {missing}."
            )
        ordered = features[list(self.columns)]
        return ordered.astype(self.dtypes)


class ModelRegistry:
    """Détient le modèle courant du processus, et sait le remplacer.

    Le verrou n'est pas décoratif : uvicorn sert plusieurs requêtes de front,
    et un rechargement qui remplacerait le modèle en pleine prédiction
    laisserait une requête lire un objet à moitié échangé.
    """

    def __init__(self, tracking_uri: str, model_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.model_uri = model_uri
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    @property
    def is_ready(self) -> bool:
        """Dit si une prévision peut être servie maintenant."""
        return self._loaded is not None

    def current(self) -> LoadedModel:
        """Retourne le modèle servi, ou refuse explicitement de répondre."""
        loaded = self._loaded
        if loaded is None:
            raise ModelUnavailable(
                f"Aucun modèle chargé depuis {self.model_uri}. Vérifier que le"
                " registre MLflow est joignable et que l'alias existe."
            )
        return loaded

    def load(self) -> LoadedModel | None:
        """Charge le modèle désigné par l'alias, sans faire tomber le service.

        Le retour est `None` en cas d'échec plutôt qu'une exception : au
        démarrage, l'appelant ne peut rien faire d'une exception sinon sortir,
        et sortir est précisément ce qu'il ne faut pas faire.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            model = mlflow.pyfunc.load_model(self.model_uri)
            loaded = _describe(model, self.model_uri)
        except Exception as exc:  # noqa: BLE001 - le registre lève large
            logger.error("modèle %s non chargé : %s", self.model_uri, exc)
            return None
        with self._lock:
            self._loaded = loaded
        logger.info(
            "modèle servi : %s (version %s, %d variable(s))",
            loaded.uri,
            loaded.version,
            len(loaded.columns),
        )
        return loaded

    def input_columns(self) -> list[str]:
        """Retourne les colonnes attendues, dans l'ordre de la signature.

        L'ordre vient du modèle et non de la configuration du service : c'est
        la seule façon d'être certain que le service présente ses variables
        comme l'entraînement les a vues.
        """
        return list(self.current().columns)


def _describe(model: Any, model_uri: str) -> LoadedModel:
    """Extrait de la signature ce qu'il faut savoir pour appeler le modèle.

    Une signature absente est un modèle enregistré hors du chemin prévu : rien
    ne dit alors quelles variables lui présenter ni dans quel ordre. Le refus
    est immédiat, au chargement, plutôt qu'à la première requête — un modèle
    inservable ne doit pas être annoncé comme servi.
    """
    signature = getattr(model.metadata, "signature", None)
    if signature is None or signature.inputs is None:
        raise UnservableModel(
            f"Le modèle {model_uri} n'a pas de signature : impossible de savoir"
            " quelles variables lui présenter, ni dans quel ordre."
        )
    inputs = signature.inputs
    # Une seule interrogation du registre pour la version et les tags : deux
    # appels pourraient tomber de part et d'autre d'une promotion et décrire
    # deux versions différentes dans un même modèle chargé.
    entry = _registry_entry(model_uri)
    return LoadedModel(
        model=model,
        uri=model_uri,
        version=_version_of(entry, model, model_uri),
        columns=tuple(inputs.input_names()),
        dtypes=dict(zip(inputs.input_names(), inputs.numpy_types(), strict=True)),
        residual_std=_residual_std_of(entry, model_uri),
    )


def _registry_entry(model_uri: str) -> Any:
    """Retourne la version de registre que l'alias désigne, ou None.

    None n'est pas une panne : un modèle chargé par chemin d'artefact plutôt
    que par alias n'a pas d'entrée de registre, et reste servable.
    """
    name, alias = _parse_alias(model_uri)
    if not name:
        return None
    try:
        return mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    except Exception as exc:  # noqa: BLE001 - le registre lève large
        logger.warning("alias %s non résolu dans le registre : %s", model_uri, exc)
        return None


def _residual_std_of(entry: Any, model_uri: str) -> float | None:
    """Lit la dispersion de l'erreur dans les tags de la version.

    Une valeur nulle ou négative est refusée plutôt que servie : elle
    produirait une bande de largeur nulle, c'est-à-dire une prévision annoncée
    comme certaine. Mieux vaut pas d'intervalle qu'un intervalle faux.
    """
    tags = dict(getattr(entry, "tags", None) or {})
    raw = tags.get(RESIDUAL_STD_TAG)
    if raw is None:
        logger.warning(
            "version servie par %s sans tag %s : les prévisions partiront"
            " sans intervalle. Réentraîner pose ce tag.",
            model_uri,
            RESIDUAL_STD_TAG,
        )
        return None
    try:
        spread = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "tag %s illisible (%r), prévisions sans intervalle",
            RESIDUAL_STD_TAG,
            raw,
        )
        return None
    if spread <= 0:
        logger.warning(
            "tag %s non strictement positif (%s), prévisions sans intervalle",
            RESIDUAL_STD_TAG,
            spread,
        )
        return None
    return spread


def _version_of(entry: Any, model: Any, model_uri: str) -> str:
    """Retourne la version du registre que l'alias désigne aujourd'hui.

    C'est cette valeur, et non l'identifiant interne du modèle, que
    `PredictionOut.model_version` doit porter : le contrat la décrit comme le
    tag MLflow du modèle, et c'est elle qu'un exploitant lit dans le registre
    pour savoir ce qui est promu. Elle est résolue au chargement, pas à chaque
    requête, sans quoi le service interrogerait le registre à chaque prévision.

    Un modèle chargé par chemin d'artefact plutôt que par alias n'a pas de
    version : l'identifiant interne prend alors le relais. Moins précis, mais
    c'est une trace, là où une chaîne vide n'en serait pas une.
    """
    version = getattr(entry, "version", None)
    if version is not None:
        return str(version)
    metadata = getattr(model, "metadata", None)
    return str(getattr(metadata, "model_uuid", None) or model_uri)


def _parse_alias(model_uri: str) -> tuple[str, str]:
    """Décompose `models:/nom@alias` en son nom et son alias."""
    if not model_uri.startswith(ALIAS_PREFIX) or "@" not in model_uri:
        return "", ""
    name, _, alias = model_uri[len(ALIAS_PREFIX) :].partition("@")
    return name, alias
