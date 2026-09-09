# **********************************************************************
# * Nom     : loader.py                                                *
# * Type    : Module                                                   *
# * Sujet   : Résolution et chargement du modèle servi depuis le       *
# *   registre MLflow                                                  *
# * Service : serving                                                  *
# **********************************************************************

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

# Préfixe d'une URI de modèle désignée par alias.
ALIAS_PREFIX = "models:/"

# Tag portant l'écart-type des résidus, base de l'intervalle.
RESIDUAL_STD_TAG = "residual_std"

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Classe : ModelUnavailable
    Description : Aucun modèle n'est chargé, le service ne peut rien prédire.
    """


class UnservableModel(ValueError):
    """Classe : UnservableModel
    Description : Le modèle chargé ne dit pas ce qu'il attend en entrée.
    """


@dataclass(frozen=True)
class LoadedModel:
    """Classe : LoadedModel
    Description : Un modèle chargé et ce qu'il faut pour lui présenter des
      variables.
    """
    model: Any
    uri: str
    version: str
    columns: tuple[str, ...]
    dtypes: dict[str, Any]
    residual_std: float | None = None

    def predict(self, features: pd.DataFrame) -> Any:
        """Méthode : predict
        Description : Prédit à partir d'un lot de variables conformé à sa
          signature.
        """
        return self.model.predict(self.conform(features))

    def conform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Méthode : conform
        Description : Réordonne et retype les variables selon la signature du
          modèle.
        """
        missing = [name for name in self.columns if name not in features.columns]
        if missing:
            raise UnservableModel(
                f"Variables absentes du lot présenté au modèle : {missing}."
            )
        ordered = features[list(self.columns)]
        return ordered.astype(self.dtypes)


class ModelRegistry:
    """Classe : ModelRegistry
    Description : Détient le modèle servi et son rechargement, sous verrou.
    """
    def __init__(self, tracking_uri: str, model_uri: str) -> None:
        """Méthode : __init__
        Description : Retient l'adresse du registre et l'URI du modèle à
          servir.
        """
        self.tracking_uri = tracking_uri
        self.model_uri = model_uri
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    @property
    def is_ready(self) -> bool:
        """Méthode : is_ready
        Description : Dit si un modèle est chargé et prêt à prédire.
        """
        return self._loaded is not None

    def current(self) -> LoadedModel:
        """Méthode : current
        Description : Rend le modèle chargé, ou refuse si aucun ne l'est.
        """
        loaded = self._loaded
        if loaded is None:
            raise ModelUnavailable(
                f"Aucun modèle chargé depuis {self.model_uri}. Vérifier que le"
                " registre MLflow est joignable et que l'alias existe."
            )
        return loaded

    def load(self) -> LoadedModel | None:
        """Méthode : load
        Description : Charge le modèle depuis le registre, sans faire tomber le
          service s'il échoue.
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
        """Méthode : input_columns
        Description : Énumère les variables que le modèle chargé attend.
        """
        return list(self.current().columns)


def _describe(model: Any, model_uri: str) -> LoadedModel:
    """Méthode : _describe
    Description : Décrit un modèle chargé : colonnes, types et écart-type des
      résidus.
    """
    signature = getattr(model.metadata, "signature", None)
    if signature is None or signature.inputs is None:
        raise UnservableModel(
            f"Le modèle {model_uri} n'a pas de signature : impossible de savoir"
            " quelles variables lui présenter, ni dans quel ordre."
        )
    inputs = signature.inputs
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
    """Méthode : _registry_entry
    Description : Retrouve l'entrée de registre correspondant à un alias.
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
    """Méthode : _residual_std_of
    Description : Lit l'écart-type des résidus, en avertissant s'il manque.
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
    """Méthode : _version_of
    Description : Détermine le numéro de version d'un modèle chargé.
    """
    version = getattr(entry, "version", None)
    if version is not None:
        return str(version)
    metadata = getattr(model, "metadata", None)
    return str(getattr(metadata, "model_uuid", None) or model_uri)


def _parse_alias(model_uri: str) -> tuple[str, str]:
    """Méthode : _parse_alias
    Description : Décompose une URI models:/nom@alias en ses deux parties.
    """
    if not model_uri.startswith(ALIAS_PREFIX) or "@" not in model_uri:
        return "", ""
    name, _, alias = model_uri[len(ALIAS_PREFIX) :].partition("@")
    return name, alias
