# **********************************************************************
# * Nom     : candidates.py                                            *
# * Type    : Module                                                   *
# * Sujet   : Familles de modèles opposées sur le banc d'arbitrage     *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.model import ModelParams
from training.model import fit as fit_boosted

# Famille apprise par un entraînement ordinaire.
DEFAULT_LEARNER = "xgboost"


class CandidateError(ValueError):
    """Classe : CandidateError
    Description : La famille demandée est inconnue ou n'a pas su apprendre.
    """


@dataclass(frozen=True)
class Learner:
    """Classe : Learner
    Description : Une famille : son nom, sa fonction d'ajustement, ses besoins.
    """
    name: str
    fit: Callable[..., Any]
    uses_validation: bool


def _fit_forest(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, object],
) -> Any:
    """Méthode : _fit_forest
    Description : Ajuste une forêt aléatoire, valeurs absentes imputées
      d'abord.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(**dict(params)),
    ).fit(features, target)


def _fit_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, object],
) -> Any:
    """Méthode : _fit_ridge
    Description : Ajuste une régression ridge sur des variables centrées
      réduites.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(**dict(params)),
    ).fit(features, target)


def _fit_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, object],
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    early_stopping_rounds: int,
) -> Any:
    """Méthode : _fit_xgboost
    Description : Ajuste un gradient boosté avec arrêt anticipé sur la
      validation.
    """
    return fit_boosted(
        features,
        target,
        valid_features,
        valid_target,
        ModelParams.from_mapping(params),
        early_stopping_rounds,
    )


LEARNERS: dict[str, Learner] = {
    DEFAULT_LEARNER: Learner(
        name=DEFAULT_LEARNER,
        fit=_fit_xgboost,
        uses_validation=True,
    ),
    "foret-aleatoire": Learner(
        name="foret-aleatoire",
        fit=_fit_forest,
        uses_validation=False,
    ),
    "ridge": Learner(
        name="ridge",
        fit=_fit_ridge,
        uses_validation=False,
    ),
}


def learner(name: str) -> Learner:
    """Méthode : learner
    Description : Retrouve une famille par son nom, et nomme les connues sinon.
    """
    try:
        return LEARNERS[name]
    except KeyError:
        raise CandidateError(
            f"Candidat inconnu : {name!r}. Connus : {sorted(LEARNERS)}."
        ) from None


def fit_candidate(
    name: str,
    params: Mapping[str, object],
    train_features: pd.DataFrame,
    train_target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    early_stopping_rounds: int,
) -> Any:
    """Méthode : fit_candidate
    Description : Ajuste la famille demandée en lui passant ce dont elle a
      besoin.
    """
    chosen = learner(name)
    try:
        if chosen.uses_validation:
            return chosen.fit(
                train_features,
                train_target,
                params,
                valid_features,
                valid_target,
                early_stopping_rounds,
            )
        return chosen.fit(train_features, train_target, params)
    except TypeError as exc:
        raise CandidateError(
            f"Hyperparamètres refusés par {name} : {exc}."
        ) from exc
