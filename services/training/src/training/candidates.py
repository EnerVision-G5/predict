"""Les algorithmes que le challenge oppose, derrière une seule interface.

Un seul algorithme était câblé dans le service : le nom du run, celui du
modèle enregistré et le module qui l'ajuste disaient tous « xgboost ». Rien
n'était faux, mais rien ne permettait de vérifier que XGBoost était le bon
choix — ADR-010 le décide en comparant trois options, et la comparaison
n'existait que dans le document.

Un candidat n'est pas un jeu d'hyperparamètres, c'est une famille de modèles.
Régler `max_depth` produit un autre XGBoost ; opposer une régression
régularisée à une forêt aléatoire dit si la structure du problème est
non linéaire, ce qu'aucun réglage ne répondra.

Les deux candidats scikit-learn passent par un pipeline, et ce n'est pas de
l'ornement. XGBoost traite les valeurs manquantes nativement ; scikit-learn
lève. Or les premières heures d'un historique n'ont pas de décalage à 168 h :
sans imputation, ces candidats échoueraient sur des données que XGBoost
apprend sans broncher, et le classement dirait « erreur » là où il doit dire
« moins bon ». La mise à l'échelle, elle, ne concerne que la régression
régularisée : une pénalité qui compare des coefficients compare aussi les
unités de leurs variables, et une heure de la journée n'a pas l'ordre de
grandeur d'un kilowatt.
"""

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

DEFAULT_LEARNER = "xgboost"


class CandidateError(ValueError):
    """Le candidat demandé n'existe pas, ou refuse ses hyperparamètres."""


@dataclass(frozen=True)
class Learner:
    """Un algorithme candidat : comment on l'ajuste, comment on le nomme.

    L'ajustement est porté par une fonction et non par une méthode à
    redéfinir : les trois familles n'ont en commun que leur signature, et une
    hiérarchie de classes ne ferait qu'ajouter un niveau d'indirection à trois
    appels différents.
    """

    name: str
    fit: Callable[..., Any]
    uses_validation: bool


def _fit_forest(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, object],
) -> Any:
    """Ajuste une forêt aléatoire, valeurs manquantes imputées en amont."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(**dict(params)),
    ).fit(features, target)


def _fit_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, object],
) -> Any:
    """Ajuste la régression régularisée, mise à l'échelle en amont."""
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
    """Ajuste le modèle boosté, avec l'arrêt anticipé sur la validation."""
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
    """Retourne le candidat portant ce nom, ou dit lesquels existent.

    Le message énumère les noms connus : une faute de frappe dans `conf/` ne
    doit pas se lire comme une panne du service, et l'exploitant n'a pas à
    ouvrir le code pour retrouver l'orthographe attendue.
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
    """Ajuste le candidat demandé et retourne le modèle prêt à prédire.

    Tous reçoivent les mêmes arguments et seul l'arrêt anticipé consomme la
    validation. Écrire deux appels selon la famille, chez l'appelant, ferait
    de chaque ajout de candidat une modification du code qui les enchaîne.
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
