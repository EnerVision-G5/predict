"""Modèle XGBoost : hyperparamètres, ajustement, métriques.

Le module ne sait rien de MLflow ni du stockage. Il reçoit trois tableaux et
rend un modèle et des métriques, ce qui le rend testable sans serveur de suivi
et sans partition sur disque. Le suivi est le métier de `training.tracking`.

L'arrêt anticipé regarde le bloc de validation, jamais celui de test. C'est ce
qui sépare les deux blocs : la validation sert à décider quand s'arrêter, donc
le modèle finit par s'y ajuster ; le test n'est regardé qu'une fois, et c'est
la seule mesure qu'on ait le droit de comparer entre deux runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# En deçà, un écart-type ne décrit rien : deux points suffisent à en produire
# un, et le service en tirerait une bande dont la largeur serait un hasard.
MINIMUM_RESIDUALS = 2

# Métrique qui porte les décisions prises sur ces mesures — promouvoir une
# version, déclarer une dérive. La MAE plutôt que la RMSE : elle s'exprime en
# kilowatts moyens, ce qui se discute avec un exploitant, là où la RMSE
# amplifie les grands écarts et se compare mal d'un site à l'autre.
#
# Définie ici et non chez ceux qui décident : l'arbitrage et la surveillance
# tranchent sur la même grandeur, et deux constantes finiraient par diverger
# le jour où l'une des deux changerait.
DECISION_METRIC = "mae"


@dataclass(frozen=True)
class ModelParams:
    """Hyperparamètres du modèle, journalisés tels quels dans MLflow.

    Un dataclass et non un dictionnaire : une faute de frappe dans un nom
    d'hyperparamètre passerait inaperçue dans un dictionnaire, XGBoost
    l'ignorerait, et le run serait enregistré avec un paramètre qui n'a rien
    réglé.
    """

    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    random_state: int = 42

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ModelParams:
        """Construit les hyperparamètres depuis le bloc `training.params`."""
        known = {
            field: values[field]
            for field in cls.__annotations__
            if field in values
        }
        unknown = sorted(set(values) - set(cls.__annotations__))
        if unknown:
            raise ValueError(
                f"Hyperparamètres inconnus dans conf/ : {unknown}. Les ajouter"
                " à ModelParams, ou corriger le fichier."
            )
        return cls(**known)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, object]:
        """Retourne les hyperparamètres sous la forme attendue par XGBoost."""
        return asdict(self)


def fit(
    features: pd.DataFrame,
    target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    params: ModelParams,
    early_stopping_rounds: int,
) -> XGBRegressor:
    """Entraîne le régresseur, en s'arrêtant quand la validation cesse de gagner.

    Sans arrêt anticipé, `n_estimators` serait un pari : trop bas, le modèle
    n'apprend pas ; trop haut, il apprend le bruit du jeu d'apprentissage. Ici
    il devient une borne haute, et le nombre d'arbres réellement retenu est
    journalisé avec le run.
    """
    model = XGBRegressor(
        **params.as_dict(),
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
    )
    model.fit(
        features, target, eval_set=[(valid_features, valid_target)], verbose=False
    )
    return model


def evaluate(observed: pd.Series, predicted: Sequence[float]) -> dict[str, float]:
    """Retourne les métriques de qualité comparables entre deux runs."""
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
    }


def residual_std(observed: pd.Series, predicted: Sequence[float]) -> float:
    """Écart-type des résidus sur le bloc de test, en kilowatts.

    C'est la dispersion de l'erreur du modèle autour de zéro, et c'est ce qui
    permet au service d'inférence d'assortir sa prévision d'un intervalle.
    Mesurée sur le test et non sur la validation : la validation a décidé de
    l'arrêt, le modèle s'y est donc ajusté et sa dispersion y est optimiste.

    L'écart-type et non l'erreur absolue moyenne, parce que c'est lui qui se
    compose : deux erreurs indépendantes s'additionnent en variance, ce dont
    la récurrence du service a besoin pour élargir sa bande avec l'horizon.

    Le degré de liberté est laissé à 1 (`ddof=1`) : la moyenne des résidus est
    estimée sur ces mêmes points, et la retenir sans le dire sous-estimerait
    la dispersion — d'autant plus que le bloc de test est court.
    """
    residuals = pd.Series(observed).reset_index(drop=True) - pd.Series(
        list(predicted)
    ).reset_index(drop=True)
    if len(residuals) < MINIMUM_RESIDUALS:
        raise ValueError(
            "Un écart-type de résidus demande au moins"
            f" {MINIMUM_RESIDUALS} points de test, {len(residuals)} fourni(s)."
        )
    return float(residuals.std(ddof=1))


def best_iteration(model: XGBRegressor) -> int:
    """Nombre d'arbres réellement retenus par l'arrêt anticipé.

    Comparé à `n_estimators`, il dit si la borne a été atteinte : si oui, le
    modèle gagnait encore quand on l'a coupé, et la borne est trop basse.
    """
    return int(getattr(model, "best_iteration", 0) or 0) + 1
