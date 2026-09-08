# **********************************************************************
# * Nom     : model.py                                                 *
# * Type    : Module                                                   *
# * Sujet   : Ajustement du modèle retenu et mesure de sa qualité      *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Nombre de résidus sous lequel un écart-type n'a pas de sens.
MINIMUM_RESIDUALS = 2

# Métrique sur laquelle se prend la décision de promotion.
DECISION_METRIC = "mae"


@dataclass(frozen=True)
class ModelParams:
    """Classe : ModelParams
    Description : Hyperparamètres du gradient boosté, avec leurs défauts.
    """
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    random_state: int = 42

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> ModelParams:
        """Méthode : from_mapping
        Description : Construit les hyperparamètres depuis la configuration, en
          refusant l'inconnu.
        """
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
        """Méthode : as_dict
        Description : Rend les hyperparamètres sous la forme attendue par la
          bibliothèque.
        """
        return asdict(self)


def fit(
    features: pd.DataFrame,
    target: pd.Series,
    valid_features: pd.DataFrame,
    valid_target: pd.Series,
    params: ModelParams,
    early_stopping_rounds: int,
) -> XGBRegressor:
    """Méthode : fit
    Description : Ajuste le modèle avec arrêt anticipé sur le bloc de
      validation.
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
    """Méthode : evaluate
    Description : Mesure l'écart entre observé et prédit : MAE, RMSE, R².
    """
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
    }


def residual_std(observed: pd.Series, predicted: Sequence[float]) -> float:
    """Méthode : residual_std
    Description : Écart-type des résidus, base de l'intervalle de confiance
      servi.
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
    """Méthode : best_iteration
    Description : Nombre d'arbres retenus par l'arrêt anticipé.
    """
    return int(getattr(model, "best_iteration", 0) or 0) + 1
