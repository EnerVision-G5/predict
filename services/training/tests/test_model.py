"""Modèle : hyperparamètres, ajustement, métriques.

Le run MLflow est un effet de bord, testé par une exécution réelle et non par
la CI. Ce qui se teste ici sans dépendance, c'est que le modèle apprend un
signal qu'on lui a mis, que l'arrêt anticipé regarde bien la validation, et
que les hyperparamètres lus dans `conf/` sont ceux qu'on croit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_common.schemas import feature_columns
from training.model import (
    ModelParams,
    best_iteration,
    evaluate,
    fit,
    residual_std,
)

COLUMNS = feature_columns((1, 24), 2)
ROWS = 300


def learnable(rows: int = ROWS) -> tuple[pd.DataFrame, pd.Series]:
    """Jeu synthétique linéaire, appris exactement par un arbre boosté."""
    generator = np.random.default_rng(seed=42)
    frame = pd.DataFrame({column: generator.normal(size=rows) for column in COLUMNS})
    return frame, 3.0 * frame["lag_1h"] + 2.0


class TestModelParams:
    """Un dataclass, et non un dictionnaire d'hyperparamètres."""

    def test_the_seed_is_fixed(self) -> None:
        # Sans graine fixe, deux runs aux mêmes paramètres ne seraient pas
        # comparables.
        assert ModelParams().random_state == 42

    def test_the_configuration_block_is_read(self) -> None:
        params = ModelParams.from_mapping({"n_estimators": 120, "max_depth": 4})
        assert params.n_estimators == 120
        assert params.max_depth == 4

    def test_the_unspecified_values_keep_their_default(self) -> None:
        assert ModelParams.from_mapping({"max_depth": 4}).learning_rate == 0.05

    def test_a_typo_in_the_configuration_is_refused(self) -> None:
        # XGBoost ignorerait un paramètre inconnu, et le run serait enregistré
        # avec un hyperparamètre qui n'a rien réglé.
        with pytest.raises(ValueError, match="n_estimator"):
            ModelParams.from_mapping({"n_estimator": 120})


def test_fit_learns_the_signal() -> None:
    features, target = learnable()
    valid_features, valid_target = learnable(rows=80)
    model = fit(
        features, target, valid_features, valid_target, ModelParams(), 20
    )
    assert evaluate(target, model.predict(features))["r2"] > 0.9


def test_early_stopping_reads_the_validation_block() -> None:
    # C'est ce qui sépare validation et test : le modèle s'ajuste à la
    # première, la seconde n'est regardée qu'une fois.
    features, target = learnable()
    valid_features, valid_target = learnable(rows=80)
    params = ModelParams(n_estimators=500)
    model = fit(features, target, valid_features, valid_target, params, 5)
    assert best_iteration(model) <= params.n_estimators


def test_evaluate_returns_the_three_comparable_metrics() -> None:
    observed = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate(observed, [1.0, 2.0, 3.0])
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_evaluate_measures_a_wrong_prediction() -> None:
    metrics = evaluate(pd.Series([1.0, 2.0, 3.0]), [2.0, 3.0, 4.0])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["r2"] < 1.0


def test_best_iteration_counts_the_trees_actually_kept() -> None:
    # Comparé à n_estimators, il dit si la borne a été atteinte — donc si le
    # modèle gagnait encore quand on l'a coupé.
    features, target = learnable()
    valid_features, valid_target = learnable(rows=80)
    model = fit(features, target, valid_features, valid_target, ModelParams(), 10)
    assert best_iteration(model) >= 1


def test_residual_std_is_zero_on_a_perfect_prediction() -> None:
    assert residual_std(pd.Series([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]) == 0.0


def test_residual_std_measures_the_spread_not_the_bias() -> None:
    # Une erreur constante ne disperse rien : le modèle se trompe, mais de
    # façon prévisible, et l'intervalle servi n'a pas à s'en élargir.
    observed = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert residual_std(observed, [2.0, 3.0, 4.0, 5.0]) == pytest.approx(0.0)
    assert residual_std(observed, [2.0, 1.0, 4.0, 3.0]) > 0.0


def test_residual_std_ignores_the_index_of_the_observations() -> None:
    # Le bloc de test est une tranche d'un tableau plus grand : son index ne
    # part pas de zéro, là où les prédictions sont une suite nue. Les aligner
    # sur l'index soustrairait des NaN et rendrait un écart-type absurde.
    observed = pd.Series([10.0, 12.0, 14.0], index=[907, 908, 909])
    assert residual_std(observed, [10.0, 12.0, 14.0]) == 0.0


def test_residual_std_refuses_a_single_point() -> None:
    with pytest.raises(ValueError, match="au moins"):
        residual_std(pd.Series([1.0]), [1.0])
