"""Entraînement du modèle, hors base et hors MLflow.

Le run MLflow et la lecture en base sont des effets de bord testés par une
exécution réelle, pas par la CI. Ce qui se teste ici sans dépendance, c'est
l'ajustement du modèle et le calcul des métriques.
"""

import numpy as np
import pandas as pd

from training.dataset import FEATURE_COLUMNS, TARGET_COLUMN
from training.train import ModelParams, evaluate, fit_model

ROWS = 200


def training_frame() -> pd.DataFrame:
    """Jeu synthétique linéaire, appris exactement par un arbre boosté."""
    generator = np.random.default_rng(seed=42)
    frame = pd.DataFrame(
        {column: generator.normal(size=ROWS) for column in FEATURE_COLUMNS}
    )
    frame[TARGET_COLUMN] = 3.0 * frame["hour"] + 2.0
    return frame


def test_model_params_carry_a_fixed_seed() -> None:
    assert ModelParams().random_state == 42


def test_fit_model_learns_the_signal() -> None:
    frame = training_frame()
    model = fit_model(frame, ModelParams())
    predicted = model.predict(frame[list(FEATURE_COLUMNS)])
    assert evaluate(frame[TARGET_COLUMN], predicted)["r2"] > 0.9


def test_evaluate_returns_the_three_comparable_metrics() -> None:
    observed = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate(observed, [1.0, 2.0, 3.0])
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
