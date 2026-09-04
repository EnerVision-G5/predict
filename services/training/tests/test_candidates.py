"""Les familles opposées par le challenge, derrière une seule interface.

Ce qui compte ici n'est pas laquelle gagne — c'est le banc qui le dit, sur des
données réelles — mais que les trois soient réellement interchangeables. Un
candidat qui échouerait là où un autre apprend produirait un classement où
l'absence vaut défaite, ce qui n'est pas une comparaison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_common.schemas import feature_columns
from training.candidates import (
    DEFAULT_LEARNER,
    LEARNERS,
    CandidateError,
    fit_candidate,
    learner,
)

COLUMNS = feature_columns((1, 24), 2)
ROWS = 240

PARAMS = {
    DEFAULT_LEARNER: {"n_estimators": 30, "max_depth": 3},
    "foret-aleatoire": {"n_estimators": 20, "max_depth": 6, "random_state": 42},
    "ridge": {"alpha": 1.0},
}


def learnable(rows: int = ROWS) -> tuple[pd.DataFrame, pd.Series]:
    """Jeu synthétique linéaire, à la portée des trois familles."""
    generator = np.random.default_rng(seed=42)
    frame = pd.DataFrame(
        {column: generator.normal(size=rows) for column in COLUMNS}
    )
    return frame, 3.0 * frame["lag_1h"] + 2.0


def fit_named(name: str, features: pd.DataFrame, target: pd.Series):
    """Ajuste un candidat sur le même jeu que ses concurrents."""
    valid_features, valid_target = learnable(rows=60)
    return fit_candidate(
        name,
        PARAMS[name],
        features,
        target,
        valid_features,
        valid_target,
        early_stopping_rounds=5,
    )


class TestLearner:
    """Le catalogue est nommé : une faute de frappe doit se lire."""

    def test_the_default_learner_is_registered(self) -> None:
        assert learner(DEFAULT_LEARNER).name == DEFAULT_LEARNER

    def test_only_the_boosted_model_consumes_the_validation(self) -> None:
        # L'arrêt anticipé est la seule raison de montrer la validation à un
        # candidat : la donner aux autres serait leur laisser voir des heures
        # qu'ils n'apprennent pas.
        assert learner(DEFAULT_LEARNER).uses_validation
        assert not learner("ridge").uses_validation

    def test_an_unknown_candidate_lists_the_known_ones(self) -> None:
        with pytest.raises(CandidateError, match="ridge"):
            learner("xgbost")


@pytest.mark.parametrize("name", sorted(LEARNERS))
def test_every_family_learns_the_signal(name: str) -> None:
    features, target = learnable()
    model = fit_named(name, features, target)
    predicted = model.predict(features)
    assert float(np.corrcoef(predicted, target)[0, 1]) > 0.9


@pytest.mark.parametrize("name", sorted(LEARNERS))
def test_every_family_tolerates_a_missing_lag(name: str) -> None:
    # Les premières heures d'un historique n'ont pas de décalage à 168 h.
    # XGBoost les traite nativement, scikit-learn lève : sans l'imputation du
    # pipeline, le classement dirait « erreur » là où il doit dire « moins
    # bon ».
    features, target = learnable()
    features.loc[:9, "lag_24h"] = np.nan
    assert fit_named(name, features, target).predict(features) is not None


def test_a_refused_hyperparameter_names_the_candidate() -> None:
    features, target = learnable(rows=60)
    with pytest.raises(CandidateError, match="ridge"):
        fit_candidate(
            "ridge",
            {"alpha": 1.0, "profondeur": 4},
            features,
            target,
            features,
            target,
            early_stopping_rounds=5,
        )
