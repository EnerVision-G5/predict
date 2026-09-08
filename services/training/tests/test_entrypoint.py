"""Aiguillage de l'entraînement : ce qui se décide avant d'apprendre.

Le run MLflow reste un effet de bord, testé par une exécution réelle et non par
la CI. Ce qui se teste ici sans dépendance, c'est ce qui décide : quels
candidats le challenge oppose, et ce qu'un refus de promotion fait réellement.
"""

from __future__ import annotations

import logging

import pytest

from predict_common.config import Config
from training.__main__ import PromotionRefused, candidate_params, enforce
from training.candidates import DEFAULT_LEARNER
from training.promotion import Verdict


def config_for(candidates: dict | None = None) -> Config:
    """Configuration réduite aux deux blocs que le challenge énumère."""
    return Config(
        values={
            "training": {
                "params": {"n_estimators": 120},
                "candidates": candidates if candidates is not None else {},
            }
        }
    )


class TestCandidateParams:
    """Le challenge énumère les familles, le modèle ordinaire en tête."""

    def test_the_registered_family_comes_first(self) -> None:
        first, _ = next(iter(candidate_params(config_for())))
        assert first == DEFAULT_LEARNER

    def test_its_hyperparameters_come_from_training_params(self) -> None:
        _, params = next(iter(candidate_params(config_for())))
        assert params["n_estimators"] == 120

    def test_the_declared_families_follow(self) -> None:
        names = [
            name
            for name, _ in candidate_params(
                config_for({"ridge": {"alpha": 1.0}, "foret-aleatoire": {}})
            )
        ]
        assert names == [DEFAULT_LEARNER, "ridge", "foret-aleatoire"]

    def test_a_family_without_hyperparameters_keeps_its_defaults(self) -> None:
        params = dict(candidate_params(config_for({"ridge": None})))
        assert params["ridge"] == {}


class TestEnforce:
    """Un refus doit arrêter la promotion, pas seulement la commenter."""

    def test_an_accepted_verdict_lets_the_promotion_through(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="training.__main__"):
            enforce(Verdict(accepted=True, reason="meilleur"), force=False)
        assert "promotion acceptée" in caplog.text

    def test_a_refusal_stops_everything(self) -> None:
        with pytest.raises(PromotionRefused, match="dégradation"):
            enforce(Verdict(accepted=False, reason="dégradation"), force=False)

    def test_forcing_passes_but_says_so(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="training.__main__"):
            enforce(Verdict(accepted=False, reason="dégradation"), force=True)
        assert "FORCÉE" in caplog.text
