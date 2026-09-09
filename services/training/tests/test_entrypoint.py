from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pytest

from predict_common.config import Config
from training import __main__ as entrypoint
from training import tracking
from training.__main__ import (
    Contender,
    PromotionRefused,
    candidate_params,
    enforce,
    promote_challenge_winner,
    register_winner,
)
from training.candidates import DEFAULT_LEARNER
from training.promotion import Verdict


def config_for(candidates: dict | None = None) -> Config:
    return Config(
        values={
            "training": {
                "params": {"n_estimators": 120},
                "candidates": candidates if candidates is not None else {},
            }
        }
    )


class TestCandidateParams:
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


@dataclass
class FakeBench:
    name: str = "foret-aleatoire"
    label: str = "2026-08-01/2026-08-14"


@dataclass
class FakeFixtures:
    bench: FakeBench = field(default_factory=FakeBench)


def contender(**overrides: object) -> Contender:
    defaults: dict[str, object] = {
        "bench": FakeBench(),
        "run_id": "run-3",
        "learned": True,
        "model_id": "m-42",
        "tags": {"candidat": "foret-aleatoire"},
    }
    return Contender(**{**defaults, **overrides})


class TestRegisterWinner:
    """Ce qui entre au registre, et surtout ce qui n'y entre pas."""

    @staticmethod
    def settings() -> tracking.TrackingSettings:
        return tracking.TrackingSettings(
            tracking_uri="http://mlflow:5000",
            experiment="enervision-consumption",
            registered_model="enervision_consommation",
        )

    def registered(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        seen: list[str] = []

        def fake_register(_settings, model_id, tags=None) -> str:  # noqa: ANN001
            seen.append(model_id)
            return "4"

        monkeypatch.setattr(tracking, "register_logged_model", fake_register)
        return seen

    def test_a_learned_winner_with_its_model_is_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self.registered(monkeypatch)
        register_winner(self.settings(), contender(), FakeFixtures())
        assert seen == ["m-42"]

    def test_a_baseline_winner_is_not_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self.registered(monkeypatch)
        register_winner(
            self.settings(), contender(learned=False), FakeFixtures()
        )
        assert seen == []

    def test_a_winner_whose_model_never_left_is_not_registered(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen = self.registered(monkeypatch)
        with caplog.at_level(logging.ERROR, logger="training.__main__"):
            register_winner(
                self.settings(), contender(model_id=""), FakeFixtures()
            )
        assert seen == []
        assert "inservable une fois promue" in caplog.text


class TestPromoteChallengeWinner:
    """La mise en service qui suit un arbitrage, et ses trois issues."""

    @staticmethod
    def captured(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool]]:
        seen: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            entrypoint,
            "promote_registered",
            lambda _c, _e, version, force=False: seen.append((version, force)),
        )
        return seen

    def test_the_registered_winner_goes_through_the_promotion_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self.captured(monkeypatch)
        promote_challenge_winner(config_for(), object(), "4", force=False)
        assert seen == [("4", False)]

    def test_forcing_is_carried_all_the_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self.captured(monkeypatch)
        promote_challenge_winner(config_for(), object(), "4", force=True)
        assert seen == [("4", True)]

    def test_an_arbitration_without_a_registered_winner_promotes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen = self.captured(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="training.__main__"):
            promote_challenge_winner(config_for(), object(), "")
        assert seen == []
        assert "le champion en place le reste" in caplog.text

    def test_promoting_without_a_database_is_refused_not_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.captured(monkeypatch)
        with pytest.raises(ValueError, match="DATABASE_URL"):
            promote_challenge_winner(config_for(), None, "4")
