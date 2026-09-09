# **********************************************************************
# * Nom     : __main__.py                                              *
# * Type    : Point d'entrée                                           *
# * Sujet   : Entraînement, arbitrage des candidats et mise en service *
# *   du modèle                                                        *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from predict_common.config import Config, ConfigError, load_config
from predict_common.db import DatabaseError, open_engine
from predict_common.paths import PathError, parse_date
from predict_common.schemas import feature_columns
from training import arbitration, promotion, registry, tracing, tracking
from training.arbitration import ArbitrationError, Bench, BenchResult
from training.baseline import BaselineError, Persistence, naive_baselines
from training.candidates import (
    DEFAULT_LEARNER,
    CandidateError,
    fit_candidate,
)
from training.dataset import (
    DatasetError,
    Split,
    exclude_window,
    matrices,
    read_features,
    select,
    split_by_time,
)
from training.model import (
    DECISION_METRIC,
    best_iteration,
    evaluate,
    residual_std,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# Profondeur d'historique apprise par défaut, en journées.
DEFAULT_HISTORY_DAYS = 90

# Code de sortie d'un entraînement abouti.
EXIT_OK = 0
# Code de sortie d'un entraînement interrompu.
EXIT_FAILED = 1
# Code de sortie d'une promotion refusée par la règle.
EXIT_REFUSED = 3

logger = logging.getLogger(__name__)


def tracking_settings(config: Config) -> tracking.TrackingSettings:
    """Méthode : tracking_settings
    Description : Compose les coordonnées du registre depuis la configuration.
    """
    return tracking.TrackingSettings(
        tracking_uri=config.get_str("mlflow.tracking_uri"),
        experiment=config.get_str("training.experiment"),
        registered_model=config.get_str("training.registered_model"),
    )


@dataclass(frozen=True)
class Fixtures:
    """Classe : Fixtures
    Description : Tout ce qu'un entraînement lit une fois : découpe, banc,
      colonnes, baselines.
    """
    version: str
    history_days: int
    sites: tuple[str, ...]
    split: Split
    bench: Bench
    bench_frame: pd.DataFrame
    columns: tuple[str, ...]
    baselines: tuple[Persistence, ...]
    naive: BenchResult

    @property
    def sites_label(self) -> str:
        """Méthode : sites_label
        Description : Sites appris, tels qu'ils apparaissent dans les tags.
        """
        return ",".join(self.sites) if self.sites else "toutes"


def load_split(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
    bench: Bench | None = None,
) -> Split:
    """Méthode : load_split
    Description : Lit la fenêtre d'apprentissage et la découpe dans le temps.
    """
    start = end - timedelta(days=history_days - 1)
    frame = read_features(config.get_str("storage.root"), version, start, end)
    if frame.empty:
        raise DatasetError(
            f"Aucune partition de variables {version} entre {start} et {end}."
            " Lancer l'ETL sur cette fenêtre avant d'entraîner."
        )
    if bench is not None:
        frame = exclude_window(frame, bench.start, bench.end)
    usable = select(frame, sites, config.get_float("training.max_imputed_ratio"))
    return split_by_time(
        usable,
        valid_ratio=config.get_float("training.valid_ratio"),
        test_ratio=config.get_float("training.test_ratio"),
    )


def load_bench_frame(
    config: Config,
    version: str,
    bench: Bench,
    sites: Sequence[str] | None,
) -> pd.DataFrame:
    """Méthode : load_bench_frame
    Description : Lit les variables du banc d'arbitrage, hors apprentissage.
    """
    frame = read_features(
        config.get_str("storage.root"), version, bench.start, bench.end
    )
    usable = select(frame, sites, config.get_float("training.max_imputed_ratio"))
    if usable.empty:
        raise ArbitrationError(
            f"Banc d'arbitrage vide entre {bench.start} et {bench.end} :"
            " aucune heure exploitable. Lancer l'ETL sur cette fenêtre, ou"
            " déplacer training.arbitration."
        )
    return usable


def prepare(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
) -> Fixtures:
    """Méthode : prepare
    Description : Rassemble en une fois tout ce dont l'entraînement aura
      besoin.
    """
    inputs = {
        "feature_version": version,
        "until": end.isoformat(),
        "history_days": history_days,
        "sites": list(sites or ()),
    }
    with tracing.span("preparation", inputs=inputs) as active:
        bench = arbitration.resolve_bench(config, end)
        columns = feature_columns(
            config.get_int_list("etl.lag_hours"),
            config.get_int("etl.rolling_window_h"),
        )
        bench_frame = load_bench_frame(config, version, bench, sites)
        baselines = naive_baselines(config.get_int_list("etl.lag_hours"))
        logger.info(
            "banc d'arbitrage %s (%d journée(s), %s) : %d heure(s)",
            bench.label,
            bench.days,
            "figé" if bench.pinned else "glissant",
            len(bench_frame),
        )
        fixtures = Fixtures(
            version=version,
            history_days=history_days,
            sites=tuple(sites or ()),
            split=load_split(config, version, end, history_days, sites, bench),
            bench=bench,
            bench_frame=bench_frame,
            columns=columns,
            baselines=baselines,
            naive=arbitration.naive_reference(
                baselines, bench_frame, columns, bench
            ),
        )
        active.set_outputs(
            {
                "bench_window": bench.label,
                "bench_pinned": bench.pinned,
                "bench_hours": len(bench_frame),
                "train_hours": len(fixtures.split.train),
                "valid_hours": len(fixtures.split.valid),
                "test_hours": len(fixtures.split.test),
                "feature_columns": len(columns),
            }
        )
    return fixtures


def promote(
    engine: Engine,
    settings: tracking.TrackingSettings,
    version: str,
    run_id: str,
    trained_at: datetime,
) -> None:
    """Méthode : promote
    Description : Pose l'alias champion et inscrit la version active en base.
    """
    inputs = {
        "registered_model": settings.registered_model,
        "version": version,
        "run_id": run_id,
    }
    with tracing.span("mise_en_service", inputs=inputs):
        tracking.set_alias(
            settings.registered_model, version, tracking.PRODUCTION_ALIAS
        )
        registry.publish_champion(
            engine, settings.registered_model, version, run_id, trained_at
        )


def promote_registered(
    config: Config,
    engine: Engine,
    version: str,
    force: bool = False,
) -> None:
    """Méthode : promote_registered
    Description : Met en service une version déjà enregistrée, sans
      réapprendre.
    """
    settings = tracking_settings(config)
    tracking.connect(settings)
    enforce(decide_registered(config, settings, version), force)
    run_id, trained_at = tracking.version_identity(settings.registered_model, version)
    promote(engine, settings, version, run_id, trained_at)


def decide_registered(
    config: Config,
    settings: tracking.TrackingSettings,
    version: str,
) -> promotion.Verdict:
    """Méthode : decide_registered
    Description : Applique la règle de promotion à une version déjà
      enregistrée.
    """
    snapshot = tracking.version_snapshot(settings.registered_model, version)
    label = f"version {version}"
    candidate = arbitration.read_bench(snapshot.metrics, snapshot.params, label)
    naive = arbitration.read_naive(snapshot.metrics, snapshot.params)
    if candidate is None or naive is None:
        return promotion.Verdict(
            accepted=False,
            reason=(
                f"la {label} ne porte aucune mesure de banc : elle est"
                " antérieure à l'arbitrage, et rien ne l'oppose au champion"
            ),
        )
    return promotion.decide(
        candidate,
        champion_bench(settings),
        naive,
        config.get_float("training.promotion.margin"),
    )


class PromotionRefused(RuntimeError):
    """Classe : PromotionRefused
    Description : La règle refuse de mettre ce candidat en service.
    """


@dataclass(frozen=True)
class Trained:
    """Classe : Trained
    Description : Un candidat appris, avec ses mesures et de quoi signer son
      modèle.
    """
    model: Any
    name: str
    metrics: dict[str, float]
    bench: BenchResult
    signature_features: pd.DataFrame
    signature_predictions: Any


def fit_and_measure(
    fixtures: Fixtures,
    name: str,
    params: Mapping[str, object],
    early_stopping: int,
) -> Trained:
    """Méthode : fit_and_measure
    Description : Ajuste un candidat et le mesure, sur le test comme sur le
      banc.
    """
    inputs = {
        "learner": name,
        "params": {key: str(value) for key, value in params.items()},
        "early_stopping_rounds": early_stopping,
        "train_hours": len(fixtures.split.train),
    }
    with tracing.span(f"apprentissage.{name}", inputs=inputs) as active:
        train_x, train_y = matrices(fixtures.split.train, fixtures.columns)
        valid_x, valid_y = matrices(fixtures.split.valid, fixtures.columns)
        test_x, test_y = matrices(fixtures.split.test, fixtures.columns)
        model = fit_candidate(
            name, params, train_x, train_y, valid_x, valid_y, early_stopping
        )
        predicted = model.predict(test_x)
        measured = arbitration.score(
            model, fixtures.bench_frame, fixtures.columns, name, fixtures.bench
        )
        active.set_outputs(
            {
                **evaluate(test_y, predicted),
                **arbitration.bench_metrics(measured, fixtures.naive),
            }
        )
    return Trained(
        model=model,
        name=name,
        metrics={
            **evaluate(test_y, predicted),
            "residual_std": residual_std(test_y, predicted),
            **arbitration.bench_metrics(measured, fixtures.naive),
        },
        bench=measured,
        signature_features=valid_x,
        signature_predictions=model.predict(valid_x),
    )


def run_params(
    fixtures: Fixtures,
    params: Mapping[str, object],
    trained: Trained,
) -> dict[str, object]:
    """Méthode : run_params
    Description : Compose les paramètres journalisés avec le run.
    """
    values: dict[str, object] = {
        **dict(params),
        **fixtures.split.sizes,
        "candidat": trained.name,
        "feature_version": fixtures.version,
        "train_window": fixtures.split.window,
        "history_days": fixtures.history_days,
        "sites": fixtures.sites_label,
        arbitration.BENCH_WINDOW_PARAM: fixtures.bench.label,
        "arbitrage_fige": fixtures.bench.pinned,
        "arbitrage_rows": len(fixtures.bench_frame),
        "reference_naive": fixtures.naive.name,
    }
    if hasattr(trained.model, "best_iteration"):
        values["best_iteration"] = best_iteration(trained.model)
    return values


def version_tags(fixtures: Fixtures, trained: Trained) -> dict[str, object]:
    """Méthode : version_tags
    Description : Compose les tags qui décrivent la version enregistrée.
    """
    return {
        "candidat": trained.name,
        "feature_version": fixtures.version,
        "train_window": fixtures.split.window,
        "sites": fixtures.sites_label,
        arbitration.BENCH_WINDOW_PARAM: fixtures.bench.label,
        "mae": round(trained.metrics["mae"], 4),
        "rmse": round(trained.metrics["rmse"], 4),
        "r2": round(trained.metrics["r2"], 4),
        "residual_std": round(trained.metrics["residual_std"], 4),
    }


def train(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
    engine: Engine | None = None,
    force: bool = False,
) -> dict[str, float]:
    """Méthode : train
    Description : Apprend, enregistre, arbitre, et promeut si la règle
      l'accepte.
    """
    fixtures = prepare(config, version, end, history_days, sites)
    settings = tracking_settings(config)
    params = config.section("training.params")
    early_stopping = config.get_int("training.early_stopping_rounds")

    run_name = f"{version}-{DEFAULT_LEARNER}"
    with tracking.run(settings, run_name=run_name) as active:
        identity = (active.info.run_id, tracking.started_at(active))
        trained = fit_and_measure(fixtures, DEFAULT_LEARNER, params, early_stopping)
        tracking.log_params(run_params(fixtures, params, trained))
        tracking.log_metrics(trained.metrics)
        registered = tracking.log_model(
            trained.model,
            settings,
            trained.signature_features,
            trained.signature_predictions,
            tags=version_tags(fixtures, trained),
            name=run_name,
        )
    report_bench(trained.bench, fixtures)
    if engine is not None and registered:
        enforce(
            decide_promotion(config, settings, trained.bench, fixtures.naive),
            force,
        )
        promote(engine, settings, registered, *identity)
    logger.info(
        "entraînement terminé sur %s : %s", fixtures.split.window, trained.metrics
    )
    return trained.metrics


def champion_bench(settings: tracking.TrackingSettings) -> BenchResult | None:
    """Méthode : champion_bench
    Description : Relit le résultat de banc du champion en place, s'il y en a
      un.
    """
    try:
        snapshot = tracking.alias_snapshot(
            settings.registered_model, tracking.PRODUCTION_ALIAS
        )
    except MlflowException as exc:
        logger.info("aucun champion en place (%s)", exc)
        return None
    return arbitration.read_bench(
        snapshot.metrics, snapshot.params, f"champion v{snapshot.version}"
    )


def decide_promotion(
    config: Config,
    settings: tracking.TrackingSettings,
    candidate: BenchResult,
    naive: BenchResult,
) -> promotion.Verdict:
    """Méthode : decide_promotion
    Description : Oppose le candidat au champion et à la baseline sur le banc.
    """
    inputs = {
        "candidat": candidate.name,
        "candidat_erreur": candidate.error,
        "baseline": naive.name,
        "baseline_erreur": naive.error,
        "banc": candidate.window,
    }
    with tracing.span("arbitrage", inputs=inputs) as active:
        champion = champion_bench(settings)
        verdict = promotion.decide(
            candidate,
            champion,
            naive,
            config.get_float("training.promotion.margin"),
        )
        active.set_outputs(
            {
                "champion": champion.name if champion else None,
                "champion_erreur": champion.error if champion else None,
                "accepte": verdict.accepted,
                "raison": verdict.reason,
            }
        )
    return verdict


def enforce(verdict: promotion.Verdict, force: bool) -> None:
    """Méthode : enforce
    Description : Applique le verdict, ou passe outre si l'exploitant l'assume.
    """
    if verdict.accepted:
        logger.info("promotion acceptée — %s", verdict.reason)
        return
    if not force:
        raise PromotionRefused(verdict.reason)
    logger.warning(
        "promotion FORCÉE malgré la règle — %s. --force porte sur ce seul geste"
        " et n'assouplit rien pour les suivants.",
        verdict.reason,
    )


def report_bench(candidate: BenchResult, fixtures: Fixtures) -> None:
    """Méthode : report_bench
    Description : Journalise l'écart entre le candidat et la meilleure
      baseline.
    """
    measured, reference = candidate.error, fixtures.naive.error
    if measured is None or reference is None:
        logger.warning("banc %s : mesure incomplète", fixtures.bench.label)
        return
    gain = (1.0 - measured / reference) * 100.0 if reference else 0.0
    level = logger.info if measured < reference else logger.warning
    level(
        "banc %s : %s à %.2f kW contre %.2f pour %s (%+.1f %%)",
        fixtures.bench.label,
        candidate.name,
        measured,
        reference,
        fixtures.naive.name,
        gain,
    )


def candidate_params(config: Config) -> Iterator[tuple[str, Mapping[str, object]]]:
    """Méthode : candidate_params
    Description : Énumère les familles à opposer et leurs hyperparamètres.
    """
    yield DEFAULT_LEARNER, config.section("training.params")
    for name, params in config.section("training.candidates").items():
        yield str(name), dict(params or {})


@dataclass(frozen=True)
class Contender:
    """Classe : Contender
    Description : Un candidat mesuré sur le banc, et le run qui le porte.

      Le run est retenu parce que le vainqueur n'est connu qu'une fois tous les
      candidats mesurés, donc bien après la fermeture du sien : l'inscrire au
      registre demande de savoir d'où reprendre son modèle.
    """
    bench: BenchResult
    run_id: str
    learned: bool
    # Identifiant du modèle journalisé, vide s'il n'est pas parti. Une famille
    # peut avoir gagné le banc sans que son modèle soit attaché : elle reste
    # alors lisible dans l'expérience, mais rien ne peut être inscrit d'elle.
    model_id: str = ""
    tags: dict[str, object] = field(default_factory=dict)


def challenge_learner(
    settings: tracking.TrackingSettings,
    fixtures: Fixtures,
    name: str,
    params: Mapping[str, object],
    early_stopping: int,
) -> Contender:
    """Méthode : challenge_learner
    Description : Apprend une famille et la mesure sur le banc, sans rien
      promouvoir.
    """
    run_name = f"{fixtures.version}-challenge-{name}"
    with tracking.run(settings, run_name=run_name, nested=True) as active:
        trained = fit_and_measure(fixtures, name, params, early_stopping)
        tracking.log_params(run_params(fixtures, params, trained))
        tracking.log_metrics(trained.metrics)
        tracking.set_tags({"challenge": True, "famille": "apprise"})
        model_id = tracking.log_candidate_model(
            trained.model,
            trained.signature_features,
            trained.signature_predictions,
            name=run_name,
        )
        run_id = active.info.run_id
        tags = {**version_tags(fixtures, trained), "role": "vainqueur-arbitrage"}
    return Contender(
        bench=trained.bench,
        run_id=run_id,
        learned=True,
        model_id=model_id,
        tags=tags,
    )


def challenge_baseline(
    settings: tracking.TrackingSettings,
    fixtures: Fixtures,
    baseline: Persistence,
) -> Contender:
    """Méthode : challenge_baseline
    Description : Mesure une baseline naïve sur le même banc que les candidats.
    """
    measured = arbitration.score(
        baseline,
        fixtures.bench_frame,
        fixtures.columns,
        baseline.name,
        fixtures.bench,
    )
    with tracking.run(
        settings,
        run_name=f"{fixtures.version}-challenge-{baseline.name}",
        nested=True,
    ) as active:
        tracking.log_params(
            {
                "candidat": baseline.name,
                "colonne": baseline.column,
                "feature_version": fixtures.version,
                "sites": fixtures.sites_label,
                arbitration.BENCH_WINDOW_PARAM: fixtures.bench.label,
                "arbitrage_fige": fixtures.bench.pinned,
                "arbitrage_rows": len(fixtures.bench_frame),
            }
        )
        tracking.log_metrics(arbitration.bench_metrics(measured, fixtures.naive))
        tracking.set_tags({"challenge": True, "famille": "naive"})
        run_id = active.info.run_id
    # Aucun modèle attaché, et rien à inscrire même en cas de victoire : une
    # persistance recopie une colonne, elle ne se sert pas. Qu'elle gagne est
    # une information sur les autres candidats, pas un modèle à déployer.
    return Contender(bench=measured, run_id=run_id, learned=False)


def challenge(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
    engine: Engine | None = None,
    promote_winner: bool = False,
    force: bool = False,
) -> tuple[BenchResult, ...]:
    """Méthode : challenge
    Description : Oppose toutes les familles et les baselines, rend le
      classement, et met le vainqueur en service si on le demande.

      Avec `promote_winner`, le vainqueur passe la même règle que tout
      candidat : battre la persistance, ne pas dégrader le champion. Gagner
      entre familles ne suffit pas.
    """
    fixtures = prepare(config, version, end, history_days, sites)
    settings = tracking_settings(config)
    early_stopping = config.get_int("training.early_stopping_rounds")
    learners = [name for name, _ in candidate_params(config)]
    with tracking.run(settings, run_name=f"{fixtures.version}-challenge"):
        contenders = [
            challenge_learner(settings, fixtures, name, params, early_stopping)
            for name, params in candidate_params(config)
        ]
        contenders.extend(
            challenge_baseline(settings, fixtures, baseline)
            for baseline in fixtures.baselines
        )
        ordered = sorted(
            contenders, key=lambda entry: entry.bench.metrics[DECISION_METRIC]
        )
        ranked = tuple(entry.bench for entry in ordered)
        report_ranking(ranked, fixtures)
        publish_ranking(ranked, fixtures, learners)
        winner = register_winner(settings, ordered[0], fixtures)
    # Hors du run parent : la promotion suit l'arbitrage, elle n'en fait pas
    # partie.
    if promote_winner:
        promote_challenge_winner(config, engine, winner, force)
    return ranked


def promote_challenge_winner(
    config: Config,
    engine: Engine | None,
    version: str,
    force: bool = False,
) -> None:
    """Méthode : promote_challenge_winner
    Description : Met en service le vainqueur d'un arbitrage, s'il y en a un
      d'inscrit.

      Un arbitrage sans vainqueur inscrit n'est pas une panne : les baselines
      ont pu tout remporter. `register_winner` l'a déjà journalisé.
    """
    if not version:
        logger.warning(
            "aucun vainqueur inscrit au registre : rien à mettre en service,"
            " le champion en place le reste"
        )
        return
    if engine is None:
        raise ValueError(
            "--challenge --promote exige DATABASE_URL : la mise en service"
            " inscrit la version active dans la table `modele`."
        )
    promote_registered(config, engine, version, force)


def register_winner(
    settings: tracking.TrackingSettings,
    winner: Contender,
    fixtures: Fixtures,
) -> str:
    """Méthode : register_winner
    Description : Inscrit au catalogue la famille qui a gagné le banc, et elle
      seule.

      C'est le partage que MLflow suppose entre ses deux moitiés : l'expérience
      porte la confrontation, le registre porte ce qui peut être servi. Une
      famille n'y entre donc que le jour où elle gagne — le jour où elle
      devient déployable — et les battues restent lisibles dans leur run, avec
      leur modèle.

      Une baseline victorieuse n'est pas inscrite : elle ne se déploie pas. Le
      journal le dit, parce qu'une persistance en tête de banc est un résultat
      qui mérite d'être vu plutôt qu'un silence.
    """
    if not winner.learned:
        logger.warning(
            "banc remporté par %s : aucune famille apprise ne bat la"
            " persistance, rien n'est inscrit au registre",
            winner.bench.name,
        )
        return ""
    # Une version sans modèle passerait la règle de promotion, puis
    # refuserait de se charger : une panne différée.
    if not winner.model_id:
        logger.error(
            "banc remporté par %s, mais son modèle n'a pas été attaché à son"
            " run : rien n'est inscrit au registre, une version sans artefact"
            " serait inservable une fois promue",
            winner.bench.name,
        )
        return ""
    version = tracking.register_logged_model(
        settings,
        winner.model_id,
        tags={**winner.tags, "banc": fixtures.bench.label},
    )
    if version:
        logger.info(
            "vainqueur %s inscrit au registre %s en version %s",
            winner.bench.name,
            settings.registered_model,
            version,
        )
    return version


def ranking_table(
    ranked: Sequence[BenchResult],
    learners: Sequence[str],
) -> pd.DataFrame:
    """Méthode : ranking_table
    Description : Met le classement du banc en tableau, du meilleur au pire.

      La famille y figure en clair. C'est la seule colonne qui ne se déduit pas
      des métriques, et c'est celle qui porte le verdict : un modèle appris
      classé derrière une persistance n'a rien appris du tout.
    """
    best = ranked[0].metrics[DECISION_METRIC] if ranked else 0.0
    return pd.DataFrame(
        [
            {
                "rang": rank,
                "candidat": result.name,
                "famille": "apprise" if result.name in learners else "naive",
                "mae": round(result.metrics["mae"], 4),
                "rmse": round(result.metrics["rmse"], 4),
                "r2": round(result.metrics["r2"], 4),
                "ecart_au_meilleur_pct": round(
                    (result.metrics[DECISION_METRIC] - best) / best * 100, 2
                )
                if best
                else None,
            }
            for rank, result in enumerate(ranked, start=1)
        ]
    )


def publish_ranking(
    ranked: Sequence[BenchResult],
    fixtures: Fixtures,
    learners: Sequence[str],
) -> None:
    """Méthode : publish_ranking
    Description : Attache au run parent ce que l'arbitrage a décidé.

      Trois formes, parce qu'elles ne servent pas au même lecteur. Le tableau
      se trie et se compare dans l'interface. Les métriques se suivent d'un
      arbitrage à l'autre, et c'est là qu'on voit une famille perdre du terrain
      sur plusieurs mois. Les tags rendent le run retrouvable sans l'ouvrir.
    """
    table = ranking_table(ranked, learners)
    tracking.log_table(table, "arbitrage/classement.json")
    # `to_string` et non `to_markdown` : ce dernier réclame tabulate, une
    # dépendance de plus pour la seule mise en forme d'un tableau que
    # l'interface affiche déjà depuis le JSON.
    tracking.log_text(table.to_string(index=False), "arbitrage/classement.txt")

    best = ranked[0]
    best_naive = next(
        (result for result in ranked if result.name not in learners), None
    )
    metrics = {
        "meilleur_mae": best.metrics["mae"],
        "candidats": float(len(ranked)),
    }
    if best_naive is not None:
        naive_mae = best_naive.metrics[DECISION_METRIC]
        metrics["meilleure_naive_mae"] = naive_mae
        if naive_mae:
            # Ce que le meilleur candidat gagne sur la meilleure baseline. Un
            # gain négatif dit qu'apprendre n'a servi à rien ce jour-là, et
            # c'est exactement le genre de chose qu'un classement enfoui dans
            # une sortie console laisse passer.
            metrics["gain_sur_naive_pct"] = (
                (naive_mae - best.metrics[DECISION_METRIC]) / naive_mae * 100
            )
    tracking.log_metrics(metrics)
    tracking.set_tags(
        {
            "challenge": True,
            "role": "arbitrage",
            "vainqueur": best.name,
            "vainqueur_famille": "apprise" if best.name in learners else "naive",
            "candidats": len(ranked),
            "banc": fixtures.bench.label,
        }
    )


def report_ranking(ranked: Sequence[BenchResult], fixtures: Fixtures) -> None:
    """Méthode : report_ranking
    Description : Journalise le classement du banc, du meilleur au pire.
    """
    logger.info(
        "classement sur le banc %s (%d journée(s), %s, %d heure(s)) :",
        fixtures.bench.label,
        fixtures.bench.days,
        "figé" if fixtures.bench.pinned else "glissant",
        len(fixtures.bench_frame),
    )
    for rank, result in enumerate(ranked, start=1):
        logger.info(
            "  %d. %-18s MAE %8.2f kW   RMSE %8.2f   R2 %6.3f",
            rank,
            result.name,
            result.metrics["mae"],
            result.metrics["rmse"],
            result.metrics["r2"],
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Méthode : parse_args
    Description : Analyse la ligne de commande de l'entraînement.
    """
    parser = argparse.ArgumentParser(
        prog="training",
        description="Entraînement du modèle de prévision EnerVision.",
    )
    parser.add_argument(
        "--feature-version",
        default=None,
        help="Version des variables lues. Défaut : etl.feature_version.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Profondeur d'historique apprise, en journées.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Dernière journée apprise, YYYY-MM-DD. Défaut : aujourd'hui.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site appris. Répétable. Par défaut : tous.",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help=(
            "Oppose toutes les familles de conf/ et les baselines naïves sur"
            " le banc d'arbitrage, les classe, et inscrit le vainqueur au"
            " registre. Combiné à --promote, met ce vainqueur en service :"
            " c'est ainsi qu'une famille autre que la famille apprise par"
            " défaut peut devenir champion sans intervention."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Promeut malgré un refus de la règle. Décision d'exploitation,"
            " journalisée comme telle."
        ),
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Expérience MLflow. Défaut : training.experiment.",
    )
    parser.add_argument(
        "--promote-version",
        default=None,
        help=(
            "Met en service une version DÉJÀ enregistrée, sans réapprendre :"
            " pose l'alias champion et met `modele` à jour. C'est aussi ce qui"
            " rattrape un alias déplacé à la main. Exige DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Pose aussi l'alias champion, donc met le modèle en service, et"
            " l'inscrit active dans la table `modele`. Exige DATABASE_URL."
            " Sans cette option, la version reste challenger. Porte sur ce"
            " que l'entraînement vient d'apprendre, ou sur le vainqueur de"
            " --challenge. La règle de promotion s'applique dans les deux"
            " cas : battre la persistance, et ne pas dégrader le champion."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    """Méthode : _configure_logging
    Description : Arme le journal et met la sortie à l'abri de l'encodage
      local.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Méthode : main
    Description : Point d'entrée : entraîne ou arbitre, et rend un code de
      sortie.
    """
    _configure_logging()
    args = parse_args(argv)
    engine: Engine | None = None
    try:
        config = load_config()
        if args.experiment:
            config = _with_experiment(config, args.experiment)
        version = args.feature_version or config.get_str("etl.feature_version")
        end = parse_date(args.until) if args.until else datetime.now(UTC).date()
        if args.history_days < 1:
            raise ValueError("--history-days doit valoir au moins 1.")
        if args.promote and args.promote_version:
            raise ValueError(
                "--promote et --promote-version s'excluent : le premier met en"
                " service ce qu'il vient d'apprendre, le second une version"
                " déjà enregistrée."
            )
        if args.challenge and args.promote_version:
            raise ValueError(
                "--challenge et --promote-version s'excluent : le premier"
                " désigne un vainqueur qu'il vient de mesurer, le second met"
                " en service une version déjà enregistrée."
            )
        if args.promote or args.promote_version:
            engine = open_engine(config.get_optional_str("database.url"))
        # Avant le premier span : sans expérience active, la trace
        # atterrirait dans `Default`.
        tracing.configure(tracking_settings(config))
        inputs = {
            "feature_version": version,
            "until": end.isoformat(),
            "history_days": args.history_days,
            "sites": list(args.sites or ()),
            "force": bool(args.force),
        }
        if args.challenge:
            with tracing.span(
                "challenge", inputs={"promote": bool(args.promote), **inputs}
            ):
                challenge(
                    config,
                    version,
                    end,
                    args.history_days,
                    args.sites,
                    engine,
                    args.promote,
                    args.force,
                )
        elif args.promote_version:
            with tracing.span(
                "promotion_version",
                inputs={"version": args.promote_version, **inputs},
            ):
                promote_registered(
                    config, engine, args.promote_version, args.force
                )
        else:
            with tracing.span(
                "entrainement", inputs={"promote": bool(args.promote), **inputs}
            ):
                train(
                    config,
                    version,
                    end,
                    args.history_days,
                    args.sites,
                    engine,
                    args.force,
                )
    except PromotionRefused as exc:
        logger.warning(
            "promotion refusée — %s. La version reste challenger ; --force"
            " passe outre si la règle a tort.",
            exc,
        )
        return EXIT_REFUSED
    except (
        ArbitrationError,
        BaselineError,
        CandidateError,
        ConfigError,
        PathError,
        DatasetError,
    ) as exc:
        logger.error("entraînement interrompu : %s", exc)
        return EXIT_FAILED
    except DatabaseError as exc:
        logger.error("promotion impossible : %s", exc)
        return EXIT_FAILED
    except MlflowException as exc:
        logger.error("registre MLflow : %s", exc)
        return EXIT_FAILED
    except SQLAlchemyError as exc:
        logger.error(
            "alias champion posé, mais `modele` non mise à jour : %s."
            " Relancer --promote une fois la base joignable.",
            exc,
        )
        return EXIT_FAILED
    except ValueError as exc:
        logger.error("erreur inattendue (%s) : %s", type(exc).__name__, exc)
        return EXIT_FAILED
    finally:
        if engine is not None:
            engine.dispose()
    return EXIT_OK


def _with_experiment(config: Config, experiment: str) -> Config:
    """Méthode : _with_experiment
    Description : Rend la même configuration avec une autre expérience MLflow.
    """
    values = {**config.values}
    values["training"] = {**values.get("training", {}), "experiment": experiment}
    return type(config)(values=values, env_name=config.env_name)


if __name__ == "__main__":
    raise SystemExit(main())
