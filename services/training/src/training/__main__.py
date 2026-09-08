"""Point d'entrée de l'entraînement : des partitions en entrée, un modèle en sortie.

    python -m training --feature-version v1
    python -m training --feature-version v1 --history-days 180 --site SITE001

Le service lit `features/{version}/dt=.../` et enregistre un modèle dans
MLflow. Il n'écrit aucun fichier que quelqu'un d'autre devrait aller chercher,
et ne connaît ni le collecteur, ni l'ETL, ni le service d'inférence.

Un modèle entraîné hors de ce chemin n'est pas déployable, et c'est
volontaire : le service d'inférence ne charge que ce que le registre lui
désigne. C'est aussi ce qui rend un modèle traçable — son run porte la version
des variables, la fenêtre apprise et les métriques du bloc de test.

L'entraînement produit un `challenger`, jamais un `champion`. Promouvoir est
une décision d'exploitation, prise en déplaçant l'alias dans MLflow, et le
service la suit sans être redéployé.

Le mot `challenger` a longtemps été le seul morceau de challenge : rien
n'opposait la version apprise à celle en service. Deux choses le font
maintenant. Tout candidat est réévalué sur un banc d'arbitrage — une fenêtre
retirée de l'apprentissage, la même pour tous — et la promotion refuse ce qui
ne bat pas la persistance naïve (ADR-010) ou ce qui dégraderait le champion.
`--force` passe outre, en le disant.

    python -m training --challenge

oppose toutes les familles déclarées dans `conf/` et les baselines naïves sur
ce banc, les classe, et ne promeut rien. Le classement est une lecture ; la
mise en service reste un geste séparé.

`--promote` fait deux choses et non une : il déplace l'alias, puis inscrit la
version dans `modele`, la table du schéma figé. Ce second geste demande la
base, que l'entraînement ne touche dans aucun autre cas — c'est pourquoi la
connexion est ouverte avant l'apprentissage et non après. Découvrir une
DATABASE_URL absente au bout d'une heure de calcul laisserait le choix entre
perdre le run et servir un modèle que rien ne référence.

`--promote-version` fait le même geste sur une version déjà enregistrée, sans
rien réapprendre :
    python -m training --promote-version 7

C'est la voie qui remet le miroir d'aplomb après un `mlflow models set-alias`,
qui déplace l'alias sans rien savoir de `modele`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from predict_common.config import Config, ConfigError, load_config
from predict_common.db import DatabaseError, open_engine
from predict_common.paths import PathError, parse_date
from predict_common.schemas import feature_columns
from training import arbitration, promotion, registry, tracking
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

DEFAULT_HISTORY_DAYS = 90

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 3

logger = logging.getLogger(__name__)


def tracking_settings(config: Config) -> tracking.TrackingSettings:
    """Lit où le run doit être écrit et sous quel nom enregistrer le modèle."""
    return tracking.TrackingSettings(
        tracking_uri=config.get_str("mlflow.tracking_uri"),
        experiment=config.get_str("training.experiment"),
        registered_model=config.get_str("training.registered_model"),
    )


@dataclass(frozen=True)
class Fixtures:
    """Ce qu'un entraînement a lu avant d'apprendre quoi que ce soit.

    Réunies dans un seul objet parce qu'elles sont indissociables : le banc
    est retiré du jeu d'apprentissage, et les baselines sont mesurées sur le
    banc. Les passer séparément laisserait la possibilité d'en apparier deux
    qui ne vont pas ensemble — un banc et un jeu qui se recouvrent, par
    exemple, ce qu'aucune signature ne signalerait.
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
        """Sites appris, tels qu'ils partent dans les paramètres du run."""
        return ",".join(self.sites) if self.sites else "toutes"


def load_split(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
    bench: Bench | None = None,
) -> Split:
    """Lit la fenêtre d'apprentissage et la découpe dans l'ordre du temps.

    Le banc est retiré avant la découpe et non après : retiré après, il
    amputerait le bloc de test de ses journées les plus récentes tout en
    laissant les ratios croire qu'il les a.
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
    """Lit les heures du banc, filtrées comme celles de l'apprentissage.

    Le même filtre d'imputation qu'à l'apprentissage, sinon le banc jugerait
    les candidats sur des heures que l'entraînement s'interdit d'apprendre :
    on mesurerait alors leur aptitude à reproduire l'interpolation de l'ETL.
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
    """Lit le jeu d'apprentissage, le banc, et la référence à battre."""
    bench = arbitration.resolve_bench(config, end)
    columns = feature_columns(
        config.get_int_list("etl.lag_hours"), config.get_int("etl.rolling_window_h")
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
    return Fixtures(
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


def promote(
    engine: Engine,
    settings: tracking.TrackingSettings,
    version: str,
    run_id: str,
    trained_at: datetime,
) -> None:
    """Met la version en service, puis l'inscrit comme telle dans `modele`.

    L'alias d'abord, le miroir ensuite. C'est l'alias qui met réellement le
    modèle en service : une ligne active pour une version que le service ne
    résout pas serait un miroir qui ment, alors qu'un alias déplacé sans
    miroir est un retard visible, que le journal nomme et qu'un second
    `--promote` rattrape.
    """
    tracking.set_alias(settings.registered_model, version, tracking.PRODUCTION_ALIAS)
    registry.publish_champion(
        engine, settings.registered_model, version, run_id, trained_at
    )


def promote_registered(
    config: Config,
    engine: Engine,
    version: str,
    force: bool = False,
) -> None:
    """Met en service une version déjà enregistrée, sans rien réapprendre.

    Deux usages, et le second est le plus fréquent. Servir une version qu'on a
    laissée décanter en `challenger`, et rattraper un alias déplacé à la main
    — `mlflow models set-alias` ne connaît pas `modele` et laisse le miroir en
    arrière. L'inscription étant un `ON CONFLICT DO UPDATE`, la rejouer sur une
    version déjà active ne fait rien de plus.

    La règle de promotion s'y applique comme ailleurs, mais sur ce que la
    version a journalisé le jour de son entraînement : rien n'est réappris ici,
    et recalculer sa mesure sur le banc d'aujourd'hui la jugerait sur des
    heures qu'elle n'a pas vues au même titre que les autres.
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
    """Oppose une version déjà enregistrée au champion en place.

    Une version antérieure au banc n'en porte aucune mesure. Le refus est alors
    franc : la mettre en service reste possible sous `--force`, ce qui est
    exactement ce qu'elle est — une décision prise sans comparaison, et qui
    doit se lire comme telle dans le journal.
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
    """La règle a refusé de mettre le candidat en service."""


@dataclass(frozen=True)
class Trained:
    """Un candidat ajusté, et tout ce qu'on a mesuré sur lui.

    Les deux mesures ne disent pas la même chose et voyagent donc ensemble.
    `metrics` juge le modèle dans sa propre fenêtre — c'est ce que le service
    d'inférence lit pour borner sa prévision. `bench` le juge sur la fenêtre
    commune, et c'est la seule des deux qui se compare à un autre candidat.
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
    """Ajuste un candidat, le mesure sur son test puis sur le banc.

    Partagé par l'entraînement ordinaire et par le challenge : les deux
    doivent mesurer exactement de la même façon, sinon le classement du second
    ne dirait rien du modèle que le premier enregistre.
    """
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
    """Retourne ce qui rend le run reproductible et comparable.

    La fenêtre du banc en fait partie, et c'est nouveau : c'est elle que la
    promotion relit pour refuser d'opposer deux mesures qui n'ont pas vu les
    mêmes heures.
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
    """Décrit la version dans le registre, à côté de son alias."""
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
    """Entraîne un modèle sur la fenêtre demandée et enregistre son run.

    Un moteur passé vaut demande de promotion : `main` ne l'ouvre que sous
    `--promote`, et la base n'a aucun autre usage dans ce service. La promotion
    n'est plus acquise pour autant — elle passe par la règle de
    `training.promotion`, et un refus laisse la version en challenger.
    """
    fixtures = prepare(config, version, end, history_days, sites)
    settings = tracking_settings(config)
    params = config.section("training.params")
    early_stopping = config.get_int("training.early_stopping_rounds")

    with tracking.run(settings, run_name=f"{version}-{DEFAULT_LEARNER}") as active:
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
    """Retourne ce que la version en service a mesuré sur son banc.

    `None` quand il n'y a pas encore de champion, et `None` aussi quand celui
    en place a été enregistré avant l'existence du banc. Les deux cas sont
    distincts pour la règle de promotion — le premier est une première mise en
    service, le second un refus — et c'est elle qui les sépare, pas cette
    lecture.
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
    """Oppose le candidat au champion en place, sur le banc."""
    return promotion.decide(
        candidate,
        champion_bench(settings),
        naive,
        config.get_float("training.promotion.margin"),
    )


def enforce(verdict: promotion.Verdict, force: bool) -> None:
    """Applique la décision, ou journalise le passage en force.

    Le refus est une exception et non un code de retour : la promotion est
    faite d'un alias puis d'une écriture en base, et il ne doit rester aucun
    chemin par lequel la première aurait lieu après un refus.
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
    """Dit ce que le candidat vaut face à la référence gratuite.

    Journalisé même sans promotion : c'est le chiffre qui dit si
    l'entraînement a servi à quelque chose, et il ne doit pas n'apparaître que
    le jour où quelqu'un demande une mise en service.
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
    """Énumère les candidats du challenge, le modèle ordinaire en tête.

    XGBoost n'est pas dans `training.candidates` et vient de `training.params`
    : c'est celui que l'entraînement ordinaire apprend, et écrire ses
    hyperparamètres à deux endroits les ferait diverger — le challenge
    classerait alors un modèle que personne n'enregistre.
    """
    yield DEFAULT_LEARNER, config.section("training.params")
    for name, params in config.section("training.candidates").items():
        yield str(name), dict(params or {})


def challenge_learner(
    settings: tracking.TrackingSettings,
    fixtures: Fixtures,
    name: str,
    params: Mapping[str, object],
    early_stopping: int,
) -> BenchResult:
    """Ajuste un candidat, journalise son run, et n'enregistre rien.

    Aucun appel à `log_model` : un challenge compare, il ne met pas en service.
    Enregistrer chaque candidat remplirait le registre de versions qu'aucun
    alias ne désigne, et la version que `serving` résout porte le nom d'une
    famille — y déposer une forêt serait un contresens de nommage avant d'être
    un contresens d'exploitation.
    """
    with tracking.run(settings, run_name=f"{fixtures.version}-challenge-{name}"):
        trained = fit_and_measure(fixtures, name, params, early_stopping)
        tracking.log_params(run_params(fixtures, params, trained))
        tracking.log_metrics(trained.metrics)
        tracking.set_tags({"challenge": True, "famille": "apprise"})
    return trained.bench


def challenge_baseline(
    settings: tracking.TrackingSettings,
    fixtures: Fixtures,
    baseline: Persistence,
) -> BenchResult:
    """Mesure une persistance sur le banc et lui donne son propre run.

    ADR-010 fait de la baseline un livrable permanent : lui donner un run,
    c'est la rendre visible dans l'interface à côté de ce qu'elle arbitre,
    plutôt que de la réduire à un nombre cité dans le journal d'un autre.
    """
    measured = arbitration.score(
        baseline,
        fixtures.bench_frame,
        fixtures.columns,
        baseline.name,
        fixtures.bench,
    )
    with tracking.run(
        settings, run_name=f"{fixtures.version}-challenge-{baseline.name}"
    ):
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
    return measured


def challenge(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
) -> tuple[BenchResult, ...]:
    """Oppose tous les candidats sur le banc, les classe, et ne promeut rien.

    Ne rien promouvoir est le propos, pas une limite. Un classement dit quelle
    famille convient au problème ; mettre en service est une autre décision,
    qui se prend après l'avoir lu et passe par `--promote-version`.
    """
    fixtures = prepare(config, version, end, history_days, sites)
    settings = tracking_settings(config)
    early_stopping = config.get_int("training.early_stopping_rounds")
    measured = [
        challenge_learner(settings, fixtures, name, params, early_stopping)
        for name, params in candidate_params(config)
    ]
    measured.extend(
        challenge_baseline(settings, fixtures, baseline)
        for baseline in fixtures.baselines
    )
    ranked = tuple(
        sorted(measured, key=lambda result: result.metrics[DECISION_METRIC])
    )
    report_ranking(ranked, fixtures)
    return ranked


def report_ranking(ranked: Sequence[BenchResult], fixtures: Fixtures) -> None:
    """Imprime le classement, du meilleur au moins bon.

    Sur le banc et non sur les blocs de test respectifs : c'est toute la
    raison d'être du banc, et un classement bâti sur des fenêtres différentes
    serait une opinion présentée comme une mesure.
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
    """Analyse la ligne de commande de l'entraînement."""
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
            " le banc d'arbitrage, les classe, et ne promeut rien."
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
            " Sans cette option, la version reste challenger."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    """Arme le journal, et met la sortie standard à l'abri de l'encodage local.

    MLflow imprime des emoji quand il rend la main ; une console Windows en
    cp1252 lève alors une UnicodeEncodeError au beau milieu d'un run qui, lui,
    s'est bien passé. On ne peut pas demander à MLflow de se taire, mais on
    peut faire en sorte qu'un caractère non représentable dégrade l'affichage
    au lieu d'interrompre le traitement.
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
    """Point d'entrée du conteneur d'entraînement."""
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
        if args.challenge and (args.promote or args.promote_version):
            raise ValueError(
                "--challenge ne promeut rien, par construction : il compare"
                " des familles, et la mise en service se décide après lecture"
                " du classement."
            )
        if args.promote or args.promote_version:
            engine = open_engine(config.get_optional_str("database.url"))
        if args.challenge:
            challenge(config, version, end, args.history_days, args.sites)
        elif args.promote_version:
            promote_registered(config, engine, args.promote_version, args.force)
        else:
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
    """Retourne la configuration avec l'expérience imposée en ligne de commande."""
    values = {**config.values}
    values["training"] = {**values.get("training", {}), "experiment": experiment}
    return type(config)(values=values, env_name=config.env_name)


if __name__ == "__main__":
    raise SystemExit(main())
