"""Surveillance de l'écart entre ce que le modèle prédit et ce qui est mesuré.

    python -m training.drift
    python -m training.drift --since 2026-09-01 --until 2026-09-07

Un modèle ne se dégrade pas d'un coup : il se dégrade parce que le monde
change sous lui — un site qui déménage sa production, un capteur remplacé, une
saison qui n'était pas dans l'historique. Les métriques du jour de
l'entraînement ne disent rien de cela, puisqu'elles ont été mesurées sur des
données du passé. Ce module rejoue le modèle en service sur les mesures
arrivées depuis, et regarde s'il se trompe davantage qu'à sa naissance.

Trois décisions structurent le fichier.

**L'écart est mesuré à un pas, pas sur l'horizon complet.** Le service
d'inférence prédit par récurrence sur 48 heures, et son erreur s'accumule
mécaniquement à chaque pas ; la mêler à la dégradation du modèle rendrait les
deux indiscernables. Ici chaque heure est prédite à partir de ses décalages
réels — la même tâche que celle mesurée à l'entraînement, donc la seule
comparable.

**Le seuil est un rapport, pas une valeur absolue.** Un écart de 20 kW n'a pas
le même sens sur un bureau de 200 kW et sur une usine de 1000. La référence
est l'erreur du modèle sur son jeu de test, lue dans le run qui l'a produit :
elle suit le modèle, et promouvoir une autre version change la référence du
même geste.

**Le verdict est un code de sortie.** Un ordonnanceur n'a pas à lire un
journal pour savoir s'il doit alerter. Le code 2 dit « dérive », distinct du 1
qui dit « le calcul lui-même a échoué » — les confondre ferait chercher un
problème de modèle là où il n'y a qu'une base injoignable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import mlflow
import mlflow.pyfunc
import pandas as pd

from predict_common.config import Config, ConfigError, load_config
from predict_common.paths import PathError, parse_date
from predict_common.schemas import feature_columns
from training import tracking
from training.dataset import DatasetError, matrices, read_features
from training.model import evaluate

EXIT_OK = 0
EXIT_FAILED = 1
# Distinct de EXIT_FAILED : une dérive est un résultat, pas une panne. Les
# confondre ferait chercher un problème d'infrastructure là où le modèle a
# simplement vieilli.
EXIT_DRIFTED = 2

# Métrique qui porte le verdict. La MAE plutôt que la RMSE : elle s'exprime en
# kilowatts moyens, ce qui se discute avec un exploitant, là où la RMSE
# amplifie les grands écarts et se compare mal d'un site à l'autre.
DECISION_METRIC = "mae"

logger = logging.getLogger(__name__)


class MonitoringError(RuntimeError):
    """La surveillance n'a pas pu être menée à son terme."""


@dataclass(frozen=True)
class DriftSettings:
    """Ce qui règle une évaluation, extrait de la configuration une fois."""

    experiment: str
    window_days: int
    alert_ratio: float
    min_rows: int
    registered_model: str
    alias: str

    @classmethod
    def from_config(cls, config: Config) -> DriftSettings:
        """Lit le bloc `monitoring` et le nom du modèle surveillé."""
        return cls(
            experiment=config.get_str("monitoring.experiment"),
            window_days=config.get_int("monitoring.window_days"),
            alert_ratio=config.get_float("monitoring.mae_alert_ratio"),
            min_rows=config.get_int("monitoring.min_rows"),
            registered_model=config.get_str("training.registered_model"),
            alias=tracking.PRODUCTION_ALIAS,
        )


@dataclass(frozen=True)
class DriftReport:
    """Ce qu'une évaluation a mesuré, et ce qu'elle en conclut."""

    metrics: dict[str, float]
    baseline: dict[str, float]
    rows: int
    version: str
    window: str

    @property
    def ratio(self) -> float | None:
        """Rapport entre l'erreur mesurée et celle de l'entraînement.

        `None` quand la référence manque : un modèle enregistré sans métrique
        de test ne permet aucune comparaison, et inventer un rapport de 1
        laisserait croire que tout va bien.
        """
        reference = self.baseline.get(DECISION_METRIC)
        if not reference:
            return None
        return self.metrics[DECISION_METRIC] / reference

    def verdict(self, settings: DriftSettings) -> str:
        """Dit ce que la mesure permet de conclure, en une ligne de journal."""
        if self.rows < settings.min_rows:
            return "indécis"
        ratio = self.ratio
        if ratio is None:
            return "sans référence"
        return "dérive" if ratio > settings.alert_ratio else "stable"


def window(
    since: date | None,
    until: date | None,
    window_days: int,
) -> tuple[date, date]:
    """Retourne la fenêtre évaluée, bornes comprises.

    Sans borne, on regarde les derniers jours : c'est ce que fait un
    ordonnanceur quotidien, pour qui la fenêtre glisse et n'a pas à être dite.
    """
    if window_days < 1:
        raise MonitoringError("La fenêtre de surveillance couvre au moins un jour.")
    end = until or since or datetime.now(UTC).date()
    start = since or end - timedelta(days=window_days - 1)
    if start > end:
        raise MonitoringError(f"Fenêtre vide : {end} précède {start}.")
    return start, end


def load_served_model(settings: DriftSettings) -> tuple[object, str]:
    """Charge le modèle en service et retourne sa version du registre.

    Le modèle surveillé est celui que le service sert, pas le dernier
    entraîné : surveiller un `challenger` que personne n'utilise ne dirait
    rien de ce que les consommateurs reçoivent.
    """
    uri = f"models:/{settings.registered_model}@{settings.alias}"
    try:
        model = mlflow.pyfunc.load_model(uri)
        version = tracking.served_version(settings.registered_model, settings.alias)
    except Exception as exc:  # noqa: BLE001 - le registre lève large
        raise MonitoringError(f"modèle {uri} non résolu : {exc}") from exc
    return model, version


def measure(
    model: object,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, float]:
    """Compare, heure par heure, ce que le modèle prédit à ce qui a été mesuré.

    Les décalages viennent des mesures réelles et non de prédictions
    antérieures : c'est la tâche à un pas, celle qu'on sait comparer à
    l'entraînement. Le service, lui, prédit par récurrence, et son erreur
    grandit avec l'horizon pour une raison qui n'a rien à voir avec la dérive.
    """
    explanatory, observed = matrices(frame, columns)
    predicted = model.predict(explanatory)
    return evaluate(observed, predicted)


def evaluate_window(
    config: Config,
    settings: DriftSettings,
    version_of_features: str,
    since: date,
    until: date,
) -> DriftReport:
    """Mesure l'écart du modèle servi sur les mesures d'une fenêtre."""
    frame = read_features(
        config.get_str("storage.root"), version_of_features, since, until
    )
    if frame.empty:
        raise MonitoringError(
            f"Aucune partition de variables {version_of_features} entre"
            f" {since} et {until}. Lancer l'ETL sur cette fenêtre."
        )
    model, version = load_served_model(settings)
    columns = feature_columns(
        config.get_int_list("etl.lag_hours"), config.get_int("etl.rolling_window_h")
    )
    return DriftReport(
        metrics=measure(model, frame, columns),
        baseline=_baseline(settings),
        rows=len(frame),
        version=version,
        window=f"{since}/{until}",
    )


def publish(report: DriftReport, settings: DriftSettings, feature_version: str) -> None:
    """Écrit la mesure dans MLflow, dans l'expérience de surveillance.

    Un run par évaluation, et non un run unique qu'on rallongerait : chaque
    fenêtre est un fait daté, et la vue « Chart » de l'expérience trace la
    série sans qu'on ait à tenir un identifiant de run entre deux exécutions.
    """
    mlflow.set_experiment(settings.experiment)
    with mlflow.start_run(run_name=f"drift-{report.window.split('/')[-1]}"):
        mlflow.log_params(
            {
                "model": settings.registered_model,
                "model_version": report.version,
                "alias": settings.alias,
                "feature_version": feature_version,
                "window": report.window,
                "rows": report.rows,
                "alert_ratio": settings.alert_ratio,
            }
        )
        mlflow.log_metrics(report.metrics)
        for name, value in report.baseline.items():
            # Préfixées : sans cela, la référence et la mesure porteraient le
            # même nom et l'une écraserait l'autre dans le même run.
            mlflow.log_metric(f"baseline_{name}", value)
        ratio = report.ratio
        if ratio is not None:
            mlflow.log_metric("mae_ratio", ratio)
        mlflow.set_tag("verdict", report.verdict(settings))


def run(config: Config, since: date | None, until: date | None, version: str) -> int:
    """Mesure l'écart, le publie, et retourne le code de sortie qui convient."""
    settings = DriftSettings.from_config(config)
    mlflow.set_tracking_uri(config.get_str("mlflow.tracking_uri"))
    first, last = window(since, until, settings.window_days)
    report = evaluate_window(config, settings, version, first, last)
    publish(report, settings, version)

    verdict = report.verdict(settings)
    ratio = report.ratio
    _log_verdict(report, settings, verdict, ratio)
    return EXIT_DRIFTED if verdict == "dérive" else EXIT_OK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande de la surveillance."""
    parser = argparse.ArgumentParser(
        prog="training.drift",
        description="Écart entre le modèle servi et les mesures arrivées depuis.",
    )
    parser.add_argument(
        "--since", default=None, help="Première journée évaluée, YYYY-MM-DD."
    )
    parser.add_argument(
        "--until", default=None, help="Dernière journée évaluée, YYYY-MM-DD."
    )
    parser.add_argument(
        "--feature-version",
        default=None,
        help="Version des variables lues. Défaut : etl.feature_version.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée de la surveillance."""
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        version = args.feature_version or config.get_str("etl.feature_version")
        since = parse_date(args.since) if args.since else None
        until = parse_date(args.until) if args.until else None
        return run(config, since, until, version)
    except (ConfigError, PathError, DatasetError, MonitoringError) as exc:
        logger.error("surveillance interrompue : %s", exc)
        return EXIT_FAILED


def _baseline(settings: DriftSettings) -> dict[str, float]:
    """Retourne les métriques de référence, ou rien si elles manquent.

    L'absence de référence n'interrompt pas la mesure : l'écart brut reste
    utile, et le journal dira qu'il n'a pas pu être rapporté à quoi que ce
    soit.
    """
    try:
        return tracking.baseline_metrics(settings.registered_model, settings.alias)
    except Exception as exc:  # noqa: BLE001 - le registre lève large
        logger.warning("référence indisponible : %s", exc)
        return {}


def _log_verdict(
    report: DriftReport,
    settings: DriftSettings,
    verdict: str,
    ratio: float | None,
) -> None:
    """Journalise la conclusion, en disant toujours sur quoi elle repose."""
    mesure = report.metrics[DECISION_METRIC]
    reference = report.baseline.get(DECISION_METRIC)
    if verdict == "indécis":
        logger.warning(
            "%d heure(s) seulement sur %s : trop peu pour conclure (minimum %d)",
            report.rows,
            report.window,
            settings.min_rows,
        )
        return
    if ratio is None:
        logger.warning(
            "version %s sur %s : MAE %.2f kW, aucune référence pour la situer",
            report.version,
            report.window,
            mesure,
        )
        return
    niveau = logger.warning if verdict == "dérive" else logger.info
    niveau(
        "version %s sur %s : MAE %.2f kW contre %.2f à l'entraînement"
        " (×%.2f, seuil ×%.2f) — %s",
        report.version,
        report.window,
        mesure,
        reference,
        ratio,
        settings.alert_ratio,
        verdict,
    )


def _configure_logging() -> None:
    """Arme le journal, et met la sortie standard à l'abri de l'encodage local.

    MLflow imprime des emoji quand il rend la main ; une console Windows en
    cp1252 lève alors une UnicodeEncodeError au beau milieu d'un run qui, lui,
    s'est bien passé.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


if __name__ == "__main__":
    raise SystemExit(main())
