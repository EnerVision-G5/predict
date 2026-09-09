# **********************************************************************
# * Nom     : drift.py                                                 *
# * Type    : Point d'entrée                                           *
# * Sujet   : Écart entre le modèle servi et les mesures arrivées      *
# *   depuis, verdict en code de sortie                                *
# * Service : training                                                 *
# **********************************************************************

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
from predict_common.schemas import SITE_COLUMN, feature_columns
from training import tracking
from training.dataset import DatasetError, matrices, read_features
from training.model import DECISION_METRIC, evaluate

# Le modèle servi tient ses métriques.
EXIT_OK = 0
# Le calcul lui-même a échoué.
EXIT_FAILED = 1
# Dérive constatée : c'est un verdict, pas une panne.
EXIT_DRIFTED = 2


# Le modèle reste dans les clous de sa référence.
VERDICT_STABLE = "stable"
# L'erreur dépasse le seuil toléré face à la référence.
VERDICT_DRIFTED = "dérive"
# Trop peu d'heures mesurées pour conclure.
VERDICT_UNDECIDED = "indécis"
# Le run de référence n'a pas de métrique comparable.
VERDICT_NO_REFERENCE = "sans référence"

logger = logging.getLogger(__name__)


class MonitoringError(RuntimeError):
    """Classe : MonitoringError
    Description : La surveillance n'a pas pu mesurer ce qu'elle devait.
    """


@dataclass(frozen=True)
class DriftSettings:
    """Classe : DriftSettings
    Description : Fenêtre, seuils et modèle surveillé, lus dans la
      configuration.
    """
    experiment: str
    window_days: int
    alert_ratio: float
    min_rows: int
    min_rows_per_site: int
    registered_model: str
    alias: str

    @classmethod
    def from_config(cls, config: Config) -> DriftSettings:
        """Méthode : from_config
        Description : Construit les réglages depuis le bloc monitoring de la
          configuration.
        """
        return cls(
            experiment=config.get_str("monitoring.experiment"),
            window_days=config.get_int("monitoring.window_days"),
            alert_ratio=config.get_float("monitoring.mae_alert_ratio"),
            min_rows=config.get_int("monitoring.min_rows"),
            min_rows_per_site=config.get_int("monitoring.min_rows_per_site"),
            registered_model=config.get_str("training.registered_model"),
            alias=tracking.PRODUCTION_ALIAS,
        )


def error_ratio(
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> float | None:
    """Méthode : error_ratio
    Description : Rapport entre l'erreur mesurée et celle de référence.
    """
    reference = baseline.get(DECISION_METRIC)
    if not reference:
        return None
    return metrics[DECISION_METRIC] / reference


def verdict_for(
    metrics: dict[str, float],
    baseline: dict[str, float],
    rows: int,
    min_rows: int,
    alert_ratio: float,
) -> str:
    """Méthode : verdict_for
    Description : Tranche entre stable, dérive et indécis selon le volume et le
      seuil.
    """
    if rows < min_rows:
        return VERDICT_UNDECIDED
    ratio = error_ratio(metrics, baseline)
    if ratio is None:
        return VERDICT_NO_REFERENCE
    return VERDICT_DRIFTED if ratio > alert_ratio else VERDICT_STABLE


@dataclass(frozen=True)
class SiteMeasure:
    """Classe : SiteMeasure
    Description : Ce qu'un site a mesuré sur la fenêtre surveillée.
    """
    site_id: str
    metrics: dict[str, float]
    rows: int

    def verdict(
        self,
        settings: DriftSettings,
        baseline: dict[str, float],
    ) -> str:
        """Méthode : verdict
        Description : Verdict propre à ce périmètre, avec ses seuils.
        """
        return verdict_for(
            self.metrics,
            baseline,
            self.rows,
            settings.min_rows_per_site,
            settings.alert_ratio,
        )


@dataclass(frozen=True)
class DriftReport:
    """Classe : DriftReport
    Description : Bilan de la surveillance : ensemble, référence et détail par
      site.
    """
    metrics: dict[str, float]
    baseline: dict[str, float]
    rows: int
    version: str
    window: str
    sites: tuple[SiteMeasure, ...] = ()

    @property
    def ratio(self) -> float | None:
        """Méthode : ratio
        Description : Rapport d'erreur de l'ensemble face à la référence.
        """
        return error_ratio(self.metrics, self.baseline)

    def verdict(self, settings: DriftSettings) -> str:
        """Méthode : verdict
        Description : Verdict propre à ce périmètre, avec ses seuils.
        """
        return verdict_for(
            self.metrics,
            self.baseline,
            self.rows,
            settings.min_rows,
            settings.alert_ratio,
        )

    def drifted_sites(self, settings: DriftSettings) -> tuple[str, ...]:
        """Méthode : drifted_sites
        Description : Sites dont l'erreur dépasse le seuil à eux seuls.
        """
        return tuple(
            site.site_id
            for site in self.sites
            if site.verdict(settings, self.baseline) == VERDICT_DRIFTED
        )


def window(
    since: date | None,
    until: date | None,
    window_days: int,
) -> tuple[date, date]:
    """Méthode : window
    Description : Détermine la fenêtre surveillée à partir des bornes
      demandées.
    """
    if window_days < 1:
        raise MonitoringError("La fenêtre de surveillance couvre au moins un jour.")
    end = until or since or datetime.now(UTC).date()
    start = since or end - timedelta(days=window_days - 1)
    if start > end:
        raise MonitoringError(f"Fenêtre vide : {end} précède {start}.")
    return start, end


def load_served_model(settings: DriftSettings) -> tuple[object, str]:
    """Méthode : load_served_model
    Description : Charge le modèle actuellement servi et son numéro de version.
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
    """Méthode : measure
    Description : Mesure le modèle sur un lot de variables déjà lu.
    """
    explanatory, observed = matrices(frame, columns)
    predicted = model.predict(explanatory)
    return evaluate(observed, predicted)


def measure_sites(
    model: object,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[SiteMeasure, ...]:
    """Méthode : measure_sites
    Description : Mesure le modèle site par site, si la partition les
      distingue.
    """
    if SITE_COLUMN not in frame.columns:
        logger.warning(
            "partition sans colonne %s : pas de détail par site", SITE_COLUMN
        )
        return ()
    return tuple(
        SiteMeasure(
            site_id=str(site_id),
            metrics=measure(model, group, columns),
            rows=len(group),
        )
        for site_id, group in frame.groupby(SITE_COLUMN, sort=True)
    )


def evaluate_window(
    config: Config,
    settings: DriftSettings,
    version_of_features: str,
    since: date,
    until: date,
) -> DriftReport:
    """Méthode : evaluate_window
    Description : Lit la fenêtre, mesure le modèle servi et compose le bilan.
    """
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
        sites=measure_sites(model, frame, columns),
    )


def publish(report: DriftReport, settings: DriftSettings, feature_version: str) -> None:
    """Méthode : publish
    Description : Enregistre le bilan dans sa propre expérience MLflow.
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
            mlflow.log_metric(f"baseline_{name}", value)
        ratio = report.ratio
        if ratio is not None:
            mlflow.log_metric("mae_ratio", ratio)
        mlflow.set_tag("verdict", report.verdict(settings))
        _publish_sites(report, settings)


def run(config: Config, since: date | None, until: date | None, version: str) -> int:
    """Méthode : run
    Description : Surveille la fenêtre demandée et rend le code de sortie du
      verdict.
    """
    settings = DriftSettings.from_config(config)
    mlflow.set_tracking_uri(config.get_str("mlflow.tracking_uri"))
    first, last = window(since, until, settings.window_days)
    report = evaluate_window(config, settings, version, first, last)
    publish(report, settings, version)

    verdict = report.verdict(settings)
    ratio = report.ratio
    _log_verdict(report, settings, verdict, ratio)
    drifted = report.drifted_sites(settings)
    _log_sites(report, settings, drifted)
    if verdict == VERDICT_DRIFTED or drifted:
        return EXIT_DRIFTED
    return EXIT_OK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Méthode : parse_args
    Description : Analyse la ligne de commande de la surveillance.
    """
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
    """Méthode : main
    Description : Point d'entrée : mesure la dérive et rend son verdict en code
      de sortie.
    """
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
    """Méthode : _baseline
    Description : Lit les métriques de référence, sans faire échouer la mesure
      si elles manquent.
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
    """Méthode : _log_verdict
    Description : Journalise le verdict d'ensemble et ce qui l'a fondé.
    """
    mesure = report.metrics[DECISION_METRIC]
    reference = report.baseline.get(DECISION_METRIC)
    if verdict == VERDICT_UNDECIDED:
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
    niveau = logger.warning if verdict == VERDICT_DRIFTED else logger.info
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


def _publish_sites(report: DriftReport, settings: DriftSettings) -> None:
    """Méthode : _publish_sites
    Description : Enregistre les mesures et verdicts de chaque site dans le
      run.
    """
    for site in report.sites:
        for name, value in site.metrics.items():
            mlflow.log_metric(f"{site.site_id}_{name}", value)
        mlflow.log_metric(f"{site.site_id}_rows", site.rows)
        mlflow.set_tag(
            f"verdict_{site.site_id}", site.verdict(settings, report.baseline)
        )


def _log_sites(
    report: DriftReport,
    settings: DriftSettings,
    drifted: Sequence[str],
) -> None:
    """Méthode : _log_sites
    Description : Journalise le détail par site, du plus dégradé au moins.
    """
    if not report.sites:
        return
    ordered = sorted(
        report.sites, key=lambda site: site.metrics[DECISION_METRIC], reverse=True
    )
    for site in ordered:
        logger.info(
            "  %s : MAE %.2f kW sur %d heure(s) — %s",
            site.site_id,
            site.metrics[DECISION_METRIC],
            site.rows,
            site.verdict(settings, report.baseline),
        )
    if drifted:
        logger.warning(
            "%d site(s) en dérive : %s", len(drifted), ", ".join(drifted)
        )


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


if __name__ == "__main__":
    raise SystemExit(main())
