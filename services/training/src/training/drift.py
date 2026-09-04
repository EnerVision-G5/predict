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

**La mesure est faite site par site autant que d'ensemble.** Une MAE unique
sur tout le parc est dominée par le plus gros consommateur : elle dit ce que
le parc coûte en erreur, pas où l'erreur se trouve, et un bureau qui double la
sienne disparaît dans la moyenne d'une usine dix fois plus grande. Le code 2
sort donc aussi quand un seul site dérive, sans quoi le détail par site serait
publié sans jamais être écouté. La référence, elle, reste celle du modèle et
n'est pas propre au site : voir `SiteMeasure.verdict`.
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
from predict_common.schemas import SITE_COLUMN, feature_columns
from training import tracking
from training.dataset import DatasetError, matrices, read_features
from training.model import DECISION_METRIC, evaluate

EXIT_OK = 0
EXIT_FAILED = 1
# Distinct de EXIT_FAILED : une dérive est un résultat, pas une panne. Les
# confondre ferait chercher un problème d'infrastructure là où le modèle a
# simplement vieilli.
EXIT_DRIFTED = 2


# Les quatre conclusions possibles. Nommées parce qu'elles voyagent : le
# journal les imprime, MLflow les pose en tag, et le code de sortie en dépend.
VERDICT_STABLE = "stable"
VERDICT_DRIFTED = "dérive"
VERDICT_UNDECIDED = "indécis"
VERDICT_NO_REFERENCE = "sans référence"

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
    min_rows_per_site: int
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
            min_rows_per_site=config.get_int("monitoring.min_rows_per_site"),
            registered_model=config.get_str("training.registered_model"),
            alias=tracking.PRODUCTION_ALIAS,
        )


def error_ratio(
    metrics: dict[str, float],
    baseline: dict[str, float],
) -> float | None:
    """Rapport entre l'erreur mesurée et celle de l'entraînement.

    `None` quand la référence manque ou vaut zéro : un modèle enregistré sans
    métrique de test ne permet aucune comparaison, inventer un rapport de 1
    laisserait croire que tout va bien, et diviser par zéro donnerait un
    infini, donc une alerte permanente.
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
    """Conclut, ou dit pourquoi il n'y a rien à conclure.

    Sortie ici plutôt que portée par le rapport : la même règle tranche pour
    l'ensemble de la fenêtre et pour chacun de ses sites, avec un plancher de
    lignes différent. L'écrire deux fois laisserait les deux dériver.
    """
    if rows < min_rows:
        return VERDICT_UNDECIDED
    ratio = error_ratio(metrics, baseline)
    if ratio is None:
        return VERDICT_NO_REFERENCE
    return VERDICT_DRIFTED if ratio > alert_ratio else VERDICT_STABLE


@dataclass(frozen=True)
class SiteMeasure:
    """Ce que le modèle servi a donné sur un seul site de la fenêtre."""

    site_id: str
    metrics: dict[str, float]
    rows: int

    def verdict(
        self,
        settings: DriftSettings,
        baseline: dict[str, float],
    ) -> str:
        """Conclut pour ce site, avec le plancher de lignes qui lui convient.

        La référence reste celle du modèle, mesurée sur tout son jeu de test :
        elle n'est pas propre au site. Un site structurellement plus difficile
        que la moyenne paraîtra donc dégradé dès le premier jour. Ce verdict
        sert à ranger les sites entre eux et à voir l'un d'eux se détacher, pas
        à juger un site dans l'absolu — une référence par site demanderait que
        l'entraînement en enregistre une, ce qu'il ne fait pas encore.
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
    """Ce qu'une évaluation a mesuré, et ce qu'elle en conclut."""

    metrics: dict[str, float]
    baseline: dict[str, float]
    rows: int
    version: str
    window: str
    # Le détail par site. Vide quand la partition n'en porte pas la colonne,
    # ce qui n'empêche pas la mesure d'ensemble d'exister.
    sites: tuple[SiteMeasure, ...] = ()

    @property
    def ratio(self) -> float | None:
        """Rapport entre l'erreur mesurée et celle de l'entraînement."""
        return error_ratio(self.metrics, self.baseline)

    def verdict(self, settings: DriftSettings) -> str:
        """Dit ce que la mesure d'ensemble permet de conclure."""
        return verdict_for(
            self.metrics,
            self.baseline,
            self.rows,
            settings.min_rows,
            settings.alert_ratio,
        )

    def drifted_sites(self, settings: DriftSettings) -> tuple[str, ...]:
        """Sites dont l'erreur dépasse le seuil, dans l'ordre alphabétique.

        C'est la raison d'être de la découpe. Une MAE d'ensemble est dominée
        par le plus gros consommateur : un site de bureau qui double son
        erreur disparaît dans la moyenne d'une usine dix fois plus grande, et
        la dérive qu'on cherche est précisément celle-là.
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

    C'est aussi pourquoi ce nombre n'est pas celui que l'API métier publie
    sous « écart prédiction / consommation réelle » : celui-là compare les
    prévisions réellement servies, récursives, à ce qui est arrivé ensuite.
    Les deux sont justes, celui-ci sera toujours le meilleur des deux, et les
    afficher sous le même libellé serait un contresens.
    """
    explanatory, observed = matrices(frame, columns)
    predicted = model.predict(explanatory)
    return evaluate(observed, predicted)


def measure_sites(
    model: object,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[SiteMeasure, ...]:
    """Refait la même mesure, site par site, dans l'ordre des identifiants.

    Une moyenne d'ensemble est dominée par le plus gros consommateur : elle
    dit ce que le parc coûte en erreur, pas où l'erreur se trouve. Le tri par
    identifiant rend le journal comparable d'un run à l'autre.

    Une partition dépourvue de la colonne de site rend un tuple vide plutôt
    qu'une erreur : la mesure d'ensemble, elle, reste valable.
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
        sites=measure_sites(model, frame, columns),
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
        _publish_sites(report, settings)


def run(config: Config, since: date | None, until: date | None, version: str) -> int:
    """Mesure l'écart, le publie, et retourne le code de sortie qui convient.

    Le code 2 sort dès qu'un site dérive, et pas seulement quand l'ensemble
    dérive. C'est ce que la découpe par site sert à voir : une MAE globale est
    dominée par le plus gros consommateur, et attendre qu'elle bouge
    reviendrait à ne jamais réagir à la dérive d'un petit site — c'est-à-dire
    à publier un détail par site sans jamais l'écouter.
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
    """Publie le détail par site : une métrique et un tag par identifiant.

    Les noms sont préfixés par le site plutôt que regroupés dans un seul
    dictionnaire : la vue « Chart » de MLflow trace une métrique nommée, et un
    dictionnaire ne s'y trace pas.
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
    """Journalise le détail par site, du plus en erreur au moins en erreur.

    Le tri par erreur décroissante, et non par identifiant : ce qu'on vient
    lire dans ce journal, c'est quel site s'est détaché.
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
