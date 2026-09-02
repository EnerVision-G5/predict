"""Ingestion continue des mesures courantes, point d'entrée du conteneur.

Ce module est le pendant temps réel de `etl.pipeline` : là où le pipeline
rattrape une fenêtre passée par pagination, le poller interroge la mesure
courante de chaque site à cadence fixe et l'écrit au fil de l'eau. Les deux
partagent volontairement les mêmes étages de transformation et de chargement,
donc la même table et les mêmes règles de qualité.

Trois choix structurent le fichier.

Le processus ne s'arrête pas sur un échec réseau. Un site injoignable est
journalisé et le tick continue avec les autres ; la vraie relance, c'est le
tick suivant, une minute plus tard. Seul un incident au démarrage fait sortir
le processus, et c'est alors la politique de redémarrage du conteneur qui
reprend la main.

La cadence est ancrée sur des instants absolus, pas sur une attente de la
durée de l'intervalle. Un tick lent décalerait sinon tous les suivants, et le
retard deviendrait invisible en se fondant dans la cadence.

Le retard est journalisé sous ses deux formes, parce qu'elles ne désignent pas
la même panne : le retard de données mesure l'âge de ce que sert la source, le
retard d'ordonnancement mesure ce que le poller lui-même a pris de retard.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import signal
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import FrameType
from typing import Any

import pandas as pd
import requests
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from etl.config import EtlConfig, load_config
from etl.extract import ExtractionError, build_session, fetch_current
from etl.load import load_frame
from etl.pipeline import resolve_sites
from etl.transform import TransformError, deduplicate, to_frame

# Le démarrage n'a pas le luxe d'attendre le tick suivant : sans référentiel
# des sites, il n'y a rien à interroger. On insiste donc plus longuement
# qu'en régime établi.
STARTUP_ATTEMPTS = 3
STARTUP_BACKOFF_S = 5.0

# Au-delà de cet écart entre l'instant prévu d'un tick et son démarrage réel,
# le poller ne tient plus la cadence : c'est un symptôme, pas un détail.
SCHEDULE_SKEW_WARNING_S = 5.0

EXIT_OK = 0
EXIT_STARTUP_FAILED = 1

# Échecs dont un site peut se remettre au tick suivant. Tout le reste est un
# défaut de programmation, qui doit remonter au lieu d'être absorbé.
RECOVERABLE = (
    ExtractionError,
    TransformError,
    SQLAlchemyError,
    requests.RequestException,
)

logger = logging.getLogger(__name__)


class PollError(RuntimeError):
    """Une opération réseau a échoué malgré ses tentatives."""


@dataclass(frozen=True)
class PollContext:
    """Ressources partagées par tous les ticks d'un même processus."""

    config: EtlConfig
    engine: Engine
    session: requests.Session
    stop: threading.Event


@dataclass(frozen=True)
class SiteTick:
    """Résultat de l'interrogation d'un site sur un tick."""

    site_id: str
    rows: int
    lag_s: float | None


@dataclass(frozen=True)
class TickReport:
    """Bilan consolidé d'un tick, tel qu'il part au journal."""

    rows: int
    lags_s: tuple[float, ...]
    failed_sites: tuple[str, ...]

    @property
    def max_lag_s(self) -> float | None:
        """Retard de données du site le plus en retard, sites muets exclus."""
        if not self.lags_s:
            return None
        return max(self.lags_s)


@dataclass
class Schedule:
    """Suite des instants de tick, ancrée sur un instant de départ."""

    interval_s: float
    due_at: datetime
    missed: int = field(default=0, init=False)

    def advance(self, now: datetime) -> None:
        """Place l'échéance suivante, en écartant les ticks déjà dépassés.

        Rejouer les ticks manqués n'aurait aucun sens ici : `/current` ne sert
        que la mesure du moment, une rafale de rattrapage relirait plusieurs
        fois la même valeur.
        """
        self.due_at += timedelta(seconds=self.interval_s)
        late_s = (now - self.due_at).total_seconds()
        if late_s < 0:
            self.missed = 0
            return
        self.missed = 1 + math.floor(late_s / self.interval_s)
        self.due_at += timedelta(seconds=self.missed * self.interval_s)


def call_with_retry(
    operation: Callable[[], Any],
    attempts: int,
    backoff_s: float,
    sleep: Callable[[float], Any],
    label: str,
) -> Any:
    """Exécute `operation`, en retentant les échecs réseau connus.

    L'attente croît avec le rang de la tentative : une coupure qui dure ne se
    règle pas en insistant à la même cadence.
    """
    last_error: Exception | None = None
    total = max(attempts, 1)
    for attempt in range(1, total + 1):
        try:
            return operation()
        except (ExtractionError, requests.RequestException) as exc:
            last_error = exc
            if attempt >= total:
                break
            delay_s = backoff_s * attempt
            logger.warning(
                "%s : tentative %d/%d échouée (%s), nouvel essai dans %.1f s",
                label,
                attempt,
                total,
                exc,
                delay_s,
            )
            sleep(delay_s)
    raise PollError(f"{label} : {total} tentative(s) échouée(s).") from last_error


def ingestion_lag_s(frame: pd.DataFrame, now: datetime) -> float | None:
    """Retourne l'âge de la mesure la plus ancienne du lot, en secondes.

    Un retard négatif n'est pas corrigé : il signale une horloge de source en
    avance sur la nôtre, information que masquer serait une faute.
    """
    if frame.empty:
        return None
    return float((pd.Timestamp(now) - frame["ts"].min()).total_seconds())


def quality_summary(frame: pd.DataFrame) -> str:
    """Résume la répartition des `data_quality` du lot pour le journal."""
    if frame.empty:
        return "aucune"
    counts = frame["data_quality"].value_counts().to_dict()
    return " ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def poll_site(context: PollContext, site_id: str, now: datetime) -> SiteTick:
    """Lit la mesure courante d'un site et l'écrit en base.

    Les échecs ne sont pas rattrapés ici : ils remontent au tick, seul niveau
    qui sache qu'un site en panne ne doit pas empêcher les autres.
    """
    config = context.config
    records = call_with_retry(
        lambda: fetch_current(config, site_id, session=context.session),
        attempts=config.poll_retries + 1,
        backoff_s=config.poll_backoff_s,
        sleep=context.stop.wait,
        label=f"site {site_id}",
    )
    frame = deduplicate(to_frame(records))
    dropped = len(records) - len(frame)
    if dropped > 0:
        logger.warning(
            "site %s : %d mesure(s) écartée(s), horodatage ou site absent",
            site_id,
            dropped,
        )
    rows = load_frame(context.engine, frame, config.batch_size)
    lag_s = ingestion_lag_s(frame, now)
    _log_site(site_id, rows, lag_s, quality_summary(frame), config.lag_warning_s)
    return SiteTick(site_id=site_id, rows=rows, lag_s=lag_s)


def run_tick(
    context: PollContext,
    sites: Sequence[str],
    now: datetime,
) -> TickReport:
    """Interroge tous les sites une fois et retourne le bilan du tick."""
    rows = 0
    lags: list[float] = []
    failed: list[str] = []
    for site_id in sites:
        try:
            tick = poll_site(context, site_id, now)
        except (PollError, *RECOVERABLE) as exc:
            failed.append(site_id)
            logger.error("site %s : tick abandonné (%s)", site_id, exc)
            continue
        rows += tick.rows
        if tick.lag_s is not None:
            lags.append(tick.lag_s)
    return TickReport(
        rows=rows,
        lags_s=tuple(lags),
        failed_sites=tuple(failed),
    )


def poll_forever(
    context: PollContext,
    sites: Sequence[str],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Boucle d'ingestion jusqu'à ce que l'arrêt du processus soit demandé.

    L'attente passe par l'événement d'arrêt et non par une temporisation
    aveugle : un conteneur qu'on stoppe rend la main tout de suite au lieu
    d'user la minute en cours.
    """
    schedule = Schedule(
        interval_s=context.config.poll_interval_s,
        due_at=clock(),
    )
    completed = 0
    while not context.stop.is_set():
        _wait_until(context.stop, schedule.due_at, clock)
        if context.stop.is_set():
            break
        started_at = clock()
        _log_skew(started_at, schedule)
        report = run_tick(context, sites, started_at)
        finished_at = clock()
        _log_tick(report, len(sites), (finished_at - started_at).total_seconds())
        completed += 1
        schedule.advance(finished_at)
    logger.info("arrêt demandé : %d tick(s) exécuté(s)", completed)
    return completed


def install_signal_handlers(stop: threading.Event) -> None:
    """Arme l'arrêt propre sur SIGTERM et SIGINT.

    Sans cela, `docker stop` couperait le processus au milieu d'un chargement
    et laisserait une transaction ouverte côté base.
    """

    def request_stop(signum: int, frame: FrameType | None) -> None:
        logger.info("signal %s reçu, arrêt après le tick en cours", signum)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def resolve_targets(
    context: PollContext,
    requested: Sequence[str] | None,
) -> list[str]:
    """Retourne les sites à interroger, ceux demandés ou tout le référentiel."""
    if requested:
        return list(requested)
    return call_with_retry(
        lambda: resolve_sites(context.config, None, session=context.session),
        attempts=STARTUP_ATTEMPTS,
        backoff_s=STARTUP_BACKOFF_S,
        sleep=context.stop.wait,
        label="référentiel des sites",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande du conteneur de polling."""
    parser = argparse.ArgumentParser(
        description="Ingestion continue des mesures courantes EnerVision.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Cadence d'interrogation en secondes. Défaut : ETL_POLL_INTERVAL_S.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à interroger. Répétable. Par défaut : tout le référentiel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée du conteneur de polling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    config = load_config()
    if args.interval is not None:
        config = dataclasses.replace(config, poll_interval_s=args.interval)
    stop = threading.Event()
    install_signal_handlers(stop)
    context = PollContext(
        config=config,
        engine=_build_engine(config),
        session=build_session(),
        stop=stop,
    )
    try:
        sites = resolve_targets(context, args.sites)
    except PollError as exc:
        logger.error("démarrage impossible : %s", exc)
        _release(context)
        return EXIT_STARTUP_FAILED
    logger.info(
        "polling de %d site(s) toutes les %.0f s : %s",
        len(sites),
        config.poll_interval_s,
        ", ".join(sites),
    )
    try:
        poll_forever(context, sites)
    finally:
        _release(context)
    return EXIT_OK


def _build_engine(config: EtlConfig) -> Engine:
    """Ouvre le moteur SQLAlchemy du processus long.

    `pool_pre_ping` n'est pas décoratif : un poller vit des jours, et une
    connexion coupée par la base entre deux ticks échouerait sur la première
    écriture au lieu d'être renouvelée.
    """
    return create_engine(config.database_url, pool_pre_ping=True)


def _release(context: PollContext) -> None:
    """Rend les ressources réseau et base du processus."""
    context.session.close()
    context.engine.dispose()


def _wait_until(
    stop: threading.Event,
    due_at: datetime,
    clock: Callable[[], datetime],
) -> None:
    """Attend l'échéance, interruptible par une demande d'arrêt."""
    delay_s = (due_at - clock()).total_seconds()
    if delay_s > 0:
        stop.wait(delay_s)


def _log_skew(started_at: datetime, schedule: Schedule) -> None:
    """Journalise l'écart entre l'instant prévu du tick et son démarrage."""
    if schedule.missed:
        logger.warning(
            "%d tick(s) sauté(s) : le tick précédent a dépassé la cadence",
            schedule.missed,
        )
    skew_s = (started_at - schedule.due_at).total_seconds()
    if skew_s > SCHEDULE_SKEW_WARNING_S:
        logger.warning(
            "retard d'ordonnancement : tick démarré %.1f s trop tard",
            skew_s,
        )


def _log_tick(report: TickReport, requested: int, duration_s: float) -> None:
    """Journalise le bilan d'un tick, retard de données compris."""
    lag = "n/a" if report.max_lag_s is None else f"{report.max_lag_s:.1f} s"
    logger.info(
        "tick : %d/%d site(s), %d mesure(s), retard données max %s, durée %.2f s",
        requested - len(report.failed_sites),
        requested,
        report.rows,
        lag,
        duration_s,
    )
    if report.failed_sites:
        logger.warning(
            "%d site(s) en échec sur ce tick : %s",
            len(report.failed_sites),
            ", ".join(report.failed_sites),
        )


def _log_site(
    site_id: str,
    rows: int,
    lag_s: float | None,
    quality: str,
    warning_s: float,
) -> None:
    """Journalise le résultat d'un site, en alertant sur un retard excessif."""
    if lag_s is None:
        logger.warning("site %s : aucune mesure courante servie", site_id)
        return
    if lag_s > warning_s:
        logger.warning(
            "site %s : retard %.1f s au-delà du seuil %.1f s, qualité %s",
            site_id,
            lag_s,
            warning_s,
            quality,
        )
        return
    logger.info(
        "site %s : %d mesure(s), retard %.1f s, qualité %s",
        site_id,
        rows,
        lag_s,
        quality,
    )


if __name__ == "__main__":
    raise SystemExit(main())
