# **********************************************************************
# * Nom     : poller.py                                                *
# * Type    : Point d'entrée                                           *
# * Sujet   : Collecte continue des mesures courantes, à cadence fixe  *
# * Service : collector                                                *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import FrameType

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from collector.__main__ import (
    DEFAULT_CATCH_UP_DAYS,
    catch_up_days,
    collect_day,
)
from collector.sink import (
    IngestionState,
    read_sensor_statuses,
    sync_sites,
    to_measures,
    to_sensor_states,
    write,
    write_alerts,
    write_sensor_episodes,
    write_sensor_states,
    write_state,
)
from predict_common.config import Config, ConfigError, load_config
from predict_common.db import (
    INGESTION_SOURCE_POLLER,
    DatabaseError,
    alerte,
    capteur_etat,
    capteur_panne,
    ingestion_etat,
    mesure,
    open_engine,
    site,
    verify_schema,
)
from predict_common.schemas import TIMESTAMP_COLUMN
from predict_common.source import SourceClient, SourceError, SourceSettings
from predict_common.timestamps import DEFAULT_SOURCE_TIMEZONE

# Tentatives de lecture du référentiel au démarrage.
STARTUP_ATTEMPTS = 3
# Attente entre deux tentatives de démarrage.
STARTUP_BACKOFF_S = 5.0

# Écart à l'heure prévue au-delà duquel on avertit.
SCHEDULE_SKEW_WARNING_S = 5.0

# Code de sortie d'un arrêt demandé.
EXIT_OK = 0
# Code de sortie d'un démarrage impossible.
EXIT_STARTUP_FAILED = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PollSettings:
    """Classe : PollSettings
    Description : Cadence, seuil de retard et taille de lot de la collecte
      continue.
    """
    interval_s: float
    lag_warning_s: float
    batch_size: int
    source_timezone: str = DEFAULT_SOURCE_TIMEZONE

    @classmethod
    def from_config(cls, config: Config) -> PollSettings:
        """Méthode : from_config
        Description : Construit les réglages depuis le bloc collector de la
          configuration.
        """
        return cls(
            interval_s=config.get_float("collector.poll_interval_s"),
            lag_warning_s=config.get_float("collector.lag_warning_s"),
            batch_size=config.get_int("database.batch_size"),
            source_timezone=config.get_str(
                "source.timezone", DEFAULT_SOURCE_TIMEZONE
            ),
        )


@dataclass(frozen=True)
class PollContext:
    """Classe : PollContext
    Description : Ce dont un tick a besoin : réglages, client, base, et le
      signal d'arrêt.
    """
    settings: PollSettings
    client: SourceClient
    engine: Engine
    stop: threading.Event


@dataclass(frozen=True)
class SiteTick:
    """Classe : SiteTick
    Description : Résultat de la collecte d'un site : lignes écrites et retard
      constaté.
    """
    site_id: str
    rows: int
    lag_s: float | None


@dataclass(frozen=True)
class TickReport:
    """Classe : TickReport
    Description : Bilan d'un tick sur tous les sites, états d'ingestion
      compris.
    """
    rows: int
    lags_s: tuple[float, ...]
    states: tuple[IngestionState, ...]

    @property
    def failed_sites(self) -> tuple[str, ...]:
        """Méthode : failed_sites
        Description : Sites dont le tick n'a rien pu écrire.
        """
        return tuple(
            state.site_id for state in self.states if not state.succeeded
        )

    @property
    def max_lag_s(self) -> float | None:
        """Méthode : max_lag_s
        Description : Retard le plus élevé constaté sur le tick.
        """
        if not self.lags_s:
            return None
        return max(self.lags_s)


@dataclass
class Schedule:
    """Classe : Schedule
    Description : Échéancier du tick suivant, et compte des ticks sautés.
    """
    interval_s: float
    due_at: datetime
    missed: int = field(default=0, init=False)

    def advance(self, now: datetime) -> None:
        """Méthode : advance
        Description : Avance l'échéance d'un pas et compte ce qui a été manqué.
        """
        self.due_at += timedelta(seconds=self.interval_s)
        late_s = (now - self.due_at).total_seconds()
        if late_s < 0:
            self.missed = 0
            return
        self.missed = 1 + math.floor(late_s / self.interval_s)
        self.due_at += timedelta(seconds=self.missed * self.interval_s)


def ingestion_lag_s(frame: pd.DataFrame, now: datetime) -> float | None:
    """Méthode : ingestion_lag_s
    Description : Retard entre la mesure la plus ancienne du lot et maintenant.
    """
    if frame.empty:
        return None
    oldest = frame[TIMESTAMP_COLUMN].min()
    return float((pd.Timestamp(now) - oldest).total_seconds())


def quality_summary(frame: pd.DataFrame) -> str:
    """Méthode : quality_summary
    Description : Résume les qualifications d'un lot en une ligne de journal.
    """
    if frame.empty:
        return "aucune"
    counts = frame["data_quality"].value_counts(dropna=False).to_dict()
    ordered = sorted(counts.items(), key=str)
    return " ".join(f"{name}={count}" for name, count in ordered)


def poll_site(context: PollContext, site_id: str, now: datetime) -> SiteTick:
    """Méthode : poll_site
    Description : Interroge un site, écrit sa mesure courante et rend son
      bilan.
    """
    settings = context.settings
    records = context.client.fetch_current(site_id)
    frame = to_measures(records, context.settings.source_timezone)
    report = write(context.engine, frame, settings.batch_size)
    lag_s = ingestion_lag_s(frame, now)
    _log_site(
        site_id,
        report.rows,
        lag_s,
        quality_summary(frame),
        settings.lag_warning_s,
    )
    return SiteTick(site_id=site_id, rows=report.rows, lag_s=lag_s)


def run_tick(
    context: PollContext,
    sites: Sequence[str],
    now: datetime,
) -> TickReport:
    """Méthode : run_tick
    Description : Interroge tous les sites une fois et retourne le bilan du
      tick.
    """
    rows = 0
    lags: list[float] = []
    states: list[IngestionState] = []
    for site_id in sites:
        try:
            tick = poll_site(context, site_id, now)
        except (SourceError, SQLAlchemyError, OSError, ValueError) as exc:
            logger.error("site %s : tick abandonné (%s)", site_id, exc)
            states.append(
                IngestionState(
                    site_id=site_id,
                    attempted_at=now,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows += tick.rows
        if tick.lag_s is not None:
            lags.append(tick.lag_s)
        states.append(
            IngestionState(
                site_id=site_id,
                attempted_at=now,
                rows=tick.rows,
                data_lag_s=tick.lag_s,
            )
        )
    return TickReport(rows=rows, lags_s=tuple(lags), states=tuple(states))


def record_tick(context: PollContext, report: TickReport) -> None:
    """Méthode : record_tick
    Description : Repose l'état de collecte du tick, sans jamais interrompre la
      boucle.
    """
    try:
        write_state(
            context.engine,
            report.states,
            context.settings.batch_size,
            INGESTION_SOURCE_POLLER,
        )
    except (SQLAlchemyError, OSError) as exc:
        logger.error("état d'ingestion non enregistré : %s", exc)


def poll_forever(
    context: PollContext,
    sites: Sequence[str],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Méthode : poll_forever
    Description : Boucle jusqu'au signal d'arrêt, un tick par échéance.
    """
    schedule = Schedule(interval_s=context.settings.interval_s, due_at=clock())
    completed = 0
    while not context.stop.is_set():
        _wait_until(context.stop, schedule.due_at, clock)
        if context.stop.is_set():
            break
        started_at = clock()
        _log_skew(started_at, schedule)
        report = run_tick(context, sites, started_at)
        record_tick(context, report)
        collect_side_channels(context)
        finished_at = clock()
        _log_tick(report, len(sites), (finished_at - started_at).total_seconds())
        completed += 1
        schedule.advance(finished_at)
    logger.info("arrêt demandé : %d tick(s) exécuté(s)", completed)
    return completed


def collect_side_channels(context: PollContext) -> None:
    """Méthode : collect_side_channels
    Description : Collecte alertes et état des capteurs, en marge des mesures.
    """
    batch_size = context.settings.batch_size
    try:
        write_alerts(
            context.engine,
            context.client.fetch_alerts(),
            batch_size,
            context.settings.source_timezone,
        )
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("alertes non collectées : %s", exc)
    try:
        _collect_sensors(context, batch_size)
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("état des capteurs non collecté : %s", exc)


def _collect_sensors(context: PollContext, batch_size: int) -> None:
    """Méthode : _collect_sensors
    Description : Repose l'état des capteurs et ouvre ou ferme leurs pannes.
    """
    payload = context.client.fetch_sensors_status()
    timezone = context.settings.source_timezone
    states = to_sensor_states(payload, timezone)
    previous = read_sensor_statuses(context.engine)
    write_sensor_states(context.engine, payload, batch_size, timezone)
    write_sensor_episodes(
        context.engine, previous, states, datetime.now(UTC), batch_size
    )


def catch_up(
    context: PollContext,
    sites: Sequence[str],
    config: Config,
    depth_days: int | None,
) -> None:
    """Méthode : catch_up
    Description : Rejoue les journées incomplètes trouvées en base au
      démarrage.
    """
    depth = depth_days or config.get_int(
        "collector.catch_up_days", DEFAULT_CATCH_UP_DAYS
    )
    batch_size = context.settings.batch_size
    try:
        days = catch_up_days(context.engine, sites, depth)
        logger.info(
            "rattrapage de %s à %s avant la boucle", days[0], days[-1]
        )
        rows = sum(
            collect_day(context.client, context.engine, batch_size, day, sites)
            for day in days
        )
        logger.info("rattrapage terminé : %d mesure(s) soumise(s)", rows)
    except (SourceError, SQLAlchemyError, OSError) as exc:
        logger.error(
            "rattrapage abandonné (%s) : la boucle démarre quand même, le"
            " trou reste à combler avec `python -m collector --catch-up`",
            exc,
        )


def install_signal_handlers(stop: threading.Event) -> None:
    """Méthode : install_signal_handlers
    Description : Fait de SIGTERM et SIGINT une demande d'arrêt après le tick.
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
    """Méthode : resolve_targets
    Description : Choisit les sites à interroger et synchronise leur
      référentiel.
    """
    referential = _fetch_referential(context, required=not requested)
    if referential is None:
        return list(requested or ())
    sync_sites(context.engine, referential, context.settings.batch_size)
    if requested:
        return list(requested)
    return [str(entry["site_id"]) for entry in referential if "site_id" in entry]


def _fetch_referential(
    context: PollContext,
    required: bool,
) -> list[dict] | None:
    """Méthode : _fetch_referential
    Description : Lit le référentiel des sites, avec quelques tentatives au
      démarrage.
    """
    last_error: Exception | None = None
    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            return context.client.fetch_sites()
        except SourceError as exc:
            last_error = exc
            if attempt < STARTUP_ATTEMPTS:
                context.stop.wait(STARTUP_BACKOFF_S * attempt)
    if required:
        raise SourceError(
            f"référentiel des sites : {STARTUP_ATTEMPTS} tentative(s) échouée(s)."
        ) from last_error
    logger.warning(
        "référentiel indisponible : boucle sur les sites demandés sans"
        " synchronisation"
    )
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Méthode : parse_args
    Description : Analyse la ligne de commande de la collecte continue.
    """
    parser = argparse.ArgumentParser(
        prog="collector.poller",
        description="Collecte continue des mesures courantes EnerVision.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Cadence en secondes. Défaut : collector.poll_interval_s.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à interroger. Répétable. Par défaut : tout le référentiel.",
    )
    parser.add_argument(
        "--no-catch-up",
        action="store_true",
        help=(
            "Démarre la boucle sans rattraper ce qui manque en base. Le"
            " rattrapage est fait par défaut : un poller qui redémarre après"
            " une coupure reprendrait sinon au présent, en laissant le trou."
        ),
    )
    parser.add_argument(
        "--catch-up-days",
        type=int,
        default=None,
        help=(
            "Profondeur maximale du rattrapage de démarrage, en journées."
            " Défaut : collector.catch_up_days."
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
    Description : Point d'entrée : monte le contexte, boucle, et rend un code
      de sortie.
    """
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        settings = PollSettings.from_config(config)
        source_settings = SourceSettings.from_config(config)
        engine = open_engine(
            config.get_optional_str("database.url"), pool_pre_ping=True
        )
        verify_schema(
            engine,
            (mesure, site, ingestion_etat, alerte, capteur_etat, capteur_panne),
        )
    except (ConfigError, DatabaseError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_STARTUP_FAILED
    if args.interval is not None:
        settings = _with_interval(settings, args.interval)

    stop = threading.Event()
    install_signal_handlers(stop)
    try:
        with SourceClient(source_settings) as client:
            context = PollContext(
                settings=settings, client=client, engine=engine, stop=stop
            )
            try:
                sites = resolve_targets(context, args.sites)
            except SourceError as exc:
                logger.error("démarrage impossible : %s", exc)
                return EXIT_STARTUP_FAILED
            if not args.no_catch_up:
                catch_up(context, sites, config, args.catch_up_days)
            logger.info(
                "collecte de %d site(s) toutes les %.0f s : %s",
                len(sites),
                settings.interval_s,
                ", ".join(sites),
            )
            poll_forever(context, sites)
    finally:
        engine.dispose()
    return EXIT_OK


def _with_interval(settings: PollSettings, interval_s: float) -> PollSettings:
    """Méthode : _with_interval
    Description : Rend les mêmes réglages avec une autre cadence, refusée si
      nulle.
    """
    if interval_s <= 0:
        raise ValueError("--interval doit être strictement positif.")
    return replace(settings, interval_s=interval_s)


def _wait_until(
    stop: threading.Event,
    due_at: datetime,
    clock: Callable[[], datetime],
) -> None:
    """Méthode : _wait_until
    Description : Attend l'échéance, interruptible par le signal d'arrêt.
    """
    delay_s = (due_at - clock()).total_seconds()
    if delay_s > 0:
        stop.wait(delay_s)


def _log_skew(started_at: datetime, schedule: Schedule) -> None:
    """Méthode : _log_skew
    Description : Signale les ticks sautés et l'écart à l'heure prévue.
    """
    if schedule.missed:
        logger.warning(
            "%d tick(s) sauté(s) : le tick précédent a dépassé la cadence",
            schedule.missed,
        )
    skew_s = (started_at - schedule.due_at).total_seconds()
    if skew_s > SCHEDULE_SKEW_WARNING_S:
        logger.warning(
            "retard d'ordonnancement : tick démarré %.1f s trop tard", skew_s
        )


def _log_tick(report: TickReport, requested: int, duration_s: float) -> None:
    """Méthode : _log_tick
    Description : Journalise le bilan d'un tick en une ligne.
    """
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
    """Méthode : _log_site
    Description : Journalise le résultat d'un site, en avertissant sur un
      retard excessif.
    """
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
