"""Collecte continue de la mesure courante, second point d'entrée du service.

    python -m collector.poller
    python -m collector.poller --interval 30 --site SITE001

Là où `python -m collector` rattrape une journée passée par pagination, le
poller interroge `/current` sur chaque site à cadence fixe et l'écrit au fil de
l'eau. Les deux alimentent la même table, avec le même schéma et la même
insertion idempotente : c'est l'ETL qui les réunit, sans savoir lequel des deux
a produit quoi.

Trois choix structurent le fichier, et ils sont les mêmes qu'avant la découpe
parce que ce sont les bons.

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

Chaque tick repose en plus son état dans `ingestion_etat`, une ligne par site.
Le journal ne suffisait pas : personne ne le lit depuis un dashboard, et
surtout un site en échec n'écrit rien dans `mesure`, si bien que rien en base
ne le distinguait d'un site dont la source n'avait rien de neuf. Cette
écriture-là ne peut jamais interrompre la boucle — voir `record_tick`.

Ni le poller ni le rattrapage n'écrasent quoi que ce soit : les deux insèrent
en `ON CONFLICT DO NOTHING`. Une minute déjà relevée au fil de l'eau n'est donc
pas réécrite par le rattrapage du lendemain, et surtout, aucun des deux ne
recouvre les colonnes que l'ETL a déduites entre-temps.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

# Le démarrage n'a pas le luxe d'attendre le tick suivant : sans référentiel
# des sites, il n'y a rien à interroger. On insiste donc plus longuement
# qu'en régime établi, où l'échec d'un site est absorbé par la cadence.
STARTUP_ATTEMPTS = 3
STARTUP_BACKOFF_S = 5.0

# Au-delà de cet écart entre l'instant prévu d'un tick et son démarrage réel,
# le poller ne tient plus la cadence : c'est un symptôme, pas un détail.
SCHEDULE_SKEW_WARNING_S = 5.0

EXIT_OK = 0
EXIT_STARTUP_FAILED = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PollSettings:
    """Ce qui règle la boucle, extrait de la configuration une seule fois."""

    interval_s: float
    lag_warning_s: float
    batch_size: int

    @classmethod
    def from_config(cls, config: Config) -> PollSettings:
        """Lit les seules clés dont la boucle a besoin."""
        return cls(
            interval_s=config.get_float("collector.poll_interval_s"),
            lag_warning_s=config.get_float("collector.lag_warning_s"),
            batch_size=config.get_int("database.batch_size"),
        )


@dataclass(frozen=True)
class PollContext:
    """Ressources partagées par tous les ticks d'un même processus."""

    settings: PollSettings
    client: SourceClient
    engine: Engine
    stop: threading.Event


@dataclass(frozen=True)
class SiteTick:
    """Résultat de l'interrogation d'un site sur un tick."""

    site_id: str
    rows: int
    lag_s: float | None


@dataclass(frozen=True)
class TickReport:
    """Bilan consolidé d'un tick, tel qu'il part au journal et en base."""

    rows: int
    lags_s: tuple[float, ...]
    # Un état par site interrogé, succès comme échec. Le journal en tire son
    # résumé, `ingestion_etat` en tire ses lignes : les deux disent la même
    # chose du même tick, ce qui n'est vrai que parce qu'ils partent d'ici.
    states: tuple[IngestionState, ...]

    @property
    def failed_sites(self) -> tuple[str, ...]:
        """Sites dont le tick a échoué, dans l'ordre d'interrogation."""
        return tuple(
            state.site_id for state in self.states if not state.succeeded
        )

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


def ingestion_lag_s(frame: pd.DataFrame, now: datetime) -> float | None:
    """Retourne l'âge de la mesure la plus ancienne du lot, en secondes.

    Un retard négatif n'est pas corrigé : il signale une horloge de source en
    avance sur la nôtre, information que masquer serait une faute.
    """
    if frame.empty:
        return None
    oldest = frame[TIMESTAMP_COLUMN].min()
    return float((pd.Timestamp(now) - oldest).total_seconds())


def quality_summary(frame: pd.DataFrame) -> str:
    """Résume la répartition des `data_quality` du lot pour le journal."""
    if frame.empty:
        return "aucune"
    counts = frame["data_quality"].value_counts(dropna=False).to_dict()
    ordered = sorted(counts.items(), key=str)
    return " ".join(f"{name}={count}" for name, count in ordered)


def poll_site(context: PollContext, site_id: str, now: datetime) -> SiteTick:
    """Lit la mesure courante d'un site et l'écrit dans `mesure`.

    Les échecs ne sont pas rattrapés ici : ils remontent au tick, seul niveau
    qui sache qu'un site en panne ne doit pas empêcher les autres.
    """
    settings = context.settings
    records = context.client.fetch_current(site_id)
    frame = to_measures(records)
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
    """Interroge tous les sites une fois et retourne le bilan du tick.

    Un état est produit pour chaque site, y compris en échec — et c'est le
    point : un site qui n'a rien écrit ne laisse aucune trace dans `mesure`,
    et sans cet état il serait indiscernable d'un site que la source n'avait
    simplement rien à dire.
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
    """Repose l'état de collecte du tick, sans jamais interrompre la boucle.

    L'écriture est rattrapée ici et nulle part ailleurs. Un tick dont la base
    vient de refuser les mesures ne pourra pas non plus y écrire son échec :
    laisser remonter l'exception ferait mourir le processus au moment précis
    où il a le plus de raisons de continuer à essayer. Le journal garde alors
    la trace, et le tick suivant retentera.
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
    """Boucle de collecte jusqu'à ce que l'arrêt du processus soit demandé.

    L'attente passe par l'événement d'arrêt et non par une temporisation
    aveugle : un conteneur qu'on stoppe rend la main tout de suite au lieu
    d'user la minute en cours.
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
    """Journalise les alertes et repose l'état des capteurs.

    Deux routes que `mesure` ne remplace pas. Les alertes disent ce que la
    source a jugé anormal, avec son seuil — information qu'aucune mesure ne
    porte, et qui disparaît de la réponse dès que l'alerte se résout. L'état
    des capteurs dit lequel est tombé et jusqu'à quand, là où `null_reasons`
    ne dit que ce qui manquait sur une ligne.

    Aucun échec ne remonte : ce sont des annexes du tick, pas le tick. Une
    route d'alertes en panne ne doit pas arrêter la collecte des mesures, qui
    est la seule chose dont la chaîne aval dépend.
    """
    batch_size = context.settings.batch_size
    try:
        write_alerts(context.engine, context.client.fetch_alerts(), batch_size)
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("alertes non collectées : %s", exc)
    try:
        _collect_sensors(context, batch_size)
    except Exception as exc:  # noqa: BLE001 - annexe : rien ne doit remonter
        logger.warning("état des capteurs non collecté : %s", exc)


def _collect_sensors(context: PollContext, batch_size: int) -> None:
    """Repose l'état des capteurs, et journalise les transitions au passage.

    L'état précédent est lu AVANT d'être écrasé : c'est lui qui date les
    débuts et les fins de panne, la source ne servant qu'un présent.
    """
    payload = context.client.fetch_sensors_status()
    states = to_sensor_states(payload)
    previous = read_sensor_statuses(context.engine)
    write_sensor_states(context.engine, payload, batch_size)
    write_sensor_episodes(
        context.engine, previous, states, datetime.now(UTC), batch_size
    )


def catch_up(
    context: PollContext,
    sites: Sequence[str],
    config: Config,
    depth_days: int | None,
) -> None:
    """Comble ce qui manque en base avant d'entrer dans la boucle.

    Sans lui, un poller qui redémarre reprend AU PRÉSENT : tout ce que la
    coupure a laissé passer reste un trou, et rien ne le signale — `mesure`
    n'a pas de ligne à montrer pour une minute qui n'a jamais été collectée.
    Le trou ne se voyait qu'au moment où l'ETL produisait une journée creuse,
    ou pas du tout.

    Le rattrapage passe par `/api/v1/readings`, la seule route qui serve du
    passé — `/current` ne connaît que l'instant présent. Sa profondeur est
    déduite de la dernière mesure de chaque site, donc de la durée réelle de
    la coupure : cinq minutes d'arrêt coûtent une journée relue, une semaine
    en coûte sept. Voir `collector.__main__.catch_up_days`.

    Son échec n'empêche pas la boucle de démarrer, et c'est délibéré : la
    collecte du présent a plus de valeur que celle du passé, et un rattrapage
    qui échoue peut être relancé à la main (`python -m collector --catch-up`)
    sans arrêter le service.
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
    """Arme l'arrêt propre sur SIGTERM et SIGINT.

    Sans cela, `docker stop` couperait le processus au milieu d'une écriture
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
    """Synchronise le référentiel et retourne les sites à interroger.

    Le référentiel est réclamé avec plus d'insistance qu'une mesure : sans
    lui, la boucle n'a rien à faire et le processus doit sortir pour que le
    conteneur le relance.
    """
    referential = _fetch_referential(context, required=not requested)
    if referential is None:
        return list(requested or ())
    # Entretient `site`, que `mesure.site_id` référence : un site absent ferait
    # rejeter ses mesures sans que rien n'explique pourquoi.
    sync_sites(context.engine, referential, context.settings.batch_size)
    if requested:
        return list(requested)
    return [str(entry["site_id"]) for entry in referential if "site_id" in entry]


def _fetch_referential(
    context: PollContext,
    required: bool,
) -> list[dict] | None:
    """Réclame le référentiel, plus longuement qu'une mesure ordinaire.

    Sans lui et sans `--site`, la boucle n'a rien à interroger : le processus
    doit sortir pour que le conteneur le relance. Avec `--site`, l'exploitant a
    nommé ses sites et une source qui ne sert pas son référentiel ne doit pas
    empêcher la boucle de tourner.
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
    """Analyse la ligne de commande du conteneur de collecte continue."""
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
    """Point d'entrée du conteneur de collecte continue."""
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        settings = PollSettings.from_config(config)
        source_settings = SourceSettings.from_config(config)
        # pool_pre_ping : le poller vit des jours, et une connexion coupée par
        # la base entre deux ticks échouerait sur la première écriture au lieu
        # d'être renouvelée.
        engine = open_engine(
            config.get_optional_str("database.url"), pool_pre_ping=True
        )
        # Le poller est un processus long : une base en retard de migration
        # doit l'empêcher de démarrer, pas le laisser journaliser le même
        # échec toutes les minutes pendant des jours.
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
    """Retourne les réglages avec la cadence imposée en ligne de commande."""
    if interval_s <= 0:
        raise ValueError("--interval doit être strictement positif.")
    return PollSettings(
        interval_s=interval_s,
        lag_warning_s=settings.lag_warning_s,
        batch_size=settings.batch_size,
    )


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
            "retard d'ordonnancement : tick démarré %.1f s trop tard", skew_s
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
