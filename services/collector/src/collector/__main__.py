# **********************************************************************
# * Nom     : __main__.py                                              *
# * Type    : Point d'entrée                                           *
# * Sujet   : Collecte datée d'une période, et rattrapage des journées *
# *   incomplètes                                                      *
# * Service : collector                                                *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy.exc import SQLAlchemyError

from collector.sink import (
    DayCoverage,
    IngestionState,
    day_coverage,
    sync_sites,
    to_measures,
    write,
    write_state,
)
from predict_common.config import ConfigError, load_config
from predict_common.db import (
    INGESTION_SOURCE_BACKFILL,
    DatabaseError,
    ingestion_etat,
    mesure,
    open_engine,
    site,
    verify_schema,
)
from predict_common.paths import PathError, date_range, parse_date
from predict_common.source import (
    MAX_PAGE_SIZE,
    SourceClient,
    SourceError,
    SourceSettings,
)
from predict_common.timestamps import DEFAULT_SOURCE_TIMEZONE

# Nombre de journées collectées quand rien n'est demandé.
DEFAULT_DAYS = 1

# Profondeur du rattrapage automatique, en journées.
DEFAULT_CATCH_UP_DAYS = 35

# Écart toléré aux bornes avant de dire une journée incomplète.
COVERAGE_TOLERANCE = timedelta(hours=1)

# Code de sortie d'une collecte aboutie.
EXIT_OK = 0
# Code de sortie d'une collecte interrompue.
EXIT_FAILED = 1

logger = logging.getLogger(__name__)


def day_window(day: date, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Méthode : day_window
    Description : Retourne les bornes UTC d'une journée, sans dépasser
      l'instant courant.
    """
    start = datetime.combine(day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    return start, min(end, now or datetime.now(UTC))


def collect_day(
    client: SourceClient,
    engine,
    batch_size: int,
    day: date,
    sites: Sequence[str],
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> int:
    """Méthode : collect_day
    Description : Collecte une journée pour chaque site et écrit son état
      d'ingestion.
    """
    start_time, end_time = day_window(day)
    records: list[dict] = []
    states: list[IngestionState] = []
    attempted_at = datetime.now(UTC)
    for site_id in sites:
        try:
            page = list(client.iter_readings(site_id, start_time, end_time))
        except SourceError as exc:
            states.append(
                IngestionState(
                    site_id=site_id,
                    attempted_at=attempted_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            record_collection(engine, states, batch_size)
            raise
        logger.info("site %s : %d mesure(s) lue(s)", site_id, len(page))
        records.extend(page)
        states.append(
            IngestionState(
                site_id=site_id,
                attempted_at=attempted_at,
                rows=len(page),
                data_lag_s=None,
            )
        )
    report = write(
        engine,
        to_measures(records, naive_timezone),
        batch_size,
        day=day,
    )
    logger.info(
        "%s : %d mesure(s) soumise(s)%s",
        day,
        report.rows,
        f", {report.dropped} écartée(s)" if report.dropped else "",
    )
    record_collection(engine, states, batch_size)
    return report.rows


def record_collection(
    engine,
    states: Sequence[IngestionState],
    batch_size: int,
) -> None:
    """Méthode : record_collection
    Description : Repose l'état d'ingestion sans jamais faire échouer la
      collecte.
    """
    try:
        write_state(engine, states, batch_size, INGESTION_SOURCE_BACKFILL)
    except (SQLAlchemyError, OSError) as exc:
        logger.error("état d'ingestion non enregistré : %s", exc)


def resolve_sites(
    client: SourceClient,
    engine,
    batch_size: int,
    requested: Sequence[str] | None,
) -> list[str]:
    """Méthode : resolve_sites
    Description : Choisit les sites à collecter, référentiel ou liste demandée.
    """
    try:
        referential = client.fetch_sites()
    except SourceError:
        if not requested:
            raise
        logger.warning(
            "référentiel indisponible : collecte des sites demandés sans"
            " synchronisation"
        )
        return list(requested)
    sync_sites(engine, referential, batch_size)
    if requested:
        return list(requested)
    return [str(entry["site_id"]) for entry in referential if "site_id" in entry]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Méthode : parse_args
    Description : Analyse la ligne de commande de la collecte.
    """
    parser = argparse.ArgumentParser(
        prog="collector",
        description="Collecte des mesures EnerVision vers TimescaleDB.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Première journée collectée, incluse, au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Dernière journée collectée, incluse. Défaut : --start.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Dernière journée collectée. Forme courte, avec --days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Nombre de journées remontées depuis --date. Défaut : 1.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à collecter. Répétable. Par défaut : tout le référentiel.",
    )
    parser.add_argument(
        "--catch-up",
        action="store_true",
        help=(
            "Rattrape ce qui manque en base, sans période à fournir : la"
            " reprise part de la dernière mesure de chaque site. Destiné au"
            " redémarrage du collecteur."
        ),
    )
    parser.add_argument(
        "--catch-up-days",
        type=int,
        default=None,
        help=(
            "Profondeur maximale du rattrapage automatique, en journées."
            f" Défaut : collector.catch_up_days, sinon {DEFAULT_CATCH_UP_DAYS}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"Mesures demandées par requête, au plus {MAX_PAGE_SIZE}."
            " Défaut : source.page_size."
        ),
    )
    return parser.parse_args(argv)


def check_period_arguments(args: argparse.Namespace) -> None:
    """Méthode : check_period_arguments
    Description : Refuse un rattrapage assorti de bornes de période.
    """
    if not args.catch_up:
        return
    given = [
        name
        for name, value in (
            ("--start", args.start),
            ("--end", args.end),
            ("--date", args.date),
        )
        if value is not None
    ]
    if given:
        raise ValueError(
            "--catch-up déduit sa période de la base :"
            f" {', '.join(given)} n'a alors aucun effet et prête à confusion."
        )


def catch_up_days(
    engine,
    sites: Sequence[str],
    depth_days: int,
    today: date | None = None,
) -> list[date]:
    """Méthode : catch_up_days
    Description : Liste les journées dont la couverture est incomplète en base.
    """
    end = today or datetime.now(UTC).date()
    first = end - timedelta(days=max(depth_days, 1) - 1)
    covered = {
        (entry.site_id, entry.day)
        for entry in day_coverage(engine, sites, first, end)
        if covers_full_day(entry, end)
    }
    return [
        day
        for day in date_range(first, end)
        if any((site_id, day) not in covered for site_id in sites)
    ]


def covers_full_day(entry: DayCoverage, today: date) -> bool:
    """Méthode : covers_full_day
    Description : Dit si une journée est couverte de bout en bout.
    """
    if entry.day >= today:
        return False
    day_start = datetime.combine(entry.day, time.min, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return (
        entry.first_at <= day_start + COVERAGE_TOLERANCE
        and entry.last_at >= day_end - COVERAGE_TOLERANCE
    )


def requested_days(args: argparse.Namespace) -> list[date]:
    """Méthode : requested_days
    Description : Traduit les arguments de période en liste de journées.
    """
    borne = args.start is not None or args.end is not None
    if borne and args.date is not None:
        raise ValueError(
            "--date et --start/--end désignent tous deux la période :"
            " en choisir une seule."
        )
    if borne:
        if args.start is None:
            raise ValueError("--end demande --start.")
        first = parse_date(args.start)
        last = parse_date(args.end) if args.end else first
        return date_range(first, last)
    if args.date is None:
        raise ValueError(
            "Période absente : donner --start/--end, ou --date avec --days."
        )
    if args.days < 1:
        raise ValueError("--days doit valoir au moins 1.")
    last = parse_date(args.date)
    return date_range(last - timedelta(days=args.days - 1), last)


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
    Description : Point d'entrée : collecte les journées demandées et rend un
      code de sortie.
    """
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        check_period_arguments(args)
        days = [] if args.catch_up else requested_days(args)
        settings = SourceSettings.from_config(config)
        if args.limit is not None:
            settings = settings.with_page_size(args.limit)
        engine = open_engine(config.get_optional_str("database.url"))
        verify_schema(engine, (mesure, site, ingestion_etat))
        batch_size = config.get_int("database.batch_size")
    except (ConfigError, DatabaseError, PathError, ValueError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_FAILED

    total = 0
    try:
        with SourceClient(settings) as client:
            sites = resolve_sites(client, engine, batch_size, args.sites)
            if args.catch_up:
                days = catch_up_days(
                    engine,
                    sites,
                    args.catch_up_days
                    or config.get_int(
                        "collector.catch_up_days", DEFAULT_CATCH_UP_DAYS
                    ),
                )
            logger.info(
                "collecte de %d site(s) du %s au %s, %d mesure(s) par requête",
                len(sites),
                days[0],
                days[-1],
                settings.page_size,
            )
            for day in days:
                total += collect_day(
                    client, engine, batch_size, day, sites, settings.timezone
                )
    except (SourceError, SQLAlchemyError) as exc:
        logger.error("collecte interrompue : %s", exc)
        return EXIT_FAILED
    finally:
        engine.dispose()
    logger.info("collecte terminée : %d mesure(s) soumise(s)", total)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
