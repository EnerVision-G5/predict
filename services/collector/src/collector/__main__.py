"""Point d'entrée du collecteur : rattrapage d'une ou plusieurs journées.

    python -m collector --date 2026-09-02
    python -m collector --date 2026-09-02 --days 7 --site SITE001

Le collecteur remplit la couche brute, qui est la table `mesure` de
TimescaleDB. Il ne connaît ni l'ETL, ni l'entraînement, ni le service : sa
sortie est une table et un schéma, et c'est tout ce que son consommateur a
besoin de savoir.

Une journée est traitée en entier avant la suivante. Le lot d'un jour tient en
mémoire — sept sites à la minute font une dizaine de milliers de lignes — là
où trois mois n'y tiendraient pas, et une journée soumise en une transaction
est soit chargée, soit absente, jamais à moitié écrite.

Relancer la même date ne double rien et n'efface rien : l'insertion est un
`ON CONFLICT DO NOTHING`. C'est ce qui rend le rejeu après incident sans effet
de bord — et le rejeu est le mode d'exploitation normal, pas l'exception : une
source indisponible pendant deux heures se rattrape en relançant la journée.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy.exc import SQLAlchemyError

from collector.client import SourceClient, SourceError, SourceSettings
from collector.sink import sync_sites, to_measures, write
from predict_common.config import ConfigError, load_config
from predict_common.db import DatabaseError, open_engine
from predict_common.paths import PathError, date_range, parse_date

DEFAULT_DAYS = 1

EXIT_OK = 0
EXIT_FAILED = 1

logger = logging.getLogger(__name__)


def day_window(day: date) -> tuple[datetime, datetime]:
    """Retourne la fenêtre UTC `[minuit, minuit du lendemain[` d'une journée.

    Les mesures rendues par la source sont ensuite filtrées sur le jour
    demandé : une source qui déborderait d'une seconde ne serait pas comptée
    dans une journée qu'elle ne concerne pas.
    """
    start = datetime.combine(day, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def collect_day(
    client: SourceClient,
    engine,
    batch_size: int,
    day: date,
    sites: Sequence[str],
) -> int:
    """Collecte une journée pour les sites demandés et la charge en base."""
    start_time, end_time = day_window(day)
    records: list[dict] = []
    for site_id in sites:
        page = list(client.iter_readings(site_id, start_time, end_time))
        logger.info("site %s : %d mesure(s) lue(s)", site_id, len(page))
        records.extend(page)
    report = write(engine, to_measures(records), batch_size, day=day)
    logger.info(
        "%s : %d mesure(s) soumise(s)%s",
        day,
        report.rows,
        f", {report.dropped} écartée(s)" if report.dropped else "",
    )
    return report.rows


def resolve_sites(
    client: SourceClient,
    engine,
    batch_size: int,
    requested: Sequence[str] | None,
) -> list[str]:
    """Synchronise le référentiel et retourne les sites à collecter.

    La synchronisation entretient `site`, que `mesure.site_id` référence : un
    site absent de la table ferait rejeter ses mesures sans que rien
    n'explique pourquoi. C'est aussi ce que le seed `02_seed_sites.sql`
    attend, ses capacités des sites 4 à 7 étant des placeholders.

    Elle n'est exigée que lorsqu'on en dépend pour savoir quoi collecter.
    Avec `--site`, l'exploitant a nommé ses sites : une source qui ne sert pas
    son référentiel ne doit pas l'empêcher de rattraper une journée, puisque
    le seed a déjà posé les sites courants. L'échec est journalisé, et c'est
    la clé étrangère qui tranchera s'il manquait vraiment quelque chose.
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
    """Analyse la ligne de commande du collecteur."""
    parser = argparse.ArgumentParser(
        prog="collector",
        description="Collecte des mesures EnerVision vers TimescaleDB.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Dernier jour collecté, au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Nombre de journées collectées en remontant depuis --date.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à collecter. Répétable. Par défaut : tout le référentiel.",
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
    """Point d'entrée du conteneur de collecte."""
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        last_day = parse_date(args.date)
        days = _requested_days(last_day, args.days)
        settings = SourceSettings.from_config(config)
        engine = open_engine(config.get_optional_str("database.url"))
        batch_size = config.get_int("database.batch_size")
    except (ConfigError, DatabaseError, PathError, ValueError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_FAILED

    total = 0
    try:
        with SourceClient(settings) as client:
            sites = resolve_sites(client, engine, batch_size, args.sites)
            logger.info(
                "collecte de %d site(s) sur %d journée(s)", len(sites), len(days)
            )
            for day in days:
                total += collect_day(client, engine, batch_size, day, sites)
    except (SourceError, SQLAlchemyError) as exc:
        logger.error("collecte interrompue : %s", exc)
        return EXIT_FAILED
    finally:
        engine.dispose()
    logger.info("collecte terminée : %d mesure(s) soumise(s)", total)
    return EXIT_OK


def _requested_days(last_day: date, days: int) -> list[date]:
    """Retourne les journées à collecter, de la plus ancienne à --date."""
    if days < 1:
        raise ValueError("--days doit valoir au moins 1.")
    return date_range(last_day - timedelta(days=days - 1), last_day)


if __name__ == "__main__":
    raise SystemExit(main())
