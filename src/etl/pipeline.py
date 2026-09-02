"""Orchestration d'un run d'ingestion, et point d'entrée du conteneur ETL.

Le module enchaîne extraction, transformation et chargement pour une fenêtre
temporelle et une liste de sites. Il ne contient aucune règle métier : chaque
étage reste testable seul.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import requests
from sqlalchemy import create_engine

from etl.config import EtlConfig, load_config
from etl.exclude import load_exclusions
from etl.extract import build_session, fetch_readings, fetch_sites
from etl.impute import impute_frame
from etl.load import load_frame
from etl.transform import deduplicate, to_frame

DEFAULT_WINDOW_HOURS = 24

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunReport:
    """Résultat d'un run, par site puis consolidé.

    Les mesures écartées sont comptées à part et non retranchées : elles ont
    bien été chargées, c'est leur usage dans les agrégats qui est refusé.
    """

    rows_per_site: dict[str, int]
    excluded_per_site: dict[str, int] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_site.values())

    @property
    def total_excluded(self) -> int:
        return sum(self.excluded_per_site.values())


def window_from_hours(
    hours: int,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Retourne la fenêtre [début, fin] couvrant les `hours` dernières heures.

    `now` est injectable pour que les tests n'aient pas à composer avec
    l'horloge réelle.
    """
    if hours <= 0:
        raise ValueError("La fenêtre d'ingestion doit couvrir au moins 1 heure.")
    end_time = now or datetime.now(UTC)
    return end_time - timedelta(hours=hours), end_time


def resolve_sites(
    config: EtlConfig,
    requested: Sequence[str] | None,
    session: requests.Session | None = None,
) -> list[str]:
    """Retourne les sites à ingérer, ceux demandés ou tout le référentiel."""
    if requested:
        return list(requested)
    return [site["site_id"] for site in fetch_sites(config, session=session)]


def run(
    config: EtlConfig,
    start_time: datetime,
    end_time: datetime,
    sites: Sequence[str] | None = None,
) -> RunReport:
    """Ingère la fenêtre demandée pour chaque site et retourne le bilan."""
    session = build_session()
    engine = create_engine(config.database_url)
    try:
        targets = resolve_sites(config, sites, session=session)
        rows_per_site: dict[str, int] = {}
        excluded_per_site: dict[str, int] = {}
        for site_id in targets:
            readings = fetch_readings(
                config, site_id, start_time, end_time, session=session
            )
            frame = impute_frame(deduplicate(to_frame(readings)))
            rows_per_site[site_id] = load_frame(engine, frame, config.batch_size)
            excluded_per_site[site_id] = load_exclusions(
                engine, frame, config.batch_size
            )
            logger.info(
                "site %s : %d mesures, %d écartée(s)",
                site_id,
                rows_per_site[site_id],
                excluded_per_site[site_id],
            )
        return RunReport(
            rows_per_site=rows_per_site,
            excluded_per_site=excluded_per_site,
        )
    finally:
        session.close()
        engine.dispose()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande du conteneur ETL."""
    parser = argparse.ArgumentParser(description="Ingestion des mesures EnerVision.")
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_WINDOW_HOURS,
        help="Profondeur de la fenêtre ingérée, en heures.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à ingérer. Répétable. Par défaut : tout le référentiel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée du conteneur ETL."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    config = load_config()
    start_time, end_time = window_from_hours(args.hours)
    report = run(config, start_time, end_time, sites=args.sites)
    logger.info(
        "run terminé : %d mesures soumises, %d écartée(s)",
        report.total_rows,
        report.total_excluded,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
