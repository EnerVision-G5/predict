# **********************************************************************
# * Nom     : __main__.py                                              *
# * Type    : Point d'entrée                                           *
# * Sujet   : Production d'une ou plusieurs journées de variables, de  *
# *   la base au stockage objet                                        *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from etl.clean import CleanError, deduplicate, to_measures
from etl.exclude import keep_usable
from etl.extract import read_measures, window
from etl.features import FeatureSpec, build
from etl.impute import impute_frame
from etl.load import LoadError, load
from etl.validate import ContractError, check_features, check_measures
from predict_common import io
from predict_common.config import Config, ConfigError, load_config
from predict_common.db import (
    DatabaseError,
    mesure,
    mesure_exclu,
    open_engine,
    verify_schema,
)
from predict_common.paths import (
    PathError,
    features_partition,
    lookback_range,
    parse_date,
)
from predict_common.schemas import TIMESTAMP_COLUMN, features_arrow_schema

# Nombre de journées produites quand rien n'est demandé.
DEFAULT_DAYS = 1

# Code de sortie d'un run abouti.
EXIT_OK = 0
# Code de sortie d'un run interrompu.
EXIT_FAILED = 1

# Messages par lesquels PostgreSQL dit qu'il démarre encore.
DB_WARMUP_MARKERS = (
    "the database system is starting up",
    "the database system is shutting down",
    "the database system is in recovery mode",
    "the database system is not yet accepting connections",
)

logger = logging.getLogger(__name__)


def feature_spec(config: Config, version: str | None = None) -> FeatureSpec:
    """Méthode : feature_spec
    Description : Construit la définition des variables demandée par la
      configuration.
    """
    return FeatureSpec(
        version=version or config.get_str("etl.feature_version"),
        resample_rule=config.get_str("etl.resample_rule"),
        lag_hours=tuple(config.get_int_list("etl.lag_hours")),
        rolling_window_h=config.get_int("etl.rolling_window_h"),
    )


def transform(
    raw: pd.DataFrame,
    spec: FeatureSpec,
    day: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Méthode : transform
    Description : Enchaîne les étages et retourne les mesures enrichies et les
      variables.
    """
    measures = impute_frame(deduplicate(to_measures(check_measures(raw))))
    features = build(keep_usable(measures), spec, day)
    return measures, check_features(features, spec)


def publish(
    features: pd.DataFrame,
    root: str,
    spec: FeatureSpec,
    day: date,
) -> str | None:
    """Méthode : publish
    Description : Écrit la partition de variables, sauf si cela revenait à en
      effacer une.
    """
    partition = features_partition(root, spec.version, day)
    if features.empty and io.exists(partition):
        logger.warning(
            "%s laissée en place : ce run n'a produit aucune heure, et la"
            " remplacer par une partition vide effacerait des variables que"
            " rien ne reproduit à cette date.",
            partition,
        )
        return None
    io.write_frame(
        features,
        partition,
        schema=features_arrow_schema(spec.lag_hours, spec.rolling_window_h),
        metadata={
            "feature_version": spec.version,
            "resample_rule": spec.resample_rule,
            "lag_hours": ",".join(str(hours) for hours in spec.lag_hours),
            "rolling_window_h": str(spec.rolling_window_h),
        },
    )
    return partition


def run(config: Config, engine: Engine, day: date, version: str | None) -> int:
    """Méthode : run
    Description : Produit la journée demandée et retourne le nombre d'heures
      publiées.
    """
    spec = feature_spec(config, version)
    root = config.get_str("storage.root")
    batch_size = config.get_int("database.batch_size")

    start_time, end_time = window(day, spec.lookback_days)
    raw = read_measures(engine, start_time, end_time)
    measures, features = transform(raw, spec, day)

    rows, excluded = load(engine, of_day(measures, day), batch_size)
    logger.info(
        "mesure %s : %d ligne(s) enrichie(s), %d exclusion(s)", day, rows, excluded
    )

    partition = publish(features, root, spec, day)
    if partition is not None:
        logger.info(
            "%s : %d heure(s) publiée(s) pour %d site(s)",
            partition,
            len(features),
            features["site_id"].nunique() if not features.empty else 0,
        )
    return len(features)


def of_day(measures: pd.DataFrame, day: date) -> pd.DataFrame:
    """Méthode : of_day
    Description : Ne garde que les mesures du jour produit, le reste n'ayant
      servi qu'aux décalages.
    """
    if measures.empty:
        return measures
    return measures[measures[TIMESTAMP_COLUMN].dt.date == day].reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Méthode : parse_args
    Description : Analyse la ligne de commande de l'ETL.
    """
    parser = argparse.ArgumentParser(
        prog="etl",
        description="Transformation des mesures TimescaleDB en variables.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Dernière journée produite, au format YYYY-MM-DD."
            " Défaut : aujourd'hui."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=(
            "Nombre de journées produites, en remontant depuis --date."
            f" Défaut : {DEFAULT_DAYS}."
        ),
    )
    parser.add_argument(
        "--feature-version",
        default=None,
        help="Version des variables. Défaut : etl.feature_version.",
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


def is_db_warming_up(error: BaseException) -> bool:
    """Méthode : is_db_warming_up
    Description : Dit si l'erreur signale une base qui démarre plutôt qu'une
      panne.
    """
    message = str(error).lower()
    return any(marker in message for marker in DB_WARMUP_MARKERS)


def db_error_line(error: BaseException) -> str:
    """Méthode : db_error_line
    Description : Réduit une erreur SQLAlchemy à sa première ligne utile.
    """
    lines = str(getattr(error, "orig", None) or error).strip().splitlines()
    return lines[0].strip() if lines else type(error).__name__


def main(argv: Sequence[str] | None = None) -> int:
    """Méthode : main
    Description : Point d'entrée : lit la configuration, produit les journées
      demandées, rend un code de sortie.
    """
    _configure_logging()
    args = parse_args(argv)
    engine: Engine | None = None
    try:
        config = load_config()
        end = parse_date(args.date) if args.date else datetime.now(UTC).date()
        engine = open_engine(config.get_optional_str("database.url"))
        verify_schema(engine, (mesure, mesure_exclu))
        for day in lookback_range(end, args.days):
            run(config, engine, day, args.feature_version)
    except (ConfigError, DatabaseError, PathError, CleanError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_FAILED
    except (ContractError, LoadError) as exc:
        logger.error("run interrompu : %s", exc)
        return EXIT_FAILED
    except OperationalError as exc:
        if is_db_warming_up(exc):
            logger.info("base en attente : elle démarre encore.")
        else:
            logger.error("base injoignable : %s", db_error_line(exc))
        return EXIT_FAILED
    except SQLAlchemyError as exc:
        logger.error("base refusant l'écriture : %s", db_error_line(exc))
        return EXIT_FAILED
    except io.StorageError as exc:
        logger.error("stockage des variables inaccessible : %s", exc)
        return EXIT_FAILED
    finally:
        if engine is not None:
            engine.dispose()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
