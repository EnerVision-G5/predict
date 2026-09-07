"""Point d'entrée de l'ETL : une journée de mesures, une partition de variables.

    python -m etl
    python -m etl --date 2026-09-02
    python -m etl --date 2026-09-02 --feature-version v1
    python -m etl --days 2

Sans `--date`, la journée produite est celle du jour : c'est ce dont une
boucle périodique a besoin, et lui faire calculer une date dans son shell
mettrait la règle ailleurs que dans le service qui l'applique.

`--days` produit plusieurs journées en remontant depuis `--date`, de la plus
ancienne à la plus récente. Une journée en cours n'est complète qu'au
lendemain : la rejouer une fois de plus est ce qui la termine, et comme
l'écriture remplace la partition au lieu de l'allonger, la rejouer ne coûte
que le calcul.

Le service lit `mesure` dans TimescaleDB, y repose ce qu'il en a déduit, et
publie `features/{version}/dt=.../` sur le stockage objet. Il ne connaît ni le
collecteur qui a rempli la table ni l'entraînement qui lira les partitions : il
ouvre une connexion et un chemin.

La fenêtre lue déborde sur les jours précédents, et c'est nécessaire. Le
décalage de 168 heures d'une heure du 2 septembre désigne une heure du
26 août : produire la journée à partir d'elle seule donnerait des décalages
vides, et elle sortirait presque entièrement écartée sans que rien ne
l'explique. La profondeur du débord est déduite du plus long décalage demandé,
jamais fixée à la main.

Seule la journée demandée est réécrite en base, jamais toute la fenêtre lue :
celle-ci ne sert qu'aux décalages, et reposer huit journées pour en produire
une ferait payer huit fois le même travail sans rien changer au résultat.

Relancer la même date reproduit le même résultat des deux côtés : l'écriture en
base repose les colonnes déduites au lieu de les ajouter, et la partition de
variables est remplacée au lieu de grossir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

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

# Journées produites quand la ligne de commande n'en demande pas
# davantage. Une seule : `--date 2026-09-02` reste ce qu'il était, et
# demander une fenêtre est un geste explicite.
DEFAULT_DAYS = 1

EXIT_OK = 0
EXIT_FAILED = 1

logger = logging.getLogger(__name__)


def feature_spec(config: Config, version: str | None = None) -> FeatureSpec:
    """Construit la définition des variables demandée par la configuration."""
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
    """Enchaîne les étages et retourne les mesures enrichies et les variables.

    Les deux sont retournés parce qu'ils ne vont pas au même endroit : les
    mesures repartent dans `mesure`, les variables en partition.
    """
    measures = impute_frame(deduplicate(to_measures(check_measures(raw))))
    features = build(keep_usable(measures), spec, day)
    return measures, check_features(features, spec)


def publish(features: pd.DataFrame, root: str, spec: FeatureSpec, day: date) -> str:
    """Écrit la partition de variables, en remplaçant celle qui existait."""
    partition = features_partition(root, spec.version, day)
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
    """Produit la journée demandée et retourne le nombre d'heures publiées."""
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
    logger.info(
        "%s : %d heure(s) publiée(s) pour %d site(s)",
        partition,
        len(features),
        features["site_id"].nunique() if not features.empty else 0,
    )
    return len(features)


def of_day(measures: pd.DataFrame, day: date) -> pd.DataFrame:
    """Ne garde que les mesures du jour produit.

    Le reste de la fenêtre n'a servi qu'à donner un passé aux décalages : le
    reposer en base à chaque run ferait réécrire huit journées pour en produire
    une, sans rien changer à ce qu'elles contiennent.
    """
    if measures.empty:
        return measures
    return measures[measures[TIMESTAMP_COLUMN].dt.date == day].reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande de l'ETL."""
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
    """Point d'entrée du conteneur ETL."""
    _configure_logging()
    args = parse_args(argv)
    engine: Engine | None = None
    try:
        config = load_config()
        end = parse_date(args.date) if args.date else datetime.now(UTC).date()
        engine = open_engine(config.get_optional_str("database.url"))
        # Avant tout calcul : une base en retard de migration ferait
        # échouer le chargement APRÈS avoir produit la journée entière,
        # sous la forme brute que remonte le driver.
        verify_schema(engine, (mesure, mesure_exclu))
        # Le rejeu d'une journée déjà produite est sans effet de bord : la
        # partition est remplacée et l'écriture en base repose les colonnes
        # déduites. Une fenêtre n'a donc pas à savoir où la précédente s'est
        # arrêtée.
        for day in lookback_range(end, args.days):
            run(config, engine, day, args.feature_version)
    except (ConfigError, DatabaseError, PathError, CleanError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_FAILED
    except (ContractError, LoadError) as exc:
        logger.error("run interrompu : %s", exc)
        return EXIT_FAILED
    except SQLAlchemyError as exc:
        logger.error("base inaccessible ou refusant l'écriture : %s", exc)
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
