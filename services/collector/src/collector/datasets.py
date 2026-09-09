# **********************************************************************
# * Nom     : datasets.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Chargement de l'historique de référence, des CSV vers la *
# *   couche brute                                                     *
# * Service : collector                                                *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from typing import Any

import pandas as pd
import pyarrow.fs
from sqlalchemy.engine import Engine

from collector import sink
from predict_common import io
from predict_common.config import Config, ConfigError, load_config
from predict_common.db import (
    DatabaseError,
    mesure,
    open_engine,
    verify_schema,
)
from predict_common.paths import join
from predict_common.schemas import (
    QUALITY_CRITICAL,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_PARTIAL,
)

logger = logging.getLogger(__name__)

# Colonnes qu'un CSV de référence doit porter.
REQUIRED_COLUMNS = (
    "timestamp",
    "site_id",
    "consumption_kwh",
    "temperature_celsius",
    "humidity_percent",
)

# Motif des fichiers retenus dans la racine des jeux de données.
FILE_PATTERN = "SITE*.csv"

# Cause posée quand la consommation manque seule.
REASON_CONSUMPTION = "consumption_sensor_failure"
# Cause posée quand la température manque.
REASON_TEMPERATURE = "temperature_sensor_failure"
# Cause posée quand l'humidité manque.
REASON_HUMIDITY = "humidity_sensor_failure"
# Cause posée quand la ligne entière est vide.
REASON_NETWORK = "network_loss"

# Taille de lot d'écriture si la configuration n'en donne pas.
DEFAULT_BATCH_SIZE = 1000


class DatasetError(RuntimeError):
    """Classe : DatasetError
    Description : Les jeux de données sont introuvables, illisibles ou mal
      formés.
    """


def find_files(root: str) -> list[str]:
    """Méthode : find_files
    Description : Liste les CSV de référence d'une racine, disque ou stockage
      objet.
    """
    try:
        filesystem, path = io.resolve(str(root))
        entries = filesystem.get_file_info(
            pyarrow.fs.FileSelector(path, recursive=False, allow_not_found=True),
        )
    except (OSError, io.StorageError) as exc:
        raise DatasetError(f"Stockage des jeux de données injoignable : {exc}") from exc
    files = sorted(
        join(str(root), entry.base_name)
        for entry in entries
        if entry.type == pyarrow.fs.FileType.File
        and fnmatch(entry.base_name, FILE_PATTERN)
    )
    if not files:
        raise DatasetError(f"Aucun fichier {FILE_PATTERN} dans {root}.")
    return files


def read_file(uri: str) -> pd.DataFrame:
    """Méthode : read_file
    Description : Lit un CSV de référence et refuse un fichier aux colonnes
      manquantes.
    """
    name = base_name(uri)
    try:
        filesystem, path = io.resolve(str(uri))
        with filesystem.open_input_stream(path) as stream:
            frame = pd.read_csv(stream)
    except (OSError, io.StorageError) as exc:
        raise DatasetError(f"Fichier {name} illisible : {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"Colonnes absentes de {name} : {missing}.")
    return frame


def base_name(uri: str) -> str:
    """Méthode : base_name
    Description : Extrait le nom de fichier d'une URI, séparateurs
      indifférents.
    """
    return str(uri).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def to_readings(frame: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Méthode : to_readings
    Description : Transforme les lignes d'un CSV en mesures, causes et qualité
      comprises.
    """
    for row in frame.to_dict(orient="records"):
        consumption = _clean(row.get("consumption_kwh"))
        temperature = _clean(row.get("temperature_celsius"))
        humidity = _clean(row.get("humidity_percent"))
        reasons = _reasons(consumption, temperature, humidity)
        yield {
            "timestamp": _iso(row["timestamp"]),
            "site_id": row["site_id"],
            "consumption_kw": consumption,
            "consumption_kwh": consumption,
            "voltage_v": None,
            "current_a": None,
            "power_factor": None,
            "temperature_celsius": temperature,
            "humidity_percent": humidity,
            "null_reasons": reasons,
            "data_quality": _quality(consumption, reasons),
        }


def load_file(engine: Engine, uri: str, batch_size: int) -> int:
    """Méthode : load_file
    Description : Charge un fichier dans la couche brute et rend le nombre de
      mesures soumises.
    """
    frame = read_file(uri)
    measures = sink.to_measures(to_readings(frame))
    report = sink.write(engine, measures, batch_size)
    logger.info(
        "%s : %d ligne(s) lue(s), %d mesure(s) soumise(s)",
        base_name(uri),
        len(frame),
        report.rows,
    )
    return report.rows


def load(engine: Engine, root: str, batch_size: int) -> int:
    """Méthode : load
    Description : Charge tous les fichiers de la racine, un par un.
    """
    total = 0
    for uri in find_files(root):
        total += load_file(engine, uri, batch_size)
    return total


def _reasons(
    consumption: float | None,
    temperature: float | None,
    humidity: float | None,
) -> list[str]:
    """Méthode : _reasons
    Description : Déduit les causes d'absence des colonnes vides d'une ligne.
    """
    if consumption is None and temperature is None and humidity is None:
        return [REASON_NETWORK]
    reasons = []
    if consumption is None:
        reasons.append(REASON_CONSUMPTION)
    if temperature is None:
        reasons.append(REASON_TEMPERATURE)
    if humidity is None:
        reasons.append(REASON_HUMIDITY)
    return reasons


def _quality(consumption: float | None, reasons: Sequence[str]) -> str:
    """Méthode : _quality
    Description : Déduit la qualification d'une ligne de ses causes d'absence.
    """
    if REASON_NETWORK in reasons:
        return QUALITY_CRITICAL
    if consumption is None:
        return QUALITY_DEGRADED
    if reasons:
        return QUALITY_PARTIAL
    return QUALITY_GOOD


def _clean(value: Any) -> float | None:
    """Méthode : _clean
    Description : Ramène une valeur absente de pandas à None.
    """
    if value is None or pd.isna(value):
        return None
    return float(value)


def _iso(value: Any) -> str:
    """Méthode : _iso
    Description : Écrit un horodatage en ISO 8601 UTC, fuseau prêté si absent.
    """
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def _batch_size(config: Config) -> int:
    """Méthode : _batch_size
    Description : Lit la taille de lot d'écriture dans la configuration.
    """
    return config.get_int("database.batch_size", DEFAULT_BATCH_SIZE)


def _datasets_root(config: Config) -> str:
    """Méthode : _datasets_root
    Description : Lit la racine des jeux de données dans la configuration.
    """
    return config.get_str("storage.datasets_root")


def build_parser() -> argparse.ArgumentParser:
    """Méthode : build_parser
    Description : Analyse la ligne de commande du chargement de l'historique.
    """
    parser = argparse.ArgumentParser(
        prog="collector.datasets",
        description=(
            "Charge l'historique de référence des sites dans la couche brute."
        ),
    )
    parser.add_argument(
        "--root",
        dest="root",
        default=None,
        help=(
            "Racine des fichiers par site, chemin ou URI s3://."
            " Défaut : storage.datasets_root."
        ),
    )
    parser.add_argument(
        "--limit",
        dest="batch_size",
        type=int,
        default=None,
        help="Taille des lots d'insertion. Défaut : database.batch_size.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Méthode : main
    Description : Point d'entrée : charge l'historique de référence et rend un
      code de sortie.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        config = load_config()
        root = args.root or _datasets_root(config)
        engine = open_engine(config.get_optional_str("database.url"))
        verify_schema(engine, (mesure,))
        batch_size = args.batch_size or _batch_size(config)
        logger.info("jeux de données lus depuis %s", root)
        written = load(engine, root, batch_size)
    except (ConfigError, DatabaseError, DatasetError, ValueError) as exc:
        logger.error("import impossible : %s", exc)
        return 1
    logger.info("import terminé : %d mesure(s) soumise(s)", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
