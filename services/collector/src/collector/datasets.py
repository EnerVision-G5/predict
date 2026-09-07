"""Chargement de l'historique de référence dans la couche brute.

Pourquoi ce module existe. `GET /api/v1/readings` ne sert pas d'historique :
la source ne remonte qu'à 48 heures, et au-delà elle répond des mesures NULLES
plutôt qu'une erreur. Un modèle a besoin de saisons, pas de deux journées. Les
jeux de données de référence — deux années horaires par site, fournis avec la
source — sont donc la SEULE origine possible de l'historique d'apprentissage,
et ce module est le chemin par lequel il entre en base.

Ce n'est pas un second collecteur. Les lignes du CSV sont converties en
lectures au format de la source, puis remises à `collector.sink`, qui les
normalise, les cale sur la grille et les écrit. Tout ce qui décide de la forme
d'une mesure reste donc écrit à un seul endroit : un chemin d'import parallèle
finirait par qualifier autrement ce que la chaîne lit ensuite pareil.

L'écriture n'écrase rien. `sink.write` insère en `ON CONFLICT DO NOTHING` : sur
un serveur qui collecte déjà, l'import comble ce qui manque et ne touche à
aucune mesure déjà présente. C'est ce qui rend la commande rejouable, et ce qui
permet de la lancer en production sans reconstruire la base.

Le pas est HORAIRE, et il le reste. Une ligne par heure, à HH:00, et non
soixante copies de la même valeur : `mesure` est une grille à la minute, mais
rien n'oblige à la remplir. L'ETL agrège à l'heure et calcule son taux
d'imputation sur les lignes PRÉSENTES (voir `etl.features._imputed_ratio`) :
une heure portant une seule ligne non imputée vaut donc `imputed_ratio = 0`.
Fabriquer les cinquante-neuf minutes absentes inventerait des mesures qui
n'ont pas eu lieu, sans rien apporter en aval.

Ce que le jeu de données ne porte pas reste NULL : ni tension, ni intensité, ni
facteur de puissance. Les déduire d'une consommation horaire serait inventer
trois grandeurs électriques à partir d'une seule.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from collector import sink
from predict_common.config import Config, ConfigError, load_config
from predict_common.db import (
    DatabaseError,
    mesure,
    open_engine,
    verify_schema,
)
from predict_common.schemas import (
    QUALITY_CRITICAL,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_PARTIAL,
)

logger = logging.getLogger(__name__)

# Colonnes exigées du CSV. Les autres qu'il porte — consumption_euros,
# solar_irradiance_wm2, hour, day_of_week, is_weekend… — sont ignorées : les
# unes n'ont pas de colonne dans `mesure`, les autres sont des variables que
# l'ETL recalcule, et les figer ici en donnerait deux versions.
REQUIRED_COLUMNS = (
    "timestamp",
    "site_id",
    "consumption_kwh",
    "temperature_celsius",
    "humidity_percent",
)

# Motif des fichiers par site. `all_sites_combined.csv` porte exactement les
# mêmes lignes et n'est volontairement pas lu : le charger en plus doublerait
# le travail pour un résultat identique, le `ON CONFLICT DO NOTHING` absorbant
# la seconde passe.
FILE_PATTERN = "SITE*.csv"

# Vocabulaire des causes, aligné sur celui que la source emploie dans
# `null_reasons`. Le jeu de données n'en fournit pas : elles sont déduites de
# ce qui manque, et doivent se lire comme celles du fil de l'eau, sans quoi une
# analyse de fiabilité aurait deux vocabulaires à connaître.
REASON_CONSUMPTION = "consumption_sensor_failure"
REASON_TEMPERATURE = "temperature_sensor_failure"
REASON_HUMIDITY = "humidity_sensor_failure"
REASON_NETWORK = "network_loss"

# Lots d'insertion. Repris de la configuration quand elle le dit, sinon cette
# valeur : sept fichiers de 17 521 lignes font 122 647 mesures, qu'il ne faut
# pas soumettre d'un bloc.
DEFAULT_BATCH_SIZE = 1000


class DatasetError(RuntimeError):
    """Le jeu de données est absent, illisible, ou incomplet."""


def find_files(directory: Path) -> list[Path]:
    """Retourne les fichiers par site, triés, et refuse un dossier muet.

    Un dossier vide est une erreur et non un import de zéro ligne : la
    commande aurait l'air d'avoir réussi, et le défaut ne se verrait qu'à
    l'entraînement, faute de variables à apprendre.
    """
    if not directory.is_dir():
        raise DatasetError(f"Répertoire de jeux de données absent : {directory}.")
    files = sorted(directory.glob(FILE_PATTERN))
    if not files:
        raise DatasetError(
            f"Aucun fichier {FILE_PATTERN} dans {directory}."
        )
    return files


def read_file(path: Path) -> pd.DataFrame:
    """Lit un fichier par site et vérifie qu'il porte ce qu'on attend."""
    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DatasetError(f"Colonnes absentes de {path.name} : {missing}.")
    return frame


def to_readings(frame: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Convertit les lignes du jeu de données en lectures de la source.

    `consumption_kw` reprend `consumption_kwh` sans conversion, et c'est exact
    au pas horaire : l'énergie consommée pendant une heure, en kWh, est
    numériquement la puissance moyenne de cette heure, en kW. C'est aussi la
    convention de la source, dont les deux champs portent la même valeur.
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
            # Absentes du jeu de données. Laissées NULL plutôt que déduites.
            "voltage_v": None,
            "current_a": None,
            "power_factor": None,
            "temperature_celsius": temperature,
            "humidity_percent": humidity,
            "null_reasons": reasons,
            "data_quality": _quality(consumption, reasons),
        }


def load_file(engine: Engine, path: Path, batch_size: int) -> int:
    """Charge un fichier et retourne le nombre de mesures soumises.

    Soumises et non écrites : le `ON CONFLICT DO NOTHING` ne remonte pas ce
    qu'il a ignoré. Un second import annonce donc les mêmes nombres que le
    premier sans avoir rien inséré — c'est le comportement attendu, et le
    seul honnête, puisque la base ne dit pas ce qu'elle a écarté.
    """
    frame = read_file(path)
    measures = sink.to_measures(to_readings(frame))
    report = sink.write(engine, measures, batch_size)
    logger.info(
        "%s : %d ligne(s) lue(s), %d mesure(s) soumise(s)",
        path.name,
        len(frame),
        report.rows,
    )
    return report.rows


def load(engine: Engine, directory: Path, batch_size: int) -> int:
    """Charge tous les fichiers par site du répertoire, dans l'ordre."""
    total = 0
    for path in find_files(directory):
        total += load_file(engine, path, batch_size)
    return total


def _reasons(
    consumption: float | None,
    temperature: float | None,
    humidity: float | None,
) -> list[str]:
    """Déduit les causes de ce qui manque sur la ligne.

    Tout absent vaut `network_loss` et non trois pannes simultanées : c'est
    ainsi que la source qualifie une coupure, et trois capteurs qui tombent à
    la même seconde décrivent le réseau, pas les capteurs.
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
    """Qualifie la ligne, avec le vocabulaire de la source.

    La consommation décide de la sévérité, parce que c'est elle que le modèle
    apprend : une température manquante gêne, une consommation manquante rend
    l'heure inapprenable.
    """
    if REASON_NETWORK in reasons:
        return QUALITY_CRITICAL
    if consumption is None:
        return QUALITY_DEGRADED
    if reasons:
        return QUALITY_PARTIAL
    return QUALITY_GOOD


def _clean(value: Any) -> float | None:
    """Ramène un manquant pandas à None, et le reste à un flottant."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _iso(value: Any) -> str:
    """Rend l'horodatage en ISO 8601 UTC, tel que `to_measures` l'attend.

    Les horodatages du jeu de données sont naïfs. Ils sont lus en UTC, comme
    ceux que la source sert : les interpréter dans le fuseau du serveur
    décalerait tout l'historique de une à deux heures selon la saison, et
    ferait apprendre au modèle des journées de travail commençant à 7 h.
    """
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat()


def _batch_size(config: Config) -> int:
    return config.get_int("database.batch_size", DEFAULT_BATCH_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collector.datasets",
        description=(
            "Charge l'historique de référence des sites dans la couche brute."
        ),
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        default="datasets",
        help="Répertoire des fichiers par site. Défaut : datasets.",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        config = load_config()
        engine = open_engine(config.get_optional_str("database.url"))
        # Avant le premier fichier : cent vingt mille lignes lues et
        # transformées pour échouer au chargement seraient du travail perdu,
        # et l'erreur brute du driver ne dirait pas que le schéma est en
        # retard sur les migrations de l'API.
        verify_schema(engine, (mesure,))
        batch_size = args.batch_size or _batch_size(config)
        written = load(engine, Path(args.directory), batch_size)
    except (ConfigError, DatabaseError, DatasetError, ValueError) as exc:
        logger.error("import impossible : %s", exc)
        return 1
    logger.info("import terminé : %d mesure(s) soumise(s)", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
