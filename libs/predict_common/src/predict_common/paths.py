# **********************************************************************
# * Nom     : paths.py                                                 *
# * Type    : Module                                                   *
# * Sujet   : Construction des chemins de partition et des fenêtres de *
# *   dates                                                            *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

import posixpath
import uuid
from datetime import date, datetime, timedelta

# Format d'une date dans un chemin comme sur la ligne de commande.
DATE_FORMAT = "%Y-%m-%d"
# Clé de partitionnement écrite dans le chemin, dt=AAAA-MM-JJ.
PARTITION_KEY = "dt"

# Préfixe d'un répertoire de travail, ignoré à la lecture.
TEMPORARY_PREFIX = "_tmp"


class PathError(ValueError):
    """Classe : PathError
    Description : Un chemin ou une date de partition est mal formé.
    """


def parse_date(text: str) -> date:
    """Méthode : parse_date
    Description : Lit une date au format attendu et refuse toute autre
      écriture.
    """
    try:
        return datetime.strptime(text.strip(), DATE_FORMAT).date()
    except ValueError as exc:
        raise PathError(
            f"Date attendue au format {DATE_FORMAT}, reçu {text!r}."
        ) from exc


def format_date(day: date) -> str:
    """Méthode : format_date
    Description : Écrit une date dans le format des chemins de partition.
    """
    return day.strftime(DATE_FORMAT)


def date_range(start: date, end: date) -> list[date]:
    """Méthode : date_range
    Description : Énumère les journées d'une fenêtre, bornes comprises.
    """
    if end < start:
        raise PathError(
            f"Fenêtre vide : {format_date(end)} précède {format_date(start)}."
        )
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def lookback_range(day: date, days: int) -> list[date]:
    """Méthode : lookback_range
    Description : Énumère les journées d'une fenêtre qui se termine au jour
      donné.
    """
    if days < 1:
        raise PathError("Une fenêtre de rattrapage couvre au moins un jour.")
    return date_range(day - timedelta(days=days - 1), day)


def features_partition(root: str, version: str, day: date) -> str:
    """Méthode : features_partition
    Description : Compose le chemin de la partition de variables d'une journée.
    """
    return join(
        root, "features", _segment(version, "feature_version"), partition_segment(day)
    )


def partition_segment(day: date) -> str:
    """Méthode : partition_segment
    Description : Compose le segment de chemin qui porte la date.
    """
    return f"{PARTITION_KEY}={format_date(day)}"


def part_file(index: int = 0) -> str:
    """Méthode : part_file
    Description : Nomme un fichier de partition, unique pour éviter toute
      collision.
    """
    return f"part-{index:05d}-{uuid.uuid4().hex}.parquet"


def temporary_sibling(target: str) -> str:
    """Méthode : temporary_sibling
    Description : Compose le répertoire de travail voisin d'une partition
      cible.
    """
    parent, name = posixpath.split(normalize(target))
    return join(parent, f"{TEMPORARY_PREFIX}-{name}-{uuid.uuid4().hex}")


def join(*parts: str) -> str:
    """Méthode : join
    Description : Assemble des morceaux de chemin en séparateurs POSIX.
    """
    cleaned = [normalize(part) for part in parts if part not in ("", None)]
    if not cleaned:
        raise PathError("Un chemin ne peut pas être vide.")
    head, *tail = cleaned
    return "/".join([head.rstrip("/"), *(part.strip("/") for part in tail)])


def normalize(path: str) -> str:
    """Méthode : normalize
    Description : Ramène les séparateurs Windows à la forme POSIX.
    """
    return str(path).replace("\\", "/")


def _segment(value: str, label: str) -> str:
    """Méthode : _segment
    Description : Valide qu'une valeur peut servir de segment de chemin.
    """
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text.startswith("."):
        raise PathError(f"{label} invalide pour un chemin de partition : {value!r}.")
    return text
