"""Construction des chemins de partition, seule frontière entre les services.

Un service ne connaît pas son voisin, il connaît un chemin. L'ETL écrit
`features/v1/dt=2026-09-02/`, l'entraînement et le service d'inférence lisent
ce chemin. Personne n'importe personne : ce module est la seule chose qu'ils
partagent, et il ne fait que concaténer des chaînes.

La couche brute n'est pas ici : c'est la table `mesure`, dont
`predict_common.db` porte la définition. Une frontière peut être un chemin ou
une table ; ce qui compte est qu'elle ne soit jamais un import.

Les chemins sont manipulés comme des URI, jamais comme des chemins système.
Un `os.path.join` sous Windows produirait `data\\raw\\dt=...`, illisible pour
un stockage objet et incohérent avec ce que la même commande écrit sous
Linux. La séparation est donc toujours `/`, quel que soit l'hôte.

Le motif `dt=YYYY-MM-DD` n'est pas décoratif : c'est le partitionnement Hive,
que pyarrow, DuckDB et Spark savent tous élaguer sans lire les fichiers.
"""

from __future__ import annotations

import posixpath
import uuid
from datetime import date, datetime, timedelta

DATE_FORMAT = "%Y-%m-%d"
PARTITION_KEY = "dt"

TEMPORARY_PREFIX = "_tmp"


class PathError(ValueError):
    """Un identifiant de partition ne peut pas entrer dans un chemin."""


def parse_date(text: str) -> date:
    """Analyse une date `YYYY-MM-DD` de ligne de commande.

    Le format est strict et sans repli : une date approximative écrirait la
    partition d'un autre jour, ce qu'aucun contrôle aval ne rattraperait.
    """
    try:
        return datetime.strptime(text.strip(), DATE_FORMAT).date()
    except ValueError as exc:
        raise PathError(
            f"Date attendue au format {DATE_FORMAT}, reçu {text!r}."
        ) from exc


def format_date(day: date) -> str:
    """Retourne la date sous la forme qu'elle prend dans un chemin."""
    return day.strftime(DATE_FORMAT)


def date_range(start: date, end: date) -> list[date]:
    """Retourne les jours de `start` à `end`, bornes comprises."""
    if end < start:
        raise PathError(
            f"Fenêtre vide : {format_date(end)} précède {format_date(start)}."
        )
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def lookback_range(day: date, days: int) -> list[date]:
    """Retourne les `days` jours qui précèdent `day`, ce jour compris.

    L'ETL en a besoin pour calculer un décalage de 168 h : la partition du
    jour seule ne porte pas la semaine passée, et un décalage calculé sur ce
    qu'elle contient serait faux sans jamais le dire.
    """
    if days < 1:
        raise PathError("Une fenêtre de rattrapage couvre au moins un jour.")
    return date_range(day - timedelta(days=days - 1), day)


def features_partition(root: str, version: str, day: date) -> str:
    """Chemin de la partition de variables d'une version pour un jour."""
    return join(
        root, "features", _segment(version, "feature_version"), partition_segment(day)
    )


def partition_segment(day: date) -> str:
    """Retourne le segment Hive d'un jour, `dt=2026-09-02`."""
    return f"{PARTITION_KEY}={format_date(day)}"


def part_file(index: int = 0) -> str:
    """Retourne le nom d'un fichier de données dans une partition.

    L'UUID sépare deux écritures concurrentes de la même partition, cas
    ordinaire du poller : deux ticks d'une même minute ne doivent pas se
    recouvrir. L'index garde l'ordre lisible quand une écriture produit
    plusieurs fichiers.
    """
    return f"part-{index:05d}-{uuid.uuid4().hex}.parquet"


def temporary_sibling(target: str) -> str:
    """Chemin de travail d'une écriture atomique, voisin de sa cible.

    Voisin et non enfant : un répertoire de travail à l'intérieur de la
    partition serait visible d'un lecteur qui liste la partition pendant
    l'écriture, et resterait sur place si le processus mourait au milieu.
    """
    parent, name = posixpath.split(normalize(target))
    return join(parent, f"{TEMPORARY_PREFIX}-{name}-{uuid.uuid4().hex}")


def join(*parts: str) -> str:
    """Assemble des segments de chemin en URI, sans jamais doubler le `/`.

    Le schéma d'une URI est préservé : `posixpath.join` ramènerait
    `s3://bucket` et `raw` à `s3:/bucket/raw`, avec une barre en moins.
    """
    cleaned = [normalize(part) for part in parts if part not in ("", None)]
    if not cleaned:
        raise PathError("Un chemin ne peut pas être vide.")
    head, *tail = cleaned
    return "/".join([head.rstrip("/"), *(part.strip("/") for part in tail)])


def normalize(path: str) -> str:
    """Ramène un chemin système à la forme URI utilisée partout ici."""
    return str(path).replace("\\", "/")


def _segment(value: str, label: str) -> str:
    """Vérifie qu'un identifiant peut entrer tel quel dans un chemin.

    Une valeur qui porterait une barre oblique déplacerait silencieusement la
    partition d'un niveau, et un `..` la ferait sortir de la racine.
    """
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text.startswith("."):
        raise PathError(f"{label} invalide pour un chemin de partition : {value!r}.")
    return text
