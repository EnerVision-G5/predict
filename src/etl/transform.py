"""Normalisation des mesures brutes en tableau prêt pour la base.

Règle structurante héritée de la doc API : les valeurs manquantes ne sont
jamais filtrées. Une mesure nulle porte une information de panne capteur, elle
est chargée telle quelle avec son `data_quality` et ses `null_reasons`.
L'imputation est un traitement aval, elle n'a pas sa place ici.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

# Colonnes de la table `mesure`, dans l'ordre du schéma figé v1.0.
MEASURE_COLUMNS = (
    "ts",
    "site_id",
    "consumption_kw",
    "consumption_kwh",
    "voltage_v",
    "current_a",
    "power_factor",
    "temperature_celsius",
    "humidity_percent",
    "null_reasons",
    "data_quality",
)

NUMERIC_COLUMNS = (
    "consumption_kw",
    "consumption_kwh",
    "voltage_v",
    "current_a",
    "power_factor",
    "temperature_celsius",
    "humidity_percent",
)

DEFAULT_DATA_QUALITY = "good"


class TransformError(ValueError):
    """Le lot reçu ne porte pas les clés minimales d'une mesure."""


def to_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Construit le tableau normalisé correspondant aux mesures reçues.

    Le tableau retourné a toujours les colonnes de `MEASURE_COLUMNS`, même
    pour un lot vide : l'étage de chargement peut donc travailler sans tester
    le cas dégénéré.
    """
    rows = list(records)
    frame = pd.DataFrame(rows, columns=list(MEASURE_COLUMNS))
    if rows:
        missing = {"timestamp", "site_id"} - set(rows[0])
        if missing:
            raise TransformError(
                f"Clés absentes de la mesure source : {sorted(missing)}."
            )
        frame["ts"] = pd.to_datetime(
            [row.get("timestamp") for row in rows],
            utc=True,
            errors="coerce",
        )
    frame = _coerce_types(frame)
    return frame[list(MEASURE_COLUMNS)]


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Ne garde qu'une ligne par clé primaire (site_id, ts), la dernière.

    La table `mesure` a une PK composite : un lot contenant deux fois la même
    clé ferait échouer l'insertion entière. La dernière occurrence gagne, elle
    correspond à la relecture la plus récente de la source.
    """
    deduplicated = frame.dropna(subset=["ts", "site_id"])
    return (
        deduplicated.drop_duplicates(subset=["site_id", "ts"], keep="last")
        .sort_values(["site_id", "ts"])
        .reset_index(drop=True)
    )


def _coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Aligne les types du tableau sur ceux des colonnes de la table."""
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["site_id"] = frame["site_id"].astype("object")
    frame["data_quality"] = (
        frame["data_quality"].fillna(DEFAULT_DATA_QUALITY).astype("object")
    )
    frame["null_reasons"] = frame["null_reasons"].map(_normalize_null_reasons)
    return frame


def _normalize_null_reasons(value: Any) -> list[str]:
    """Ramène `null_reasons` au TEXT[] NOT NULL attendu par la base."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]
