"""Rangement des mesures inexploitables dans `mesure_exclu`.

Une mesure dont la puissance est nulle et qu'aucune imputation n'a pu
reconstruire ne peut entrer dans aucun agrégat : la moyenne d'une panne reste
une panne. La supprimer serait pourtant perdre la panne elle-même, qui est une
information de terrain. Elle reste donc en base, brute, et c'est `mesure_exclu`
qui porte la décision de l'écarter et la cause de cette décision.

Une mesure imputée n'est jamais rangée ici : elle porte une valeur exploitable
et `imputation_method` dit d'où cette valeur vient. Écarter ce qu'on vient de
reconstruire n'aurait servi à rien.

L'exclusion écrite par l'ETL est automatique, donc `exclu_par` reste NULL : la
colonne est réservée aux exclusions décidées par un analyste, et les confondre
rendrait impossible de savoir qui a jugé quoi.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import Column, DateTime, String, Table, Text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from etl.impute import IMPUTED_COLUMN, SOURCE_COLUMN
from etl.load import metadata, write_batches

# Cause retenue quand la source n'a fourni aucun motif. Elle ne prétend pas
# expliquer la panne, seulement dire ce que l'ETL a constaté.
DEFAULT_REASON = "valeur nulle non imputable"

EXCLUSION_COLUMNS = ("site_id", "ts", "raison")

# Reflet minimal du schéma figé v1.0. La clé naturelle (site_id, ts) est
# déclarée primaire parce que c'est elle que vise le ON CONFLICT ; la vraie
# clé primaire, exclusion_id, est laissée à l'IDENTITY de la base, comme
# exclu_le est laissé à son DEFAULT now().
mesure_exclu = Table(
    "mesure_exclu",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("raison", Text, nullable=False),
)


def is_excluded(frame: pd.DataFrame) -> pd.Series:
    """Sélectionne les mesures sans valeur brute ni valeur reconstruite."""
    raw = pd.to_numeric(frame[SOURCE_COLUMN], errors="coerce")
    imputed = pd.to_numeric(frame[IMPUTED_COLUMN], errors="coerce")
    return raw.isna() & imputed.isna() & frame["ts"].notna()


def exclusion_reason(null_reasons: Any) -> str:
    """Reprend les motifs de la source, seule à savoir pourquoi le capteur s'est tu."""
    if isinstance(null_reasons, list) and null_reasons:
        return ", ".join(str(reason) for reason in null_reasons)
    return DEFAULT_REASON


def to_exclusions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Retourne les exclusions automatiques portées par un lot imputé."""
    if frame.empty:
        return []
    excluded = frame[is_excluded(frame)]
    return [
        {
            "site_id": row.site_id,
            "ts": row.ts.to_pydatetime(),
            "raison": exclusion_reason(row.null_reasons),
        }
        for row in excluded.itertuples(index=False)
    ]


def build_exclusion_upsert(records: list[dict[str, Any]]) -> Any:
    """Construit l'insertion idempotente d'un lot d'exclusions.

    Le rejeu d'une fenêtre repasse sur des mesures déjà écartées, et la
    contrainte UNIQUE (site_id, ts) dit qu'une mesure n'est exclue qu'une
    fois : réécrire l'exclusion échouerait au lieu de ne rien faire.
    """
    statement = insert(mesure_exclu).values(records)
    return statement.on_conflict_do_nothing(index_elements=["site_id", "ts"])


def load_exclusions(
    engine: Engine,
    frame: pd.DataFrame,
    batch_size: int,
) -> int:
    """Écrit les exclusions du lot et retourne leur nombre.

    Cet appel suit celui de `load_frame` et ne le précède jamais :
    `mesure_exclu` porte une clé étrangère vers `mesure`, et une exclusion
    insérée avant sa mesure serait rejetée par la base.
    """
    records = to_exclusions(frame)
    return write_batches(engine, records, batch_size, build_exclusion_upsert)
