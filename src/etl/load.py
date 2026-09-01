"""Chargement idempotent des mesures dans la table `mesure`.

L'idempotence n'est pas un confort : le pipeline est rejoué à la main après
incident, et la fenêtre rejouée recouvre toujours des mesures déjà chargées.
La PK composite (site_id, ts) plus un ON CONFLICT DO NOTHING rendent ce rejeu
sans effet de bord.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from etl.transform import MEASURE_COLUMNS

metadata = MetaData()

# Reflet minimal du schéma figé v1.0 : seules les colonnes que l'ETL écrit
# sont déclarées. inserted_at est laissé au DEFAULT now() de la base, qui date
# le chargement et non la mesure.
mesure = Table(
    "mesure",
    metadata,
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("site_id", String(20), primary_key=True),
    Column("consumption_kw", Numeric(10, 2)),
    Column("consumption_kwh", Numeric(10, 2)),
    Column("voltage_v", Numeric(8, 2)),
    Column("current_a", Numeric(8, 2)),
    Column("power_factor", Numeric(4, 3)),
    Column("temperature_celsius", Numeric(5, 2)),
    Column("humidity_percent", Numeric(5, 2)),
    Column("null_reasons", ARRAY(Text)),
    Column("data_quality", String(10)),
)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertit le tableau normalisé en lignes acceptables par SQLAlchemy."""
    projected = frame[list(MEASURE_COLUMNS)]
    return [
        {key: _to_sql_value(value) for key, value in row.items()}
        for row in projected.to_dict(orient="records")
    ]


def build_upsert(records: list[dict[str, Any]]) -> Any:
    """Construit l'insertion idempotente pour un lot de mesures."""
    statement = insert(mesure).values(records)
    return statement.on_conflict_do_nothing(index_elements=["site_id", "ts"])


def _to_sql_value(value: Any) -> Any:
    """Ramène les manquants pandas (NaN, NaT) au NULL attendu par la base.

    Sans cette conversion, le driver écrirait un NaN flottant dans une colonne
    NUMERIC, ce que PostgreSQL accepte et qui pollue silencieusement les
    agrégats en aval.
    """
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return None
    return value


def load_frame(engine: Engine, frame: pd.DataFrame, batch_size: int) -> int:
    """Écrit le tableau en base par lots et retourne le nombre de lignes vues.

    Le compteur porte sur les lignes soumises, pas sur les lignes réellement
    insérées : ON CONFLICT DO NOTHING ne remonte pas les doublons ignorés, et
    faire croire le contraire fausserait le suivi d'ingestion.
    """
    records = to_records(frame)
    if not records:
        return 0
    with engine.begin() as connection:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            connection.execute(build_upsert(chunk))
    return len(records)
