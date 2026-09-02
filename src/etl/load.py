"""Chargement idempotent des mesures dans la table `mesure`.

L'idempotence n'est pas un confort : le pipeline est rejoué à la main après
incident, et la fenêtre rejouée recouvre toujours des mesures déjà chargées.
La PK composite (site_id, ts) plus un ON CONFLICT DO NOTHING rendent ce rejeu
sans effet de bord.
"""

from __future__ import annotations

from collections.abc import Callable
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

from etl.impute import IMPUTATION_COLUMNS
from etl.transform import MEASURE_COLUMNS

# Colonnes réellement écrites : celles que porte la source, puis celles que
# l'ETL calcule. Les secondes viennent de la migration d'EV-08, sans laquelle
# la base rejettera l'insertion : voir migrations/03_mesure_imputation.sql.
STORED_COLUMNS = (*MEASURE_COLUMNS, *IMPUTATION_COLUMNS)

metadata = MetaData()


class LoadError(ValueError):
    """Le tableau soumis n'a pas la forme attendue par la table `mesure`."""

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
    Column("consumption_kw_imputed", Numeric(10, 2)),
    Column("imputation_method", String(20)),
)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertit le tableau normalisé en lignes acceptables par SQLAlchemy.

    Un lot qui n'aurait pas traversé l'imputation est refusé ici plutôt que
    chargé amputé : `imputation_method` est NOT NULL en base, et une colonne
    silencieusement absente ferait échouer l'insertion sans dire pourquoi.
    """
    absent = [name for name in STORED_COLUMNS if name not in frame.columns]
    if absent:
        raise LoadError(
            f"Colonnes absentes du tableau à charger : {absent}. Le lot doit"
            " passer par etl.impute.impute_frame avant le chargement."
        )
    projected = frame[list(STORED_COLUMNS)]
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


def write_batches(
    engine: Engine,
    records: list[dict[str, Any]],
    batch_size: int,
    build: Callable[[list[dict[str, Any]]], Any],
) -> int:
    """Écrit les lignes par lots, tous dans la même transaction.

    `build` produit l'instruction d'un lot : la même mécanique sert `mesure`
    et `mesure_exclu`, qui n'ont en commun que d'être écrites de façon
    idempotente et par paquets bornés.
    """
    if not records:
        return 0
    with engine.begin() as connection:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            connection.execute(build(chunk))
    return len(records)


def load_frame(engine: Engine, frame: pd.DataFrame, batch_size: int) -> int:
    """Écrit le tableau en base par lots et retourne le nombre de lignes vues.

    Le compteur porte sur les lignes soumises, pas sur les lignes réellement
    insérées : ON CONFLICT DO NOTHING ne remonte pas les doublons ignorés, et
    faire croire le contraire fausserait le suivi d'ingestion.
    """
    return write_batches(engine, to_records(frame), batch_size, build_upsert)
