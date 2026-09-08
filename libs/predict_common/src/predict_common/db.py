# **********************************************************************
# * Nom     : db.py                                                    *
# * Type    : Module                                                   *
# * Sujet   : Reflet des tables écrites par la chaîne et écriture par  *
# *   lots bornés                                                      *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    inspect,
)
from sqlalchemy.engine import Engine

from predict_common.schemas import (
    QUALITY_SOURCE_COLUMN,
    SITE_COLUMN,
    TIMESTAMP_COLUMN,
)

# Clé naturelle d'une mesure, sur laquelle porte l'upsert.
CONFLICT_KEY = (SITE_COLUMN, TIMESTAMP_COLUMN)

metadata = MetaData()

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Classe : DatabaseError
    Description : La base est absente de la configuration, ou refuse
      l'écriture.
    """


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
    Column("quality_source", String(10)),
)

mesure_exclu = Table(
    "mesure_exclu",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("raison", Text, nullable=False),
)

site = Table(
    "site",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("site_type", String(50), nullable=False),
    Column("site_name", String(150), nullable=False),
    Column("location", String(150)),
    Column("capacity_kw", Numeric(10, 2), nullable=False),
    Column("status", String(20), nullable=False),
)

# Colonnes du référentiel des sites, dans l'ordre écrit.
SITE_COLUMNS = (
    "site_id",
    "site_type",
    "site_name",
    "location",
    "capacity_kw",
    "status",
)

ingestion_etat = Table(
    "ingestion_etat",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("last_attempt_at", DateTime(timezone=True), nullable=False),
    Column("last_success_at", DateTime(timezone=True)),
    Column("last_rows", Integer, nullable=False),
    Column("last_data_lag_s", Numeric(10, 2)),
    Column("consecutive_failures", Integer, nullable=False),
    Column("last_error", Text),
    Column("source", String(20), nullable=False),
)

alerte = Table(
    "alerte",
    metadata,
    Column("alert_id", String(64), primary_key=True),
    Column("site_id", String(20), nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("severity", String(10), nullable=False),
    Column("type_alerte", String(20), nullable=False),
    Column("message", Text, nullable=False),
    Column("valeur", Numeric(12, 2)),
    Column("seuil", Numeric(12, 2)),
)

# Clé d'une alerte : son identifiant, donné par la source.
ALERTE_KEY = ("alert_id",)

capteur_etat = Table(
    "capteur_etat",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("capteur", String(20), primary_key=True),
    Column("statut", String(10), nullable=False),
    Column("failing_until", DateTime(timezone=True)),
    Column("overall", String(10), nullable=False),
    Column("releve_le", DateTime(timezone=True)),
)

# Clé de l'état courant d'un capteur.
CAPTEUR_ETAT_KEY = ("site_id", "capteur")


capteur_panne = Table(
    "capteur_panne",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("capteur", String(20), primary_key=True),
    Column("debut_le", DateTime(timezone=True), primary_key=True),
    Column("fin_le", DateTime(timezone=True)),
    Column("failing_until", DateTime(timezone=True)),
)

# Clé d'une panne : un capteur et l'instant où elle s'ouvre.
CAPTEUR_PANNE_KEY = ("site_id", "capteur", "debut_le")

# Clé de l'état d'ingestion : un état par site.
INGESTION_KEY = ("site_id",)

# Marque un état posé par la collecte au fil de l'eau.
INGESTION_SOURCE_POLLER = "poller"
# Marque un état posé par un rattrapage daté.
INGESTION_SOURCE_BACKFILL = "backfill"

modele = Table(
    "modele",
    metadata,
    Column("nom", String(100), primary_key=True),
    Column("version", String(20), primary_key=True),
    Column("mlflow_run_id", String(64)),
    Column("date_entrainement", DateTime(timezone=True)),
    Column("actif", Boolean, nullable=False),
)

# Clé d'un modèle enregistré : son nom et sa version.
MODELE_KEY = ("nom", "version")

# Colonnes réécrites quand un modèle déjà inscrit est remis à jour.
MODELE_UPDATED_COLUMNS = (
    "mlflow_run_id",
    "date_entrainement",
    "actif",
)


# Colonnes appartenant au collecteur, que l'ETL ne réécrit jamais.
SOURCE_COLUMNS = (
    TIMESTAMP_COLUMN,
    SITE_COLUMN,
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

# Colonnes déduites par l'ETL, seules réécrites par lui.
DERIVED_COLUMNS = (
    "null_reasons",
    "data_quality",
    "consumption_kw_imputed",
    "imputation_method",
    QUALITY_SOURCE_COLUMN,
)


def is_configured(database_url: str) -> bool:
    """Méthode : is_configured
    Description : Dit si une URL de base a été fournie.
    """
    return bool(database_url.strip())


def open_engine(database_url: str, pool_pre_ping: bool = False) -> Engine:
    """Méthode : open_engine
    Description : Ouvre le moteur SQLAlchemy, en refusant une configuration
      vide.
    """
    if not is_configured(database_url):
        raise DatabaseError(
            "DATABASE_URL est obligatoire pour cette opération : la base porte"
            " la couche brute de la chaîne et le registre applicatif des"
            " modèles. Copier .env.example en .env et la renseigner."
        )
    return create_engine(database_url, pool_pre_ping=pool_pre_ping)


def verify_schema(engine: Engine, tables: Sequence[Table]) -> None:
    """Méthode : verify_schema
    Description : Vérifie que la base porte les tables et colonnes attendues,
      et nomme ce qui manque.
    """
    expected: dict[str, set[str]] = {
        table.name: {column.name for column in table.columns} for table in tables
    }
    inspector = inspect(engine)
    present = set(inspector.get_table_names())

    absent_tables = sorted(name for name in expected if name not in present)
    absent_columns = sorted(
        f"{name}.{column}"
        for name, columns in expected.items()
        if name in present
        for column in columns
        - {info["name"] for info in inspector.get_columns(name)}
    )
    if not absent_tables and not absent_columns:
        return

    missing = ", ".join([*absent_tables, *absent_columns])
    raise DatabaseError(
        f"La base ne porte pas le schéma attendu par la chaîne : {missing}."
        " Appliquer les migrations avant de relancer — elles vivent dans le"
        " dépôt api (`alembic upgrade head`), qui détient le schéma."
    )


def write_batches(
    engine: Engine,
    records: Sequence[dict[str, Any]],
    batch_size: int,
    build: Callable[[list[dict[str, Any]]], Any],
) -> int:
    """Méthode : write_batches
    Description : Écrit les lignes par lots bornés, tous dans la même
      transaction.
    """
    if not records:
        return 0
    rows = list(records)
    with engine.begin() as connection:
        for start in range(0, len(rows), batch_size):
            connection.execute(build(rows[start : start + batch_size]))
    return len(rows)
