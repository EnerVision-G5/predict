"""Reflet du schéma figé v1.0 : la table `mesure` et sa table d'exclusions.

Ce module est ici, et non dans un service, parce que deux services écrivent
la même table. Le collecteur y dépose la mesure telle que la source l'a
servie ; l'ETL y repose ce qu'il en a déduit. Deux définitions de `mesure`
donneraient deux vérités sur la même table, et c'est précisément ce que le
repo `enervision-db` interdit en détenant le schéma.

Le partage s'arrête là. Les *instructions* ne sont pas ici : le collecteur
insère sans écraser, l'ETL complète sans toucher aux colonnes de la source, et
ces deux politiques n'ont rien en commun sinon la table qu'elles visent. Les
mélanger ici rendrait possible qu'un service écrive avec la politique de
l'autre.

L'idempotence n'est pas un confort. Le pipeline est rejoué à la main après
incident, et la fenêtre rejouée recouvre toujours des lignes déjà écrites. La
clé primaire composite (site_id, ts) porte tout : c'est elle que vise chaque
ON CONFLICT.

Attention : trois migrations du repo enervision-db doivent être appliquées.
Le schéma figé v1.0 n'a ni `consumption_kw_imputed` ni `imputation_method`
(`03_mesure_imputation.sql`), ni la table `ingestion_etat`
(`06_ingestion_etat.sql`), ni `mesure.quality_source`
(`07_mesure_quality_source.sql`) — les trois sont écrites par cette chaîne.

Les déclarations ci-dessous sont relevées sur `enervision-db/initdb/01_schema.sql`
et sur ces migrations. Seules les tables que cette chaîne écrit sont déclarées,
et dans chacune, seules les colonnes qu'elle remplit : `inserted_at` et
`ingestion_etat.updated_at` ont un DEFAULT côté base, qui date l'écriture mieux
que nous.

`prediction` n'est donc pas ici : c'est l'API EnerVision qui l'écrit, elle
sert le contrat de prédiction à ses consommateurs et archive ce qu'elle rend.
Le service d'inférence de ce repo calcule et répond ; ce qu'on fait de sa
réponse ne le regarde pas.

`modele`, en revanche, y est. Elle décrit le modèle en service, pas la
prévision qu'il produit, et le seul geste qui change ce qu'elle doit dire est
la promotion d'un alias dans MLflow — un geste de l'entraînement. L'API ne
peut que la déduire en interrogeant le registre, ce qui donnerait deux
sources pour un fait dont une seule est autoritaire.
"""

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

CONFLICT_KEY = (SITE_COLUMN, TIMESTAMP_COLUMN)

metadata = MetaData()

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """La base est absente de la configuration, ou refuse l'écriture."""


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

CAPTEUR_PANNE_KEY = ("site_id", "capteur", "debut_le")

INGESTION_KEY = ("site_id",)

INGESTION_SOURCE_POLLER = "poller"
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

MODELE_KEY = ("nom", "version")

MODELE_UPDATED_COLUMNS = (
    "mlflow_run_id",
    "date_entrainement",
    "actif",
)


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

DERIVED_COLUMNS = (
    "null_reasons",
    "data_quality",
    "consumption_kw_imputed",
    "imputation_method",
    QUALITY_SOURCE_COLUMN,
)


def is_configured(database_url: str) -> bool:
    """Dit si une URL de base a été fournie.

    Une URL vide n'est jamais une désactivation ici : la base est la couche
    brute de la chaîne. C'est l'appelant qui transforme ce faux en refus de
    démarrer, avec un message qui nomme la variable manquante.
    """
    return bool(database_url.strip())


def open_engine(database_url: str, pool_pre_ping: bool = False) -> Engine:
    """Ouvre le moteur SQLAlchemy, en refusant une URL absente.

    `pool_pre_ping` est réservé aux processus longs : un poller vit des jours,
    et une connexion coupée par la base entre deux ticks échouerait sur la
    première écriture au lieu d'être renouvelée. Un traitement daté n'en a pas
    besoin et paierait un aller-retour par connexion.
    """
    if not is_configured(database_url):
        raise DatabaseError(
            "DATABASE_URL est obligatoire pour cette opération : la base porte"
            " la couche brute de la chaîne et le registre applicatif des"
            " modèles. Copier .env.example en .env et la renseigner."
        )
    return create_engine(database_url, pool_pre_ping=pool_pre_ping)


def verify_schema(engine: Engine, tables: Sequence[Table]) -> None:
    """Vérifie que la base porte les colonnes que la chaîne va écrire.

    Le schéma vit dans un autre dépôt — les migrations Alembic de l'API — et
    rien ne garantit qu'une base rencontrée sur un poste ou un environnement
    de démonstration soit à jour. Quand elle ne l'est pas, l'échec arrivait au
    milieu du chargement, sous la forme brute que remonte le driver :
        UndefinedColumn: column "quality_source" of relation "mesure" does
        not exist

    Ce message ne dit ni quelle migration manque, ni combien de colonnes sont
    concernées, ni que le reste de la chaîne fonctionnera de nouveau une fois
    la base remise à niveau. Il arrive en plus APRÈS le calcul complet de la
    journée, qui est alors perdu.

    Le contrôle est une seule requête sur `information_schema`, faite au
    démarrage. Il ne vérifie que ce que la chaîne écrit : une colonne ajoutée
    au schéma et qu'aucun service ne remplit n'a pas à faire échouer un run.
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
    """Écrit les lignes par lots bornés, tous dans la même transaction.

    `build` produit l'instruction d'un lot : la même mécanique sert les trois
    écritures de la chaîne — la mesure brute, la mesure enrichie et
    l'exclusion — qui n'ont en commun que d'être idempotentes et bornées.

    Le compteur porte sur les lignes soumises, pas sur les lignes réellement
    écrites : un ON CONFLICT ne remonte pas ce qu'il a ignoré, et faire croire
    le contraire fausserait le suivi d'ingestion.
    """
    if not records:
        return 0
    rows = list(records)
    with engine.begin() as connection:
        for start in range(0, len(rows), batch_size):
            connection.execute(build(rows[start : start + batch_size]))
    return len(rows)
