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

Attention : la migration `03_mesure_imputation.sql` du repo enervision-db doit
être appliquée. Le schéma figé v1.0 n'a ni `consumption_kw_imputed` ni
`imputation_method`, que l'ETL écrit à chaque lot.

Les déclarations ci-dessous sont relevées sur `enervision-db/initdb/01_schema.sql`
et sa migration `03_mesure_imputation.sql`. Seules les tables que cette chaîne
écrit sont déclarées, et dans chacune, seules les colonnes qu'elle remplit :
`inserted_at` a un DEFAULT côté base, qui date le chargement mieux que nous.

`modele` et `prediction` ne sont donc pas ici. Elles existent bien dans le
schéma figé, mais c'est l'API EnerVision qui les écrit — elle sert le contrat
de prédiction à ses consommateurs et archive ce qu'elle rend. Le service
d'inférence de ce repo calcule et répond ; ce qu'on fait de sa réponse ne le
regarde pas.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import (
    ARRAY,
    Column,
    DateTime,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine

from predict_common.schemas import SITE_COLUMN, TIMESTAMP_COLUMN

# Clé naturelle de `mesure`, et cible de tous les ON CONFLICT de la chaîne.
CONFLICT_KEY = (SITE_COLUMN, TIMESTAMP_COLUMN)

metadata = MetaData()

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """La base est absente de la configuration, ou refuse l'écriture."""


# Reflet minimal du schéma figé v1.0 : seules les colonnes que la chaîne écrit
# sont déclarées. `inserted_at` est laissé au DEFAULT now() de la base, qui
# date le chargement et non la mesure.
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

# La clé naturelle (site_id, ts) est déclarée primaire parce que c'est elle que
# vise le ON CONFLICT ; la vraie clé primaire, exclusion_id, est laissée à
# l'IDENTITY de la base, comme exclu_le est laissé à son DEFAULT now().
#
# `exclu_par` reste NULL : une exclusion écrite par l'ETL est automatique, et
# la colonne est réservée aux exclusions décidées par un analyste. Les
# confondre rendrait impossible de savoir qui a jugé quoi.
mesure_exclu = Table(
    "mesure_exclu",
    metadata,
    Column("site_id", String(20), primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("raison", Text, nullable=False),
)

# Référentiel des sites. `mesure.site_id` et `prediction.site_id` le
# référencent : une mesure dont le site n'y est pas est rejetée par la base,
# quelle que soit sa qualité. Le seed `02_seed_sites.sql` pose sept sites dont
# quatre avec des capacités marquées « à synchroniser » — c'est le collecteur
# qui les remplace, seul service à joindre `GET /api/v1/sites`.
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


# Colonnes que le collecteur écrit : celles que la source sert, et rien de
# plus. Les colonnes déduites appartiennent à l'ETL.
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

# Colonnes que l'ETL repose sur une mesure déjà présente. Elles ne recouvrent
# jamais une valeur de la source : `null_reasons` et `data_quality` sont
# complétées, pas remplacées — voir `etl.quality`.
DERIVED_COLUMNS = (
    "null_reasons",
    "data_quality",
    "consumption_kw_imputed",
    "imputation_method",
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
            "DATABASE_URL est obligatoire : la table `mesure` est la couche"
            " brute de la chaîne. Copier .env.example en .env et la renseigner."
        )
    return create_engine(database_url, pool_pre_ping=pool_pre_ping)


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
