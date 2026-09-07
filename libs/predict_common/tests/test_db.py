"""Contrôle du schéma avant écriture.

Le schéma vit dans un autre dépôt — les migrations Alembic de l'API — et rien
ne garantit qu'une base rencontrée sur un poste soit à jour. Sans ce contrôle,
l'échec arrivait au milieu du chargement, sous la forme brute du driver
(`UndefinedColumn: column "quality_source" ... does not exist`), et APRÈS que
la journée entière ait été calculée.

Un SQLite en mémoire suffit : `verify_schema` passe par l'introspection de
SQLAlchemy, qui parle tous les dialectes.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine

from predict_common.db import DatabaseError, verify_schema


def engine_with(*tables: Table):
    """Monte une base en mémoire portant exactement les tables données."""
    engine = create_engine("sqlite://")
    for table in tables:
        table.create(engine)
    return engine


def declared(name: str, *columns: str) -> Table:
    """Déclare une table attendue, colonnes nommées."""
    return Table(
        name,
        MetaData(),
        *(Column(column, String(20)) for column in columns),
    )


def test_une_base_a_jour_passe_sans_rien_dire() -> None:
    expected = declared("mesure", "ts", "site_id", "quality_source")
    engine = engine_with(declared("mesure", "ts", "site_id", "quality_source"))
    verify_schema(engine, (expected,))


def test_une_colonne_manquante_est_nommee() -> None:
    """C'est exactement la dérive rencontrée : la migration 07 non appliquée.

    Le message doit nommer la colonne, pas seulement échouer.
    """
    expected = declared("mesure", "ts", "site_id", "quality_source")
    engine = engine_with(declared("mesure", "ts", "site_id"))

    with pytest.raises(DatabaseError, match=r"mesure\.quality_source"):
        verify_schema(engine, (expected,))


def test_une_table_manquante_est_nommee_sans_ses_colonnes() -> None:
    """Une table absente se dit une fois, pas une fois par colonne.

    Lister ses huit colonnes noierait l'information utile — c'est la
    migration qui crée la table qui manque, pas huit correctifs distincts.
    """
    expected = declared("ingestion_etat", "site_id", "last_rows", "source")
    engine = engine_with(declared("mesure", "ts"))

    with pytest.raises(DatabaseError) as raised:
        verify_schema(engine, (expected,))

    message = str(raised.value)
    assert "ingestion_etat" in message
    assert "ingestion_etat.site_id" not in message


def test_le_message_nomme_la_commande_qui_repare() -> None:
    """Un contrôle qui dit ce qui manque sans dire quoi faire est à moitié fait."""
    expected = declared("mesure", "ts", "quality_source")
    engine = engine_with(declared("mesure", "ts"))

    with pytest.raises(DatabaseError, match="alembic upgrade head"):
        verify_schema(engine, (expected,))


def test_une_colonne_en_trop_dans_la_base_ne_gene_pas() -> None:
    """La chaîne vérifie ce qu'elle écrit, pas ce que la base porte.

    Une colonne ajoutée au schéma et qu'aucun service ne remplit n'a aucune
    raison de faire échouer un run.
    """
    expected = declared("mesure", "ts")
    engine = engine_with(declared("mesure", "ts", "colonne_future"))
    verify_schema(engine, (expected,))
