from __future__ import annotations

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine

from predict_common.db import DatabaseError, verify_schema


def engine_with(*tables: Table):
    engine = create_engine("sqlite://")
    for table in tables:
        table.create(engine)
    return engine


def declared(name: str, *columns: str) -> Table:
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
    expected = declared("mesure", "ts", "site_id", "quality_source")
    engine = engine_with(declared("mesure", "ts", "site_id"))

    with pytest.raises(DatabaseError, match=r"mesure\.quality_source"):
        verify_schema(engine, (expected,))


def test_une_table_manquante_est_nommee_sans_ses_colonnes() -> None:
    expected = declared("ingestion_etat", "site_id", "last_rows", "source")
    engine = engine_with(declared("mesure", "ts"))

    with pytest.raises(DatabaseError) as raised:
        verify_schema(engine, (expected,))

    message = str(raised.value)
    assert "ingestion_etat" in message
    assert "ingestion_etat.site_id" not in message


def test_le_message_nomme_la_commande_qui_repare() -> None:
    expected = declared("mesure", "ts", "quality_source")
    engine = engine_with(declared("mesure", "ts"))

    with pytest.raises(DatabaseError, match="alembic upgrade head"):
        verify_schema(engine, (expected,))


def test_une_colonne_en_trop_dans_la_base_ne_gene_pas() -> None:
    expected = declared("mesure", "ts")
    engine = engine_with(declared("mesure", "ts", "colonne_future"))
    verify_schema(engine, (expected,))
