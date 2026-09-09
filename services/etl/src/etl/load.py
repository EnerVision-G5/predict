# **********************************************************************
# * Nom     : load.py                                                  *
# * Type    : Module                                                   *
# * Sujet   : Écriture des colonnes déduites dans mesure, sans toucher *
# *   à la source                                                      *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sqlalchemy import or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from etl.exclude import to_exclusions
from etl.impute import IMPUTATION_COLUMNS
from predict_common.db import (
    CONFLICT_KEY,
    DERIVED_COLUMNS,
    SOURCE_COLUMNS,
    mesure,
    mesure_exclu,
    write_batches,
)
from predict_common.schemas import QUALITY_SOURCE_COLUMN

# Colonnes soumises à l'écriture, source et déduites réunies.
STORED_COLUMNS = (*SOURCE_COLUMNS, *IMPUTATION_COLUMNS, QUALITY_SOURCE_COLUMN)

logger = logging.getLogger(__name__)


class LoadError(ValueError):
    """Classe : LoadError
    Description : Le tableau soumis n'a pas la forme attendue par la table
      mesure.
    """


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Méthode : to_records
    Description : Convertit le tableau enrichi en lignes acceptables par
      SQLAlchemy.
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
    """Méthode : build_upsert
    Description : Construit l'écriture des colonnes déduites, réservée aux
      lignes qui changent.
    """
    statement = insert(mesure).values(records)
    excluded = statement.excluded
    changed = or_(
        *(
            mesure.c[name].is_distinct_from(getattr(excluded, name))
            for name in DERIVED_COLUMNS
        )
    )
    return statement.on_conflict_do_update(
        index_elements=list(CONFLICT_KEY),
        set_={name: getattr(excluded, name) for name in DERIVED_COLUMNS},
        where=changed,
    )


def build_exclusion_upsert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_exclusion_upsert
    Description : Construit l'insertion idempotente d'un lot d'exclusions.
    """
    return insert(mesure_exclu).values(records).on_conflict_do_nothing(
        index_elements=list(CONFLICT_KEY)
    )


def load(engine: Engine, frame: pd.DataFrame, batch_size: int) -> tuple[int, int]:
    """Méthode : load
    Description : Repose les colonnes déduites puis écrit les exclusions, dans
      cet ordre.
    """
    rows = write_batches(engine, to_records(frame), batch_size, build_upsert)
    excluded = write_batches(
        engine, to_exclusions(frame), batch_size, build_exclusion_upsert
    )
    return rows, excluded


def _to_sql_value(value: Any) -> Any:
    """Méthode : _to_sql_value
    Description : Ramène les manquants pandas au NULL attendu par la base.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and getattr(value, "ndim", 0) == 1:
        return [str(item) for item in value.tolist()]
    if value is None or pd.isna(value):
        return None
    return value
