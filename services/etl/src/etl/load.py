"""Repose dans `mesure` ce que l'ETL a déduit, sans toucher à la source.

C'est le `Load` du schéma : la mesure ressort enrichie de ce que les étages
précédents ont établi. Cinq colonnes seulement sont écrites — la cause de
chaque valeur absente, la qualification que les données imposent, la valeur
reconstruite, la méthode qui l'a produite, et la signature du passage. Les
sept colonnes de mesure ne sont jamais réécrites : elles appartiennent au
collecteur, et à lui seul.

La cinquième, `quality_source`, ne décrit pas la mesure mais le traitement.
Elle bascule de `source` à `etl` et répond à une question que rien d'autre ne
tranche : ce `good` est-il celui que le collecteur a posé faute de mieux, ou
celui que la qualification a confirmé. Sans elle, une fenêtre fraîchement
collectée passerait pour une fenêtre saine.

La distinction n'est pas théorique. Le `DO UPDATE` ci-dessous ne liste que les
colonnes déduites : même si le lot soumis portait une consommation différente
de celle en base — un bug de projection, une mauvaise jointure — la base
garderait celle de la source. La panne capteur ne peut donc pas être effacée
par l'étage qui a justement pour métier de la décrire.

Les exclusions sont écrites après les mesures et jamais avant, `mesure_exclu`
portant une clé étrangère vers `mesure`.
"""

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

STORED_COLUMNS = (*SOURCE_COLUMNS, *IMPUTATION_COLUMNS, QUALITY_SOURCE_COLUMN)

logger = logging.getLogger(__name__)


class LoadError(ValueError):
    """Le tableau soumis n'a pas la forme attendue par la table `mesure`."""


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertit le tableau enrichi en lignes acceptables par SQLAlchemy.

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
    """Construit l'écriture des colonnes déduites sur une mesure existante.

    `DO UPDATE` et non `DO NOTHING` : contrairement au collecteur, l'ETL a
    quelque chose de nouveau à dire sur une ligne déjà présente. Un rejeu doit
    reposer la qualification et l'imputation, sans quoi corriger une règle
    n'aurait aucun effet sur l'historique déjà traité.

    Mais seulement si le résultat diffère. Sans la garde ci-dessous, rejouer
    une fenêtre réécrit chacune de ses lignes à l'identique : PostgreSQL n'a
    pas de mise à jour sans écriture, chaque UPDATE laisse un tuple mort,
    remplit le WAL et donne du travail à l'autovacuum. Un rattrapage sur deux
    ans a ainsi porté un seul chunk à 361 000 UPDATE pour 88 000 lignes.

    `IS DISTINCT FROM` et non `<>` : `data_quality` peut passer de NULL à une
    valeur, et une comparaison ordinaire rendrait NULL — donc faux — laissant
    la correction sur le carreau précisément là où elle compte.
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
    """Construit l'insertion idempotente d'un lot d'exclusions.

    Le rejeu d'une fenêtre repasse sur des mesures déjà écartées, et la
    contrainte UNIQUE (site_id, ts) dit qu'une mesure n'est exclue qu'une
    fois : réécrire l'exclusion échouerait au lieu de ne rien faire.
    """
    return insert(mesure_exclu).values(records).on_conflict_do_nothing(
        index_elements=list(CONFLICT_KEY)
    )


def load(engine: Engine, frame: pd.DataFrame, batch_size: int) -> tuple[int, int]:
    """Repose les colonnes déduites puis écrit les exclusions, dans cet ordre.

    L'ordre n'est pas négociable : `mesure_exclu` porte une clé étrangère vers
    `mesure`, et une exclusion insérée avant sa mesure serait rejetée.
    """
    rows = write_batches(engine, to_records(frame), batch_size, build_upsert)
    excluded = write_batches(
        engine, to_exclusions(frame), batch_size, build_exclusion_upsert
    )
    return rows, excluded


def _to_sql_value(value: Any) -> Any:
    """Ramène les manquants pandas (NaN, NaT) au NULL attendu par la base.

    Sans cette conversion, le driver écrirait un NaN flottant dans une colonne
    NUMERIC, ce que PostgreSQL accepte et qui pollue silencieusement les
    agrégats en aval.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and getattr(value, "ndim", 0) == 1:
        return [str(item) for item in value.tolist()]
    if value is None or pd.isna(value):
        return None
    return value
