# **********************************************************************
# * Nom     : quality.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Qualification des mesures : ce qui manque, et à quel     *
# *   point c'est grave                                                *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from predict_common.schemas import (
    DATA_QUALITY_VALUES,
    QUALITY_CRITICAL,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    QUALITY_SOURCE_COLUMN,
    QUALITY_SOURCE_ETL,
    TARGET_COLUMN,
)

# Marque une cause déduite par l'ETL, non déclarée par la source.
DERIVED_REASON_SUFFIX = ":undeclared"

# Nombre d'absences à partir duquel l'heure est dite dégradée.
DEGRADED_NULL_COUNT = 3


def derive_data_quality(missing: Sequence[str]) -> str:
    """Méthode : derive_data_quality
    Description : Déduit la qualification d'une heure des colonnes qui lui
      manquent.
    """
    if not missing:
        return QUALITY_GOOD
    if TARGET_COLUMN in missing:
        return QUALITY_CRITICAL
    if len(missing) >= DEGRADED_NULL_COUNT:
        return QUALITY_DEGRADED
    return QUALITY_PARTIAL


def complete_null_reasons(
    declared: Sequence[str],
    missing: Sequence[str],
) -> list[str]:
    """Méthode : complete_null_reasons
    Description : Garde les causes déclarées, ou les déduit des colonnes
      absentes.
    """
    if declared:
        return [str(reason) for reason in declared]
    return [f"{column}{DERIVED_REASON_SUFFIX}" for column in missing]


def qualify(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Méthode : qualify
    Description : Pose sur tout le lot les causes, la qualification et son
      auteur.
    """
    if frame.empty:
        return frame
    missing = missing_columns(frame, columns)
    frame["null_reasons"] = [
        complete_null_reasons(declared, absent)
        for declared, absent in zip(frame["null_reasons"], missing, strict=True)
    ]
    frame["data_quality"] = [
        _settle_quality(declared, absent)
        for declared, absent in zip(frame["data_quality"], missing, strict=True)
    ]
    frame[QUALITY_SOURCE_COLUMN] = QUALITY_SOURCE_ETL
    return frame


def missing_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[tuple[str, ...]]:
    """Méthode : missing_columns
    Description : Liste, ligne par ligne, les colonnes surveillées qui sont
      vides.
    """
    watched = list(columns)
    absent = frame[watched].isna().to_numpy()
    return [
        tuple(
            column for column, is_absent in zip(watched, row, strict=True) if is_absent
        )
        for row in absent
    ]


def worst(qualities: Sequence[str]) -> str:
    """Méthode : worst
    Description : Retient la plus sévère de plusieurs qualifications.
    """
    known = [value for value in qualities if value in DATA_QUALITY_VALUES]
    if not known:
        return QUALITY_CRITICAL
    return max(known, key=DATA_QUALITY_VALUES.index)


def _settle_quality(declared: object, missing: Sequence[str]) -> str:
    """Méthode : _settle_quality
    Description : Arbitre entre la qualification déclarée et celle que les
      données imposent.
    """
    derived = derive_data_quality(missing)
    if declared not in DATA_QUALITY_VALUES:
        return derived
    return max(str(declared), derived, key=DATA_QUALITY_VALUES.index)
