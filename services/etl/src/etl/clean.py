# **********************************************************************
# * Nom     : clean.py                                                 *
# * Type    : Module                                                   *
# * Sujet   : Normalisation du lot lu dans mesure : types, doublons,   *
# *   qualification                                                    *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import pandas as pd

from etl.quality import qualify
from predict_common.schemas import (
    NUMERIC_COLUMNS,
    QUALITY_SOURCE_COLUMN,
    SITE_COLUMN,
    TIMESTAMP_COLUMN,
)

# Colonnes de la couche brute, dans l'ordre attendu en aval.
MEASURE_COLUMNS = (
    TIMESTAMP_COLUMN,
    SITE_COLUMN,
    *NUMERIC_COLUMNS,
    "null_reasons",
    "data_quality",
    QUALITY_SOURCE_COLUMN,
)


class CleanError(ValueError):
    """Classe : CleanError
    Description : Le lot lu n'a pas la forme attendue de la couche brute.
    """


def to_measures(raw: pd.DataFrame) -> pd.DataFrame:
    """Méthode : to_measures
    Description : Projette un lot brut sur les colonnes de la couche, types
      forcés et qualité posée.
    """
    if not raw.empty:
        missing = {TIMESTAMP_COLUMN, SITE_COLUMN} - set(raw.columns)
        if missing:
            raise CleanError(
                f"Colonnes absentes du lot lu dans `mesure` : {sorted(missing)}."
            )
    frame = pd.DataFrame(index=raw.index, columns=list(MEASURE_COLUMNS))
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        raw.get(TIMESTAMP_COLUMN), utc=True, errors="coerce"
    )
    frame[SITE_COLUMN] = raw.get(SITE_COLUMN)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(raw.get(column), errors="coerce")
    frame["data_quality"] = _as_object(raw.get("data_quality"), raw.index)
    frame["null_reasons"] = _as_object(raw.get("null_reasons"), raw.index).map(
        _normalize_null_reasons
    )
    return qualify(frame, NUMERIC_COLUMNS)[list(MEASURE_COLUMNS)]


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : deduplicate
    Description : Ne garde qu'une mesure par site et par instant, la plus
      récente.
    """
    dated = frame.dropna(subset=[TIMESTAMP_COLUMN, SITE_COLUMN])
    return (
        dated.drop_duplicates(subset=[SITE_COLUMN, TIMESTAMP_COLUMN], keep="last")
        .sort_values([SITE_COLUMN, TIMESTAMP_COLUMN])
        .reset_index(drop=True)
    )


def _as_object(column: pd.Series | None, index: pd.Index) -> pd.Series:
    """Méthode : _as_object
    Description : Rend une colonne en type objet, même absente du lot lu.
    """
    if column is None:
        return pd.Series([None] * len(index), index=index, dtype="object")
    return column.astype("object")


def _normalize_null_reasons(value: object) -> list[str]:
    """Méthode : _normalize_null_reasons
    Description : Ramène les causes d'absence à une liste de chaînes.
    """
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return [str(item) for item in list(value)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]
