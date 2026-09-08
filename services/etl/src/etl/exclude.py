# **********************************************************************
# * Nom     : exclude.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Mise à l'écart des mesures qu'aucune reconstruction ne   *
# *   rattrape                                                         *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import pandas as pd

from etl.impute import IMPUTED_COLUMN, SOURCE_COLUMN
from predict_common.schemas import SITE_COLUMN, TIMESTAMP_COLUMN

# Cause retenue quand la source n'en déclare aucune.
DEFAULT_REASON = "valeur nulle non imputable"

# Colonnes écrites dans mesure_exclu.
EXCLUSION_COLUMNS = (SITE_COLUMN, TIMESTAMP_COLUMN, "raison")


def is_excluded(frame: pd.DataFrame) -> pd.Series:
    """Méthode : is_excluded
    Description : Marque les heures sans valeur brute ni valeur reconstruite.
    """
    raw = pd.to_numeric(frame[SOURCE_COLUMN], errors="coerce")
    imputed = pd.to_numeric(frame[IMPUTED_COLUMN], errors="coerce")
    return raw.isna() & imputed.isna() & frame[TIMESTAMP_COLUMN].notna()


def exclusion_reason(null_reasons: object) -> str:
    """Méthode : exclusion_reason
    Description : Compose la cause lisible d'une exclusion.
    """
    if isinstance(null_reasons, (list, tuple)) and len(null_reasons):
        return ", ".join(str(reason) for reason in null_reasons)
    if hasattr(null_reasons, "tolist"):
        return exclusion_reason(list(null_reasons))
    return DEFAULT_REASON


def to_exclusions(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Méthode : to_exclusions
    Description : Transforme les heures écartées en lignes de mesure_exclu.
    """
    if frame.empty:
        return []
    excluded = frame[is_excluded(frame)]
    return [
        {
            SITE_COLUMN: row.site_id,
            TIMESTAMP_COLUMN: row.ts.to_pydatetime(),
            "raison": exclusion_reason(row.null_reasons),
        }
        for row in excluded.itertuples(index=False)
    ]


def keep_usable(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : keep_usable
    Description : Ne garde que les heures qu'un modèle peut apprendre.
    """
    if frame.empty:
        return frame
    return frame[~is_excluded(frame)].reset_index(drop=True)
