"""Mise à l'écart des mesures inexploitables, et cause de chaque écart.

Une mesure dont la puissance est nulle et qu'aucune imputation n'a pu
reconstruire ne peut entrer dans aucun agrégat : la moyenne d'une panne reste
une panne. La supprimer serait pourtant perdre la panne elle-même, qui est une
information de terrain. Elle reste donc dans la couche brute, intacte, et
c'est ici qu'est portée la décision de l'écarter et la cause de cette
décision.

La séparation compte : ce module décide, il n'écrit pas. La liste qu'il
produit part vers la couche des variables — d'où les lignes écartées sont
absentes — et vers `mesure_exclu` quand la sortie annexe TimescaleDB est
active. Mélanger la décision et l'écriture rendrait l'une intestable sans
l'autre.

Une mesure imputée n'est jamais écartée : elle porte une valeur exploitable et
`imputation_method` dit d'où cette valeur vient. Écarter ce qu'on vient de
reconstruire n'aurait servi à rien.
"""

from __future__ import annotations

import pandas as pd

from etl.impute import IMPUTED_COLUMN, SOURCE_COLUMN
from predict_common.schemas import SITE_COLUMN, TIMESTAMP_COLUMN

# Cause retenue quand la source n'a fourni aucun motif. Elle ne prétend pas
# expliquer la panne, seulement dire ce que l'ETL a constaté.
DEFAULT_REASON = "valeur nulle non imputable"

EXCLUSION_COLUMNS = (SITE_COLUMN, TIMESTAMP_COLUMN, "raison")


def is_excluded(frame: pd.DataFrame) -> pd.Series:
    """Sélectionne les mesures sans valeur brute ni valeur reconstruite."""
    raw = pd.to_numeric(frame[SOURCE_COLUMN], errors="coerce")
    imputed = pd.to_numeric(frame[IMPUTED_COLUMN], errors="coerce")
    return raw.isna() & imputed.isna() & frame[TIMESTAMP_COLUMN].notna()


def exclusion_reason(null_reasons: object) -> str:
    """Reprend les motifs de la source, seule à savoir pourquoi le capteur s'est tu."""
    if isinstance(null_reasons, (list, tuple)) and len(null_reasons):
        return ", ".join(str(reason) for reason in null_reasons)
    if hasattr(null_reasons, "tolist"):
        return exclusion_reason(list(null_reasons))
    return DEFAULT_REASON


def to_exclusions(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Retourne les exclusions automatiques portées par un lot imputé."""
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
    """Retourne le lot privé des mesures qu'aucun agrégat ne peut utiliser.

    C'est ce tableau, et non le lot complet, qui alimente les variables. Un
    modèle entraîné sur des valeurs que la chaîne a jugées inexploitables
    apprendrait ces pannes comme s'il s'agissait de consommation.
    """
    if frame.empty:
        return frame
    return frame[~is_excluded(frame)].reset_index(drop=True)
