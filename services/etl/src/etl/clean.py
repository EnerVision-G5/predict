"""Normalisation des mesures lues en base, avant qualification.

Les lignes arrivent déjà nommées et typées par la table : le renommage a eu
lieu une fois pour toutes chez le collecteur. Ce qui reste à faire ici est ce
que la base ne garantit pas — un `NUMERIC` relu par le driver n'est pas un
flottant, un `TEXT[]` n'est pas une liste Python, et deux lignes de la fenêtre
peuvent porter la même clé si le lot déborde.

Règle structurante héritée du contrat de la source : les valeurs manquantes ne
sont jamais filtrées. Une mesure nulle porte une information de panne capteur,
elle traverse cette étape telle quelle avec son `data_quality` et ses
`null_reasons`. L'imputation est un traitement aval, elle n'a pas sa place
ici : elle relève de `etl.impute`, qui écrit dans une colonne séparée sans
jamais toucher à la valeur brute normalisée ici.

La qualification, elle, appartient bien à cette étape : un lot qui sortirait
d'ici avec un null sans motif aurait déjà perdu la panne, et aucune étape aval
ne saurait la retrouver.
"""

from __future__ import annotations

import pandas as pd

from etl.quality import qualify
from predict_common.schemas import (
    NUMERIC_COLUMNS,
    SITE_COLUMN,
    TIMESTAMP_COLUMN,
)

# Colonnes de la table `mesure`, dans l'ordre du schéma figé v1.0. L'ordre
# n'est pas cosmétique : c'est celui dans lequel la sortie annexe vers
# TimescaleDB écrit ses lignes.
MEASURE_COLUMNS = (
    TIMESTAMP_COLUMN,
    SITE_COLUMN,
    *NUMERIC_COLUMNS,
    "null_reasons",
    "data_quality",
)


class CleanError(ValueError):
    """Le lot reçu ne porte pas les colonnes de la table `mesure`."""


def to_measures(raw: pd.DataFrame) -> pd.DataFrame:
    """Convertit un lot lu en base en tableau normalisé et qualifié.

    Le tableau retourné a toujours les colonnes de `MEASURE_COLUMNS`, même
    pour un lot vide : les étages suivants travaillent sans tester le cas
    dégénéré.
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
    """Ne garde qu'une ligne par clé naturelle (site_id, ts), la dernière.

    La clé primaire de `mesure` interdit déjà le doublon, mais l'étage reste :
    il protège l'écriture de retour, où `ON CONFLICT` arbitre entre le lot et
    la table et non à l'intérieur d'un même lot. Une projection qui
    dupliquerait une ligne ferait échouer l'insertion entière.
    """
    dated = frame.dropna(subset=[TIMESTAMP_COLUMN, SITE_COLUMN])
    return (
        dated.drop_duplicates(subset=[SITE_COLUMN, TIMESTAMP_COLUMN], keep="last")
        .sort_values([SITE_COLUMN, TIMESTAMP_COLUMN])
        .reset_index(drop=True)
    )


def _as_object(column: pd.Series | None, index: pd.Index) -> pd.Series:
    """Retourne la colonne demandée, ou une colonne vide de même longueur."""
    if column is None:
        return pd.Series([None] * len(index), index=index, dtype="object")
    return column.astype("object")


def _normalize_null_reasons(value: object) -> list[str]:
    """Ramène `null_reasons` au TEXT[] NOT NULL attendu par la base.

    Le driver rend un TEXT[] tantôt en liste, tantôt en tableau numpy selon le
    chemin de lecture : les deux disent la même chose, une seule doit traverser
    l'étage.
    """
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return [str(item) for item in list(value)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]
