# **********************************************************************
# * Nom     : impute.py                                                *
# * Type    : Module                                                   *
# * Sujet   : Reconstruction des consommations absentes, sans toucher  *
# *   à la valeur brute                                                *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import pandas as pd

from predict_common.schemas import (
    METHOD_INTERPOLATION,
    METHOD_LOCF,
    METHOD_NONE,
    SITE_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)

# Colonne brute lue, jamais réécrite par cet étage.
SOURCE_COLUMN = TARGET_COLUMN
# Colonne portant la meilleure valeur exploitable.
IMPUTED_COLUMN = "consumption_kw_imputed"
# Colonne disant comment la valeur a été reconstruite.
METHOD_COLUMN = "imputation_method"

# Les deux colonnes que cet étage ajoute au lot.
IMPUTATION_COLUMNS = (IMPUTED_COLUMN, METHOD_COLUMN)


def impute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : impute_frame
    Description : Reconstruit les valeurs absentes, par interpolation quand
      c'est possible, par report sinon.
    """
    imputed = frame.copy()
    raw = pd.to_numeric(imputed[SOURCE_COLUMN], errors="coerce")
    imputed[IMPUTED_COLUMN] = raw
    imputed[METHOD_COLUMN] = METHOD_NONE
    if imputed.empty:
        return imputed
    interpolated, carried = _candidates(imputed, raw)
    eligible = raw.isna() & _is_imputable(imputed)
    _fill(imputed, eligible & interpolated.notna(), interpolated, METHOD_INTERPOLATION)
    _fill(
        imputed,
        eligible & interpolated.isna() & carried.notna(),
        carried,
        METHOD_LOCF,
    )
    return imputed


def _is_imputable(frame: pd.DataFrame) -> pd.Series:
    """Méthode : _is_imputable
    Description : Écarte les lignes sans horodatage, qu'aucune méthode ne sait
      placer.
    """
    return frame[TIMESTAMP_COLUMN].notna()


def _candidates(
    frame: pd.DataFrame,
    raw: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Méthode : _candidates
    Description : Calcule, site par site, la valeur interpolée et la valeur
      reportée.
    """
    interpolated = pd.Series(float("nan"), index=frame.index, dtype="float64")
    carried = interpolated.copy()
    dated = frame[frame[TIMESTAMP_COLUMN].notna()]
    for _, group in dated.groupby(SITE_COLUMN, sort=False):
        ordered = group.sort_values(TIMESTAMP_COLUMN)
        values = pd.Series(
            raw.loc[ordered.index].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(ordered[TIMESTAMP_COLUMN]),
        )
        interpolated.loc[ordered.index] = values.interpolate(
            method="time", limit_area="inside"
        ).to_numpy()
        carried.loc[ordered.index] = values.ffill().to_numpy()
    return interpolated, carried


def _fill(
    frame: pd.DataFrame,
    selected: pd.Series,
    values: pd.Series,
    method: str,
) -> None:
    """Méthode : _fill
    Description : Pose une série de valeurs reconstruites et la méthode qui les
      a produites.
    """
    if not selected.any():
        return
    frame.loc[selected, IMPUTED_COLUMN] = values[selected]
    frame.loc[selected, METHOD_COLUMN] = method
