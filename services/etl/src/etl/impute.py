"""Imputation de la puissance, dans une colonne à part de la valeur brute.

Le contrat gelé entre les équipes ne laisse aucune place à l'interprétation :
`consumption_kw` reste la valeur de la source, `consumption_kw_imputed` porte
celle que l'ETL a calculée, et `imputation_method` dit laquelle des deux on
lit. Écraser la valeur brute par une valeur imputée effacerait la panne
capteur, qui est précisément l'information à conserver.

`consumption_kw_imputed` porte toujours la meilleure valeur exploitable : la
valeur brute quand elle existe, la valeur reconstruite sinon. Un agrégat en
aval n'a donc qu'une colonne à lire, et `imputation_method` lui dit à quoi
s'en tenir. `none` couvre les deux cas où rien n'a été inventé — la mesure
brute est exploitable, ou rien ne permettait de la reconstruire — que la
nullité de la colonne imputée sépare.

Deux méthodes, dans cet ordre. Une valeur encadrée par deux voisines connues
est interpolée sur le temps. Une valeur qui n'a qu'un passé est reportée
(last observation carried forward). Une valeur sans passé ni futur dans le lot
n'est pas inventée : elle reste nulle, et `etl.exclude` écartera la mesure.

L'imputation ne consulte pas `data_quality` et ne le modifie pas : la panne
reste écrite dans la mesure, quelle que soit la qualité de la reconstruction.
Un agrégat qui refuse les mesures dégradées le fait sur `data_quality`, un
agrégat qui refuse les valeurs reconstruites le fait sur `imputation_method` ;
mélanger les deux critères dans une seule colonne priverait l'un des deux.
"""

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

SOURCE_COLUMN = TARGET_COLUMN
IMPUTED_COLUMN = "consumption_kw_imputed"
METHOD_COLUMN = "imputation_method"

IMPUTATION_COLUMNS = (IMPUTED_COLUMN, METHOD_COLUMN)


def impute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la valeur imputée et sa méthode, sans toucher à la valeur brute.

    Le tableau d'entrée n'est pas modifié : le lot brut reste disponible tel
    qu'il est sorti de la normalisation, y compris pour un test qui voudrait
    comparer les deux.
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
    """Écarte les mesures qui n'ont pas de place dans la série temporelle.

    Sans horodatage exploitable, une mesure n'a ni passé ni futur : lui donner
    une valeur reviendrait à la placer au hasard dans la série.
    """
    return frame[TIMESTAMP_COLUMN].notna()


def _candidates(
    frame: pd.DataFrame,
    raw: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Calcule, par site, les deux valeurs de remplacement possibles.

    Chaque site est traité seul : reporter la valeur d'un site sur un autre
    n'aurait aucun sens, ils ne mesurent pas la même installation. Les séries
    sont indexées sur le temps et non sur les positions, pour qu'un trou dans
    la série ne fasse pas interpoler comme si les mesures étaient contiguës.
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
    """Reporte les valeurs retenues dans la colonne imputée et sa méthode."""
    if not selected.any():
        return
    frame.loc[selected, IMPUTED_COLUMN] = values[selected]
    frame.loc[selected, METHOD_COLUMN] = method
