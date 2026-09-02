"""Imputation de la puissance dans une colonne séparée de la valeur brute.

La garantie testée est double : la valeur brute ressort intacte de l'étage,
et la valeur reconstruite dit toujours par quelle méthode elle a été obtenue.
"""

import pandas as pd

from etl.impute import (
    IMPUTED_COLUMN,
    METHOD_COLUMN,
    METHOD_INTERPOLATION,
    METHOD_LOCF,
    METHOD_NONE,
    SOURCE_COLUMN,
    impute_frame,
)
from etl.transform import deduplicate, to_frame


def series(make_reading, values, site_id="SITE001"):
    """Construit un lot horaire d'un site à partir de ses puissances."""
    readings = [
        make_reading(
            f"2026-01-15T{hour:02d}:00:00Z",
            site_id=site_id,
            consumption_kw=value,
        )
        for hour, value in enumerate(values)
    ]
    return deduplicate(to_frame(readings))


def test_impute_frame_never_touches_the_raw_value(make_reading) -> None:
    frame = series(make_reading, [10.0, None, 30.0])
    imputed = impute_frame(frame)
    assert imputed.loc[1, IMPUTED_COLUMN] == 20.0
    assert pd.isna(imputed.loc[1, SOURCE_COLUMN])


def test_impute_frame_leaves_the_input_batch_alone(make_reading) -> None:
    frame = series(make_reading, [10.0, None, 30.0])
    impute_frame(frame)
    assert IMPUTED_COLUMN not in frame.columns


def test_impute_frame_interpolates_a_framed_gap(make_reading) -> None:
    imputed = impute_frame(series(make_reading, [10.0, None, 30.0]))
    assert imputed.loc[1, METHOD_COLUMN] == METHOD_INTERPOLATION
    assert imputed.loc[1, IMPUTED_COLUMN] == 20.0


def test_impute_frame_weighs_the_interpolation_on_time(make_reading) -> None:
    # Trou de deux heures : la valeur reconstruite doit tomber au tiers et aux
    # deux tiers, pas à la moitié comme si les mesures étaient contiguës.
    imputed = impute_frame(series(make_reading, [10.0, None, None, 40.0]))
    assert imputed.loc[1, IMPUTED_COLUMN] == 20.0
    assert imputed.loc[2, IMPUTED_COLUMN] == 30.0


def test_impute_frame_carries_the_last_known_value_forward(
    make_reading,
) -> None:
    imputed = impute_frame(series(make_reading, [10.0, 20.0, None]))
    assert imputed.loc[2, METHOD_COLUMN] == METHOD_LOCF
    assert imputed.loc[2, IMPUTED_COLUMN] == 20.0


def test_impute_frame_invents_nothing_without_a_past(make_reading) -> None:
    imputed = impute_frame(series(make_reading, [None, None]))
    assert list(imputed[METHOD_COLUMN]) == [METHOD_NONE, METHOD_NONE]
    assert imputed[IMPUTED_COLUMN].isna().all()


def test_impute_frame_marks_an_untouched_measure_as_none(make_reading) -> None:
    imputed = impute_frame(series(make_reading, [10.0]))
    assert imputed.loc[0, METHOD_COLUMN] == METHOD_NONE
    assert imputed.loc[0, IMPUTED_COLUMN] == 10.0


def test_impute_frame_never_borrows_from_another_site(make_reading) -> None:
    frame = pd.concat(
        [
            series(make_reading, [10.0, 20.0], site_id="SITE001"),
            series(make_reading, [None, None], site_id="SITE002"),
        ],
        ignore_index=True,
    )
    imputed = impute_frame(frame)
    absent = imputed[imputed["site_id"] == "SITE002"]
    assert absent[IMPUTED_COLUMN].isna().all()


def test_impute_frame_adds_its_columns_to_an_empty_batch() -> None:
    imputed = impute_frame(to_frame([]))
    assert imputed.empty
    assert IMPUTED_COLUMN in imputed.columns
    assert METHOD_COLUMN in imputed.columns


def test_impute_frame_ignores_a_row_without_a_usable_timestamp(
    make_reading,
) -> None:
    frame = to_frame(
        [
            make_reading("2026-01-15T08:00:00Z", consumption_kw=10.0),
            make_reading("pas une date", consumption_kw=None),
        ]
    )
    imputed = impute_frame(frame)
    assert imputed.loc[1, METHOD_COLUMN] == METHOD_NONE
    assert pd.isna(imputed.loc[1, IMPUTED_COLUMN])
