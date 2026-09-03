"""Panne capteur simulée : la mesure est rangée, jamais perdue (EV-08).

Ce fichier est le test d'acceptation du ticket, et il survit à la découpe en
services parce que la garantie, elle, n'a pas changé. Il ne teste aucun étage
en particulier : il rejoue une panne de capteur de bout en bout, de la mesure
servie par la source jusqu'aux lignes soumises à la base, et vérifie qu'à
aucun moment l'information de panne n'a été écrasée.

Trois pannes, parce qu'elles ne se rangent pas au même endroit. Une coupure au
milieu d'une série est reconstruite et garde sa valeur brute nulle. Une
coupure en fin de série est reportée depuis la dernière valeur connue. Une
coupure sans passé exploitable n'est pas inventée : la mesure reste brute en
base et part dans `mesure_exclu` avec sa cause.
"""

from __future__ import annotations

import pandas as pd

from etl.clean import deduplicate, to_measures
from etl.exclude import to_exclusions
from etl.impute import IMPUTED_COLUMN, METHOD_COLUMN, impute_frame
from etl.load import to_records
from predict_common.schemas import (
    METHOD_INTERPOLATION,
    METHOD_LOCF,
    METHOD_NONE,
    QUALITY_CRITICAL,
)

OUTAGE_REASON = "sensor_failure"


def outage(make_reading, hour):
    """Mesure d'une panne capteur telle que la source la sert.

    La source annonce `good` tout en n'envoyant pas de puissance : c'est le
    cas qui piège, et celui que le ticket demande de ne pas laisser passer.
    """
    return make_reading(
        f"2026-09-02T{hour:02d}:00:00Z",
        consumption_kw=None,
        null_reasons=[OUTAGE_REASON],
        data_quality="good",
    )


def healthy(make_reading, hour, consumption_kw):
    """Mesure nominale d'un capteur en état de marche."""
    return make_reading(f"2026-09-02T{hour:02d}:00:00Z", consumption_kw=consumption_kw)


def ingest(make_raw, readings):
    """Rejoue la chaîne de transformation complète sur un lot de mesures."""
    return impute_frame(deduplicate(to_measures(make_raw(readings))))


def test_an_outage_inside_a_series_keeps_its_raw_null(make_raw, make_reading) -> None:
    frame = ingest(
        make_raw,
        [
            healthy(make_reading, 8, 100.0),
            outage(make_reading, 9),
            healthy(make_reading, 10, 200.0),
        ],
    )
    row = frame.loc[1]
    assert pd.isna(row["consumption_kw"])
    assert row["null_reasons"] == [OUTAGE_REASON]
    assert row["data_quality"] == QUALITY_CRITICAL
    assert row[IMPUTED_COLUMN] == 150.0
    assert row[METHOD_COLUMN] == METHOD_INTERPOLATION


def test_an_outage_inside_a_series_is_not_excluded(make_raw, make_reading) -> None:
    # La mesure porte une valeur exploitable et dit d'où elle vient : l'écarter
    # des agrégats reviendrait à jeter ce qu'on vient de reconstruire.
    frame = ingest(
        make_raw,
        [
            healthy(make_reading, 8, 100.0),
            outage(make_reading, 9),
            healthy(make_reading, 10, 200.0),
        ],
    )
    assert to_exclusions(frame) == []


def test_an_outage_ending_a_series_carries_the_last_value_forward(
    make_raw, make_reading
) -> None:
    frame = ingest(make_raw, [healthy(make_reading, 8, 100.0), outage(make_reading, 9)])
    assert frame.loc[1, IMPUTED_COLUMN] == 100.0
    assert frame.loc[1, METHOD_COLUMN] == METHOD_LOCF
    assert to_exclusions(frame) == []


def test_an_outage_without_a_past_is_filed_with_its_cause(
    make_raw, make_reading
) -> None:
    frame = ingest(make_raw, [outage(make_reading, 8)])
    assert pd.isna(frame.loc[0, IMPUTED_COLUMN])
    assert frame.loc[0, METHOD_COLUMN] == METHOD_NONE

    exclusions = to_exclusions(frame)
    assert len(exclusions) == 1
    assert exclusions[0]["site_id"] == "SITE001"
    assert exclusions[0]["raison"] == OUTAGE_REASON


def test_an_unfiled_outage_still_reaches_the_base_raw(make_raw, make_reading) -> None:
    # L'exclusion range la mesure, elle ne la remplace pas : la ligne brute est
    # chargée dans `mesure` avec sa cause, panne comprise.
    frame = ingest(make_raw, [outage(make_reading, 8)])
    record = to_records(frame)[0]
    assert record["consumption_kw"] is None
    assert record["null_reasons"] == [OUTAGE_REASON]
    assert record["data_quality"] == QUALITY_CRITICAL
    assert record[IMPUTED_COLUMN] is None
    assert record[METHOD_COLUMN] == METHOD_NONE


def test_a_prolonged_outage_keeps_every_raw_null(make_raw, make_reading) -> None:
    frame = ingest(
        make_raw,
        [
            healthy(make_reading, 8, 100.0),
            *(outage(make_reading, hour) for hour in (9, 10, 11)),
            healthy(make_reading, 12, 200.0),
        ],
    )
    outages = frame.loc[1:3]
    assert outages["consumption_kw"].isna().all()
    assert (outages[METHOD_COLUMN] == METHOD_INTERPOLATION).all()
    assert list(outages[IMPUTED_COLUMN]) == [125.0, 150.0, 175.0]
    assert (outages["data_quality"] == QUALITY_CRITICAL).all()


def test_a_healthy_series_carries_no_trace_of_imputation(
    make_raw, make_reading
) -> None:
    frame = ingest(make_raw, [healthy(make_reading, hour, 100.0) for hour in (8, 9)])
    assert (frame[METHOD_COLUMN] == METHOD_NONE).all()
    assert list(frame["null_reasons"]) == [[], []]
    assert to_exclusions(frame) == []
