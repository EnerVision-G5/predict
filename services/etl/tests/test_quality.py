"""Qualification des mesures : aucune valeur nulle sans cause.

La garantie testée est celle du ticket EV-08 : une mesure nulle qui
traverserait l'ETL sans motif ni qualification aurait perdu la panne capteur,
et aucune étape aval ne saurait la reconstituer.
"""

from __future__ import annotations

import pandas as pd

from etl.clean import to_measures
from etl.quality import (
    DERIVED_REASON_SUFFIX,
    complete_null_reasons,
    derive_data_quality,
    missing_columns,
    qualify,
    worst,
)
from predict_common.schemas import (
    NUMERIC_COLUMNS,
    QUALITY_CRITICAL,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    QUALITY_SOURCE_COLUMN,
    QUALITY_SOURCE_ETL,
)


def test_derive_data_quality_qualifies_a_complete_measure() -> None:
    assert derive_data_quality(()) == QUALITY_GOOD


def test_derive_data_quality_treats_a_missing_target_as_critical() -> None:
    assert derive_data_quality(("consumption_kw",)) == QUALITY_CRITICAL


def test_derive_data_quality_degrades_beyond_three_silent_sensors() -> None:
    silent = ("voltage_v", "current_a", "humidity_percent")
    assert derive_data_quality(silent) == QUALITY_DEGRADED


def test_derive_data_quality_stays_partial_below_the_threshold() -> None:
    assert derive_data_quality(("voltage_v",)) == QUALITY_PARTIAL


def test_complete_null_reasons_names_every_unexplained_column() -> None:
    completed = complete_null_reasons([], ("voltage_v", "current_a"))
    assert completed == [
        f"voltage_v{DERIVED_REASON_SUFFIX}",
        f"current_a{DERIVED_REASON_SUFFIX}",
    ]


def test_complete_null_reasons_keeps_the_source_motives_alone() -> None:
    completed = complete_null_reasons(["network_loss"], ("voltage_v",))
    assert completed == ["network_loss"]


def test_missing_columns_names_the_silent_sensors_of_each_row(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw(
            [
                make_reading("2026-09-02T08:00:00Z"),
                make_reading("2026-09-02T08:01:00Z", voltage_v=None),
            ]
        )
    )
    assert missing_columns(frame, NUMERIC_COLUMNS) == [(), ("voltage_v",)]


def test_qualify_never_lets_a_null_pass_without_a_reason(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw([make_reading("2026-09-02T08:00:00Z", consumption_kw=None)])
    )
    assert frame.loc[0, "null_reasons"] == [f"consumption_kw{DERIVED_REASON_SUFFIX}"]
    assert frame.loc[0, "data_quality"] == QUALITY_CRITICAL


def test_qualify_keeps_a_source_alert_the_data_alone_would_not_show(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw(
            [
                make_reading(
                    "2026-09-02T08:00:00Z",
                    voltage_v=None,
                    data_quality=QUALITY_DEGRADED,
                )
            ]
        )
    )
    assert frame.loc[0, "data_quality"] == QUALITY_DEGRADED


def test_qualify_refuses_a_qualification_kinder_than_the_data(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw(
            [
                make_reading(
                    "2026-09-02T08:00:00Z",
                    consumption_kw=None,
                    data_quality=QUALITY_GOOD,
                )
            ]
        )
    )
    assert frame.loc[0, "data_quality"] == QUALITY_CRITICAL


def test_qualify_recomputes_a_qualification_the_base_would_reject(
    make_raw, make_reading
) -> None:
    frame = to_measures(
        make_raw([make_reading("2026-09-02T08:00:00Z", data_quality="ok")])
    )
    assert frame.loc[0, "data_quality"] == QUALITY_GOOD


def test_qualify_leaves_a_complete_measure_untouched(make_raw, make_reading) -> None:
    frame = to_measures(make_raw([make_reading("2026-09-02T08:00:00Z")]))
    assert frame.loc[0, "null_reasons"] == []
    assert frame.loc[0, "data_quality"] == QUALITY_GOOD


def test_qualify_accepts_an_empty_batch(make_raw) -> None:
    assert qualify(to_measures(make_raw([])), NUMERIC_COLUMNS).empty


def test_qualify_counts_an_unreadable_value_as_a_silent_sensor(
    make_raw, make_reading
) -> None:
    raw = make_raw([make_reading("2026-09-02T08:00:00Z", voltage_v="n/a")])
    frame = to_measures(raw)
    assert pd.isna(frame.loc[0, "voltage_v"])
    assert frame.loc[0, "null_reasons"] == [f"voltage_v{DERIVED_REASON_SUFFIX}"]


class TestWorst:
    """L'agrégation horaire retient la pire qualification, pas leur moyenne."""

    def test_a_critical_minute_makes_the_hour_critical(self) -> None:
        assert worst([QUALITY_GOOD, QUALITY_CRITICAL, QUALITY_GOOD]) == QUALITY_CRITICAL

    def test_an_hour_of_healthy_minutes_stays_good(self) -> None:
        assert worst([QUALITY_GOOD, QUALITY_GOOD]) == QUALITY_GOOD

    def test_an_unknown_value_is_ignored(self) -> None:
        assert worst(["ok", QUALITY_PARTIAL]) == QUALITY_PARTIAL

    def test_an_hour_without_any_known_value_is_critical(self) -> None:
        assert worst([]) == QUALITY_CRITICAL


class TestQualitySource:
    """La marque qui distingue un `good` posé d'un `good` confirmé.

    `data_quality` est NOT NULL DEFAULT 'good' : le collecteur retombe sur le
    défaut quand la source se tait. Sans cette marque, l'API métier compterait
    0 % de mesures dégradées sur une journée que l'ETL n'a pas encore vue, et
    le site paraîtrait parfait — l'inverse de ce que l'indicateur doit dire.
    """

    def test_a_qualified_batch_is_signed_by_the_etl(
        self, make_raw, make_reading
    ) -> None:
        frame = to_measures(make_raw([make_reading("2026-09-02T08:00:00Z")]))
        assert frame.loc[0, QUALITY_SOURCE_COLUMN] == QUALITY_SOURCE_ETL

    def test_an_empty_batch_still_carries_the_column(self, make_raw) -> None:
        assert QUALITY_SOURCE_COLUMN in to_measures(make_raw([])).columns
