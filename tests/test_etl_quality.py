"""Qualification des mesures : aucune valeur nulle sans cause.

La garantie testée ici est celle du ticket : une mesure nulle qui atteindrait
la base sans motif ni qualification aurait perdu la panne capteur, et aucune
étape aval ne saurait la reconstituer.
"""

import pandas as pd

from etl.quality import (
    DERIVED_REASON_SUFFIX,
    QUALITY_CRITICAL,
    QUALITY_DEGRADED,
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    complete_null_reasons,
    derive_data_quality,
    missing_columns,
    qualify,
)
from etl.transform import NUMERIC_COLUMNS, to_frame


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
    make_reading,
) -> None:
    frame = to_frame(
        [
            make_reading("2026-01-15T08:00:00Z"),
            make_reading("2026-01-15T08:01:00Z", voltage_v=None),
        ]
    )
    assert missing_columns(frame, NUMERIC_COLUMNS) == [(), ("voltage_v",)]


def test_qualify_never_lets_a_null_pass_without_a_reason(make_reading) -> None:
    frame = to_frame(
        [make_reading("2026-01-15T08:00:00Z", consumption_kw=None)]
    )
    assert frame.loc[0, "null_reasons"] == [
        f"consumption_kw{DERIVED_REASON_SUFFIX}"
    ]
    assert frame.loc[0, "data_quality"] == QUALITY_CRITICAL


def test_qualify_keeps_a_source_alert_the_data_alone_would_not_show(
    make_reading,
) -> None:
    # Un seul capteur muet ferait déduire 'partial' ; la source en sait plus.
    frame = to_frame(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                voltage_v=None,
                data_quality=QUALITY_DEGRADED,
            )
        ]
    )
    assert frame.loc[0, "data_quality"] == QUALITY_DEGRADED


def test_qualify_refuses_a_qualification_kinder_than_the_data(
    make_reading,
) -> None:
    # La source annonce 'good' en n'envoyant pas de puissance : la garder
    # sortirait la panne de idx_mesure_quality, qui n'indexe que le non-'good'.
    frame = to_frame(
        [
            make_reading(
                "2026-01-15T08:00:00Z",
                consumption_kw=None,
                data_quality=QUALITY_GOOD,
            )
        ]
    )
    assert frame.loc[0, "data_quality"] == QUALITY_CRITICAL


def test_qualify_recomputes_a_qualification_the_base_would_reject(
    make_reading,
) -> None:
    # 'ok' n'est pas dans le CHECK de la colonne : le soumettre tel quel
    # ferait échouer l'insertion du lot entier, pas seulement de la ligne.
    frame = to_frame([make_reading("2026-01-15T08:00:00Z", data_quality="ok")])
    assert frame.loc[0, "data_quality"] == QUALITY_GOOD


def test_qualify_leaves_a_complete_measure_untouched(make_reading) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z")])
    assert frame.loc[0, "null_reasons"] == []
    assert frame.loc[0, "data_quality"] == QUALITY_GOOD


def test_qualify_accepts_an_empty_batch() -> None:
    frame = qualify(to_frame([]), NUMERIC_COLUMNS)
    assert frame.empty


def test_qualify_counts_an_unreadable_value_as_a_silent_sensor(
    make_reading,
) -> None:
    frame = to_frame([make_reading("2026-01-15T08:00:00Z", voltage_v="n/a")])
    assert pd.isna(frame.loc[0, "voltage_v"])
    assert frame.loc[0, "null_reasons"] == [
        f"voltage_v{DERIVED_REASON_SUFFIX}"
    ]
