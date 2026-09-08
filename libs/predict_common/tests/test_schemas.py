from __future__ import annotations

import numpy
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from predict_common.schemas import (
    MEASURE_SCHEMA,
    NUMERIC_COLUMNS,
    feature_columns,
    features_arrow_schema,
    features_schema,
    is_sequence,
    lag_column,
    published_columns,
    rolling_column,
)

LAGS = (1, 24, 168)
WINDOW = 24


def raw_row(**overrides) -> pd.DataFrame:
    row = {
        "ts": pd.Timestamp("2026-09-02T08:00:00Z"),
        "site_id": "SITE001",
        **{name: 1.0 for name in NUMERIC_COLUMNS},
        "null_reasons": [],
        "data_quality": "good",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def features_row(**overrides) -> pd.DataFrame:
    row = {
        "ts": pd.Timestamp("2026-09-02T08:00:00Z"),
        "site_id": "SITE001",
        "consumption_kw": 50.0,
        "hour": 8,
        "day_of_week": 2,
        "is_weekend": 0,
        "temperature_celsius": 20.0,
        **{lag_column(hours): 50.0 for hours in LAGS},
        rolling_column(WINDOW): 50.0,
        "data_quality": "good",
        "imputed_ratio": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_raw_schema_accepts_a_complete_measure() -> None:
    assert len(MEASURE_SCHEMA.validate(raw_row())) == 1


def test_raw_schema_keeps_a_null_measure() -> None:
    row = raw_row(consumption_kw=None, data_quality="critical")
    validated = MEASURE_SCHEMA.validate(row)
    assert pd.isna(validated.loc[0, "consumption_kw"])


def test_raw_schema_refuses_a_measure_without_a_timestamp() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(ts=None), lazy=True)


def test_raw_schema_refuses_a_quality_outside_the_check() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(data_quality="ok"), lazy=True)


def test_raw_schema_refuses_a_site_id_the_base_would_truncate() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(site_id="S" * 21), lazy=True)


def test_raw_schema_accepts_a_numpy_array_of_motives() -> None:
    frame = raw_row()
    frame["null_reasons"] = [numpy.array(["sensor_failure"], dtype=object)]
    assert len(MEASURE_SCHEMA.validate(frame, lazy=True)) == 1


def test_raw_schema_refuses_a_lone_motive() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(null_reasons="sensor_failure"), lazy=True)


def test_raw_schema_tolerates_a_column_the_source_added() -> None:
    frame = raw_row()
    frame["nouveau_capteur"] = 1.0
    assert len(MEASURE_SCHEMA.validate(frame, lazy=True)) == 1


def test_measure_schema_refuses_a_duplicated_key() -> None:
    doubled = pd.concat([raw_row(), raw_row()], ignore_index=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(doubled, lazy=True)


def test_measure_schema_refuses_an_unqualified_measure() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(data_quality=None), lazy=True)


def test_measure_schema_refuses_a_power_factor_out_of_range() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(power_factor=1.5), lazy=True)


def test_measure_schema_refuses_an_impossible_humidity() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(humidity_percent=120.0), lazy=True)


def test_is_sequence_accepts_the_three_forms_of_a_motive_list() -> None:
    assert is_sequence([])
    assert is_sequence(("a",))
    assert is_sequence(numpy.array(["a"], dtype=object))
    assert not is_sequence("a")


def test_features_schema_accepts_a_complete_hour() -> None:
    schema = features_schema(LAGS, WINDOW)
    assert len(schema.validate(features_row())) == 1


def test_features_schema_refuses_a_missing_target() -> None:
    schema = features_schema(LAGS, WINDOW)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(features_row(consumption_kw=None), lazy=True)


def test_features_schema_accepts_a_site_without_a_thermometer() -> None:
    schema = features_schema(LAGS, WINDOW)
    assert len(schema.validate(features_row(temperature_celsius=None))) == 1


def test_features_schema_refuses_an_unknown_column() -> None:
    schema = features_schema(LAGS, WINDOW)
    frame = features_row()
    frame["surprise"] = 1.0
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(frame, lazy=True)


def test_features_schema_refuses_an_impossible_hour() -> None:
    schema = features_schema(LAGS, WINDOW)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(features_row(hour=24), lazy=True)


def test_features_schema_refuses_a_ratio_outside_zero_one() -> None:
    schema = features_schema(LAGS, WINDOW)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(features_row(imputed_ratio=1.5), lazy=True)


def test_features_schema_refuses_a_duplicated_hour() -> None:
    schema = features_schema(LAGS, WINDOW)
    doubled = pd.concat([features_row(), features_row()], ignore_index=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(doubled, lazy=True)


def test_feature_columns_puts_the_calendar_before_the_lags() -> None:
    assert feature_columns(LAGS, WINDOW) == (
        "hour",
        "day_of_week",
        "is_weekend",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "roll_mean_24h",
    )


def test_the_model_is_not_given_the_temperature() -> None:
    assert "temperature_celsius" not in feature_columns(LAGS, WINDOW)


def test_the_partition_keeps_the_temperature() -> None:
    published = published_columns(LAGS, WINDOW)
    assert "temperature_celsius" in published
    assert set(feature_columns(LAGS, WINDOW)) < set(published)


def test_published_columns_puts_the_calendar_before_the_lags() -> None:
    assert published_columns(LAGS, WINDOW) == (
        "hour",
        "day_of_week",
        "is_weekend",
        "temperature_celsius",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "roll_mean_24h",
    )


def test_changing_the_lags_changes_the_columns() -> None:
    assert feature_columns((1,), WINDOW) != feature_columns(LAGS, WINDOW)


def test_the_arrow_schema_carries_every_declared_column() -> None:
    arrow = features_arrow_schema(LAGS, WINDOW)
    assert set(arrow.names) == set(features_row().columns)
