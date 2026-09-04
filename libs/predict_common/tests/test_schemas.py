"""Schémas des couches : ce qu'un artefact doit contenir pour être lisible.

Un schéma est un contrat entre deux services qui ne se connaissent pas. Les
tests fixent donc ce qu'il accepte autant que ce qu'il refuse : un schéma trop
permissif laisserait passer une partition fausse, un schéma trop strict
refuserait une partition que le producteur vient d'écrire valide — et ce
second cas est le pire, parce qu'il décrédibilise le contrôle.
"""

from __future__ import annotations

import numpy
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from predict_common.schemas import (
    MEASURE_SCHEMA,
    MODEL_DERIVED_COLUMNS,
    NUMERIC_COLUMNS,
    add_derived_calendar,
    cyclic_hour,
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
    """Construit une ligne de `mesure` conforme, avant surcharge."""
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
    """Construit une ligne de variables conforme, avant surcharge."""
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
    # Une valeur nulle est un capteur qui dit qu'il est tombé : la refuser à
    # l'entrée perdrait la panne, qui est l'information à conserver.
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
    # C'est sous cette forme que parquet rend une colonne list<string> :
    # n'accepter que la liste refuserait ce que le producteur a écrit valide.
    frame = raw_row()
    frame["null_reasons"] = [numpy.array(["sensor_failure"], dtype=object)]
    assert len(MEASURE_SCHEMA.validate(frame, lazy=True)) == 1


def test_raw_schema_refuses_a_lone_motive() -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(null_reasons="sensor_failure"), lazy=True)


def test_raw_schema_tolerates_a_column_the_source_added() -> None:
    # Le collecteur projette ce que la table accepte ; une colonne de plus
    # dans le lot lu n'est pas une rupture de contrat.
    frame = raw_row()
    frame["nouveau_capteur"] = 1.0
    assert len(MEASURE_SCHEMA.validate(frame, lazy=True)) == 1


def test_measure_schema_refuses_a_duplicated_key() -> None:
    # (site_id, ts) est la clé primaire de `mesure` : deux fois la même ferait
    # échouer l'insertion du lot entier, et pas seulement de sa ligne.
    doubled = pd.concat([raw_row(), raw_row()], ignore_index=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(doubled, lazy=True)


def test_measure_schema_refuses_an_unqualified_measure() -> None:
    # `NOT NULL DEFAULT 'good'` en base : le schéma figé ne sait pas dire
    # « non qualifiée ». C'est le collecteur qui retombe sur le défaut, et
    # l'ETL qui repose la qualification à son passage.
    with pytest.raises((SchemaError, SchemaErrors)):
        MEASURE_SCHEMA.validate(raw_row(data_quality=None), lazy=True)


def test_measure_schema_refuses_a_power_factor_out_of_range() -> None:
    # CHECK (power_factor BETWEEN 0 AND 1) en base : détectée ici, la valeur
    # aberrante nomme sa ligne ; laissée à PostgreSQL, elle fait échouer le lot.
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
    # Une heure sans consommation exploitable n'est pas une ligne
    # d'apprentissage dégradée : elle n'a pas sa place dans la couche.
    schema = features_schema(LAGS, WINDOW)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(features_row(consumption_kw=None), lazy=True)


def test_features_schema_accepts_a_site_without_a_thermometer() -> None:
    # XGBoost gère nativement l'absence : exiger la température viderait la
    # partition d'un site parfaitement exploitable.
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
    # (site_id, ts) est la clé naturelle : deux fois la même heure donnerait
    # deux cibles pour le même instant.
    schema = features_schema(LAGS, WINDOW)
    doubled = pd.concat([features_row(), features_row()], ignore_index=True)
    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(doubled, lazy=True)


def test_feature_columns_puts_the_calendar_before_the_lags() -> None:
    # L'ordre est celui de la signature MLflow : le service d'inférence le
    # déduit du même appel, ce qui empêche les deux côtés de diverger. Les
    # variables dérivées suivent celle dont elles sortent.
    assert feature_columns(LAGS, WINDOW) == (
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "roll_mean_24h",
    )


def test_the_model_is_not_given_the_temperature() -> None:
    # Le service ne connaît pas la météo des heures qu'il prédit : il la
    # présenterait vide à chaque requête, et le modèle aurait appris des
    # séparations qu'il ne pourrait plus emprunter.
    assert "temperature_celsius" not in feature_columns(LAGS, WINDOW)


def test_the_partition_keeps_the_temperature() -> None:
    # Elle reste une mesure réelle, et servira le jour où une prévision météo
    # alimentera l'inférence. La sortir de la partition changerait le contrat
    # de la couche, donc imposerait une feature_version.
    published = published_columns(LAGS, WINDOW)
    assert "temperature_celsius" in published
    assert "temperature_celsius" not in feature_columns(LAGS, WINDOW)


def test_the_derived_columns_are_never_published() -> None:
    # C'est ce qui dispense d'une feature_version : elles ne sont écrites dans
    # aucune partition, elles sont reconstruites à chaque lecture depuis une
    # colonne qui, elle, est publiée.
    published = published_columns(LAGS, WINDOW)
    for name in MODEL_DERIVED_COLUMNS:
        assert name in feature_columns(LAGS, WINDOW)
        assert name not in published
    assert "hour" in published


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
    # C'est pour cela qu'un tel changement s'accompagne d'une feature_version.
    assert feature_columns((1,), WINDOW) != feature_columns(LAGS, WINDOW)


def test_the_hour_encoding_closes_the_circle() -> None:
    # La raison d'être de l'encodage : minuit est le tour suivant de la même
    # heure, pas un point situé vingt-trois unités plus loin.
    assert cyclic_hour(24) == pytest.approx(cyclic_hour(0))


def test_midnight_is_the_near_neighbour_of_eleven_pm() -> None:
    # Sur l'échelle entière, 23 et 0 sont les deux extrémités. Sur le cercle,
    # ils sont aussi proches que 0 et 1 — c'est ce que le modèle doit voir.
    def distance(left: int, right: int) -> float:
        left_sin, left_cos = cyclic_hour(left)
        right_sin, right_cos = cyclic_hour(right)
        return float((left_sin - right_sin) ** 2 + (left_cos - right_cos) ** 2)

    assert distance(23, 0) == pytest.approx(distance(0, 1))


def test_the_two_components_tell_the_hours_apart() -> None:
    # Le sinus seul confondrait deux heures symétriques de la journée : c'est
    # la raison pour laquelle il en faut deux et non une.
    hours = range(24)
    assert len({cyclic_hour(hour) for hour in hours}) == len(hours)


def test_add_derived_calendar_reads_the_published_hour() -> None:
    frame = pd.DataFrame({"hour": [0, 6, 23]})
    enriched = add_derived_calendar(frame)
    expected_sin, expected_cos = cyclic_hour(frame["hour"])
    assert list(enriched["hour_sin"]) == pytest.approx(list(expected_sin))
    assert list(enriched["hour_cos"]) == pytest.approx(list(expected_cos))
    # Le lot reçu n'est pas modifié : l'appelant garde ce qu'il a lu.
    assert "hour_sin" not in frame.columns


def test_add_derived_calendar_leaves_a_lot_without_the_hour_alone() -> None:
    # Nommer la colonne manquante est le métier du contrôle en aval, qui sait
    # quelles colonnes l'appelant a demandées.
    frame = pd.DataFrame({"lag_1h": [1.0]})
    assert list(add_derived_calendar(frame).columns) == ["lag_1h"]


def test_the_arrow_schema_carries_every_declared_column() -> None:
    # Seule la couche des variables est un fichier : la couche brute est une
    # table, dont les types sont ceux de PostgreSQL.
    arrow = features_arrow_schema(LAGS, WINDOW)
    assert set(arrow.names) == set(features_row().columns)
