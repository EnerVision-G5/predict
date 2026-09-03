"""Vérification des contrats de couche, à la lecture comme à l'écriture.

Ce qui est testé n'est pas pandera, mais la position du contrôle. Une
ligne de `mesure` cassée doit faire échouer l'ETL au moment où il la lit, avec
un message qui nomme la colonne fautive — pas trois étapes plus loin, quand un
modèle entraîné dessus prédira n'importe quoi et que plus rien ne remontera à
la cause.
"""

from __future__ import annotations

import pandas as pd
import pytest

from etl.features import FeatureSpec, build
from etl.validate import ContractError, check_features, check_measures
from predict_common.schemas import (
    METHOD_NONE,
    QUALITY_GOOD,
    lag_column,
    rolling_column,
)

SPEC = FeatureSpec(
    version="v1", resample_rule="1h", lag_hours=(1,), rolling_window_h=2
)


def conforming_features() -> pd.DataFrame:
    """Produit une partition de variables réellement issue de l'ETL."""
    stamps = pd.date_range("2026-09-01T00:00:00Z", periods=72, freq="h", tz="UTC")
    measures = pd.DataFrame(
        {
            "ts": stamps,
            "site_id": "SITE001",
            "consumption_kw": [50.0 + index % 12 for index in range(72)],
            "consumption_kw_imputed": [50.0 + index % 12 for index in range(72)],
            "temperature_celsius": 20.0,
            "data_quality": QUALITY_GOOD,
            "imputation_method": METHOD_NONE,
        }
    )
    return build(measures, SPEC, stamps[-1].date())


def test_check_measures_accepts_a_conforming_batch(make_raw, make_reading) -> None:
    assert len(check_measures(make_raw([make_reading("2026-09-02T08:00:00Z")]))) == 1


def test_check_measures_refuses_a_broken_contract(make_raw, make_reading) -> None:
    raw = make_raw([make_reading("2026-09-02T08:00:00Z", data_quality="excellent")])
    with pytest.raises(ContractError):
        check_measures(raw)


def test_the_failure_names_the_faulty_column(make_raw, make_reading) -> None:
    # Un message qui ne dirait que « schéma invalide » obligerait à rejouer le
    # lot à la main pour trouver la colonne.
    raw = make_raw([make_reading("2026-09-02T08:00:00Z", data_quality="excellent")])
    with pytest.raises(ContractError, match="data_quality"):
        check_measures(raw)


def test_the_failure_counts_every_violation(make_raw, make_reading) -> None:
    # `lazy=True` rassemble tout le lot : réparer un schéma une colonne par
    # exécution serait une perte de temps pure.
    raw = make_raw(
        [
            make_reading(f"2026-09-02T0{hour}:00:00Z", data_quality="ok")
            for hour in range(3)
        ]
    )
    with pytest.raises(ContractError, match="3 violation"):
        check_measures(raw)


def test_an_empty_batch_is_accepted(make_raw) -> None:
    # Une journée sans mesure exploitable est un fait d'exploitation — une
    # source arrêtée, un site neuf — pas une rupture de contrat.
    assert check_measures(make_raw([])).empty


def test_check_features_accepts_what_the_etl_produces() -> None:
    features = conforming_features()
    assert not features.empty
    assert len(check_features(features, SPEC)) == len(features)


def test_check_features_refuses_a_null_target() -> None:
    features = conforming_features()
    features.loc[0, "consumption_kw"] = None
    with pytest.raises(ContractError):
        check_features(features, SPEC)


def test_check_features_refuses_a_missing_lag_column() -> None:
    features = conforming_features().drop(columns=[lag_column(1)])
    with pytest.raises(ContractError):
        check_features(features, SPEC)


def test_check_features_refuses_a_partition_of_another_version() -> None:
    # Les colonnes d'une version ne sont pas celles d'une autre : publier v2
    # sous le préfixe v1 rendrait la partition illisible pour l'entraînement.
    features = conforming_features()
    other = FeatureSpec("v2", "1h", (1, 24), rolling_window_h=2)
    with pytest.raises(ContractError):
        check_features(features, other)


def test_the_error_names_the_layer_and_its_version() -> None:
    features = conforming_features().drop(columns=[rolling_column(2)])
    with pytest.raises(ContractError, match="variables v1"):
        check_features(features, SPEC)
