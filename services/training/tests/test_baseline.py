"""Baselines naïves : la référence gratuite qu'un modèle doit battre.

ADR-010 range la baseline parmi les livrables permanents. Ce qui se teste ici
n'est pas sa qualité de prévision — elle n'en a aucune ambition — mais qu'elle
soit bien ce qu'elle prétend : une recopie exacte d'une colonne déjà publiée,
et non un calcul qui aurait dérivé.
"""

from __future__ import annotations

import pandas as pd
import pytest

from training.baseline import BaselineError, Persistence, naive_baselines


def frame() -> pd.DataFrame:
    """Trois heures portant les décalages que l'ETL publie."""
    return pd.DataFrame(
        {
            "lag_1h": [10.0, 11.0, 12.0],
            "lag_24h": [20.0, 21.0, 22.0],
            "consumption_kw": [30.0, 31.0, 32.0],
        }
    )


class TestPersistence:
    """Une persistance est une lecture de colonne, pas un modèle."""

    def test_the_name_carries_the_lag(self) -> None:
        # Le nom voyage dans le classement et dans le nom du run : deux
        # persistances indiscernables y seraient illisibles.
        assert Persistence(hours=24).name == "persistance-24h"

    def test_the_column_is_the_one_the_etl_publishes(self) -> None:
        # Déduite du même appel que les variables du modèle : l'écrire à la
        # main ici la laisserait diverger du jour où l'ETL renommerait.
        assert Persistence(hours=168).column == "lag_168h"

    def test_it_predicts_the_observed_value(self) -> None:
        predicted = Persistence(hours=24).predict(frame())
        assert list(predicted) == [20.0, 21.0, 22.0]

    def test_a_missing_column_is_refused(self) -> None:
        # Une persistance qui rendrait des zéros paraîtrait très mauvaise et
        # laisserait croire que le modèle l'écrase, alors que la comparaison
        # n'aurait pas eu lieu.
        with pytest.raises(BaselineError, match="lag_168h"):
            Persistence(hours=168).predict(frame())


def test_one_baseline_per_published_lag() -> None:
    baselines = naive_baselines([1, 24, 168])
    assert [baseline.hours for baseline in baselines] == [1, 24, 168]


def test_no_lag_gives_no_baseline() -> None:
    # Le cas est laissé à l'arbitrage, qui refuse : c'est lui qui sait
    # qu'ADR-010 exige une comparaison systématique.
    assert naive_baselines([]) == ()
