"""Jeu d'apprentissage : lecture des partitions et découpe temporelle.

La garantie centrale est que le modèle ne voit jamais le futur. Une coupe au
hasard le laisserait apprendre la fin d'une journée dont il doit prédire le
début : ses métriques seraient excellentes en validation et fausses en
production, ce qui est la pire des deux erreurs possibles — celle qui ne se
voit qu'une fois déployée.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from predict_common import io
from predict_common.paths import features_partition
from predict_common.schemas import feature_columns, features_arrow_schema
from training.dataset import (
    DatasetError,
    matrices,
    read_features,
    select,
    split_by_time,
)

LAGS = (1, 24)
WINDOW = 2
COLUMNS = feature_columns(LAGS, WINDOW)
END = date(2026, 9, 10)


def features(day: date, hours: int = 24, site_id: str = "SITE001", **columns):
    """Partition de variables d'une journée, telle que l'ETL la publie."""
    stamps = pd.date_range(
        f"{day.isoformat()}T00:00:00Z", periods=hours, freq="h", tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "ts": stamps,
            "site_id": site_id,
            "consumption_kw": [50.0 + index for index in range(hours)],
            "hour": stamps.hour,
            "day_of_week": stamps.dayofweek,
            "is_weekend": (stamps.dayofweek >= 5).astype(int),
            "temperature_celsius": 20.0,
            "lag_1h": 50.0,
            "lag_24h": 50.0,
            "roll_mean_2h": 50.0,
            "data_quality": "good",
            "imputed_ratio": 0.0,
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def seed(root: Path, days: int = 6, **columns) -> None:
    """Publie `days` partitions de variables se terminant le jour de référence."""
    for offset in range(days):
        day = END - timedelta(days=offset)
        io.write_frame(
            features(day, **columns),
            features_partition(str(root), "v1", day),
            schema=features_arrow_schema(LAGS, WINDOW),
        )


def test_read_features_gathers_the_window(tmp_path: Path) -> None:
    seed(tmp_path, days=3)
    frame = read_features(str(tmp_path), "v1", END - timedelta(days=2), END)
    assert len(frame) == 72


def test_read_features_skips_a_missing_day(tmp_path: Path) -> None:
    # La source a pu être arrêtée, ou la chaîne démarrée en cours de fenêtre :
    # c'est le volume total qui décide si l'apprentissage est possible.
    seed(tmp_path, days=2)
    frame = read_features(str(tmp_path), "v1", END - timedelta(days=5), END)
    assert len(frame) == 48


def test_read_features_says_how_much_of_the_window_is_missing(
    tmp_path: Path, caplog
) -> None:
    # `--history-days 90` sur une chaîne qui n'a qu'un mois produit un modèle
    # appris sur un mois, et rien ne l'annonçait : les journées absentes sont
    # journalisées en debug par la couche de stockage, sous le niveau que les
    # services configurent.
    seed(tmp_path, days=2)
    with caplog.at_level(logging.WARNING):
        read_features(str(tmp_path), "v1", END - timedelta(days=5), END)
    assert "6 journée(s)" in caplog.text
    assert "2 seulement" in caplog.text


def test_read_features_stays_quiet_on_a_complete_window(
    tmp_path: Path, caplog
) -> None:
    # Un avertissement à chaque run le rendrait illisible le jour où il compte.
    seed(tmp_path, days=3)
    with caplog.at_level(logging.WARNING):
        read_features(str(tmp_path), "v1", END - timedelta(days=2), END)
    assert caplog.text == ""


def test_read_features_warns_when_the_whole_window_is_absent(
    tmp_path: Path, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        assert read_features(str(tmp_path), "v1", END - timedelta(days=2), END).empty
    assert "aucune" in caplog.text


def test_read_features_returns_nothing_for_an_unknown_version(
    tmp_path: Path,
) -> None:
    seed(tmp_path, days=2)
    assert read_features(str(tmp_path), "v2", END - timedelta(days=5), END).empty


def test_select_keeps_the_requested_sites(tmp_path: Path) -> None:
    frame = pd.concat(
        [features(END), features(END, site_id="SITE002")], ignore_index=True
    )
    assert set(select(frame, ["SITE001"], 1.0)["site_id"]) == {"SITE001"}


def test_select_drops_a_target_the_etl_mostly_invented() -> None:
    # Une heure reconstruite à 90 % enseigne l'interpolation de l'ETL, pas la
    # consommation du site.
    frame = features(END)
    frame.loc[0, "imputed_ratio"] = 0.9
    assert len(select(frame, None, 0.5)) == 23


def test_select_keeps_a_lightly_imputed_target() -> None:
    frame = features(END)
    frame["imputed_ratio"] = 0.2
    assert len(select(frame, None, 0.5)) == 24


class TestSplit:
    """Trois blocs, dans l'ordre du temps, et jamais au hasard."""

    def test_the_blocks_follow_one_another_in_time(self) -> None:
        frame = pd.concat(
            [features(END - timedelta(days=offset)) for offset in range(6)],
            ignore_index=True,
        )
        split = split_by_time(frame, valid_ratio=0.15, test_ratio=0.15)
        assert split.train["ts"].max() <= split.valid["ts"].min()
        assert split.valid["ts"].max() <= split.test["ts"].min()

    def test_the_training_block_is_the_largest(self) -> None:
        frame = pd.concat(
            [features(END - timedelta(days=offset)) for offset in range(6)],
            ignore_index=True,
        )
        sizes = split_by_time(frame, 0.15, 0.15).sizes
        assert sizes["train_rows"] > sizes["valid_rows"] + sizes["test_rows"]

    def test_the_window_names_the_learnt_period(self) -> None:
        # Deux modèles aux mêmes métriques ne sont pas comparables s'ils n'ont
        # pas vu la même période : c'est un paramètre du run.
        frame = pd.concat(
            [features(END - timedelta(days=offset)) for offset in range(6)],
            ignore_index=True,
        )
        assert "/" in split_by_time(frame, 0.15, 0.15).window

    def test_two_sites_are_cut_at_the_same_instant(self) -> None:
        # Une coupe par rang mettrait la fin d'un site dans l'apprentissage et
        # le début d'un autre dans le test.
        frame = pd.concat(
            [
                features(END - timedelta(days=offset), site_id=site)
                for offset in range(6)
                for site in ("SITE001", "SITE002")
            ],
            ignore_index=True,
        )
        split = split_by_time(frame, 0.15, 0.15)
        assert set(split.test["site_id"]) == {"SITE001", "SITE002"}

    def test_an_empty_window_is_refused(self) -> None:
        with pytest.raises(DatasetError):
            split_by_time(pd.DataFrame(), 0.15, 0.15)

    def test_a_window_too_short_for_the_ratios_is_refused(self) -> None:
        # Un test vide donnerait un modèle enregistré sans rien qui atteste sa
        # qualité — pire qu'un échec, puisqu'il serait promouvable.
        with pytest.raises(DatasetError):
            split_by_time(features(END, hours=2), 0.15, 0.15)

    def test_impossible_ratios_are_refused(self) -> None:
        with pytest.raises(DatasetError):
            split_by_time(features(END), valid_ratio=0.6, test_ratio=0.6)


def test_matrices_follow_the_shared_column_order() -> None:
    # L'ordre est celui de la signature MLflow : le fixer à la main ici
    # laisserait le service d'inférence diverger sans que rien ne le dise.
    explanatory, target = matrices(features(END), COLUMNS)
    assert list(explanatory.columns) == list(COLUMNS)
    assert target.name == "consumption_kw"


def test_matrices_refuse_a_partition_missing_a_variable() -> None:
    with pytest.raises(DatasetError):
        matrices(features(END).drop(columns=["lag_24h"]), COLUMNS)
