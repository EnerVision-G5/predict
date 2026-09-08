# **********************************************************************
# * Nom     : dataset.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Lecture des variables et découpe temporelle en           *
# *   apprentissage, validation, test                                  *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from predict_common import io
from predict_common.paths import date_range, features_partition
from predict_common.schemas import SITE_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN

# Part reconstruite d'une heure, qui décide de la garder ou non.
IMPUTED_RATIO_COLUMN = "imputed_ratio"

logger = logging.getLogger(__name__)


class DatasetError(ValueError):
    """Classe : DatasetError
    Description : Le jeu lu est vide, incomplet, ou impossible à découper.
    """


@dataclass(frozen=True)
class Split:
    """Classe : Split
    Description : Les trois blocs d'un jeu découpé dans le temps.
    """
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        """Méthode : sizes
        Description : Nombre d'heures de chaque bloc.
        """
        return {
            "train_rows": len(self.train),
            "valid_rows": len(self.valid),
            "test_rows": len(self.test),
        }

    @property
    def window(self) -> str:
        """Méthode : window
        Description : Fenêtre couverte par le bloc d'apprentissage.
        """
        if self.train.empty:
            return ""
        stamps = self.train[TIMESTAMP_COLUMN]
        return f"{stamps.min().date()}/{stamps.max().date()}"


def read_features(
    root: str,
    version: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Méthode : read_features
    Description : Lit les partitions de variables d'une fenêtre de journées.
    """
    requested = date_range(start, end)
    partitions = [features_partition(root, version, day) for day in requested]
    frame = io.read_frames(partitions, missing_ok=True)
    if frame.empty:
        logger.warning(
            "aucune des %d journée(s) demandées entre %s et %s n'est publiée"
            " en %s",
            len(requested),
            start,
            end,
            version,
        )
        return frame
    _warn_on_gaps(frame, requested, version)
    return frame.sort_values([SITE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


def _warn_on_gaps(
    frame: pd.DataFrame,
    requested: Sequence[date],
    version: str,
) -> None:
    """Méthode : _warn_on_gaps
    Description : Signale les journées demandées qui ne sont pas publiées.
    """
    present = frame[TIMESTAMP_COLUMN].dt.date.nunique()
    if present >= len(requested):
        return
    logger.warning(
        "fenêtre demandée : %d journée(s) du %s au %s, %d seulement"
        " publiée(s) en %s. Le modèle n'apprendra que sur celles-là.",
        len(requested),
        requested[0],
        requested[-1],
        present,
        version,
    )


def select(
    frame: pd.DataFrame,
    sites: Sequence[str] | None,
    max_imputed_ratio: float,
) -> pd.DataFrame:
    """Méthode : select
    Description : Ne garde que les sites demandés et les heures assez peu
      reconstruites.
    """
    selected = frame
    if sites:
        selected = selected[selected[SITE_COLUMN].isin(list(sites))]
    kept = selected[selected[IMPUTED_RATIO_COLUMN] <= max_imputed_ratio]
    dropped = len(selected) - len(kept)
    if dropped:
        logger.info(
            "%d heure(s) écartée(s) : plus de %.0f %% de valeurs reconstruites",
            dropped,
            max_imputed_ratio * 100,
        )
    return kept.reset_index(drop=True)


def exclude_window(
    frame: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Méthode : exclude_window
    Description : Retire du jeu les journées réservées au banc d'arbitrage.
    """
    if frame.empty:
        return frame
    days = frame[TIMESTAMP_COLUMN].dt.date
    kept = frame[(days < start) | (days > end)].reset_index(drop=True)
    removed = len(frame) - len(kept)
    if removed:
        logger.info(
            "%d heure(s) retirée(s) de l'apprentissage : réservées au banc"
            " d'arbitrage du %s au %s",
            removed,
            start,
            end,
        )
    return kept


def split_by_time(
    frame: pd.DataFrame,
    valid_ratio: float,
    test_ratio: float,
) -> Split:
    """Méthode : split_by_time
    Description : Découpe dans l'ordre du temps, jamais au hasard.
    """
    if not 0.0 < valid_ratio + test_ratio < 1.0:
        raise DatasetError(
            "valid_ratio + test_ratio doit être strictement entre 0 et 1."
        )
    if frame.empty:
        raise DatasetError("Aucune heure exploitable dans la fenêtre demandée.")
    ordered = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    stamps = ordered[TIMESTAMP_COLUMN]
    train_end = stamps.quantile(1.0 - valid_ratio - test_ratio)
    valid_end = stamps.quantile(1.0 - test_ratio)
    split = Split(
        train=ordered[stamps <= train_end].reset_index(drop=True),
        valid=ordered[(stamps > train_end) & (stamps <= valid_end)].reset_index(
            drop=True
        ),
        test=ordered[stamps > valid_end].reset_index(drop=True),
    )
    _require_non_empty(split)
    return split


def matrices(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Méthode : matrices
    Description : Sépare les variables explicatives de la cible à prédire.
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DatasetError(f"Variables absentes de la partition : {missing}.")
    return frame[list(columns)], frame[TARGET_COLUMN]


def _require_non_empty(split: Split) -> None:
    """Méthode : _require_non_empty
    Description : Refuse une découpe qui laisserait un bloc vide.
    """
    empty = [name for name, size in split.sizes.items() if size == 0]
    if empty:
        raise DatasetError(
            f"Découpe impossible, bloc(s) vide(s) : {empty}. La fenêtre"
            " d'apprentissage est trop courte pour les ratios demandés."
        )
