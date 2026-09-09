# **********************************************************************
# * Nom     : features.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Construction des variables explicatives : grille         *
# *   horaire, décalages, calendrier                                   *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from etl.impute import IMPUTED_COLUMN, METHOD_COLUMN
from etl.quality import worst
from predict_common.schemas import (
    METHOD_NONE,
    QUALITY_CRITICAL,
    SITE_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    lag_column,
    published_columns,
    rolling_column,
)

# Part de l'heure qui a été reconstruite.
IMPUTED_RATIO_COLUMN = "imputed_ratio"
# Premier jour du week-end, lundi valant zéro.
WEEKEND_FIRST_DAY = 5
# Sert à vérifier que le pas de grille divise l'heure.
SECONDS_PER_HOUR = 3600

logger = logging.getLogger(__name__)


class FeatureError(ValueError):
    """Classe : FeatureError
    Description : Les variables demandées ne peuvent pas être calculées telles
      quelles.
    """


@dataclass(frozen=True)
class FeatureSpec:
    """Classe : FeatureSpec
    Description : Définition d'une version de variables : pas, décalages,
      fenêtre glissante.
    """
    version: str
    resample_rule: str
    lag_hours: tuple[int, ...]
    rolling_window_h: int

    @property
    def columns(self) -> tuple[str, ...]:
        """Méthode : columns
        Description : Énumère les colonnes publiées par cette version.
        """
        return published_columns(self.lag_hours, self.rolling_window_h)

    @property
    def periods_per_hour(self) -> int:
        """Méthode : periods_per_hour
        Description : Nombre de pas de grille dans une heure, refusé s'il n'est
          pas entier.
        """
        step = pd.Timedelta(self.resample_rule)
        seconds = step.total_seconds()
        if seconds <= 0 or SECONDS_PER_HOUR % seconds:
            raise FeatureError(
                f"Le pas {self.resample_rule!r} ne divise pas l'heure : les"
                " décalages exprimés en heures ne tomberaient pas sur la grille."
            )
        return int(SECONDS_PER_HOUR // seconds)

    @property
    def lookback_days(self) -> int:
        """Méthode : lookback_days
        Description : Profondeur à lire pour que le plus long décalage ait de
          quoi désigner.
        """
        deepest_h = max((*self.lag_hours, self.rolling_window_h))
        return 2 + -(-deepest_h // 24)


def resample(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Méthode : resample
    Description : Ramène les mesures d'une fenêtre sur la grille horaire, site
      par site.
    """
    if frame.empty:
        return _empty_hourly()
    dated = frame.dropna(subset=[TIMESTAMP_COLUMN, SITE_COLUMN])
    if dated.empty:
        return _empty_hourly()
    grouped = [
        _resample_site(str(site_id), group, spec)
        for site_id, group in dated.groupby(SITE_COLUMN, sort=True)
    ]
    return pd.concat(grouped, ignore_index=True)


def build(frame: pd.DataFrame, spec: FeatureSpec, day: date) -> pd.DataFrame:
    """Méthode : build
    Description : Produit les variables du jour demandé à partir d'une fenêtre
      de mesures.
    """
    hourly = resample(frame, spec)
    if hourly.empty:
        return _empty_features(spec)
    enriched = pd.concat(
        [
            _derive_site(group, spec)
            for _, group in hourly.groupby(SITE_COLUMN, sort=True)
        ],
        ignore_index=True,
    )
    selected = enriched[enriched[TIMESTAMP_COLUMN].dt.date == day]
    complete = selected.dropna(subset=list(_required_columns(spec)))
    _report_incomplete(selected, complete, spec)
    return complete[list(_output_columns(spec))].reset_index(drop=True)


def _report_incomplete(
    selected: pd.DataFrame,
    complete: pd.DataFrame,
    spec: FeatureSpec,
) -> None:
    """Méthode : _report_incomplete
    Description : Dit quelles colonnes ont fait retirer des heures, et combien.
    """
    removed = len(selected) - len(complete)
    if not removed:
        return
    causes = {
        name: int(selected[name].isna().sum())
        for name in _required_columns(spec)
        if selected[name].isna().any()
    }
    logger.warning(
        "%d heure(s) sur %d retirée(s), faute d'un historique suffisant : %s",
        removed,
        len(selected),
        ", ".join(f"{name} absent sur {count}" for name, count in causes.items()),
    )


def _resample_site(
    site_id: str,
    group: pd.DataFrame,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """Méthode : _resample_site
    Description : Agrège les mesures d'un site sur la grille, une colonne à la
      fois.
    """
    indexed = group.set_index(pd.DatetimeIndex(group[TIMESTAMP_COLUMN])).sort_index()
    buckets = indexed.resample(spec.resample_rule)
    hourly = pd.DataFrame(
        {
            TARGET_COLUMN: buckets[IMPUTED_COLUMN].mean(),
            "temperature_celsius": buckets["temperature_celsius"].mean(),
            IMPUTED_RATIO_COLUMN: buckets[METHOD_COLUMN].agg(_imputed_ratio),
            "data_quality": buckets["data_quality"].agg(_worst_quality),
        }
    )
    hourly.index.name = TIMESTAMP_COLUMN
    hourly = hourly.reset_index()
    hourly.insert(1, SITE_COLUMN, site_id)
    return hourly


def _derive_site(group: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Méthode : _derive_site
    Description : Ajoute calendrier, décalages et moyenne glissante à un site.
    """
    frame = _on_complete_grid(group, spec)
    stamps = frame[TIMESTAMP_COLUMN]
    frame["hour"] = stamps.dt.hour
    frame["day_of_week"] = stamps.dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= WEEKEND_FIRST_DAY).astype(int)
    target = frame[TARGET_COLUMN]
    for hours in spec.lag_hours:
        frame[lag_column(hours)] = target.shift(hours * spec.periods_per_hour)
    window = spec.rolling_window_h * spec.periods_per_hour
    frame[rolling_column(spec.rolling_window_h)] = (
        target.shift(1).rolling(window=window, min_periods=window).mean()
    )
    return frame


def _on_complete_grid(group: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Méthode : _on_complete_grid
    Description : Réindexe un site sur une grille sans trou, pour que les
      décalages tombent juste.
    """
    ordered = group.sort_values(TIMESTAMP_COLUMN)
    grid = pd.date_range(
        start=ordered[TIMESTAMP_COLUMN].min(),
        end=ordered[TIMESTAMP_COLUMN].max(),
        freq=spec.resample_rule,
        tz="UTC",
    )
    site_id = ordered[SITE_COLUMN].iloc[0]
    reindexed = (
        ordered.set_index(TIMESTAMP_COLUMN)
        .reindex(grid)
        .rename_axis(TIMESTAMP_COLUMN)
        .reset_index()
    )
    reindexed[SITE_COLUMN] = site_id
    return reindexed


def _imputed_ratio(methods: pd.Series) -> float:
    """Méthode : _imputed_ratio
    Description : Part des minutes reconstruites dans une heure.
    """
    known = methods.dropna()
    if known.empty:
        return 0.0
    return float((known != METHOD_NONE).mean())


def _worst_quality(qualities: pd.Series) -> str:
    """Méthode : _worst_quality
    Description : Retient la qualification la plus sévère des minutes d'une
      heure.
    """
    known = qualities.dropna()
    if known.empty:
        return QUALITY_CRITICAL
    return worst(list(known.astype(str)))


def _required_columns(spec: FeatureSpec) -> Sequence[str]:
    """Méthode : _required_columns
    Description : Colonnes sans lesquelles une heure n'est pas publiable.
    """
    return (
        TARGET_COLUMN,
        *(lag_column(hours) for hours in spec.lag_hours),
        rolling_column(spec.rolling_window_h),
    )


def _output_columns(spec: FeatureSpec) -> Sequence[str]:
    """Méthode : _output_columns
    Description : Colonnes écrites dans la partition, dans l'ordre.
    """
    return (
        TIMESTAMP_COLUMN,
        SITE_COLUMN,
        TARGET_COLUMN,
        *spec.columns,
        "data_quality",
        IMPUTED_RATIO_COLUMN,
    )


def _empty_hourly() -> pd.DataFrame:
    """Méthode : _empty_hourly
    Description : Tableau horaire vide, mais aux bonnes colonnes.
    """
    return pd.DataFrame(
        columns=[
            TIMESTAMP_COLUMN,
            SITE_COLUMN,
            TARGET_COLUMN,
            "temperature_celsius",
            IMPUTED_RATIO_COLUMN,
            "data_quality",
        ]
    )


def _empty_features(spec: FeatureSpec) -> pd.DataFrame:
    """Méthode : _empty_features
    Description : Partition de variables vide, mais aux bonnes colonnes.
    """
    return pd.DataFrame(columns=list(_output_columns(spec)))
