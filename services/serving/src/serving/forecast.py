# **********************************************************************
# * Nom     : forecast.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Construction de la série prédite, pas à pas, depuis      *
# *   l'historique des variables                                       *
# * Service : serving                                                  *
# **********************************************************************

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from predict_common import io
from predict_common.paths import features_partition, lookback_range
from predict_common.schemas import (
    SITE_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    lag_column,
    rolling_column,
)

# Premier jour du week-end, lundi valant zéro.
WEEKEND_FIRST_DAY = 5

# Quantile normal de l'intervalle de confiance à 95 %.
CONFIDENCE_Z = 1.96

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastPoint:
    """Classe : ForecastPoint
    Description : Un point de la série prédite, avec son intervalle.
    """
    stamp: pd.Timestamp
    value: float
    lower: float | None
    upper: float | None


class NoHistory(LookupError):
    """Classe : NoHistory
    Description : Le site demandé n'a pas d'historique exploitable.
    """


@dataclass(frozen=True)
class ForecastSpec:
    """Classe : ForecastSpec
    Description : Ce dont la prévision a besoin : racine, version, décalages,
      profondeur.
    """
    root: str
    feature_version: str
    lag_hours: tuple[int, ...]
    rolling_window_h: int
    lookback_days: int
    reference_years: int = 0

    @property
    def deepest_lag_h(self) -> int:
        """Méthode : deepest_lag_h
        Description : Plus long décalage à couvrir, fenêtre glissante comprise.
        """
        return max((*self.lag_hours, self.rolling_window_h))


def read_history(spec: ForecastSpec, today: date | None = None) -> pd.DataFrame:
    """Méthode : read_history
    Description : Lit les partitions de variables de la fenêtre qui précède le
      jour donné.
    """
    end = today or datetime.now(UTC).date()
    partitions = [
        features_partition(spec.root, spec.feature_version, day)
        for day in lookback_range(end, spec.lookback_days)
    ]
    frame = io.read_frames(partitions, missing_ok=True)
    if frame.empty:
        return frame
    return frame.sort_values([SITE_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)


@dataclass
class _HistoryCache:
    """Classe : _HistoryCache
    Description : Dernier historique lu, gardé le temps de son TTL.
    """
    key: tuple[str, str, int, date] | None = None
    frame: pd.DataFrame | None = None
    read_at: float = 0.0


_cache = _HistoryCache()
_cache_lock = threading.Lock()


def cached_history(
    spec: ForecastSpec,
    ttl_s: float,
    today: date | None = None,
) -> pd.DataFrame:
    """Méthode : cached_history
    Description : Rend l'historique du jour, relu seulement quand son TTL a
      expiré.
    """
    if ttl_s <= 0:
        return read_history(spec, today)
    day = today or datetime.now(UTC).date()
    key = (spec.root, spec.feature_version, spec.lookback_days, day)
    now = time.monotonic()
    with _cache_lock:
        fresh = (
            _cache.frame is not None
            and _cache.key == key
            and (now - _cache.read_at) < ttl_s
        )
        if fresh:
            return _cache.frame
    frame = read_history(spec, day)
    with _cache_lock:
        _cache.key, _cache.frame, _cache.read_at = key, frame, time.monotonic()
    return frame


def reset_history_cache() -> None:
    """Méthode : reset_history_cache
    Description : Vide le cache. Point de reprise des tests, et rien d'autre.
    """
    with _cache_lock:
        _cache.key, _cache.frame, _cache.read_at = None, None, 0.0


# Décalage d'une année de référence, en journées entières de semaines.
REFERENCE_SHIFT_DAYS = 364


def reference_history(
    spec: ForecastSpec,
    site_id: str,
    today: date | None = None,
) -> tuple[pd.Series, date]:
    """Méthode : reference_history
    Description : Cherche l'historique d'une année de référence quand le récent
      manque.
    """
    end = today or datetime.now(UTC).date()
    for years in range(1, spec.reference_years + 1):
        origin = end - timedelta(days=REFERENCE_SHIFT_DAYS * years)
        frame = read_history(spec, today=origin)
        try:
            history = site_history(frame, site_id, spec)
        except NoHistory:
            continue
        shifted = history.copy()
        shifted.index = shifted.index + timedelta(
            days=REFERENCE_SHIFT_DAYS * years
        )
        now = pd.Timestamp(datetime.now(UTC)).floor("h")
        shifted = shifted[shifted.index <= now]
        if shifted.empty:
            continue
        return shifted, origin
    raise NoHistory(
        f"Aucun historique de référence pour {site_id} sur les"
        f" {spec.reference_years} année(s) précédentes."
    )


def site_history(frame: pd.DataFrame, site_id: str, spec: ForecastSpec) -> pd.Series:
    """Méthode : site_history
    Description : Extrait la série d'un site, indexée par instant.
    """
    rows = (
        frame[frame[SITE_COLUMN] == site_id]
        if SITE_COLUMN in frame.columns
        else frame
    )
    if rows.empty:
        raise NoHistory(
            f"Aucune heure récente pour {site_id} dans les"
            f" {spec.lookback_days} dernière(s) journée(s) de variables"
            f" {spec.feature_version}."
        )
    ordered = rows.sort_values(TIMESTAMP_COLUMN)
    series = pd.Series(
        ordered[TARGET_COLUMN].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(ordered[TIMESTAMP_COLUMN]),
    )
    grid = pd.date_range(series.index.min(), series.index.max(), freq="1h", tz="UTC")
    return series.reindex(grid)


def confidence_band(
    value: float,
    step: int,
    residual_std: float | None,
) -> tuple[float | None, float | None]:
    """Méthode : confidence_band
    Description : Encadre une valeur prédite, l'incertitude croissant avec
      l'horizon.
    """
    if residual_std is None or step < 1:
        return None, None
    margin = CONFIDENCE_Z * residual_std * (step**0.5)
    return value - margin, value + margin


def horizon_stamps(history: pd.Series, hours: int) -> list[pd.Timestamp]:
    """Méthode : horizon_stamps
    Description : Énumère les instants à prédire après le dernier connu.
    """
    last = history.index.max()
    return [last + timedelta(hours=step) for step in range(1, hours + 1)]


def build_row(
    history: pd.Series,
    stamp: pd.Timestamp,
    spec: ForecastSpec,
) -> dict[str, float | int] | None:
    """Méthode : build_row
    Description : Compose les variables d'un instant à partir de l'historique
      courant.
    """
    row: dict[str, float | int] = {
        "hour": int(stamp.hour),
        "day_of_week": int(stamp.dayofweek),
        "is_weekend": int(stamp.dayofweek >= WEEKEND_FIRST_DAY),
    }
    for hours in spec.lag_hours:
        value = _at(history, stamp - timedelta(hours=hours))
        if value is None:
            return None
        row[lag_column(hours)] = value
    window = _window(history, stamp, spec.rolling_window_h)
    if window is None:
        return None
    row[rolling_column(spec.rolling_window_h)] = window
    return row


def predict_series(
    predict: callable,
    history: pd.Series,
    hours: int,
    spec: ForecastSpec,
    columns: Sequence[str],
    residual_std: float | None = None,
) -> list[ForecastPoint]:
    """Méthode : predict_series
    Description : Prédit pas à pas, chaque prédiction nourrissant la suivante.
    """
    working = history.copy()
    points: list[ForecastPoint] = []
    for step, stamp in enumerate(horizon_stamps(history, hours), start=1):
        row = build_row(working, stamp, spec)
        if row is None:
            logger.warning("horizon interrompu à %s : décalage manquant", stamp)
            break
        value = float(predict(pd.DataFrame([row])[list(columns)])[0])
        working.loc[stamp] = value
        lower, upper = confidence_band(value, step, residual_std)
        points.append(ForecastPoint(stamp=stamp, value=value, lower=lower, upper=upper))
    return points


def _at(history: pd.Series, stamp: pd.Timestamp) -> float | None:
    """Méthode : _at
    Description : Valeur de l'historique à un instant, None si elle manque.
    """
    if stamp not in history.index:
        return None
    value = history.loc[stamp]
    return None if pd.isna(value) else float(value)


def _window(
    history: pd.Series,
    stamp: pd.Timestamp,
    hours: int,
) -> float | None:
    """Méthode : _window
    Description : Moyenne des heures précédant un instant, None si la fenêtre
      est incomplète.
    """
    start = stamp - timedelta(hours=hours)
    values = history.loc[(history.index >= start) & (history.index < stamp)].dropna()
    if len(values) < hours:
        return None
    return float(values.mean())
