"""Prévision multi-pas : historique récent, puis récurrence heure par heure.

Prédire une heure demande de connaître la précédente, la veille et la semaine
passée. Le service ne recalcule pas ces décalages depuis les mesures brutes —
ce serait refaire le travail de l'ETL, avec un second jeu de règles qui
finirait par diverger du premier. Il lit la dernière partition de variables
publiée, qui les porte déjà, calculées une seule fois et de la même façon pour
l'apprentissage et pour la prévision.

C'est une quatrième frontière, et elle a la même forme que les trois autres :
un chemin, un schéma, aucun import. L'ETL ne sait pas que le service le lit.

Prévoir vingt-quatre heures se fait ensuite par récurrence. La prévision de
h+1 devient le `lag_1h` de h+2, et ainsi de suite ; les décalages plus longs
restent puisés dans l'historique observé tant qu'il en reste. L'erreur
s'accumule donc avec l'horizon, ce qui est la nature de l'exercice et non un
défaut de l'implémentation : c'est la raison pour laquelle le contrat borne
l'horizon à quarante-huit heures.
"""

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

WEEKEND_FIRST_DAY = 5

# Quantile normal bilatéral à 95 %. Nommé plutôt qu'écrit dans le calcul : la
# largeur de l'intervalle servi est une décision, et une décision se relit.
CONFIDENCE_Z = 1.96

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastPoint:
    """Une heure prédite, et l'intervalle qui dit ce qu'elle vaut.

    Les bornes sont nulles quand la version servie ne déclare pas la
    dispersion de son erreur : le contrat les décrit comme optionnelles, et un
    intervalle inventé serait pire qu'un intervalle absent — un consommateur
    ne saurait pas qu'il ne repose sur rien.
    """

    stamp: pd.Timestamp
    value: float
    lower: float | None
    upper: float | None


class NoHistory(LookupError):
    """Le site demandé n'a aucune heure récente dans les partitions lues."""


@dataclass(frozen=True)
class ForecastSpec:
    """Ce que le service doit savoir des variables pour les reconstruire."""

    root: str
    feature_version: str
    lag_hours: tuple[int, ...]
    rolling_window_h: int
    lookback_days: int

    @property
    def deepest_lag_h(self) -> int:
        """Décalage le plus profond, qui décide de l'historique à charger."""
        return max((*self.lag_hours, self.rolling_window_h))


def read_history(spec: ForecastSpec, today: date | None = None) -> pd.DataFrame:
    """Lit les dernières partitions de variables publiées.

    Les partitions manquantes sont ignorées : l'ETL de la nuit peut ne pas
    avoir tourné, et le service doit alors servir sur l'historique de la
    veille plutôt que de refuser. C'est l'absence du site, et non celle d'une
    journée, qui empêche de prédire.
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
    """Dernière fenêtre de variables lue, et l'instant où elle l'a été.

    Le service relisait le stockage à chaque prévision : `lookback_days`
    partitions journalières téléchargées, décompressées, concaténées et
    triées, pour n'en extraire ensuite qu'un seul site. Le job de
    rafraîchissement boucle sur les sept sites, donc sept lectures complètes
    de la même fenêtre par cycle — et, le service étant ouvert avant
    `serving.auth`, autant d'amplification offerte à qui voulait le saturer.

    L'ETL ne publie qu'une fois par cycle : relire plus souvent ne peut rien
    apprendre de neuf. Le cache est donc une fenêtre de temps, pas une
    invalidation fine — une partition republiée entre deux expirations sera
    vue au plus tard au bout du TTL, ce qui est exactement la fraîcheur que
    `PredictionOut.feature_lag_hours` publie déjà par ailleurs.
    """

    key: tuple[str, str, int, date] | None = None
    frame: pd.DataFrame | None = None
    read_at: float = 0.0


_cache = _HistoryCache()
# uvicorn sert plusieurs requêtes de front et `predict` est un `def` synchrone,
# donc exécuté dans un fil du pool : deux requêtes peuvent entrer ici ensemble.
_cache_lock = threading.Lock()


def cached_history(
    spec: ForecastSpec,
    ttl_s: float,
    today: date | None = None,
) -> pd.DataFrame:
    """Rend la fenêtre de variables, relue seulement si elle a vieilli.

    Un TTL nul ou négatif désactive le cache et rend le comportement d'avant :
    c'est ce que règle `serving.feature_cache_ttl_s`, et ce que les tests qui
    éprouvent la lecture elle-même utilisent.

    Le tableau rendu est partagé entre appelants et ne doit jamais être muté.
    Aucun consommateur ne le fait — `site_history` filtre puis trie, ce qui
    copie — mais la règle vaut d'être écrite.
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
    # La lecture se fait HORS du verrou : elle peut durer des secondes sur un
    # stockage objet, et la tenir bloquerait toutes les requêtes du service.
    # Deux lectures concurrentes au même instant sont possibles et sans
    # conséquence — elles produisent le même tableau, la dernière gagne.
    frame = read_history(spec, day)
    with _cache_lock:
        _cache.key, _cache.frame, _cache.read_at = key, frame, time.monotonic()
    return frame


def reset_history_cache() -> None:
    """Vide le cache. Point de reprise des tests, et rien d'autre."""
    with _cache_lock:
        _cache.key, _cache.frame, _cache.read_at = None, None, 0.0


def site_history(frame: pd.DataFrame, site_id: str, spec: ForecastSpec) -> pd.Series:
    """Retourne la série de consommation observée d'un site, indexée sur le temps.

    La série est réindexée sur une grille horaire complète : c'est elle qui
    donne son sens à un décalage, exactement comme dans l'ETL. Un trou reste
    un trou, il n'est pas comblé ici — l'imputation appartient à l'ETL, et la
    refaire dans le service en donnerait deux versions.
    """
    # Le tableau peut être vide et sans colonnes : c'est ce que rend la lecture
    # quand aucune partition n'existe sur la fenêtre. Le cas n'est pas
    # distingué de celui du site inconnu, parce qu'il donne la même réponse au
    # consommateur — il n'y a rien à prédire — et que le journal du service
    # dira laquelle des deux causes l'a produit.
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
    """Encadre une valeur prédite au pas `step` de la récurrence.

    La largeur croît en racine du pas, et non linéairement : le service prédit
    par récurrence, chaque heure repartant des heures qu'il vient lui-même de
    prédire. Les erreurs de deux pas successifs s'additionnent en variance, pas
    en écart-type — d'où le `sqrt`. Une croissance linéaire donnerait à
    l'horizon 48 une bande quatre fois trop large, et personne ne la lirait.

    C'est une approximation, et elle est optimiste sur deux points : elle
    suppose les erreurs successives indépendantes, et elle ignore que le modèle
    se trompe davantage sur ses propres prédictions que sur des mesures. Elle
    dit un ordre de grandeur, pas une garantie.

    La borne basse n'est pas ramenée à zéro. Le schéma de la couche brute
    n'interdit pas une consommation négative — un site qui produit localement
    peut afficher un soutirage net négatif — et rogner la borne masquerait un
    modèle qui prédit une valeur aberrante au lieu de la laisser voir.
    """
    if residual_std is None or step < 1:
        return None, None
    margin = CONFIDENCE_Z * residual_std * (step**0.5)
    return value - margin, value + margin


def horizon_stamps(history: pd.Series, hours: int) -> list[pd.Timestamp]:
    """Retourne les heures prédites, à la suite de la dernière observée."""
    last = history.index.max()
    return [last + timedelta(hours=step) for step in range(1, hours + 1)]


def build_row(
    history: pd.Series,
    stamp: pd.Timestamp,
    spec: ForecastSpec,
) -> dict[str, float | int] | None:
    """Construit la ligne de variables d'une heure à prédire.

    `None` quand un décalage manque : l'heure ne peut pas être prédite, et
    inventer sa valeur donnerait une prévision dont rien ne dirait qu'elle
    repose sur du vide.
    """
    # Les variables calendaires sont entières, et pas seulement par élégance :
    # la signature du modèle les déclare `integer`, et MLflow refuse une
    # conversion float64 vers int32 qu'il ne peut pas garantir sans perte.
    # La température n'est pas ici, et le modèle ne l'attend pas : elle a été
    # sortie des variables explicatives dans `predict_common.schemas`. Le
    # service ne connaît pas la météo des heures à venir, et un modèle entraîné
    # sur une colonne toujours vide en production apprend un biais, pas un
    # signal. La partition la porte toujours : le jour où une prévision météo
    # alimentera l'inférence, elle est déjà là.
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
    """Prédit les `hours` heures suivantes, chacune nourrissant la suivante.

    La série d'historique grandit à chaque pas : la valeur prédite pour h+1 y
    est écrite avant de construire la ligne de h+2, qui la lira comme son
    `lag_1h`. C'est ce qui rend la récurrence possible, et c'est aussi ce qui
    fait croître l'erreur avec l'horizon.
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
    """Retourne la valeur d'une heure de la série, ou rien si elle manque."""
    if stamp not in history.index:
        return None
    value = history.loc[stamp]
    return None if pd.isna(value) else float(value)


def _window(
    history: pd.Series,
    stamp: pd.Timestamp,
    hours: int,
) -> float | None:
    """Moyenne des `hours` heures précédant l'heure à prédire.

    La fenêtre s'arrête à l'heure précédente et n'inclut jamais celle qu'on
    prédit : c'est la même règle qu'à l'entraînement, où la moyenne glissante
    est décalée d'un pas. La rompre ici présenterait au modèle une variable
    qu'il n'a jamais vue sous cette forme.
    """
    start = stamp - timedelta(hours=hours)
    values = history.loc[(history.index >= start) & (history.index < stamp)].dropna()
    if len(values) < hours:
        return None
    return float(values.mean())
