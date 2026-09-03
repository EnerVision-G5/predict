"""Construction des variables explicatives : agrégats, décalages, calendrier.

La source produit à la minute, le modèle prédit à l'heure. C'est ici que la
série change de pas, et ce changement n'est pas un détail de mise en forme :
il corrige une faute que la chaîne portait tant que les décalages étaient
calculés en nombre de lignes. Un décalage compté en lignes suppose une série
sans trou ; une coupure de capteur d'un quart d'heure décalait alors
silencieusement tout l'historique, et `lag_24h` désignait autre chose que la
veille sans que rien ne le dise.

Ici, la série est rééchantillonnée puis réindexée sur une grille horaire
complète, trous compris. Un décalage devient une position sur cette grille :
`lag_24h` est la veille à la même heure, ou rien du tout. Les lignes dont un
décalage manque sont retirées — elles n'ont pas d'historique suffisant, et les
garder reviendrait à imputer une valeur que le modèle prendrait pour une
observation.

La moyenne glissante est décalée d'un pas avant d'être calculée. Sans ce
décalage, elle contiendrait la cible de l'heure courante : le modèle lirait la
réponse dans la question, ses métriques seraient excellentes à
l'apprentissage et fausses en production.

Chaque heure porte enfin ce qu'elle doit à l'ETL. `imputed_ratio` dit quelle
part de l'heure a été reconstruite et `data_quality` retient la plus sévère
des qualifications des minutes qui la composent — jamais leur moyenne, qui ne
voudrait rien dire. Les deux informent l'entraînement sans décider à sa place.
"""

from __future__ import annotations

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

IMPUTED_RATIO_COLUMN = "imputed_ratio"
WEEKEND_FIRST_DAY = 5
SECONDS_PER_HOUR = 3600


class FeatureError(ValueError):
    """Les variables demandées ne peuvent pas être calculées telles quelles."""


@dataclass(frozen=True)
class FeatureSpec:
    """Définition d'une version de variables, telle que `conf/` la porte.

    Changer l'un de ces champs change les colonnes produites : c'est pour cela
    qu'un tel changement s'accompagne d'une nouvelle `feature_version`, et non
    d'une réécriture des partitions existantes. Deux versions coexistent alors
    sous deux préfixes, et un modèle entraîné sur la première reste
    reproductible après la sortie de la seconde.
    """

    version: str
    resample_rule: str
    lag_hours: tuple[int, ...]
    rolling_window_h: int

    @property
    def columns(self) -> tuple[str, ...]:
        """Colonnes calculées écrites dans la partition, dans l'ordre du schéma.

        Sur-ensemble des variables du modèle : l'ETL publie aussi ce que le
        modèle ne consomme pas, la température au premier chef. Ce que le
        modèle lit est décidé par `feature_columns`, pas ici — la couche des
        variables décrit ce qu'elle sait, elle n'arbitre pas à la place de
        l'entraînement.
        """
        return published_columns(self.lag_hours, self.rolling_window_h)

    @property
    def periods_per_hour(self) -> int:
        """Nombre de pas de la grille dans une heure.

        Les décalages sont exprimés en heures dans `conf/`, parce que c'est
        ainsi qu'on raisonne sur une saisonnalité. La grille, elle, a le pas
        du rééchantillonnage : la conversion doit être exacte, sinon `lag_24h`
        ne tomberait pas sur la veille à la même heure.
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
        """Profondeur d'historique nécessaire pour remplir une journée.

        Le décalage le plus long décide : produire le 2 septembre avec un
        `lag_168h` demande de lire jusqu'au 26 août. La journée elle-même
        compte pour un, et un jour de marge absorbe le décalage de fuseau
        d'une source qui ne serait pas exactement en UTC.
        """
        deepest_h = max((*self.lag_hours, self.rolling_window_h))
        return 2 + -(-deepest_h // 24)


def resample(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Ramène les mesures d'un lot au pas de la grille, site par site.

    La cible agrégée est `consumption_kw_imputed` et non `consumption_kw` :
    c'est elle qui porte la meilleure valeur exploitable de chaque minute, la
    brute quand elle existe et la reconstruite sinon. Agréger la brute
    trouerait les heures que l'imputation venait précisément de combler.
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
    """Produit les variables du jour demandé à partir d'une fenêtre de mesures.

    La fenêtre reçue déborde volontairement sur les jours précédents : sans
    eux, `lag_168h` n'aurait rien à désigner. Seul le jour demandé est
    retourné, et c'est ce qui rend la production d'une journée indépendante
    des journées voisines — donc rejouable.
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
    return complete[list(_output_columns(spec))].reset_index(drop=True)


def _resample_site(
    site_id: str,
    group: pd.DataFrame,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """Agrège les mesures d'un site sur la grille, une colonne à la fois.

    Chaque colonne a sa règle, et aucune n'est la moyenne par défaut :
    `data_quality` retient la plus sévère, `imputed_ratio` est la part des
    minutes reconstruites. Une agrégation uniforme perdrait les deux.
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
    """Ajoute calendrier, décalages et moyenne glissante à un site.

    Le site est d'abord réindexé sur une grille complète : c'est cette grille,
    et non la suite des lignes présentes, qui donne son sens à un décalage.
    """
    frame = _on_complete_grid(group, spec)
    stamps = frame[TIMESTAMP_COLUMN]
    frame["hour"] = stamps.dt.hour
    frame["day_of_week"] = stamps.dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= WEEKEND_FIRST_DAY).astype(int)
    target = frame[TARGET_COLUMN]
    for hours in spec.lag_hours:
        frame[lag_column(hours)] = target.shift(hours * spec.periods_per_hour)
    # Décalée d'un pas avant d'être moyennée : sans cela, la fenêtre
    # contiendrait la cible de l'instant courant.
    window = spec.rolling_window_h * spec.periods_per_hour
    frame[rolling_column(spec.rolling_window_h)] = (
        target.shift(1).rolling(window=window, min_periods=window).mean()
    )
    return frame


def _on_complete_grid(group: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Réindexe un site sur une grille sans trou, du premier au dernier pas."""
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
    """Part des mesures du pas dont la valeur a été reconstruite."""
    known = methods.dropna()
    if known.empty:
        return 0.0
    return float((known != METHOD_NONE).mean())


def _worst_quality(qualities: pd.Series) -> str:
    """Qualification du pas : la plus sévère de celles qu'il contient."""
    known = qualities.dropna()
    if known.empty:
        return QUALITY_CRITICAL
    return worst(list(known.astype(str)))


def _required_columns(spec: FeatureSpec) -> Sequence[str]:
    """Colonnes dont l'absence retire la ligne du jeu d'apprentissage.

    La température n'en fait pas partie, et c'est délibéré : un site sans
    capteur thermique produit une série de consommation parfaitement
    exploitable, et XGBoost gère nativement l'absence d'une variable. L'exiger
    ici viderait la partition de ce site sans rien dire.
    """
    return (
        TARGET_COLUMN,
        *(lag_column(hours) for hours in spec.lag_hours),
        rolling_column(spec.rolling_window_h),
    )


def _output_columns(spec: FeatureSpec) -> Sequence[str]:
    """Colonnes de la partition produite, dans l'ordre du schéma."""
    return (
        TIMESTAMP_COLUMN,
        SITE_COLUMN,
        TARGET_COLUMN,
        *spec.columns,
        "data_quality",
        IMPUTED_RATIO_COLUMN,
    )


def _empty_hourly() -> pd.DataFrame:
    """Tableau horaire vide, aux colonnes attendues par la suite."""
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
    """Tableau de variables vide, aux colonnes de la version demandée."""
    return pd.DataFrame(columns=list(_output_columns(spec)))
