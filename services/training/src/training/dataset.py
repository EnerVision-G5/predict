"""Jeu d'apprentissage : lecture des partitions et découpe temporelle.

L'entraînement ne lit plus la base. Il lit `features/{version}/dt=.../`, que
l'ETL a produit et validé, et c'est tout ce qu'il connaît de l'amont. Le gain
n'est pas seulement architectural : un jeu d'apprentissage est désormais un
ensemble de fichiers immuables et datés, donc un run est reproductible en
rejouant la même liste de partitions, ce qu'une requête sur une base vivante
ne permettait pas.

La découpe est temporelle, jamais aléatoire. Une coupe au hasard laisserait le
modèle voir le futur d'une même journée pendant l'apprentissage et gonflerait
artificiellement le score : le modèle paraîtrait excellent en validation et
serait mauvais en production, ce qui est la pire des deux erreurs possibles.

Trois blocs et non deux. La validation sert l'arrêt anticipé, donc le modèle
la regarde à chaque tour et finit par s'y ajuster ; le test n'est regardé
qu'une fois, à la fin. Mesurer la qualité sur le jeu qui a servi à décider
quand s'arrêter reviendrait à se noter soi-même.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from predict_common import io
from predict_common.paths import date_range, features_partition
from predict_common.schemas import (
    SITE_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    add_derived_calendar,
)

IMPUTED_RATIO_COLUMN = "imputed_ratio"

logger = logging.getLogger(__name__)


class DatasetError(ValueError):
    """Le jeu d'apprentissage demandé est vide ou impossible à découper."""


@dataclass(frozen=True)
class Split:
    """Les trois blocs d'un jeu d'apprentissage, dans l'ordre du temps."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        """Volume de chaque bloc, tel qu'il part dans les paramètres du run."""
        return {
            "train_rows": len(self.train),
            "valid_rows": len(self.valid),
            "test_rows": len(self.test),
        }

    @property
    def window(self) -> str:
        """Fenêtre couverte par l'apprentissage, `debut/fin`.

        Journalisée comme paramètre du run : deux modèles aux mêmes
        hyperparamètres et aux mêmes métriques ne sont pas comparables s'ils
        n'ont pas vu la même période.
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
    """Lit les partitions de variables couvrant une fenêtre de journées.

    Une journée absente est ignorée : la source a pu être arrêtée, ou la
    chaîne démarrée en cours de fenêtre. C'est le volume total qui décide si
    l'apprentissage est possible, pas la présence de chaque jour.

    Ignorée, mais pas passée sous silence. `--history-days 90` sur une chaîne
    qui n'a qu'un mois de partitions produit un modèle appris sur un mois, et
    rien n'en avertissait : les journées manquantes sont journalisées en
    `debug` par la couche de stockage, sous le niveau que les services
    configurent. La proportion manquante est écrite ici, au niveau où elle
    sera lue, parce qu'elle change ce que le modèle a pu apprendre — une
    saisonnalité absente de la fenêtre ne s'apprend pas.
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
    """Signale l'écart entre la fenêtre demandée et celle réellement lue.

    Les journées présentes sont comptées dans le tableau lu plutôt que sur le
    stockage : l'information est déjà là, et une seconde interrogation du
    stockage objet coûterait un appel réseau par journée pour redire ce que
    les données disent.
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
    """Retient les lignes que le modèle a le droit d'apprendre.

    Le filtre sur l'imputation n'est pas une précaution cosmétique : une heure
    dont la cible a été reconstruite à 90 % enseigne l'interpolation de l'ETL,
    pas la consommation du site. Le seuil est dans `conf/`, parce que c'est un
    arbitrage entre volume et fidélité, et qu'il se règle.
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


def split_by_time(
    frame: pd.DataFrame,
    valid_ratio: float,
    test_ratio: float,
) -> Split:
    """Coupe le jeu en apprentissage, validation et test, dans l'ordre du temps.

    La coupe se fait sur l'horodatage et non sur le rang des lignes : avec
    plusieurs sites, une coupe par rang mettrait la fin de l'historique d'un
    site dans l'apprentissage et le début d'un autre dans le test.
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
    """Sépare les variables explicatives de la cible, dans l'ordre du modèle.

    L'ordre des colonnes est celui de la signature MLflow, et il vient d'un
    unique appel partagé avec le service d'inférence. Le fixer ici à la main
    laisserait les deux diverger sans que rien ne le signale.

    Les variables dérivées sont reconstruites ici, et ici seulement. C'est le
    passage obligé de l'entraînement comme de la surveillance de dérive : les
    dériver dans chacun des deux laisserait deux jeux de variables se former
    sous le même nom, et la surveillance mesurerait alors un modèle qu'elle ne
    nourrit pas comme l'entraînement l'a nourri.
    """
    enriched = add_derived_calendar(frame)
    missing = [name for name in columns if name not in enriched.columns]
    if missing:
        raise DatasetError(f"Variables absentes de la partition : {missing}.")
    return enriched[list(columns)], enriched[TARGET_COLUMN]


def _require_non_empty(split: Split) -> None:
    """Refuse une découpe dont un bloc serait vide.

    Un test vide donnerait un run sans métrique et un modèle enregistré sans
    rien qui atteste sa qualité — pire qu'un échec, puisqu'il serait
    promouvable.
    """
    empty = [name for name, size in split.sizes.items() if size == 0]
    if empty:
        raise DatasetError(
            f"Découpe impossible, bloc(s) vide(s) : {empty}. La fenêtre"
            " d'apprentissage est trop courte pour les ratios demandés."
        )
