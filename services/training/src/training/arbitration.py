"""Le banc d'arbitrage : la fenêtre commune où les candidats se comparent.

Deux modèles ne sont comparables que s'ils ont été jugés sur la même période.
`tracking` le dit depuis toujours, mais rien ne le garantissait : chaque run
mesurait sa qualité sur SON bloc de test, découpé dans SA fenêtre
d'apprentissage. Deux entraînements espacés d'une semaine produisaient donc
deux MAE que rien n'autorisait à mettre côte à côte — et c'est pourtant ce
qu'un exploitant fait en ouvrant l'interface MLflow.

Le banc corrige cela. C'est une fenêtre de journées retirée de
l'apprentissage, sur laquelle tout candidat est réévalué : le modèle qu'on
vient d'apprendre, la version en service, et les baselines naïves. Les trois
reçoivent les mêmes heures, la même cible, les mêmes métriques.

Le banc est glissant par défaut et figé sur demande, et la différence n'est
pas cosmétique. Glissant, il vaut pour comparer entre eux les candidats d'un
même entraînement, qui partagent la fenêtre du jour. Figé
(`training.arbitration.start` et `.end`), il vaut en plus d'un entraînement à
l'autre, parce que la fenêtre ne bouge plus. La promotion refuse de comparer
deux mesures faites sur des bancs différents plutôt que de produire un
classement qui n'en est pas un.

Retirer le banc de l'apprentissage coûte des données, et c'est le prix d'une
mesure qui veut dire quelque chose : un modèle évalué sur des heures qu'il a
apprises annonce la qualité de sa mémoire, pas celle de ses prévisions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from predict_common.config import Config
from predict_common.paths import lookback_range, parse_date
from training.baseline import Persistence
from training.dataset import matrices
from training.model import DECISION_METRIC, evaluate

BENCH_PREFIX = "arbitrage_"

NAIVE_METRIC = f"naif_{DECISION_METRIC}"

BENCH_WINDOW_PARAM = "arbitrage_window"

logger = logging.getLogger(__name__)


class ArbitrationError(ValueError):
    """Le banc demandé est vide, mal borné, ou absent du stockage."""


@dataclass(frozen=True)
class Bench:
    """La fenêtre du banc, et le fait qu'elle soit figée ou glissante."""

    start: date
    end: date
    pinned: bool

    @property
    def label(self) -> str:
        """Fenêtre telle qu'elle voyage dans les paramètres et les tags."""
        return f"{self.start}/{self.end}"

    @property
    def days(self) -> int:
        """Nombre de journées couvertes, bornes comprises."""
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class BenchResult:
    """Ce qu'un candidat a donné sur le banc, et sur quel banc.

    La fenêtre voyage avec les métriques et non à côté : c'est ce qui permet
    de refuser une comparaison entre deux mesures qui n'ont pas vu les mêmes
    heures, au lieu de la faire sans le savoir.
    """

    name: str
    window: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def error(self) -> float | None:
        """Erreur de décision, ou rien si la mesure ne la porte pas."""
        return self.metrics.get(DECISION_METRIC)


def resolve_bench(config: Config, end: date) -> Bench:
    """Retourne le banc : celui de `conf/` s'il est figé, sinon le glissant.

    Les deux bornes sont exigées ensemble. Une seule renseignée est une
    configuration à moitié écrite, et deviner l'autre produirait un banc que
    personne n'a décidé — donc des comparaisons qu'on croirait figées.
    """
    start_text = config.get_optional_str("training.arbitration.start")
    end_text = config.get_optional_str("training.arbitration.end")
    if bool(start_text) != bool(end_text):
        raise ArbitrationError(
            "training.arbitration.start et .end vont ensemble : un banc figé"
            " a deux bornes, sinon il n'est pas figé."
        )
    if start_text:
        first, last = parse_date(start_text), parse_date(end_text)
        if first > last:
            raise ArbitrationError(f"Banc vide : {last} précède {first}.")
        return Bench(start=first, end=last, pinned=True)
    days = config.get_int("training.arbitration.days")
    if days < 1:
        raise ArbitrationError("Un banc d'arbitrage couvre au moins un jour.")
    window = lookback_range(end, days)
    return Bench(start=window[0], end=window[-1], pinned=False)


def score(
    predictor: Any,
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    bench: Bench,
) -> BenchResult:
    """Mesure un candidat sur le banc, quel que soit ce qu'il est.

    Un modèle appris et une persistance passent par le même appel : la seconde
    n'est qu'un `predict` qui recopie une colonne. Les mesurer par deux
    chemins différents laisserait s'installer un écart de traitement entre la
    référence et ce qu'elle est censée arbitrer.
    """
    explanatory, observed = matrices(frame, columns)
    return BenchResult(
        name=name,
        window=bench.label,
        metrics=evaluate(observed, predictor.predict(explanatory)),
    )


def naive_reference(
    baselines: Sequence[Persistence],
    frame: pd.DataFrame,
    columns: Sequence[str],
    bench: Bench,
) -> BenchResult:
    """Retourne la meilleure des persistances, celle qu'il faut battre.

    La meilleure et non la première : prendre une persistance au hasard ferait
    une référence qu'on choisirait, donc une barre qu'on pourrait s'arranger
    pour placer bas. ADR-010 demande une comparaison systématique, ce qui n'a
    de sens que contre la plus dure des références gratuites.
    """
    if not baselines:
        raise ArbitrationError(
            "Aucune baseline naïve : etl.lag_hours n'en déclare aucune, et"
            " ADR-010 exige une comparaison systématique."
        )
    measured = [
        score(baseline, frame, columns, baseline.name, bench)
        for baseline in baselines
    ]
    return min(measured, key=lambda result: result.metrics[DECISION_METRIC])


def bench_metrics(result: BenchResult, naive: BenchResult) -> dict[str, float]:
    """Retourne les métriques du banc telles qu'elles partent dans le run."""
    metrics = {
        f"{BENCH_PREFIX}{name}": value for name, value in result.metrics.items()
    }
    metrics[NAIVE_METRIC] = naive.metrics[DECISION_METRIC]
    return metrics


def read_bench(
    metrics: Mapping[str, float],
    params: Mapping[str, str],
    name: str,
) -> BenchResult | None:
    """Relit d'un run passé ce qu'il a mesuré sur son banc.

    Rend `None` quand le run n'en porte pas. Un modèle enregistré avant
    l'existence du banc n'a rien de comparable : inventer une valeur ferait
    passer une promotion pour une décision alors qu'elle serait un pari.
    """
    measured = {
        key.removeprefix(BENCH_PREFIX): value
        for key, value in metrics.items()
        if key.startswith(BENCH_PREFIX)
    }
    window = params.get(BENCH_WINDOW_PARAM, "")
    if not measured or not window:
        return None
    return BenchResult(name=name, window=window, metrics=measured)


def read_naive(
    metrics: Mapping[str, float],
    params: Mapping[str, str],
) -> BenchResult | None:
    """Relit la référence naïve qu'un run a journalisée à côté de sa mesure.

    Elle est enregistrée avec le run et non recalculée : recalculer
    aujourd'hui la persistance d'une fenêtre passée demanderait de relire des
    partitions qui ont pu être régénérées depuis, et la référence d'une
    décision doit être celle qui a servi à la prendre.
    """
    error = metrics.get(NAIVE_METRIC)
    window = params.get(BENCH_WINDOW_PARAM, "")
    if error is None or not window:
        return None
    return BenchResult(
        name="la meilleure persistance",
        window=window,
        metrics={DECISION_METRIC: error},
    )
