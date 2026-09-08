# **********************************************************************
# * Nom     : arbitration.py                                           *
# * Type    : Module                                                   *
# * Sujet   : Banc d'arbitrage : la fenêtre commune où tous les        *
# *   candidats sont jugés                                             *
# * Service : training                                                 *
# **********************************************************************

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

# Préfixe des métriques mesurées sur le banc.
BENCH_PREFIX = "arbitrage_"

# Métrique de la meilleure baseline sur le même banc.
NAIVE_METRIC = f"naif_{DECISION_METRIC}"

# Paramètre portant la fenêtre, relu par la promotion.
BENCH_WINDOW_PARAM = "arbitrage_window"

logger = logging.getLogger(__name__)


class ArbitrationError(ValueError):
    """Classe : ArbitrationError
    Description : Le banc demandé est vide, mal borné, ou absent du stockage.
    """


@dataclass(frozen=True)
class Bench:
    """Classe : Bench
    Description : La fenêtre du banc, et le fait qu'elle soit figée ou
      glissante.
    """
    start: date
    end: date
    pinned: bool

    @property
    def label(self) -> str:
        """Méthode : label
        Description : Fenêtre telle qu'elle voyage dans les paramètres et les
          tags.
        """
        return f"{self.start}/{self.end}"

    @property
    def days(self) -> int:
        """Méthode : days
        Description : Nombre de journées que couvre le banc.
        """
        return (self.end - self.start).days + 1


@dataclass(frozen=True)
class BenchResult:
    """Classe : BenchResult
    Description : Ce qu'un candidat a mesuré, et sur quelle fenêtre.
    """
    name: str
    window: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def error(self) -> float | None:
        """Méthode : error
        Description : Métrique de décision de ce résultat.
        """
        return self.metrics.get(DECISION_METRIC)


def resolve_bench(config: Config, end: date) -> Bench:
    """Méthode : resolve_bench
    Description : Détermine le banc : figé si la configuration le borne,
      glissant sinon.
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
    """Méthode : score
    Description : Mesure un candidat sur le banc et rend son résultat.
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
    """Méthode : naive_reference
    Description : Retient la meilleure des baselines naïves sur le banc.
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
    """Méthode : bench_metrics
    Description : Compose les métriques journalisées avec le run.
    """
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
    """Méthode : read_bench
    Description : Relit le résultat de banc d'un run déjà enregistré.
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
    """Méthode : read_naive
    Description : Relit la référence naïve d'un run déjà enregistré.
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
