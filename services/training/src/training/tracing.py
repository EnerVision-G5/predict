# **********************************************************************
# * Nom     : tracing.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Spans d'observabilité des étapes d'un entraînement       *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.entities import SpanType

from training.tracking import TrackingSettings

logger = logging.getLogger(__name__)


def configure(settings: TrackingSettings) -> None:
    """Méthode : configure
    Description : Pointe le traçage sur le registre et sur l'expérience où
      les runs sont déjà écrits.

      À appeler avant le premier span : une trace ouverte sans expérience
      active atterrit dans `Default`, loin des runs qu'elle décrit.
    """
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment)


@contextmanager
def span(
    name: str,
    span_type: str = SpanType.CHAIN,
    inputs: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Méthode : span
    Description : Ouvre un span et lui pose les entrées choisies.

      Choisies, et non capturées : `@mlflow.trace` sérialiserait ce que ces
      étapes se passent — des tableaux de milliers d'heures et un modèle
      ajusté — et la trace pèserait plus que le modèle qu'elle décrit.

      Rien n'est intercepté : une étape qui échoue échoue, et MLflow marque
      le span en erreur de lui-même.
    """
    with mlflow.start_span(name=name, span_type=span_type) as active:
        if inputs is not None:
            active.set_inputs(dict(inputs))
        yield active
