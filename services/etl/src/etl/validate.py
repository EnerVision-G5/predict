"""Vérification des contrats de couche, à la lecture comme à l'écriture.

L'ETL est entre deux frontières et il les vérifie toutes les deux.

À la lecture, il valide ce que le collecteur a écrit dans `mesure`. Les
contraintes de la base font foi, mais elles ne couvrent pas tout : un lot lu
puis mal projeté, une colonne rendue dans un type inattendu par le driver, une
qualification devenue impossible passeraient sans bruit, seraient agrégés en
variables plausibles, et l'anomalie ne se verrait qu'au moment où un modèle
entraîné dessus prédirait n'importe quoi. Trois étapes plus loin, et sans rien
pour remonter à la cause.

À l'écriture, il valide ce que l'entraînement lira. Une partition de variables
qui casse son contrat ne doit pas être publiée : elle est reproductible, on
peut la refaire, alors que la retirer après coup demande de savoir qui l'a
déjà lue.

L'échec est immédiat et bavard. `lazy=True` rassemble toutes les violations
d'un lot avant de lever, plutôt que de s'arrêter à la première : réparer un
schéma une colonne par exécution serait une perte de temps pure.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from etl.features import FeatureSpec
from predict_common.schemas import MEASURE_SCHEMA, features_schema

# Nombre de violations détaillées dans le message d'erreur. Au-delà, la liste
# devient illisible dans un journal de conteneur et le compte total suffit à
# dire l'ampleur du problème.
REPORTED_FAILURES = 10

logger = logging.getLogger(__name__)


class ContractError(RuntimeError):
    """Un artefact ne respecte pas le schéma de sa couche."""


def check_measures(frame: pd.DataFrame) -> pd.DataFrame:
    """Valide un lot lu dans `mesure` avant de le transformer."""
    return _check(frame, MEASURE_SCHEMA, "couche brute")


def check_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Valide une partition de variables avant de la publier."""
    schema = features_schema(spec.lag_hours, spec.rolling_window_h)
    return _check(frame, schema, f"couche des variables {spec.version}")


def _check(frame: pd.DataFrame, schema, layer: str) -> pd.DataFrame:
    """Applique un schéma et transforme son échec en erreur de contrat.

    Un lot vide est accepté sans validation : pandera ne peut pas contrôler le
    type d'une colonne sans valeur, et une journée sans mesure exploitable est
    un fait d'exploitation — une source arrêtée, un site neuf — pas une
    rupture de contrat.
    """
    if frame.empty:
        logger.info("%s : lot vide, contrat non applicable", layer)
        return frame
    try:
        return schema.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        raise ContractError(f"{layer} : {_summarize(exc)}") from exc


def _summarize(error: SchemaError | SchemaErrors) -> str:
    """Résume les violations d'un lot en un message lisible en journal."""
    cases = getattr(error, "failure_cases", None)
    if cases is None or not hasattr(cases, "head"):
        return str(error)
    total = len(cases)
    columns = [name for name in ("column", "check", "failure_case") if name in cases]
    extract = cases.head(REPORTED_FAILURES)[columns].to_dict(orient="records")
    hidden = total - REPORTED_FAILURES
    suffix = f" (et {hidden} autre(s))" if hidden > 0 else ""
    return f"{total} violation(s) du schéma : {extract}{suffix}"
