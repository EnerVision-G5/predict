# **********************************************************************
# * Nom     : validate.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Vérification des contrats de couche, à la lecture comme  *
# *   à l'écriture                                                     *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import logging

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from etl.features import FeatureSpec
from predict_common.schemas import MEASURE_SCHEMA, features_schema

# Nombre de violations détaillées dans le message d'erreur.
REPORTED_FAILURES = 10

logger = logging.getLogger(__name__)


class ContractError(RuntimeError):
    """Classe : ContractError
    Description : Un artefact ne respecte pas le schéma de sa couche.
    """


def check_measures(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : check_measures
    Description : Valide un lot lu dans mesure avant de le transformer.
    """
    return _check(frame, MEASURE_SCHEMA, "couche brute")


def check_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Méthode : check_features
    Description : Valide une partition de variables avant de la publier.
    """
    schema = features_schema(spec.lag_hours, spec.rolling_window_h)
    return _check(frame, schema, f"couche des variables {spec.version}")


def _check(frame: pd.DataFrame, schema, layer: str) -> pd.DataFrame:
    """Méthode : _check
    Description : Applique un schéma et transforme son échec en erreur de
      contrat.
    """
    if frame.empty:
        logger.info("%s : lot vide, contrat non applicable", layer)
        return frame
    try:
        return schema.validate(frame, lazy=True)
    except (SchemaErrors, SchemaError) as exc:
        raise ContractError(f"{layer} : {_summarize(exc)}") from exc


def _summarize(error: SchemaError | SchemaErrors) -> str:
    """Méthode : _summarize
    Description : Résume les violations d'un lot en un message lisible en
      journal.
    """
    cases = getattr(error, "failure_cases", None)
    if cases is None or not hasattr(cases, "head"):
        return str(error)
    total = len(cases)
    columns = [name for name in ("column", "check", "failure_case") if name in cases]
    extract = cases.head(REPORTED_FAILURES)[columns].to_dict(orient="records")
    hidden = total - REPORTED_FAILURES
    suffix = f" (et {hidden} autre(s))" if hidden > 0 else ""
    return f"{total} violation(s) du schéma : {extract}{suffix}"
