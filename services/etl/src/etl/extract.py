# **********************************************************************
# * Nom     : extract.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Lecture des mesures brutes sur une fenêtre, exclusions   *
# *   déjà écartées                                                    *
# * Service : etl                                                      *
# **********************************************************************

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from predict_common.db import SOURCE_COLUMNS

logger = logging.getLogger(__name__)

# Colonnes projetées par la requête, celles de la source seules.
SELECTED = ", ".join(f"m.{name}" for name in SOURCE_COLUMNS)

# Lit la fenêtre en écartant ce qu'un analyste a exclu.
MEASURES_QUERY = text(
    f"""
    SELECT {SELECTED}
    FROM mesure AS m
    LEFT JOIN mesure_exclu AS e
      ON e.site_id = m.site_id AND e.ts = m.ts
    WHERE m.ts >= :start_time
      AND m.ts < :end_time
      AND e.exclusion_id IS NULL
    ORDER BY m.site_id, m.ts
    """
)


def window(day: date, lookback_days: int) -> tuple[datetime, datetime]:
    """Méthode : window
    Description : Retourne la fenêtre UTC couvrant le jour produit et son
      historique.
    """
    if lookback_days < 1:
        raise ValueError("La fenêtre de lecture couvre au moins un jour.")
    end = datetime.combine(day, time.min, tzinfo=UTC) + timedelta(days=1)
    return end - timedelta(days=lookback_days), end


def read_measures(
    engine: Engine,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Méthode : read_measures
    Description : Lit les mesures retenues de la fenêtre et rend toujours les
      colonnes attendues.
    """
    with engine.connect() as connection:
        frame = pd.read_sql_query(
            MEASURES_QUERY,
            connection,
            params={"start_time": start_time, "end_time": end_time},
        )
    logger.info(
        "%d mesure(s) lue(s) entre %s et %s", len(frame), start_time, end_time
    )
    if frame.empty:
        return pd.DataFrame(columns=list(SOURCE_COLUMNS))
    return frame
