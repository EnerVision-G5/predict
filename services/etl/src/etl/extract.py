"""Lecture des mesures brutes dans `mesure`, sur une fenêtre de journées.

L'ETL lit ce que le collecteur a déposé. Il ne rappelle jamais la source : le
collecteur est seul à la joindre, et une seconde lecture donnerait une seconde
version de la vérité amont — celle de l'API à l'instant de l'ETL, qui n'est pas
celle qu'on a stockée.

La fenêtre lue déborde volontairement sur les journées précédentes. Le décalage
de 168 heures d'une heure du 2 septembre désigne une heure du 26 août : lire la
seule journée demandée donnerait des décalages vides, et elle sortirait presque
entièrement écartée sans que rien ne l'explique.

Les mesures déjà écartées par un run précédent sont exclues à la lecture, et
c'est le seul endroit où `mesure_exclu` est consultée. Une mesure qu'un
analyste a jugée aberrante ne doit pas revenir nourrir les variables au
prochain rejeu, sous prétexte qu'elle est toujours dans `mesure` — elle y est
justement parce qu'on ne supprime pas les pannes.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from predict_common.db import SOURCE_COLUMNS

logger = logging.getLogger(__name__)

SELECTED = ", ".join(f"m.{name}" for name in SOURCE_COLUMNS)

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
    """Retourne la fenêtre UTC couvrant le jour produit et son historique.

    Bornes `[début, fin[` : la borne haute est exclue pour qu'une mesure de
    minuit pile appartienne au lendemain et à lui seul, sans quoi elle serait
    lue deux fois par deux runs voisins.
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
    """Lit les mesures retenues de la fenêtre, exclusions déjà écartées.

    Le tableau retourné porte toujours les colonnes de la source, même vide :
    les étages suivants n'ont pas de cas dégénéré à traiter.
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
