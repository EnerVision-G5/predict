"""Construction du jeu d'apprentissage à partir de la table `mesure`.

Les mesures exclues (`mesure_exclu`) sont écartées ici et nulle part ailleurs :
un modèle entraîné sur des valeurs que l'équipe a jugées aberrantes apprendrait
ces aberrations.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

TARGET_COLUMN = "consumption_kw"

# Décalages horaires retenus comme variables explicatives : l'heure précédente
# porte l'inertie thermique, 24 h et 168 h portent les saisonnalités
# journalière et hebdomadaire des sites tertiaires.
LAG_HOURS = (1, 24, 168)

FEATURE_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_celsius",
    *(f"lag_{lag}h" for lag in LAG_HOURS),
)

TRAINING_QUERY = text(
    """
    SELECT m.ts, m.site_id, m.consumption_kw, m.temperature_celsius
    FROM mesure AS m
    LEFT JOIN mesure_exclu AS e
      ON e.site_id = m.site_id AND e.ts = m.ts
    WHERE m.site_id = :site_id
      AND m.ts >= :start_time
      AND m.ts < :end_time
      AND e.exclusion_id IS NULL
    ORDER BY m.ts
    """
)


def read_measures(
    engine: Engine,
    site_id: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Lit les mesures retenues d'un site sur une fenêtre."""
    with engine.connect() as connection:
        return pd.read_sql_query(
            TRAINING_QUERY,
            connection,
            params={
                "site_id": site_id,
                "start_time": start_time.to_pydatetime(),
                "end_time": end_time.to_pydatetime(),
            },
            parse_dates=["ts"],
        )


def build_features(measures: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables calendaires et les décalages au jeu de mesures.

    Les lignes dont un décalage manque sont retirées : elles n'ont pas
    d'historique suffisant, et les garder reviendrait à imputer une valeur que
    le modèle prendrait pour une observation.

    Les décalages sont calculés en nombre de lignes, ce qui suppose une série
    horaire sans trou. Une coupure de capteur décalerait donc silencieusement
    l'historique : rééchantillonner sur un pas horaire avant d'appeler cette
    fonction est le travail d'EV-20, pas de ce squelette.
    """
    frame = measures.sort_values("ts").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(frame["ts"], utc=True)
    frame["hour"] = timestamps.dt.hour
    frame["day_of_week"] = timestamps.dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    for lag in LAG_HOURS:
        frame[f"lag_{lag}h"] = frame[TARGET_COLUMN].shift(lag)
    return frame.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS]).reset_index(
        drop=True
    )


def split_train_test(
    frame: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coupe le jeu en apprentissage et test, sans mélanger les dates.

    Une coupe aléatoire laisserait le modèle voir le futur d'une même journée
    pendant l'apprentissage et gonflerait artificiellement le score.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio doit être strictement entre 0 et 1.")
    cutoff = int(len(frame) * (1.0 - test_ratio))
    return frame.iloc[:cutoff].copy(), frame.iloc[cutoff:].copy()
