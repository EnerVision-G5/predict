"""Baselines naïves : ce qu'un modèle doit battre pour mériter d'être servi.

ADR-010 décide « XGBoost comparé systématiquement à une baseline naïve » et
range la baseline parmi les livrables permanents, pas parmi les étapes
jetables. Ce module est cette décision.

Une baseline naïve n'apprend rien : elle recopie une valeur déjà observée.
C'est précisément ce qui en fait une référence honnête — elle ne coûte ni
entraînement, ni registre, ni surveillance, et un modèle qui ne la bat pas ne
paie pas ce qu'il coûte. Sur une consommation horaire, la persistance est
redoutable : la consommation d'un bureau à 9 h ressemble beaucoup à celle
d'hier à 9 h, et un modèle qui n'apporte que quelques pour cent sur elle
apporte, en réalité, quelques pour cent.

Les prédicteurs sont des lectures de colonnes, pas des calculs : l'ETL a déjà
publié `lag_1h`, `lag_24h` et `lag_168h`, et les relire est exactement ce que
la persistance signifie. Ils n'ont donc aucune donnée d'apprentissage à
recevoir, et se mesurent sur le banc d'arbitrage comme n'importe quel
candidat — même fenêtre, même cible, mêmes métriques.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from predict_common.schemas import lag_column


class BaselineError(ValueError):
    """La partition ne porte pas la colonne dont la persistance a besoin."""


@dataclass(frozen=True)
class Persistence:
    """Prédit la consommation observée `hours` heures plus tôt.

    Le décalage n'est pas un hyperparamètre à régler mais le choix d'une
    saisonnalité : 1 h suit l'inertie, 24 h le cycle jour/nuit, 168 h le cycle
    hebdomadaire d'un site tertiaire fermé le week-end.
    """

    hours: int

    @property
    def name(self) -> str:
        """Nom porté par le run et par le classement du challenge."""
        return f"persistance-{self.hours}h"

    @property
    def column(self) -> str:
        """Colonne recopiée, celle que l'ETL publie pour ce décalage."""
        return lag_column(self.hours)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Retourne la prévision, c'est-à-dire la colonne de décalage.

        L'absence de la colonne est une erreur et non un repli : une
        persistance qui rendrait des zéros paraîtrait simplement très mauvaise
        et laisserait croire que le modèle appris l'écrase, alors que la
        comparaison n'aurait pas eu lieu.
        """
        if self.column not in features.columns:
            raise BaselineError(
                f"Colonne {self.column} absente : la persistance à"
                f" {self.hours} h ne peut pas être mesurée."
            )
        return features[self.column]


def naive_baselines(lag_hours: Sequence[int]) -> tuple[Persistence, ...]:
    """Retourne une persistance par décalage publié par l'ETL.

    La liste vient de `conf/` et non d'une énumération écrite ici : ajouter un
    décalage aux variables doit ajouter la baseline correspondante, sinon la
    référence vieillirait pendant que le modèle, lui, apprendrait la nouvelle.
    """
    return tuple(Persistence(hours=hours) for hours in lag_hours)
