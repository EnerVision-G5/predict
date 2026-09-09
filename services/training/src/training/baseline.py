# **********************************************************************
# * Nom     : baseline.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Baselines naïves auxquelles tout modèle doit se comparer *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from predict_common.schemas import lag_column


class BaselineError(ValueError):
    """Classe : BaselineError
    Description : La baseline demandée ne peut pas être mesurée sur ce lot.
    """


@dataclass(frozen=True)
class Persistence:
    """Classe : Persistence
    Description : Recopie la valeur d'il y a N heures, sans rien apprendre.
    """
    hours: int

    @property
    def name(self) -> str:
        """Méthode : name
        Description : Nom lisible de la baseline, tel qu'il apparaît dans les
          journaux.
        """
        return f"persistance-{self.hours}h"

    @property
    def column(self) -> str:
        """Méthode : column
        Description : Colonne de décalage que cette persistance recopie.
        """
        return lag_column(self.hours)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Méthode : predict
        Description : Rend la valeur décalée, ou refuse si la colonne manque.
        """
        if self.column not in features.columns:
            raise BaselineError(
                f"Colonne {self.column} absente : la persistance à"
                f" {self.hours} h ne peut pas être mesurée."
            )
        return features[self.column]


def naive_baselines(lag_hours: Sequence[int]) -> tuple[Persistence, ...]:
    """Méthode : naive_baselines
    Description : Construit une persistance par décalage déclaré.
    """
    return tuple(Persistence(hours=hours) for hours in lag_hours)
