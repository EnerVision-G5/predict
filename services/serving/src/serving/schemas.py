"""DTO du service d'inférence EnerVision déployé sur Azure.

Source de vérité du contrat. Toute modification exige une PR sur
enervision/docs/contracts et la relecture des trois consommateurs.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Corps de réponse commun à toutes les erreurs du service d'inférence."""

    model_config = ConfigDict(from_attributes=True)

    detail: str = Field(
        description="Message d'erreur destiné au consommateur de l'API."
    )


class HealthOut(BaseModel):
    """État de disponibilité du service d'inférence."""

    model_config = ConfigDict(from_attributes=True)

    status: Literal["ok"] = Field(description="Statut du service, ok s'il répond.")
    timestamp: datetime = Field(description="Horodatage de la réponse, ISO 8601 UTC.")


class PredictionRequest(BaseModel):
    """Demande de prévision de consommation pour un site."""

    model_config = ConfigDict(from_attributes=True)

    site_id: str = Field(description="Identifiant du site à prédire.")
    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=48,
        description="Profondeur de la prévision en heures, entre 1 et 48.",
    )


class PredictionPoint(BaseModel):
    """Point de la série prédite, avec son intervalle de confiance."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime = Field(
        description="Horodatage du point prédit, ISO 8601 UTC."
    )
    predicted_consumption_kw: float = Field(
        description="Puissance prédite en kilowatts.",
    )
    lower_bound_kw: float | None = Field(
        description="Borne basse de l'intervalle de confiance, nulle si non calculée.",
    )
    upper_bound_kw: float | None = Field(
        description="Borne haute de l'intervalle de confiance, nulle si non calculée.",
    )


class PredictionOut(BaseModel):
    """Prévision complète renvoyée pour un site."""

    model_config = ConfigDict(from_attributes=True)

    site_id: str = Field(description="Identifiant du site prédit.")
    model_version: str = Field(
        description="Tag MLflow du modèle ayant servi la réponse.",
    )
    generated_at: datetime = Field(
        description="Horodatage de production de la prévision, ISO 8601 UTC.",
    )
    points: list[PredictionPoint] = Field(
        description="Série prédite, triée par horodatage croissant.",
    )
