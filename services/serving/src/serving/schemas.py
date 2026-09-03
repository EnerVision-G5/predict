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
    history_end: datetime = Field(
        description=(
            "Derniere heure observee sur laquelle la prevision s'appuie, ISO"
            " 8601 UTC. Distincte de generated_at : le service predit a partir"
            " des variables publiees par l'ETL, qui peuvent dater."
        ),
    )
    feature_lag_hours: float = Field(
        description=(
            "Age des variables ayant servi la prevision, en heures, ecart"
            " entre generated_at et history_end. Une prevision calculee sur"
            " des variables vieilles de trois jours n'est pas fausse, elle est"
            " aveugle : ce champ est la seule chose qui le dise."
        ),
    )
    points: list[PredictionPoint] = Field(
        description="Série prédite, triée par horodatage croissant.",
    )


class ReadinessOut(BaseModel):
    """Ce dont le service dispose pour servir une prévision.

    Distincte de HealthOut, et la distinction n'est pas cosmétique. `/health`
    est la sonde de vivacité de l'hébergeur et ne consulte rien : la lier au
    registre ferait redémarrer un service en parfait état chaque fois que
    MLflow tousse. Cette route-ci consulte, et personne ne redémarre rien
    dessus — elle sert à savoir POURQUOI une prévision est indisponible, ce
    qu'un 503 nu ne dit pas.
    """

    model_config = ConfigDict(from_attributes=True)

    ready: bool = Field(
        description=(
            "Vrai quand le service peut servir une prevision : un modele est"
            " resolu et des variables existent."
        ),
    )
    model_resolved: bool = Field(
        description="Vrai quand le registre MLflow a resolu un modele.",
    )
    model_version: str | None = Field(
        description="Version servie, nulle quand aucun modele n'est resolu.",
    )
    features_available: bool = Field(
        description=(
            "Vrai quand au moins une partition de variables est lisible sur la"
            " fenetre de recul configuree."
        ),
    )
    history_end: datetime | None = Field(
        description=(
            "Derniere heure disponible dans les variables, ISO 8601 UTC."
            " Nulle quand aucune partition n'est lisible."
        ),
    )
    detail: str = Field(
        description="Cause de l'indisponibilite, vide quand le service est pret.",
    )
