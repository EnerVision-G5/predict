"""DTO du service d'inférence EnerVision, déployé on-premise.

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
    history_source: str = Field(
        default="recent",
        description=(
            "Origine des observations qui ont nourri les decalages du modele."
            " 'recent' : les variables publiees par l'ETL sur la fenetre"
            " courante, cas normal. 'reference' : faute d'heures recentes"
            " continues, la prevision rejoue l'historique de reference decale"
            " d'un nombre entier d'annees de 52 semaines. Les heures predites"
            " restent celles qui viennent, mais la serie observee sur laquelle"
            " elles s'appuient ne decrit pas cette semaine-ci. Un consommateur"
            " qui affiche une prevision de repli sans le dire ferait passer"
            " pour une mesure du site ce qui est un profil de l'an dernier."
        ),
    )
    history_origin: datetime | None = Field(
        default=None,
        description=(
            "Derniere heure REELLEMENT observee dont la prevision descend,"
            " ISO 8601 UTC. Renseignee seulement quand history_source vaut"
            " 'reference' : history_end porte alors la date decalee, et ce"
            " champ est le seul a dire d'ou vient la serie."
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


class SourceSiteOut(BaseModel):
    """Site du référentiel de la source, relayé tel qu'elle le sert.

    Ce n'est pas le référentiel de la base : c'est celui de l'API Mock, que
    l'API métier vient chercher ici parce qu'elle ne connaît pas la source.
    Les champs sont ceux du contrat de la source, sans traduction.

    Tout est optionnel sauf l'identifiant, et ce n'est pas du laxisme : un
    relais qui exigerait la forme complète rendrait 500 dès qu'un seul site
    est mal décrit, emportant les six autres avec lui. Le consommateur écarte
    ce qu'il ne peut pas exploiter ; le relais, lui, rend ce qu'il a reçu.
    """

    model_config = ConfigDict(from_attributes=True)

    site_id: str = Field(description="Identifiant du site.")
    site_type: str | None = Field(
        default=None, description="Type de site : office, factory, datacenter."
    )
    site_name: str | None = Field(default=None, description="Nom lisible du site.")
    location: str | None = Field(
        default=None,
        description="Localisation du site, absente si la source ne la sert pas.",
    )
    capacity_kw: float | None = Field(
        default=None, description="Puissance installée en kilowatts."
    )
    status: str | None = Field(
        default=None, description="État déclaré : active ou inactive."
    )


class SpikeReadingOut(BaseModel):
    """Mesure constatée sur le site juste après le déclenchement d'un pic.

    Elle est lue et non calculée : la simulation agit sur la source, et c'est
    la source qui dit ce qu'elle sert désormais. Les null et leurs motifs
    traversent intacts, comme partout ailleurs dans la chaîne.
    """

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime = Field(description="Horodatage de la mesure, ISO 8601.")
    site_id: str = Field(description="Identifiant du site mesuré.")
    consumption_kw: float | None = Field(
        default=None, description="Puissance instantanée en kilowatts."
    )
    consumption_kwh: float | None = Field(
        default=None, description="Énergie sur la période en kilowattheures."
    )
    voltage_v: float | None = Field(default=None, description="Tension en volts.")
    current_a: float | None = Field(default=None, description="Intensité en ampères.")
    power_factor: float | None = Field(
        default=None, description="Facteur de puissance, entre 0 et 1."
    )
    temperature_celsius: float | None = Field(
        default=None, description="Température extérieure en degrés Celsius."
    )
    humidity_percent: float | None = Field(
        default=None, description="Humidité relative en pourcentage."
    )
    null_reasons: list[str] = Field(
        default_factory=list,
        description="Causes des valeurs manquantes, telles que la source les donne.",
    )
    data_quality: str = Field(
        description="Qualification de la source : good, partial, degraded, critical."
    )


class SpikeSimulationOut(BaseModel):
    """Résultat d'une simulation de pic : ce qui a été demandé, et ce qui suit.

    `reading` est la mesure relue immédiatement après le déclenchement. Elle
    est optionnelle parce qu'une source qui accepte le pic puis se tait sur
    `/current` a quand même déclenché le pic : rendre la simulation en échec
    ferait croire le contraire, et un second appel doublerait le pic.
    """

    model_config = ConfigDict(from_attributes=True)

    site_id: str = Field(description="Site sur lequel le pic a été déclenché.")
    status: str = Field(
        description="Statut rendu par la source, simulated en cas de succès."
    )
    event: str = Field(description="Nature de l'événement, consumption_spike.")
    duration_minutes: int = Field(description="Durée demandée du pic, en minutes.")
    message: str = Field(description="Message rendu par la source, lisible tel quel.")
    simulated_at: datetime = Field(
        description="Horodatage du déclenchement, ISO 8601 UTC."
    )
    reading: SpikeReadingOut | None = Field(
        default=None,
        description="Mesure relue après le pic, absente si la source s'est tue.",
    )
