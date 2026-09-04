"""Schémas des couches d'artefacts : le contrat entre deux services.

Chaque frontière de la chaîne est un chemin plus un schéma. Le chemin est
construit par `paths`, le schéma est déclaré ici, et c'est leur couple qui
remplace l'appel de fonction que la découpe a supprimé.

Deux formes coexistent, et elles ne servent pas à la même chose.

Le schéma pandera valide le contenu : les types, les bornes, les valeurs
admises, ce qui peut être nul. Il est vérifié à l'écriture par le producteur
et à la lecture par le consommateur. Cette double vérification n'est pas une
redite : validée à l'écriture seule, une rupture de contrat serait imputée au
consommateur trois étapes plus loin ; validée à la lecture seule, une
partition fausse serait déjà publiée.

Le schéma pyarrow fixe les types du fichier. Sans lui, une colonne entièrement
nulle sur une journée s'écrirait en type `null`, et la lecture de deux
partitions dont l'une a des valeurs et l'autre non échouerait à la
concaténation — panne fréquente, tardive, et dont la cause est invisible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy
import pandas
import pandera.pandas as pa
import pyarrow

# Colonnes de mesure servies par la source. L'ordre est celui du schéma figé
# v1.0 de la table `mesure`, qui reste la référence commune aux équipes.
NUMERIC_COLUMNS = (
    "consumption_kw",
    "consumption_kwh",
    "voltage_v",
    "current_a",
    "power_factor",
    "temperature_celsius",
    "humidity_percent",
)

# Colonne cible des prévisions.
TARGET_COLUMN = "consumption_kw"

# Nom de l'horodatage tel que la source le sert. La table `mesure` l'appelle
# `ts` : le collecteur traduit à l'entrée, une fois, et c'est la seule
# frontière de renommage de la chaîne.
SOURCE_TIMESTAMP_COLUMN = "timestamp"
TIMESTAMP_COLUMN = "ts"
SITE_COLUMN = "site_id"

# Valeurs admises par la contrainte CHECK de `data_quality`, de la moins
# sévère à la plus sévère : cet ordre est celui de la comparaison.
QUALITY_GOOD = "good"
QUALITY_PARTIAL = "partial"
QUALITY_DEGRADED = "degraded"
QUALITY_CRITICAL = "critical"
DATA_QUALITY_VALUES = (
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    QUALITY_DEGRADED,
    QUALITY_CRITICAL,
)

# Qui a posé `data_quality` et `null_reasons`, migration
# 07_mesure_quality_source.sql.
#
# `data_quality` est NOT NULL DEFAULT 'good' : le schéma figé ne sait pas dire
# « pas encore qualifiée ». Le collecteur retombe donc sur le défaut quand la
# source se tait, et l'ETL repose la qualification à son passage. Sans cette
# colonne, les deux 'good' sont indiscernables, et une fenêtre non encore
# traitée par l'ETL passe pour une fenêtre saine — ce que l'indicateur de
# mesures dégradées publié par l'API métier lirait comme 0 % de dégradation.
QUALITY_SOURCE_COLUMN = "quality_source"
QUALITY_SOURCE_SOURCE = "source"
QUALITY_SOURCE_ETL = "etl"
QUALITY_SOURCE_VALUES = (QUALITY_SOURCE_SOURCE, QUALITY_SOURCE_ETL)

# Valeurs admises par la contrainte CHECK de `imputation_method`.
METHOD_NONE = "none"
METHOD_LOCF = "locf"
METHOD_INTERPOLATION = "interpolation"
IMPUTATION_METHODS = (METHOD_NONE, METHOD_LOCF, METHOD_INTERPOLATION)

UTC_DTYPE = "datetime64[ns, UTC]"

# Longueur maximale d'un identifiant de site, reprise du VARCHAR(20) de la
# table `mesure` : une valeur plus longue serait tronquée par la base.
SITE_ID_MAX_LENGTH = 20

# Bornes des CHECK que porte `mesure`. Les rejouer ici n'est pas une redite :
# une violation détectée avant l'insertion nomme la colonne et la ligne, là où
# PostgreSQL fait échouer le lot entier sans dire laquelle des mille est
# fautive.
POWER_FACTOR_RANGE = (0.0, 1.0)
HUMIDITY_RANGE = (0.0, 100.0)

FEATURES_LAYER = "features"


def is_sequence(value: object) -> bool:
    """Dit si une valeur est un tableau de motifs, quelle que soit sa forme.

    `null_reasons` est écrit en `list<string>` et relu par pyarrow sous forme
    de tableau numpy, pas de liste Python. Les deux sont le même contrat :
    n'accepter que la liste ferait échouer la validation sur une partition que
    le producteur venait pourtant d'écrire valide, ce qui est le pire des
    faux positifs — celui qui décrédibilise le contrôle.
    """
    return isinstance(value, (list, tuple, numpy.ndarray))


def _numeric_columns(nullable: bool) -> dict[str, pa.Column]:
    """Déclare les sept colonnes de mesure, toutes décimales."""
    return {
        name: pa.Column(float, nullable=nullable, required=True)
        for name in NUMERIC_COLUMNS
    }


# --- Couche brute : la table `mesure`, écrite par le collecteur, lue par l'ETL
#
# La couche brute n'est pas un fichier : c'est TimescaleDB. Le schéma ci-dessous
# ne remplace pas les contraintes de la base — elles font foi et vivent dans le
# repo enervision-db — il les rejoue en amont, là où l'erreur est encore
# imputable au bon service. Une violation détectée ici nomme la colonne ; la
# même violation laissée à PostgreSQL fait échouer l'insertion du lot entier
# sans dire laquelle des mille lignes est fautive.
#
# Tout y est nullable sauf l'horodatage et le site. Une valeur nulle n'est pas
# une valeur qui manque, c'est un capteur qui dit qu'il est tombé : la refuser
# ici perdrait la panne, qui est l'information à conserver. Le tri des nulls
# est le métier de l'ETL, pas celui du collecteur.

MEASURE_SCHEMA = pa.DataFrameSchema(
    {
        TIMESTAMP_COLUMN: pa.Column(UTC_DTYPE, nullable=False),
        SITE_COLUMN: pa.Column(
            str,
            nullable=False,
            checks=pa.Check.str_length(1, SITE_ID_MAX_LENGTH),
        ),
        **_numeric_columns(nullable=True),
        # Deux colonnes portent un CHECK en base. Une valeur hors bornes est
        # une donnée de source aberrante, pas une panne capteur : la laisser
        # passer ferait échouer l'insertion du lot entier.
        "power_factor": pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(*POWER_FACTOR_RANGE),
        ),
        "humidity_percent": pa.Column(
            float, nullable=True, checks=pa.Check.in_range(*HUMIDITY_RANGE)
        ),
        # Liste de chaînes : pandera n'a pas de type de colonne pour un
        # tableau, la contrainte porte donc sur la valeur. `TEXT[] NOT NULL`
        # en base, donc jamais nulle — le vide se dit par une liste vide.
        "null_reasons": pa.Column(
            object,
            nullable=False,
            checks=pa.Check(
                lambda series: series.map(is_sequence),
                error="null_reasons doit être une liste, jamais une valeur seule.",
            ),
        ),
        # `NOT NULL DEFAULT 'good'` en base : le schéma figé ne sait pas dire
        # « non qualifiée ». Le collecteur retombe donc sur le défaut quand la
        # source se tait, et l'ETL repose la qualification à son passage — voir
        # UNQUALIFIED dans collector.sink.
        "data_quality": pa.Column(
            str, nullable=False, checks=pa.Check.isin(DATA_QUALITY_VALUES)
        ),
    },
    # Une colonne inconnue de la source n'est pas une rupture de contrat : le
    # collecteur projette ce que la table accepte. Une colonne manquante, elle,
    # en est une.
    strict=False,
    coerce=True,
    unique=[SITE_COLUMN, TIMESTAMP_COLUMN],
    name="mesure",
)


# --- Couche des variables : ce que l'ETL écrit, ce que l'entraînement lit ---
#
# La cible y est obligatoire et non nulle. C'est ce qui distingue cette couche
# de la précédente : une heure sans consommation exploitable n'est pas une
# ligne d'apprentissage dégradée, c'est une ligne qui n'a pas sa place. Elle a
# été écartée par `etl.exclude`, en amont, avec sa cause.

# Deux jeux de colonnes, et non un seul, parce que « ce que l'ETL publie » et
# « ce que le modèle consomme » ne sont pas la même chose.
#
# La température est une mesure réelle, et la partition la garde : elle sert à
# l'analyse, et elle servira au modèle le jour où une prévision météo
# alimentera l'inférence. Mais le service d'inférence, lui, ne connaît pas la
# température des heures à venir — il la présenterait vide à chaque prédiction.
# Un modèle entraîné dessus apprendrait des séparations qu'il ne pourrait plus
# emprunter en production : chaque arbre qui teste la température enverrait
# toutes les lignes servies dans sa branche par défaut. Ce n'est pas une
# information perdue proprement, c'est un biais fixe que rien ne signale.
#
# Les deux listes se recouvrent donc partiellement, et c'est voulu : les sortir
# d'ici plutôt que de les écrire deux fois est ce qui empêche l'ETL et
# l'entraînement de diverger sans que rien ne le dise.
#
# Une troisième catégorie s'y ajoute : les colonnes que le modèle consomme sans
# qu'elles soient écrites nulle part, parce qu'elles se déduisent d'une colonne
# déjà publiée. L'encodage cyclique de l'heure en est le cas : il est une pure
# fonction de `hour`, que la partition porte et que le service d'inférence
# reconstruit seul depuis l'horodatage. Les écrire en parquet stockerait deux
# fois la même information et imposerait une nouvelle `feature_version` pour un
# calcul de deux lignes.

PUBLISHED_FIXED_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_celsius",
)

MODEL_FIXED_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
)

# Variables dérivées à la lecture, jamais écrites dans la partition. Voir
# `add_derived_calendar` pour la raison d'être de cet encodage.
MODEL_DERIVED_COLUMNS = (
    "hour_sin",
    "hour_cos",
)

# Bornes calendaires, écrites une fois pour que la contrainte du schéma et le
# calcul qui la remplit ne puissent pas diverger.
HOUR_RANGE = (0, 23)
DAY_OF_WEEK_RANGE = (0, 6)

# Période de l'encodage cyclique de l'heure. Vingt-quatre et non `HOUR_RANGE`
# : c'est le nombre d'heures d'un tour de cadran, pas la borne haute de la
# colonne. Les confondre décalerait tout l'encodage d'un vingt-quatrième.
HOURS_PER_DAY = 24


def lag_column(hours: int) -> str:
    """Nom de la colonne portant le décalage de `hours` heures."""
    return f"lag_{hours}h"


def rolling_column(hours: int) -> str:
    """Nom de la colonne portant la moyenne glissante sur `hours` heures."""
    return f"roll_mean_{hours}h"


def cyclic_hour(hour: int | pandas.Series) -> tuple[Any, Any]:
    """Projette une heure sur le cercle des vingt-quatre heures.

    Rend le couple (sinus, cosinus) de l'angle correspondant, de la même forme
    que ce qu'elle reçoit : deux flottants pour une heure, deux séries pour une
    colonne. C'est la même fonction qui sert à l'entraînement, qui la lit sur
    une colonne, et au service d'inférence, qui la calcule pour une heure à la
    fois.

    Pourquoi cet encodage. `hour` va de 0 à 23, et cette échelle dit que 23 h
    et 0 h sont séparées de vingt-trois unités alors qu'elles se suivent. Un
    modèle qui découpe cette échelle par seuils ne peut jamais placer une
    frontière qui réunit la fin d'une journée et le début de la suivante : il
    lui faut deux règles séparées, apprises indépendamment, pour un seul
    phénomène. Le couple sinus/cosinus rétablit cette continuité, et il en faut
    deux : le sinus seul confondrait deux heures symétriques de la journée.

    Écrite ici, dans la bibliothèque partagée, et non dans l'un des deux
    services : deux implémentations de la même formule finiraient par diverger,
    et l'écart ne se verrait que dans la qualité des prévisions servies.
    """
    angle = 2 * numpy.pi * hour / HOURS_PER_DAY
    return numpy.sin(angle), numpy.cos(angle)


def add_derived_calendar(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Ajoute au lot les variables calendaires dérivées de `hour`.

    Rend le lot inchangé s'il ne porte pas `hour` : c'est au contrôle des
    colonnes manquantes, en aval, de dire ce qui manque et de nommer la
    colonne. Échouer ici donnerait la même panne sous un message qui parle
    d'une colonne que l'appelant n'a jamais demandée.

    Les colonnes ainsi produites ne sont dans aucune partition : elles sont
    reconstruites à chaque lecture, ce qui est exactement ce qui dispense d'une
    nouvelle `feature_version`.
    """
    if "hour" not in frame.columns:
        return frame
    enriched = frame.copy()
    sin, cos = cyclic_hour(enriched["hour"])
    enriched["hour_sin"] = sin
    enriched["hour_cos"] = cos
    return enriched


def feature_columns(lag_hours: Sequence[int], rolling_window_h: int) -> tuple[str, ...]:
    """Retourne les variables explicatives, dans l'ordre attendu du modèle.

    L'ordre compte : c'est celui de la signature MLflow, donc celui dans lequel
    le service d'inférence doit présenter ses colonnes. Le déduire d'un même
    appel des deux côtés est ce qui empêche l'entraînement et le service de
    diverger sans que rien ne le dise.

    La température n'en fait pas partie : voir `MODEL_FIXED_COLUMNS`. La
    surveillance de dérive lit cette même liste, si bien qu'elle mesure la
    tâche que le service rend vraiment, et non une tâche plus facile.

    Les colonnes dérivées suivent immédiatement les colonnes calendaires dont
    elles sortent, et précèdent les décalages : l'ordre suit la provenance, ce
    qui rend la signature lisible pour qui ouvre le registre.
    """
    return (
        *MODEL_FIXED_COLUMNS,
        *MODEL_DERIVED_COLUMNS,
        *(lag_column(hours) for hours in lag_hours),
        rolling_column(rolling_window_h),
    )


def published_columns(
    lag_hours: Sequence[int],
    rolling_window_h: int,
) -> tuple[str, ...]:
    """Retourne les colonnes calculées que l'ETL écrit dans la partition.

    Les deux listes se recouvrent sans que l'une contienne l'autre. La
    partition porte en plus ce que le modèle ne consomme pas encore, la
    température au premier chef ; le modèle consomme en plus ce qui se déduit
    d'une colonne publiée sans être écrit, `MODEL_DERIVED_COLUMNS`.

    Retirer une colonne d'ici change le contrat de la couche, donc impose une
    nouvelle `feature_version` ; en retirer une de `feature_columns` ne change
    que le modèle, que MLflow versionne déjà. C'est pour cela qu'une variable
    dérivée à la lecture ne coûte pas de version : elle n'entre jamais ici.
    """
    return (
        *PUBLISHED_FIXED_COLUMNS,
        *(lag_column(hours) for hours in lag_hours),
        rolling_column(rolling_window_h),
    )


def features_schema(
    lag_hours: Sequence[int],
    rolling_window_h: int,
) -> pa.DataFrameSchema:
    """Construit le schéma pandera de la couche des variables.

    Le schéma dépend de la configuration parce que les décalages en dépendent :
    changer `etl.lag_hours` change les colonnes produites, et c'est pour cela
    que ce changement s'accompagne d'une nouvelle `feature_version`.
    """
    derived = {
        name: pa.Column(float, nullable=False)
        for name in (
            *(lag_column(hours) for hours in lag_hours),
            rolling_column(rolling_window_h),
        )
    }
    return pa.DataFrameSchema(
        {
            TIMESTAMP_COLUMN: pa.Column(UTC_DTYPE, nullable=False),
            SITE_COLUMN: pa.Column(
                str,
                nullable=False,
                checks=pa.Check.str_length(1, SITE_ID_MAX_LENGTH),
            ),
            TARGET_COLUMN: pa.Column(float, nullable=False),
            "hour": pa.Column(
                int, nullable=False, checks=pa.Check.in_range(*HOUR_RANGE)
            ),
            "day_of_week": pa.Column(
                int, nullable=False, checks=pa.Check.in_range(*DAY_OF_WEEK_RANGE)
            ),
            "is_weekend": pa.Column(int, nullable=False, checks=pa.Check.isin((0, 1))),
            # La température reste nullable : un site sans capteur thermique
            # produit quand même une série de consommation exploitable, et
            # XGBoost gère nativement l'absence.
            "temperature_celsius": pa.Column(float, nullable=True),
            **derived,
            "data_quality": pa.Column(
                str, nullable=False, checks=pa.Check.isin(DATA_QUALITY_VALUES)
            ),
            # Part de l'heure reconstruite par l'ETL, entre 0 et 1. Elle dit à
            # l'entraînement ce que la cible doit à l'imputation, sans le
            # décider à sa place.
            "imputed_ratio": pa.Column(
                float, nullable=False, checks=pa.Check.in_range(0.0, 1.0)
            ),
        },
        strict=True,
        coerce=True,
        unique=[SITE_COLUMN, TIMESTAMP_COLUMN],
        name=f"features[{','.join(str(hours) for hours in lag_hours)}]",
    )


def features_arrow_schema(
    lag_hours: Sequence[int],
    rolling_window_h: int,
) -> pyarrow.Schema:
    """Construit le schéma pyarrow correspondant, qui fixe les types écrits."""
    derived = [
        (name, pyarrow.float64())
        for name in (
            *(lag_column(hours) for hours in lag_hours),
            rolling_column(rolling_window_h),
        )
    ]
    return pyarrow.schema(
        [
            (TIMESTAMP_COLUMN, pyarrow.timestamp("us", tz="UTC")),
            (SITE_COLUMN, pyarrow.string()),
            (TARGET_COLUMN, pyarrow.float64()),
            ("hour", pyarrow.int32()),
            ("day_of_week", pyarrow.int32()),
            ("is_weekend", pyarrow.int32()),
            ("temperature_celsius", pyarrow.float64()),
            *derived,
            ("data_quality", pyarrow.string()),
            ("imputed_ratio", pyarrow.float64()),
        ]
    )
