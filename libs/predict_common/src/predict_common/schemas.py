# **********************************************************************
# * Nom     : schemas.py                                               *
# * Type    : Module                                                   *
# * Sujet   : Contrats des couches : noms de colonnes, valeurs admises *
# *   et schémas de validation                                         *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

from collections.abc import Sequence

import numpy
import pandera.pandas as pa
import pyarrow

# Colonnes numériques de la couche brute.
NUMERIC_COLUMNS = (
    "consumption_kw",
    "consumption_kwh",
    "voltage_v",
    "current_a",
    "power_factor",
    "temperature_celsius",
    "humidity_percent",
)

# Colonne que le modèle apprend à prédire.
TARGET_COLUMN = "consumption_kw"

# Nom de l'horodatage tel que la source le rend.
SOURCE_TIMESTAMP_COLUMN = "timestamp"
# Nom de l'horodatage dans toute la chaîne.
TIMESTAMP_COLUMN = "ts"
# Nom de l'identifiant de site dans toute la chaîne.
SITE_COLUMN = "site_id"

# Qualification d'une heure sans défaut constaté.
QUALITY_GOOD = "good"
# Qualification d'une heure incomplète mais exploitable.
QUALITY_PARTIAL = "partial"
# Qualification d'une heure dont la valeur est douteuse.
QUALITY_DEGRADED = "degraded"
# Qualification d'une heure inexploitable.
QUALITY_CRITICAL = "critical"
# Qualifications admises, de la meilleure à la pire.
DATA_QUALITY_VALUES = (
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    QUALITY_DEGRADED,
    QUALITY_CRITICAL,
)

# Colonne disant QUI a posé la qualification.
QUALITY_SOURCE_COLUMN = "quality_source"
# Qualification posée par le collecteur, faute de mieux.
QUALITY_SOURCE_SOURCE = "source"
# Qualification déduite des données par l'ETL.
QUALITY_SOURCE_ETL = "etl"
# Auteurs admis d'une qualification.
QUALITY_SOURCE_VALUES = (QUALITY_SOURCE_SOURCE, QUALITY_SOURCE_ETL)

# Valeur brute, ou rien à reconstruire.
METHOD_NONE = "none"
# Valeur reconstruite par report de la dernière connue.
METHOD_LOCF = "locf"
# Valeur reconstruite entre deux valeurs connues.
METHOD_INTERPOLATION = "interpolation"
# Méthodes de reconstruction admises.
IMPUTATION_METHODS = (METHOD_NONE, METHOD_LOCF, METHOD_INTERPOLATION)

# Type pandas exigé de tout horodatage de la chaîne.
UTC_DTYPE = "datetime64[ns, UTC]"

# Longueur maximale d'un identifiant de site en base.
SITE_ID_MAX_LENGTH = 20

# Bornes physiques du facteur de puissance.
POWER_FACTOR_RANGE = (0.0, 1.0)
# Bornes physiques de l'humidité relative, en pourcent.
HUMIDITY_RANGE = (0.0, 100.0)

# Nom de la couche des variables dans les chemins.
FEATURES_LAYER = "features"


def is_sequence(value: object) -> bool:
    """Méthode : is_sequence
    Description : Dit si une valeur est une liste, un tuple ou un tableau.
    """
    return isinstance(value, (list, tuple, numpy.ndarray))


def _numeric_columns(nullable: bool) -> dict[str, pa.Column]:
    """Méthode : _numeric_columns
    Description : Décrit les colonnes numériques de la couche brute.
    """
    return {
        name: pa.Column(float, nullable=nullable, required=True)
        for name in NUMERIC_COLUMNS
    }


# Contrat de la couche brute, vérifié à la lecture.
MEASURE_SCHEMA = pa.DataFrameSchema(
    {
        TIMESTAMP_COLUMN: pa.Column(UTC_DTYPE, nullable=False),
        SITE_COLUMN: pa.Column(
            str,
            nullable=False,
            checks=pa.Check.str_length(1, SITE_ID_MAX_LENGTH),
        ),
        **_numeric_columns(nullable=True),
        "power_factor": pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(*POWER_FACTOR_RANGE),
        ),
        "humidity_percent": pa.Column(
            float, nullable=True, checks=pa.Check.in_range(*HUMIDITY_RANGE)
        ),
        "null_reasons": pa.Column(
            object,
            nullable=False,
            checks=pa.Check(
                lambda series: series.map(is_sequence),
                error="null_reasons doit être une liste, jamais une valeur seule.",
            ),
        ),
        "data_quality": pa.Column(
            str, nullable=False, checks=pa.Check.isin(DATA_QUALITY_VALUES)
        ),
    },
    strict=False,
    coerce=True,
    unique=[SITE_COLUMN, TIMESTAMP_COLUMN],
    name="mesure",
)


# Colonnes de calendrier et de météo publiées telles quelles.
PUBLISHED_FIXED_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "temperature_celsius",
)

# Colonnes de calendrier que le modèle reçoit.
MODEL_FIXED_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
)

# Bornes de l'heure du jour.
HOUR_RANGE = (0, 23)
# Bornes du jour de la semaine, lundi valant zéro.
DAY_OF_WEEK_RANGE = (0, 6)


def lag_column(hours: int) -> str:
    """Méthode : lag_column
    Description : Nomme la colonne d'un décalage exprimé en heures.
    """
    return f"lag_{hours}h"


def rolling_column(hours: int) -> str:
    """Méthode : rolling_column
    Description : Nomme la colonne d'une moyenne glissante en heures.
    """
    return f"roll_mean_{hours}h"


def feature_columns(lag_hours: Sequence[int], rolling_window_h: int) -> tuple[str, ...]:
    """Méthode : feature_columns
    Description : Énumère les colonnes que le modèle reçoit en entrée.
    """
    return (
        *MODEL_FIXED_COLUMNS,
        *(lag_column(hours) for hours in lag_hours),
        rolling_column(rolling_window_h),
    )


def published_columns(
    lag_hours: Sequence[int],
    rolling_window_h: int,
) -> tuple[str, ...]:
    """Méthode : published_columns
    Description : Énumère les colonnes écrites dans une partition de variables.
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
    """Méthode : features_schema
    Description : Construit le contrat d'une partition de variables pour des
      décalages donnés.
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
            "temperature_celsius": pa.Column(float, nullable=True),
            **derived,
            "data_quality": pa.Column(
                str, nullable=False, checks=pa.Check.isin(DATA_QUALITY_VALUES)
            ),
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
    """Méthode : features_arrow_schema
    Description : Construit le schéma parquet correspondant, types figés.
    """
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
