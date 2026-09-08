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

import numpy
import pandera.pandas as pa
import pyarrow

NUMERIC_COLUMNS = (
    "consumption_kw",
    "consumption_kwh",
    "voltage_v",
    "current_a",
    "power_factor",
    "temperature_celsius",
    "humidity_percent",
)

TARGET_COLUMN = "consumption_kw"

SOURCE_TIMESTAMP_COLUMN = "timestamp"
TIMESTAMP_COLUMN = "ts"
SITE_COLUMN = "site_id"

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

QUALITY_SOURCE_COLUMN = "quality_source"
QUALITY_SOURCE_SOURCE = "source"
QUALITY_SOURCE_ETL = "etl"
QUALITY_SOURCE_VALUES = (QUALITY_SOURCE_SOURCE, QUALITY_SOURCE_ETL)

METHOD_NONE = "none"
METHOD_LOCF = "locf"
METHOD_INTERPOLATION = "interpolation"
IMPUTATION_METHODS = (METHOD_NONE, METHOD_LOCF, METHOD_INTERPOLATION)

UTC_DTYPE = "datetime64[ns, UTC]"

SITE_ID_MAX_LENGTH = 20

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

HOUR_RANGE = (0, 23)
DAY_OF_WEEK_RANGE = (0, 6)


def lag_column(hours: int) -> str:
    """Nom de la colonne portant le décalage de `hours` heures."""
    return f"lag_{hours}h"


def rolling_column(hours: int) -> str:
    """Nom de la colonne portant la moyenne glissante sur `hours` heures."""
    return f"roll_mean_{hours}h"


def feature_columns(lag_hours: Sequence[int], rolling_window_h: int) -> tuple[str, ...]:
    """Retourne les variables explicatives, dans l'ordre attendu du modèle.

    L'ordre compte : c'est celui de la signature MLflow, donc celui dans lequel
    le service d'inférence doit présenter ses colonnes. Le déduire d'un même
    appel des deux côtés est ce qui empêche l'entraînement et le service de
    diverger sans que rien ne le dise.

    La température n'en fait pas partie : voir `MODEL_FIXED_COLUMNS`. La
    surveillance de dérive lit cette même liste, si bien qu'elle mesure la
    tâche que le service rend vraiment, et non une tâche plus facile.
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
    """Retourne les colonnes calculées que l'ETL écrit dans la partition.

    Sur-ensemble de `feature_columns` : la partition porte en plus ce que le
    modèle ne consomme pas encore. Retirer une colonne d'ici change le contrat
    de la couche, donc impose une nouvelle `feature_version` ; en retirer une
    de `feature_columns` ne change que le modèle, que MLflow versionne déjà.
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
