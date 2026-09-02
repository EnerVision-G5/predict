"""Qualification des mesures : nommer la cause de chaque valeur absente.

Une valeur nulle n'est pas une valeur qui manque, c'est un capteur qui parle :
il dit qu'il est tombé. La perdre, ou la ranger sous un `data_quality` à
`good`, revient à effacer la panne. Ce module garantit donc deux choses avant
tout chargement — toute colonne nulle est nommée dans `null_reasons`, et
`data_quality` décrit ce que la mesure vaut réellement.

Les motifs de la source ne sont jamais réécrits : elle seule sait pourquoi son
capteur s'est tu, l'ETL ne fait que combler son silence.

Sa qualification, elle, ne peut pas être meilleure que ce que ses données
montrent. Des deux — celle de la source et celle que le lot laisse déduire —
c'est la plus sévère qui est retenue. La source peut donc alerter au-delà de ce
que l'ETL voit, jamais en deçà : un `good` posé sur une puissance nulle ferait
disparaître la panne de `idx_mesure_quality`, l'index d'audit des capteurs, qui
ne regarde justement que les mesures non `good`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

QUALITY_GOOD = "good"
QUALITY_PARTIAL = "partial"
QUALITY_DEGRADED = "degraded"
QUALITY_CRITICAL = "critical"

# Valeurs admises par la contrainte CHECK de la colonne data_quality, de la
# moins sévère à la plus sévère : cet ordre est celui de la comparaison.
DATA_QUALITY_VALUES = (
    QUALITY_GOOD,
    QUALITY_PARTIAL,
    QUALITY_DEGRADED,
    QUALITY_CRITICAL,
)

# Colonne cible des prévisions : son absence dégrade la mesure bien plus
# qu'une hygrométrie manquante, qui n'entre dans aucun agrégat de puissance.
TARGET_COLUMN = "consumption_kw"

# Suffixe des motifs ajoutés par l'ETL faute de motif de la source. Il dit ce
# qu'il est : un constat de l'ETL, et non une cause remontée du terrain.
DERIVED_REASON_SUFFIX = ":undeclared"

# Au-delà de ce nombre de capteurs muets, la mesure ne décrit plus le site :
# elle décrit la panne. En deçà, il reste de quoi l'exploiter partiellement.
DEGRADED_NULL_COUNT = 3


def derive_data_quality(missing: Sequence[str]) -> str:
    """Déduit la qualité d'une mesure des colonnes restées nulles."""
    if not missing:
        return QUALITY_GOOD
    if TARGET_COLUMN in missing:
        return QUALITY_CRITICAL
    if len(missing) >= DEGRADED_NULL_COUNT:
        return QUALITY_DEGRADED
    return QUALITY_PARTIAL


def complete_null_reasons(
    declared: Sequence[str],
    missing: Sequence[str],
) -> list[str]:
    """Nomme les colonnes nulles d'une mesure que la source n'a pas expliquée.

    Un seul motif de la source suffit à couvrir la ligne : `network_loss`
    explique aussi bien un capteur muet que sept. Y ajouter un motif par
    colonne inventerait des causes distinctes là où il n'y en a qu'une.
    """
    if declared:
        return [str(reason) for reason in declared]
    return [f"{column}{DERIVED_REASON_SUFFIX}" for column in missing]


def qualify(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Complète `null_reasons` et `data_quality` de chaque ligne du lot.

    `columns` porte les colonnes de mesure surveillées. Les passer en argument
    évite à ce module de connaître le schéma de la table, dont la
    normalisation reste seule responsable.
    """
    if frame.empty:
        return frame
    missing = missing_columns(frame, columns)
    frame["null_reasons"] = [
        complete_null_reasons(declared, absent)
        for declared, absent in zip(frame["null_reasons"], missing, strict=True)
    ]
    frame["data_quality"] = [
        _settle_quality(declared, absent)
        for declared, absent in zip(frame["data_quality"], missing, strict=True)
    ]
    return frame


def missing_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> list[tuple[str, ...]]:
    """Nomme, pour chaque ligne, les colonnes surveillées restées nulles."""
    watched = list(columns)
    absent = frame[watched].isna().to_numpy()
    return [
        tuple(
            column
            for column, is_absent in zip(watched, row, strict=True)
            if is_absent
        )
        for row in absent
    ]


def _settle_quality(declared: object, missing: Sequence[str]) -> str:
    """Retient la plus sévère des deux qualifications possibles.

    Une valeur hors du CHECK de la colonne est ignorée plutôt que soumise :
    elle ferait échouer l'insertion du lot entier, et pas seulement la sienne.
    """
    derived = derive_data_quality(missing)
    if declared not in DATA_QUALITY_VALUES:
        return derived
    return max(
        str(declared),
        derived,
        key=DATA_QUALITY_VALUES.index,
    )
