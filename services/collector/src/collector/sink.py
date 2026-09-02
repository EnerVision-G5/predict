"""Écriture de la couche brute : la réponse de la source, dans `mesure`.

Ce module ne transforme rien. Il ne comble aucun trou et ne juge aucune
valeur : une consommation nulle reste nulle, avec ses motifs. C'est ce qui
permet de relire une ligne des mois plus tard et de la comparer à ce que l'API
a servi, sans avoir à rejouer l'ETL et sans se demander laquelle des deux
étapes a écrit quoi.

Deux choses seulement lui sont faites, et aucune ne touche à la mesure.

Le renommage `timestamp` → `ts`, parce que la table s'appelle ainsi. C'est la
seule frontière de renommage de la chaîne, et elle est traversée une fois.

La mise en conformité avec les contraintes de la table. `null_reasons` est
`TEXT[] NOT NULL` : l'absence de motif s'y écrit par une liste vide, pas par un
NULL. `data_quality` est `NOT NULL DEFAULT 'good'` avec un CHECK sur quatre
valeurs : le schéma figé ne sait pas dire « non qualifiée ».

Ce dernier point mérite d'être dit franchement, parce qu'il heurte la règle du
projet. Quand la source se tait ou envoie une valeur hors CHECK, le collecteur
écrit `good` — le défaut de la colonne — alors que rien ne l'atteste. C'est un
mensonge borné dans le temps : l'ETL repose la qualification à son passage, à
partir de ce que les données montrent, et un `good` sur une puissance absente
redevient `critical`. Entre les deux, une mesure non encore transformée
n'apparaît pas dans `idx_mesure_quality`. Les deux alternatives étaient pires :
refuser la mesure jetterait la panne qu'on cherche justement à garder, et un
NULL serait rejeté par la base.

L'écriture n'écrase jamais. Un `ON CONFLICT DO NOTHING` sur (site_id, ts) rend
le rejeu d'une journée sans effet de bord, et surtout : il empêche une
recollecte de recouvrir les colonnes que l'ETL a déduites depuis. Relancer le
collecteur sur une journée déjà transformée ne défait donc pas la
transformation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from predict_common.db import (
    CONFLICT_KEY,
    SITE_COLUMNS,
    SOURCE_COLUMNS,
    mesure,
    site,
    write_batches,
)
from predict_common.schemas import (
    DATA_QUALITY_VALUES,
    MEASURE_SCHEMA,
    NUMERIC_COLUMNS,
    QUALITY_GOOD,
    SITE_COLUMN,
    SOURCE_TIMESTAMP_COLUMN,
    TIMESTAMP_COLUMN,
)

# Qualification écrite faute de mieux quand la source se tait. C'est le DEFAULT
# de la colonne, repris explicitement plutôt que laissé à la base : un lot
# soumis en une instruction porte les mêmes clés pour toutes ses lignes, et
# omettre la colonne pour certaines n'est pas possible.
UNQUALIFIED = QUALITY_GOOD

logger = logging.getLogger(__name__)


# Colonnes que le seed `02_seed_sites.sql` marque « à synchroniser » : il pose
# sept sites dont quatre avec des capacités placeholders, et attend que le
# premier passage de la chaîne les remplace par ce que sert la source.
SITE_REQUIRED = ("site_id", "site_type", "site_name", "capacity_kw")

DEFAULT_SITE_STATUS = "active"


@dataclass(frozen=True)
class WriteReport:
    """Ce qu'une écriture a réellement soumis, et ce qu'elle a écarté."""

    rows: int
    dropped: int = 0


def to_measures(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Construit le tableau de mesures correspondant à ce que la source a servi.

    Le tableau retourné porte toujours les colonnes de `SOURCE_COLUMNS`, même
    pour un lot vide : l'appelant n'a pas de cas dégénéré à traiter.
    """
    rows = [_rename(record) for record in records]
    frame = pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))
    # Format ISO imposé, et non deviné : le contrat de la source l'annonce en
    # ISO 8601, et laisser pandas interpréter au cas par cas ferait accepter un
    # `03/09/2026` dont personne ne saurait dire si c'est mars ou septembre.
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], utc=True, errors="coerce", format="ISO8601"
    )
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[SITE_COLUMN] = frame[SITE_COLUMN].map(_clean_site_id)
    frame["null_reasons"] = frame["null_reasons"].map(_as_list)
    frame["data_quality"] = frame["data_quality"].map(_admitted_quality)
    return frame[list(SOURCE_COLUMNS)]


def drop_unplaceable(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Écarte les lignes qu'aucune clé primaire ne pourrait identifier.

    (site_id, ts) est la clé de `mesure` : sans l'un des deux, la ligne serait
    refusée par la base sans que rien ne dise laquelle du lot est fautive.
    Elle est comptée et journalisée ici, pas perdue en silence.
    """
    placeable = frame[frame[TIMESTAMP_COLUMN].notna() & frame[SITE_COLUMN].notna()]
    return placeable.reset_index(drop=True), len(frame) - len(placeable)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Ne garde qu'une ligne par clé, la dernière.

    Un lot qui porterait deux fois la même clé ferait échouer l'insertion
    entière : `ON CONFLICT` arbitre entre le lot et la table, pas à
    l'intérieur d'un même lot. La dernière occurrence gagne, elle correspond à
    la relecture la plus récente de la source.
    """
    return frame.drop_duplicates(
        subset=[SITE_COLUMN, TIMESTAMP_COLUMN], keep="last"
    ).reset_index(drop=True)


def select_day(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """Ne garde que les mesures du jour collecté.

    Une fenêtre demandée à la source déborde couramment d'un jour sur l'autre,
    par arrondi ou par fuseau. Écrire ce débord en croyant collecter une
    journée fausserait le compte rendu du run, seul moyen de savoir si la
    journée est complète.
    """
    if frame.empty:
        return frame
    return frame[frame[TIMESTAMP_COLUMN].dt.date == day].reset_index(drop=True)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Vérifie le contrat de la couche brute avant de soumettre le lot.

    La validation est faite par le producteur, à l'écriture : une mesure qui
    casse le contrat ne doit pas entrer dans la table. L'ETL la refera à la
    lecture, et ce n'est pas une redite — voir `predict_common.schemas`.
    """
    return MEASURE_SCHEMA.validate(frame, lazy=True)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertit le tableau en lignes acceptables par SQLAlchemy."""
    projected = frame[list(SOURCE_COLUMNS)]
    return [
        {key: _to_sql_value(value) for key, value in row.items()}
        for row in projected.to_dict(orient="records")
    ]


def build_insert(records: list[dict[str, Any]]) -> Any:
    """Construit l'insertion de mesures brutes, qui n'écrase jamais.

    `DO NOTHING` et non `DO UPDATE` : la ligne présente peut déjà porter les
    colonnes que l'ETL a déduites, et une recollecte n'a aucune raison de les
    effacer. La source ne réécrit pas le passé, elle le confirme.
    """
    return insert(mesure).values(records).on_conflict_do_nothing(
        index_elements=list(CONFLICT_KEY)
    )


def write(
    engine: Engine,
    frame: pd.DataFrame,
    batch_size: int,
    day: date | None = None,
) -> WriteReport:
    """Soumet un lot de mesures brutes et retourne ce qu'il a déposé."""
    placeable, dropped = drop_unplaceable(frame)
    selected = deduplicate(placeable)
    if day is not None:
        selected = select_day(selected, day)
    rows = write_batches(
        engine, to_records(validate(selected)), batch_size, build_insert
    )
    if dropped:
        logger.warning("%d mesure(s) écartée(s) : horodatage ou site absent", dropped)
    return WriteReport(rows=rows, dropped=dropped)


def to_sites(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retient les sites que la table `site` peut accueillir.

    Trois de ses colonnes sont NOT NULL sans défaut : un site que la source
    décrirait à moitié serait rejeté, et avec lui le lot entier. Il est donc
    écarté ici plutôt que soumis — le seed en a déjà posé une version
    placeholder, qui vaut mieux qu'une insertion en échec.
    """
    complete: list[dict[str, Any]] = []
    for record in records:
        if any(record.get(name) in (None, "") for name in SITE_REQUIRED):
            logger.warning(
                "site %s non synchronisé : description incomplète",
                record.get("site_id", "?"),
            )
            continue
        complete.append(
            {
                "site_id": str(record["site_id"]).strip(),
                "site_type": str(record["site_type"]),
                "site_name": str(record["site_name"]),
                "location": record.get("location"),
                "capacity_kw": record["capacity_kw"],
                "status": str(record.get("status") or DEFAULT_SITE_STATUS),
            }
        )
    return complete


def build_site_upsert(records: list[dict[str, Any]]) -> Any:
    """Construit la mise à jour du référentiel, qui écrase celle du seed.

    `DO UPDATE` et non `DO NOTHING`, contrairement aux mesures : le seed pose
    des capacités marquées « à synchroniser », et c'est précisément le rôle de
    cette écriture de les remplacer. Le référentiel n'est pas un historique,
    il décrit ce que les sites sont aujourd'hui.
    """
    statement = insert(site).values(records)
    return statement.on_conflict_do_update(
        index_elements=["site_id"],
        set_={
            name: getattr(statement.excluded, name)
            for name in SITE_COLUMNS
            if name != "site_id"
        },
    )


def sync_sites(
    engine: Engine,
    records: Iterable[dict[str, Any]],
    batch_size: int,
) -> int:
    """Met le référentiel à jour avant toute écriture de mesure.

    Avant, et jamais après : `mesure.site_id` référence `site`, et une mesure
    d'un site absent du référentiel est rejetée par la base quelle que soit sa
    qualité. Sans cette étape, un huitième site apparu chez la source ferait
    échouer chaque collecte sans que rien n'explique pourquoi.
    """
    sites = to_sites(records)
    written = write_batches(engine, sites, batch_size, build_site_upsert)
    logger.info("référentiel : %d site(s) synchronisé(s)", written)
    return written


def _rename(record: dict[str, Any]) -> dict[str, Any]:
    """Traduit le champ d'horodatage de la source vers celui de la table."""
    renamed = dict(record)
    if SOURCE_TIMESTAMP_COLUMN in renamed:
        renamed[TIMESTAMP_COLUMN] = renamed.pop(SOURCE_TIMESTAMP_COLUMN)
    return renamed


def _clean_site_id(value: Any) -> Any:
    """Ramène un identifiant de site à une chaîne, ou à rien."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: Any) -> list[str]:
    """Ramène `null_reasons` au TEXT[] NOT NULL attendu par la table.

    La source écrit tantôt une liste, tantôt rien, tantôt un motif seul. Les
    trois formes disent la même chose, et une seule doit entrer en base.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def _admitted_quality(value: Any) -> str:
    """Ne laisse passer qu'une valeur que le CHECK de la colonne accepte.

    Une qualification inconnue retombe sur le défaut de la colonne, faute de
    pouvoir dire « non qualifiée » dans le schéma figé. La soumettre telle
    quelle ferait échouer l'insertion du lot entier, et un NULL serait rejeté.
    L'ETL la reposera de toute façon.
    """
    if value in DATA_QUALITY_VALUES:
        return str(value)
    return UNQUALIFIED


def _to_sql_value(value: Any) -> Any:
    """Ramène les manquants pandas (NaN, NaT) au NULL attendu par la base.

    Sans cette conversion, le driver écrirait un NaN flottant dans une colonne
    NUMERIC, ce que PostgreSQL accepte et qui pollue silencieusement les
    agrégats en aval.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or pd.isna(value):
        return None
    return value
