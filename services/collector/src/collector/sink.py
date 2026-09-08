# **********************************************************************
# * Nom     : sink.py                                                  *
# * Type    : Module                                                   *
# * Sujet   : Écriture de tout ce que le collecteur dépose : mesures,  *
# *   états, alertes, capteurs, sites                                  *
# * Service : collector                                                *
# **********************************************************************

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from predict_common.db import (
    ALERTE_KEY,
    CAPTEUR_ETAT_KEY,
    CONFLICT_KEY,
    INGESTION_KEY,
    SITE_COLUMNS,
    SOURCE_COLUMNS,
    alerte,
    capteur_etat,
    capteur_panne,
    ingestion_etat,
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
from predict_common.timestamps import (
    DEFAULT_SOURCE_TIMEZONE,
    parse_timestamp,
    to_utc,
)

# Qualification retenue quand la source n'en déclare aucune.
UNQUALIFIED = QUALITY_GOOD

logger = logging.getLogger(__name__)


# Champs sans lesquels un site n'est pas synchronisable.
SITE_REQUIRED = ("site_id", "site_type", "site_name", "capacity_kw")

# Statut posé sur un site que la source ne qualifie pas.
DEFAULT_SITE_STATUS = "active"

# Pas sur lequel les horodatages sont alignés avant écriture.
GRID_RESOLUTION = "1min"


@dataclass(frozen=True)
class WriteReport:
    """Classe : WriteReport
    Description : Bilan d'une écriture : lignes soumises et lignes écartées.
    """
    rows: int
    dropped: int = 0


def to_measures(
    records: Iterable[dict[str, Any]],
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> pd.DataFrame:
    """Méthode : to_measures
    Description : Transforme les mesures de la source en tableau de la couche
      brute.
    """
    rows = [_rename(record) for record in records]
    frame = pd.DataFrame(rows, columns=list(SOURCE_COLUMNS))
    frame[TIMESTAMP_COLUMN] = to_utc(frame[TIMESTAMP_COLUMN], naive_timezone)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[SITE_COLUMN] = frame[SITE_COLUMN].map(_clean_site_id)
    frame["null_reasons"] = frame["null_reasons"].map(_as_list)
    frame["data_quality"] = frame["data_quality"].map(_admitted_quality)
    return frame[list(SOURCE_COLUMNS)]


def snap_to_grid(stamps: pd.Series) -> pd.Series:
    """Méthode : snap_to_grid
    Description : Aligne les horodatages sur le pas de la grille.
    """
    return stamps.dt.floor(GRID_RESOLUTION)


def drop_unplaceable(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Méthode : drop_unplaceable
    Description : Écarte les mesures sans instant ni site, qu'aucune clé ne
      place.
    """
    placeable = frame[frame[TIMESTAMP_COLUMN].notna() & frame[SITE_COLUMN].notna()]
    return placeable.reset_index(drop=True), len(frame) - len(placeable)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : deduplicate
    Description : Ne garde qu'une mesure par site et par instant.
    """
    return frame.drop_duplicates(
        subset=[SITE_COLUMN, TIMESTAMP_COLUMN], keep="last"
    ).reset_index(drop=True)


def select_day(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """Méthode : select_day
    Description : Ne garde que les mesures d'une journée donnée.
    """
    if frame.empty:
        return frame
    return frame[frame[TIMESTAMP_COLUMN].dt.date == day].reset_index(drop=True)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Méthode : validate
    Description : Vérifie le lot contre le contrat de la couche brute.
    """
    return MEASURE_SCHEMA.validate(frame, lazy=True)


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Méthode : to_records
    Description : Projette le tableau sur les colonnes de la source pour
      l'écriture.
    """
    projected = frame[list(SOURCE_COLUMNS)]
    return [
        {key: _to_sql_value(value) for key, value in row.items()}
        for row in projected.to_dict(orient="records")
    ]


def build_insert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_insert
    Description : Construit l'insertion qui ne touche jamais une mesure déjà
      écrite.
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
    """Méthode : write
    Description : Nettoie puis insère un lot de mesures, et rend son bilan.
    """
    placeable, dropped = drop_unplaceable(frame)
    placeable = placeable.assign(
        **{TIMESTAMP_COLUMN: snap_to_grid(placeable[TIMESTAMP_COLUMN])}
    )
    selected = deduplicate(placeable)
    if day is not None:
        selected = select_day(selected, day)
    rows = write_batches(
        engine, to_records(validate(selected)), batch_size, build_insert
    )
    if dropped:
        logger.warning("%d mesure(s) écartée(s) : horodatage ou site absent", dropped)
    return WriteReport(rows=rows, dropped=dropped)


@dataclass(frozen=True)
class IngestionState:
    """Classe : IngestionState
    Description : Ce qu'un site a donné lors d'une tentative de collecte.
    """
    site_id: str
    attempted_at: datetime
    rows: int = 0
    data_lag_s: float | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Méthode : succeeded
        Description : Dit si la tentative s'est terminée sans erreur.
        """
        return self.error is None


def to_success_states(
    states: Iterable[IngestionState],
    source: str,
) -> list[dict[str, Any]]:
    """Méthode : to_success_states
    Description : Traduit les tentatives réussies en lignes d'ingestion_etat.
    """
    return [
        {
            "site_id": state.site_id,
            "last_attempt_at": state.attempted_at,
            "last_success_at": state.attempted_at,
            "last_rows": state.rows,
            "last_data_lag_s": state.data_lag_s,
            "consecutive_failures": 0,
            "source": source,
        }
        for state in states
        if state.succeeded
    ]


def to_failure_states(
    states: Iterable[IngestionState],
    source: str,
) -> list[dict[str, Any]]:
    """Méthode : to_failure_states
    Description : Traduit les tentatives échouées en lignes d'ingestion_etat.
    """
    return [
        {
            "site_id": state.site_id,
            "last_attempt_at": state.attempted_at,
            "consecutive_failures": 1,
            "last_error": state.error,
            "source": source,
        }
        for state in states
        if not state.succeeded
    ]


def build_state_success_upsert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_state_success_upsert
    Description : Écrit un état de réussite et remet le compte d'échecs à zéro.
    """
    statement = insert(ingestion_etat).values(records)
    return statement.on_conflict_do_update(
        index_elements=list(INGESTION_KEY),
        set_={
            "last_attempt_at": statement.excluded.last_attempt_at,
            "last_success_at": statement.excluded.last_success_at,
            "last_rows": statement.excluded.last_rows,
            "last_data_lag_s": statement.excluded.last_data_lag_s,
            "consecutive_failures": 0,
            "source": statement.excluded.source,
        },
    )


def build_state_failure_upsert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_state_failure_upsert
    Description : Écrit un état d'échec et incrémente le compte consécutif.
    """
    statement = insert(ingestion_etat).values(records)
    return statement.on_conflict_do_update(
        index_elements=list(INGESTION_KEY),
        set_={
            "last_attempt_at": statement.excluded.last_attempt_at,
            "consecutive_failures": ingestion_etat.c.consecutive_failures + 1,
            "last_error": statement.excluded.last_error,
            "source": statement.excluded.source,
        },
    )


def write_state(
    engine: Engine,
    states: Iterable[IngestionState],
    batch_size: int,
    source: str,
) -> int:
    """Méthode : write_state
    Description : Repose l'état d'ingestion des sites, réussites et échecs
      séparés.
    """
    collected = list(states)
    rows = write_batches(
        engine,
        to_success_states(collected, source),
        batch_size,
        build_state_success_upsert,
    )
    rows += write_batches(
        engine,
        to_failure_states(collected, source),
        batch_size,
        build_state_failure_upsert,
    )
    return rows


# Champs sans lesquels une alerte n'est pas exploitable.
ALERTE_REQUIRED = ("alert_id", "site_id", "timestamp", "severity", "type", "message")

# Capteurs dont l'état est suivi, site par site.
CAPTEURS = ("consumption", "electrical", "temperature", "humidity", "network")


def to_alerts(
    records: Iterable[dict[str, Any]],
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> list[dict[str, Any]]:
    """Méthode : to_alerts
    Description : Traduit les alertes de la source en lignes, incomplètes
      écartées.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if any(record.get(name) in (None, "") for name in ALERTE_REQUIRED):
            logger.warning(
                "alerte %s ignorée : description incomplète",
                record.get("alert_id", "?"),
            )
            continue
        stamp = parse_timestamp(record["timestamp"], naive_timezone)
        if stamp is None:
            logger.warning(
                "alerte %s ignorée : horodatage illisible", record["alert_id"]
            )
            continue
        rows.append(
            {
                "alert_id": str(record["alert_id"]),
                "site_id": str(record["site_id"]).strip(),
                "ts": stamp,
                "severity": str(record["severity"]),
                "type_alerte": str(record["type"]),
                "message": str(record["message"]),
                "valeur": record.get("value"),
                "seuil": record.get("threshold"),
            }
        )
    return rows


def build_alerte_insert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_alerte_insert
    Description : Construit l'insertion idempotente d'un lot d'alertes.
    """
    return (
        insert(alerte)
        .values(records)
        .on_conflict_do_nothing(index_elements=list(ALERTE_KEY))
    )


def write_alerts(
    engine: Engine,
    records: Iterable[dict[str, Any]],
    batch_size: int,
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> int:
    """Méthode : write_alerts
    Description : Écrit les alertes actives servies par la source.
    """
    rows = to_alerts(records, naive_timezone)
    written = write_batches(engine, rows, batch_size, build_alerte_insert)
    logger.info("alertes : %d ligne(s) soumise(s)", written)
    return written


def to_sensor_states(
    payload: dict[str, Any],
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> list[dict[str, Any]]:
    """Méthode : to_sensor_states
    Description : Traduit l'état des capteurs en une ligne par site et par
      capteur.
    """
    rows: list[dict[str, Any]] = []
    for site_id, description in payload.items():
        if not isinstance(description, dict):
            continue
        overall = str(description.get("overall") or "ok")
        sensors = description.get("sensors")
        if not isinstance(sensors, dict):
            continue
        for capteur, etat in sensors.items():
            if capteur not in CAPTEURS:
                logger.warning("capteur inconnu ignoré : %s", capteur)
                continue
            if not isinstance(etat, dict):
                continue
            rows.append(
                {
                    "site_id": str(site_id).strip(),
                    "capteur": str(capteur),
                    "statut": str(etat.get("status") or "ok"),
                    "failing_until": parse_timestamp(
                        etat.get("failing_until"), naive_timezone
                    ),
                    "overall": overall,
                }
            )
    return rows


def build_capteur_etat_upsert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_capteur_etat_upsert
    Description : Écrit l'état courant d'un capteur en l'horodatant.
    """
    statement = insert(capteur_etat).values(records)
    updated = {
        name: getattr(statement.excluded, name)
        for name in ("statut", "failing_until", "overall")
    }
    updated["releve_le"] = func.now()
    return statement.on_conflict_do_update(
        index_elements=list(CAPTEUR_ETAT_KEY),
        set_=updated,
    )


def write_sensor_states(
    engine: Engine,
    payload: dict[str, Any],
    batch_size: int,
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> int:
    """Méthode : write_sensor_states
    Description : Repose l'état courant de tous les capteurs.
    """
    rows = to_sensor_states(payload, naive_timezone)
    written = write_batches(engine, rows, batch_size, build_capteur_etat_upsert)
    logger.info("capteurs : %d état(s) reposé(s)", written)
    return written


# Statut d'un capteur en panne, tel que la source le nomme.
SENSOR_FAILING = "failing"


def read_sensor_statuses(engine: Engine) -> dict[tuple[str, str], str]:
    """Méthode : read_sensor_statuses
    Description : Relit l'état connu des capteurs, pour détecter les
      changements.
    """
    statement = select(
        capteur_etat.c.site_id, capteur_etat.c.capteur, capteur_etat.c.statut
    )
    with engine.begin() as connection:
        rows = connection.execute(statement)
        return {(row[0], row[1]): row[2] for row in rows}


@dataclass(frozen=True)
class DayCoverage:
    """Classe : DayCoverage
    Description : Ce qu'une journée porte réellement en base, pour un site.
    """
    site_id: str
    day: date
    first_at: datetime
    last_at: datetime


def day_coverage(
    engine: Engine,
    site_ids: Sequence[str],
    start: date,
    end: date,
) -> list[DayCoverage]:
    """Méthode : day_coverage
    Description : Interroge la base sur ce que chaque journée couvre
      réellement.
    """
    day = func.date_trunc("day", mesure.c.ts)
    statement = (
        select(
            mesure.c.site_id,
            day.label("day"),
            func.min(mesure.c.ts),
            func.max(mesure.c.ts),
        )
        .where(
            mesure.c.site_id.in_(list(site_ids)),
            mesure.c.ts >= datetime.combine(start, time.min, tzinfo=UTC),
            mesure.c.ts < datetime.combine(end, time.min, tzinfo=UTC)
            + timedelta(days=1),
        )
        .group_by(mesure.c.site_id, day)
    )
    with engine.begin() as connection:
        return [
            DayCoverage(
                site_id=row[0],
                day=row[1].date() if hasattr(row[1], "date") else row[1],
                first_at=row[2],
                last_at=row[3],
            )
            for row in connection.execute(statement)
        ]


def to_sensor_episodes(
    previous: dict[tuple[str, str], str],
    states: list[dict[str, Any]],
    now: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Méthode : to_sensor_episodes
    Description : Compare l'état connu au nouveau et en déduit les pannes
      ouvertes ou closes.
    """
    opened: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for state in states:
        key = (state["site_id"], state["capteur"])
        was_failing = previous.get(key) == SENSOR_FAILING
        is_failing = state["statut"] == SENSOR_FAILING
        if is_failing and not was_failing:
            opened.append(
                {
                    "site_id": state["site_id"],
                    "capteur": state["capteur"],
                    "debut_le": now,
                    "fin_le": None,
                    "failing_until": state.get("failing_until"),
                }
            )
        elif was_failing and not is_failing:
            closed.append({"site_id": state["site_id"], "capteur": state["capteur"]})
    return opened, closed


def build_panne_insert(records: list[dict[str, Any]]) -> Any:
    """Méthode : build_panne_insert
    Description : Construit l'ouverture idempotente d'une panne capteur.
    """
    return insert(capteur_panne).values(records).on_conflict_do_nothing()


def write_sensor_episodes(
    engine: Engine,
    previous: dict[tuple[str, str], str],
    states: list[dict[str, Any]],
    now: datetime,
    batch_size: int,
) -> tuple[int, int]:
    """Méthode : write_sensor_episodes
    Description : Ouvre les pannes nouvelles et ferme celles qui ont cessé.
    """
    opened, closed = to_sensor_episodes(previous, states, now)
    written = write_batches(engine, opened, batch_size, build_panne_insert)
    if closed:
        with engine.begin() as connection:
            for episode in closed:
                connection.execute(
                    update(capteur_panne)
                    .where(
                        capteur_panne.c.site_id == episode["site_id"],
                        capteur_panne.c.capteur == episode["capteur"],
                        capteur_panne.c.fin_le.is_(None),
                    )
                    .values(fin_le=now)
                )
    if opened or closed:
        logger.info(
            "pannes capteur : %d ouverte(s), %d clos(es)", written, len(closed)
        )
    return written, len(closed)


def to_sites(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Méthode : to_sites
    Description : Traduit le référentiel de la source en lignes, incomplètes
      écartées.
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
    """Méthode : build_site_upsert
    Description : Écrit un site en réécrivant sa description à chaque passage.
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
    """Méthode : sync_sites
    Description : Synchronise le référentiel des sites depuis la source.
    """
    sites = to_sites(records)
    written = write_batches(engine, sites, batch_size, build_site_upsert)
    logger.info("référentiel : %d site(s) synchronisé(s)", written)
    return written


def _rename(record: dict[str, Any]) -> dict[str, Any]:
    """Méthode : _rename
    Description : Renomme l'horodatage de la source vers celui de la chaîne.
    """
    renamed = dict(record)
    if SOURCE_TIMESTAMP_COLUMN in renamed:
        renamed[TIMESTAMP_COLUMN] = renamed.pop(SOURCE_TIMESTAMP_COLUMN)
    return renamed


def _clean_site_id(value: Any) -> Any:
    """Méthode : _clean_site_id
    Description : Ramène un identifiant de site vide ou absent à None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: Any) -> list[str]:
    """Méthode : _as_list
    Description : Ramène une valeur à une liste de chaînes.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def _admitted_quality(value: Any) -> str:
    """Méthode : _admitted_quality
    Description : N'accepte qu'une qualification connue, sinon retombe sur le
      défaut.
    """
    if value in DATA_QUALITY_VALUES:
        return str(value)
    return UNQUALIFIED


def _to_sql_value(value: Any) -> Any:
    """Méthode : _to_sql_value
    Description : Ramène les manquants pandas au NULL attendu par la base.
    """
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or pd.isna(value):
        return None
    return value
