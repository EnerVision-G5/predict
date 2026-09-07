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

Une seconde table est écrite ici, et une seule chose la distingue : son contenu
ne vient pas de la source. `ingestion_etat` dit ce que le collecteur a fait,
site par site, et c'est la seule chose que `mesure` ne saura jamais dire. Un
capteur mort y dépose quand même une ligne — nulle, avec ses motifs — donc
`max(inserted_at)` avance ; un poller arrêté ou une source en 500 n'en dépose
aucune, et `max(inserted_at)` se fige exactement comme si le site avait cessé
d'exister. C'est la panne la plus grave, et c'était la plus discrète.

Sa politique d'écriture est l'inverse de celle des mesures : `DO UPDATE`, parce
qu'elle décrit le présent et non un historique. Voir `write_state`.
"""

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

# Pas de la grille de `mesure` : au plus une ligne par site et par minute.
#
# La minute est la résolution la plus fine que la chaîne produise — c'est la
# cadence du poller, que `collector.poll_interval_s` fixe à 60 s. Elle n'est
# pas celle de l'historique : `GET /api/v1/readings` sert la source par pas de
# 30 minutes (voir sa documentation), donc sur des horodatages déjà alignés à
# la minute. Les deux points d'entrée tiennent ainsi dans la même grille sans
# qu'aucun n'ait à connaître le pas de l'autre.
#
# Voir `snap_to_grid` pour ce que cette constante fait respecter.
GRID_RESOLUTION = "1min"


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


def snap_to_grid(stamps: pd.Series) -> pd.Series:
    """Ramène les horodatages sur la grille à la minute de `mesure`.

    La table est une grille : une ligne par site et par minute, et
    `(site_id, ts)` en est la clé. Les deux points d'entrée du collecteur y
    écrivent, et ils ne datent pas de la même façon :

    - le rattrapage lit `/readings`, que la source sert par pas de 30 minutes,
      donc sur des horodatages déjà alignés — `13:30:00` ;
    - le poller lit `/current`, que la source date de l'instant de l'appel —
      `13:53:31.587801`, et l'offset change à chaque tick comme à chaque
      redémarrage du processus.

    Sans cette normalisation, une minute couverte par les DEUX routes entre en
    base sous deux clés différentes, et `ON CONFLICT DO NOTHING` n'a alors
    rien à arbitrer puisqu'il n'y a pas de conflit. Le recouvrement porte sur
    les minutes `:00` et `:30` de chaque heure — les seules que `/readings`
    sert — soit 48 doublons par jour et par site sur une journée collectée au
    fil de l'eau puis rattrapée. Un poller redémarré deux fois dans la même
    minute produisait de même deux lignes au lieu d'une.

    Ce n'est pas réécrire une mesure. La valeur, sa qualité et ses motifs
    d'absence traversent intacts ; seul l'instant est ramené sur la grille où
    la table le range. L'écart absorbé est borné par la cadence du poller, que
    `collector.poll_interval_s` fixe à la minute.

    L'arrondi est vers le bas : une mesure appartient à la minute qui a
    commencé, jamais à celle qui n'a pas encore eu lieu.
    """
    return stamps.dt.floor(GRID_RESOLUTION)


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
    """Soumet un lot de mesures brutes et retourne ce qu'il a déposé.

    Le calage sur la grille se fait ICI et pas dans `to_measures`, et la
    distinction n'est pas cosmétique : la grille est une propriété de la
    TABLE, pas de la mesure. Le poller lit l'âge de ce que la source vient de
    servir sur l'horodatage brut — caler avant de le mesurer quantifierait ce
    retard à la minute et ferait paraître en retard de cinquante secondes un
    site parfaitement à l'heure.
    """
    placeable, dropped = drop_unplaceable(frame)
    placeable = placeable.assign(
        **{TIMESTAMP_COLUMN: snap_to_grid(placeable[TIMESTAMP_COLUMN])}
    )
    # Après le calage, et pas avant : deux relevés de la même minute ne
    # deviennent des doublons qu'une fois ramenés sur la grille.
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
    """Ce qu'un essai de collecte a donné pour un site.

    `error` porte le verdict : rempli, l'essai a échoué. Les deux cas ne
    s'écrivent pas de la même façon — un échec ne doit toucher ni la date du
    dernier succès, ni le nombre de lignes, ni le retard mesuré, qui décrivent
    tous le dernier essai *abouti* et restent la seule chose vraie qu'on
    sache du site.
    """

    site_id: str
    attempted_at: datetime
    rows: int = 0
    data_lag_s: float | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def to_success_states(
    states: Iterable[IngestionState],
    source: str,
) -> list[dict[str, Any]]:
    """Projette les essais aboutis vers les colonnes de `ingestion_etat`.

    `last_error` n'y figure pas : la cause du dernier échec est conservée
    après un succès. Savoir de quoi un site relève a une valeur, et l'effacer
    au premier tick réussi ferait disparaître la panne au moment précis où
    quelqu'un vient la regarder.
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
    """Projette les essais en échec vers les colonnes de `ingestion_etat`.

    Trois colonnes sont volontairement absentes — `last_success_at`,
    `last_rows`, `last_data_lag_s`. Un échec n'a rien à en dire, et les poser
    à zéro ou à NULL effacerait ce que le dernier succès avait établi. Sur une
    première insertion, ce sont les DEFAULT de la table qui s'appliquent.
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
    """Construit la mise à jour d'état d'un essai abouti.

    `DO UPDATE` et non `DO NOTHING` : la table décrit le présent, pas un
    historique. Une ligne existe déjà pour chaque site dès le deuxième tick,
    et ne rien faire figerait l'état au premier.
    """
    statement = insert(ingestion_etat).values(records)
    return statement.on_conflict_do_update(
        index_elements=list(INGESTION_KEY),
        set_={
            "last_attempt_at": statement.excluded.last_attempt_at,
            "last_success_at": statement.excluded.last_success_at,
            "last_rows": statement.excluded.last_rows,
            "last_data_lag_s": statement.excluded.last_data_lag_s,
            # Remis à zéro et non décrémenté : le compteur distingue l'à-coup
            # de la panne installée, et un succès clôt la série.
            "consecutive_failures": 0,
            "source": statement.excluded.source,
        },
    )


def build_state_failure_upsert(records: list[dict[str, Any]]) -> Any:
    """Construit la mise à jour d'état d'un essai en échec.

    Le compteur est incrémenté depuis la valeur en base et non depuis le lot :
    c'est le seul endroit qui sache combien d'essais ont déjà échoué, et le
    calculer côté processus donnerait un compte remis à un à chaque
    redémarrage du conteneur — c'est-à-dire précisément quand la panne est la
    plus probable.
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
    """Repose l'état de collecte des sites et retourne le nombre de lignes.

    Deux instructions et non une : succès et échecs ne posent pas les mêmes
    colonnes, et un lot mixte devrait choisir une forme pour les deux. Elles
    partagent la transaction de `write_batches`, si bien qu'un tick est
    enregistré en entier ou pas du tout.
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


# Colonnes sans lesquelles une alerte ne veut rien dire. La source les sert
# toutes, mais une réponse tronquée ferait échouer le lot entier : l'alerte
# incomplète est écartée seule.
ALERTE_REQUIRED = ("alert_id", "site_id", "timestamp", "severity", "type", "message")

# Capteurs décrits par la source. La liste est fermée à dessein : un capteur
# inconnu apparaîtrait en base sans que personne ne sache l'interpréter, et le
# journal est un meilleur endroit pour le signaler qu'une ligne muette.
CAPTEURS = ("consumption", "electrical", "temperature", "humidity", "network")


def to_alerts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Traduit les alertes de la source vers les colonnes de `alerte`.

    `type` devient `type_alerte` : le mot est trop générique pour une colonne,
    et la traduction est portée ici une fois plutôt que dans chaque requête.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if any(record.get(name) in (None, "") for name in ALERTE_REQUIRED):
            logger.warning(
                "alerte %s ignorée : description incomplète",
                record.get("alert_id", "?"),
            )
            continue
        stamp = _parse_stamp(record["timestamp"])
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
    """Construit l'insertion idempotente d'un lot d'alertes.

    `DO NOTHING` : le poller repasse toutes les minutes sur des alertes encore
    actives, et `alert_id` est stable côté source. Une alerte d'une heure
    serait sinon enregistrée soixante fois. Et `DO NOTHING` plutôt que
    `DO UPDATE` parce qu'une alerte ne change pas : elle est déclenchée, elle
    ne se corrige pas.
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
) -> int:
    """Journalise les alertes actives et retourne le nombre de lignes neuves."""
    rows = to_alerts(records)
    written = write_batches(engine, rows, batch_size, build_alerte_insert)
    logger.info("alertes : %d ligne(s) soumise(s)", written)
    return written


def to_sensor_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Met à plat l'état des capteurs, un enregistrement par couple site/capteur.

    La source indexe par site et imbrique les capteurs. La table, elle, est
    plate : c'est la seule forme qui permet de demander « quels capteurs sont
    tombés » sans déplier un JSON en SQL.
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
                    "failing_until": _parse_stamp(etat.get("failing_until")),
                    "overall": overall,
                }
            )
    return rows


def build_capteur_etat_upsert(records: list[dict[str, Any]]) -> Any:
    """Construit la mise à jour de l'état des capteurs.

    `DO UPDATE`, contrairement aux alertes : cette table décrit le présent.
    L'historique des pannes vit dans `mesure.null_reasons`, une ligne par
    minute avec sa cause ; le rejouer ici en ferait une seconde vérité.
    """
    statement = insert(capteur_etat).values(records)
    updated = {
        name: getattr(statement.excluded, name)
        for name in ("statut", "failing_until", "overall")
    }
    # `releve_le` est laissée au DEFAULT now() à l'insertion, mais une table du
    # présent doit dire quand elle a été rafraîchie : sans ce repos explicite,
    # la ligne garderait la date de son premier tick pour toujours, et un
    # collecteur arrêté depuis trois jours paraîtrait à jour.
    updated["releve_le"] = func.now()
    return statement.on_conflict_do_update(
        index_elements=list(CAPTEUR_ETAT_KEY),
        set_=updated,
    )


def write_sensor_states(
    engine: Engine,
    payload: dict[str, Any],
    batch_size: int,
) -> int:
    """Repose l'état des capteurs et retourne le nombre de lignes."""
    rows = to_sensor_states(payload)
    written = write_batches(engine, rows, batch_size, build_capteur_etat_upsert)
    logger.info("capteurs : %d état(s) reposé(s)", written)
    return written


SENSOR_FAILING = "failing"


def read_sensor_statuses(engine: Engine) -> dict[tuple[str, str], str]:
    """Lit l'état des capteurs déjà en base, indexé par (site, capteur).

    C'est la seule lecture de tout le collecteur, et elle a une raison : la
    source dit `failing` ou `ok` au présent, jamais depuis quand. Sans l'état
    précédent, aucune transition n'est détectable, et un capteur en panne
    depuis trois jours ouvrirait un épisode à chaque tick.
    """
    statement = select(
        capteur_etat.c.site_id, capteur_etat.c.capteur, capteur_etat.c.statut
    )
    with engine.begin() as connection:
        rows = connection.execute(statement)
        return {(row[0], row[1]): row[2] for row in rows}


@dataclass(frozen=True)
class DayCoverage:
    """Ce que la base porte pour un site et une journée."""

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
    """Retourne, par site et par journée, l'étendue de ce qui est déjà en base.

    Le rattrapage a besoin de savoir où sont les TROUS, et non où s'arrête la
    donnée. La différence décide de tout quand un poller tourne déjà : la
    dernière mesure est alors « maintenant » quelle que soit l'ampleur de ce
    qui manque derrière, et un repère de reprise ne verrait rien à combler.

    Les journées sans aucune ligne n'apparaissent pas dans le résultat — c'est
    leur absence qui les signale, et la demander à SQL coûterait une jointure
    sur une série générée pour n'apprendre que ce que l'appelant sait déjà.

    Une seule agrégation sur la fenêtre, faite au démarrage. Elle balaie les
    journées demandées et rien d'autre : c'est `mesure.ts` qui porte le
    partitionnement de l'hypertable.
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
    """Retourne les épisodes à ouvrir et ceux à clore, sans toucher à la base.

    Fonction pure, et c'est délibéré : toute la difficulté est ici, dans la
    lecture des transitions, et elle s'éprouve sans moteur.

    Quatre cas, trois conduites. Sain puis en panne ouvre un épisode. En panne
    puis en panne le prolonge — seule `failing_until` bouge, la source pouvant
    repousser sa date de rétablissement. En panne puis sain le clot. Sain puis
    sain ne dit rien.

    Un capteur inconnu de `capteur_etat` compte comme sain : c'est le premier
    tick sur ce site, et sa panne éventuelle est bien un début.
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
    """Construit l'ouverture idempotente d'un lot d'épisodes.

    `DO NOTHING` sans cible : la clé naturelle n'est pas la seule contrainte à
    protéger. Un épisode déjà ouvert sur le même capteur doit être ignoré lui
    aussi, faute de quoi deux processus concurrents en créeraient deux.
    """
    return insert(capteur_panne).values(records).on_conflict_do_nothing()


def write_sensor_episodes(
    engine: Engine,
    previous: dict[tuple[str, str], str],
    states: list[dict[str, Any]],
    now: datetime,
    batch_size: int,
) -> tuple[int, int]:
    """Ouvre et clot les épisodes de panne, et retourne les deux comptes.

    Les deux écritures partagent une transaction avec l'état courant, plus
    haut dans le tick : un épisode ouvert sans que `capteur_etat` le suive
    serait rouvert au tick suivant.
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


def _parse_stamp(value: Any) -> datetime | None:
    """Lit un horodatage ISO de la source, ou rend None s'il est illisible."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


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
