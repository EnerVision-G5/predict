"""Point d'entrée du collecteur : rattrapage de l'historique par lot.

    python -m collector --start 2026-08-01 --end 2026-09-01
    python -m collector --start 2026-08-01 --end 2026-09-01 --site SITE001
    python -m collector --date 2026-09-02 --days 7

Deux façons de dire la même chose, parce qu'elles ne servent pas au même
usage. `--start/--end` nomme une période, ce que fait un analyste qui rattrape
un historique. `--date/--days` nomme une journée et sa profondeur, ce que fait
un ordonnanceur qui rejoue la veille : la date y est un paramètre, et la
profondeur une constante.

Le collecteur remplit la couche brute, qui est la table `mesure` de
TimescaleDB. Il ne connaît ni l'ETL, ni l'entraînement, ni le service : sa
sortie est une table et un schéma, et c'est tout ce que son consommateur a
besoin de savoir.

Une journée est traitée en entier avant la suivante. Le lot d'un jour tient en
mémoire — sept sites à la minute font une dizaine de milliers de lignes — là
où trois mois n'y tiendraient pas, et une journée soumise en une transaction
est soit chargée, soit absente, jamais à moitié écrite.

Relancer la même date ne double rien et n'efface rien : l'insertion est un
`ON CONFLICT DO NOTHING`. C'est ce qui rend le rejeu après incident sans effet
de bord — et le rejeu est le mode d'exploitation normal, pas l'exception : une
source indisponible pendant deux heures se rattrape en relançant la journée.

Le rattrapage repose son état dans `ingestion_etat` comme le poller, mais sous
`source='backfill'`. La distinction compte : le rattrapage est justement ce
qu'on lance quand la collecte continue est arrêtée, et une ligne qui ne dirait
pas d'où elle vient ferait passer une journée rejouée à la main pour une
ingestion vivante.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy.exc import SQLAlchemyError

from collector.sink import (
    DayCoverage,
    IngestionState,
    day_coverage,
    sync_sites,
    to_measures,
    write,
    write_state,
)
from predict_common.config import ConfigError, load_config
from predict_common.db import (
    INGESTION_SOURCE_BACKFILL,
    DatabaseError,
    ingestion_etat,
    mesure,
    open_engine,
    site,
    verify_schema,
)
from predict_common.paths import PathError, date_range, parse_date
from predict_common.source import (
    MAX_PAGE_SIZE,
    SourceClient,
    SourceError,
    SourceSettings,
)

DEFAULT_DAYS = 1

# Profondeur du rattrapage automatique, en journées. C'est la fenêtre dans
# laquelle les trous sont cherchés — pas la quantité recollectée : une fenêtre
# sans trou ne coûte qu'une requête d'agrégation.
#
# 35 et non 30 : « il y a un mois » doit tomber DANS la fenêtre et non sur son
# bord. Une profondeur de 30 partant du 5 septembre ne remonte qu'au 7 août, et
# laisserait dehors les deux premières journées d'un trou qui commence le 5.
DEFAULT_CATCH_UP_DAYS = 35

# Écart toléré entre les bornes d'une journée et ce que la base en porte, de
# part et d'autre. Voir `covers_full_day` : elle absorbe le pas de la source
# sans avoir à le connaître.
COVERAGE_TOLERANCE = timedelta(hours=1)

EXIT_OK = 0
EXIT_FAILED = 1

logger = logging.getLogger(__name__)


def day_window(day: date, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Retourne la fenêtre UTC d'une journée, JAMAIS au-delà de maintenant.

    Les mesures rendues par la source sont ensuite filtrées sur le jour
    demandé : une source qui déborderait d'une seconde ne serait pas comptée
    dans une journée qu'elle ne concerne pas.

    La borne haute est ramenée à l'instant courant, et ce n'est pas un détail
    d'exactitude : sans elle, rattraper la journée EN COURS demande à la
    source les heures qui n'ont pas encore eu lieu. Elle ne répond pas une
    erreur — elle répond des mesures nulles, que le collecteur écrit, et que
    son `ON CONFLICT DO NOTHING` rend alors DÉFINITIVES.

    Le poller collecte ensuite ces minutes-là pour de vrai, une par une, et
    ses valeurs sont silencieusement rejetées : la ligne existe déjà, vide. Un
    rattrapage lancé à 02:30 stérilisait ainsi les vingt et une heures
    suivantes, chaque jour, sans qu'aucun journal ne le dise.
    """
    start = datetime.combine(day, time.min, tzinfo=UTC)
    end = start + timedelta(days=1)
    return start, min(end, now or datetime.now(UTC))


def collect_day(
    client: SourceClient,
    engine,
    batch_size: int,
    day: date,
    sites: Sequence[str],
) -> int:
    """Collecte une journée pour les sites demandés et la charge en base."""
    start_time, end_time = day_window(day)
    records: list[dict] = []
    states: list[IngestionState] = []
    attempted_at = datetime.now(UTC)
    for site_id in sites:
        try:
            page = list(client.iter_readings(site_id, start_time, end_time))
        except SourceError as exc:
            # L'état est posé avant que l'exception ne remonte : le rattrapage
            # s'arrête sur un échec de source, mais ce qu'il savait à cet
            # instant a plus de valeur écrit que perdu.
            states.append(
                IngestionState(
                    site_id=site_id,
                    attempted_at=attempted_at,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            record_collection(engine, states, batch_size)
            raise
        logger.info("site %s : %d mesure(s) lue(s)", site_id, len(page))
        records.extend(page)
        states.append(
            IngestionState(
                site_id=site_id,
                attempted_at=attempted_at,
                rows=len(page),
                # Aucun retard de données n'est mesuré ici, et c'en est le
                # sens : un rattrapage relit une journée passée, l'âge de ce
                # qu'il reçoit ne dit rien de la santé de la source.
                data_lag_s=None,
            )
        )
    report = write(engine, to_measures(records), batch_size, day=day)
    logger.info(
        "%s : %d mesure(s) soumise(s)%s",
        day,
        report.rows,
        f", {report.dropped} écartée(s)" if report.dropped else "",
    )
    record_collection(engine, states, batch_size)
    return report.rows


def record_collection(
    engine,
    states: Sequence[IngestionState],
    batch_size: int,
) -> None:
    """Repose l'état de collecte du rattrapage, sans jamais le faire échouer.

    `source='backfill'` et non `'poller'` : un rattrapage lancé à la main
    pendant que la collecte continue est arrêtée ne doit pas faire paraître
    l'ingestion vivante. C'est exactement le cas où la fraîcheur affichée
    deviendrait un mensonge, puisqu'il se produit quand quelque chose ne va
    déjà pas.

    L'échec de cette écriture n'est pas celui du rattrapage : les mesures,
    elles, sont chargées. Il est journalisé et n'emporte pas le run.
    """
    try:
        write_state(engine, states, batch_size, INGESTION_SOURCE_BACKFILL)
    except (SQLAlchemyError, OSError) as exc:
        logger.error("état d'ingestion non enregistré : %s", exc)


def resolve_sites(
    client: SourceClient,
    engine,
    batch_size: int,
    requested: Sequence[str] | None,
) -> list[str]:
    """Synchronise le référentiel et retourne les sites à collecter.

    La synchronisation entretient `site`, que `mesure.site_id` référence : un
    site absent de la table ferait rejeter ses mesures sans que rien
    n'explique pourquoi. C'est aussi ce que le seed `02_seed_sites.sql`
    attend, ses capacités des sites 4 à 7 étant des placeholders.

    Elle n'est exigée que lorsqu'on en dépend pour savoir quoi collecter.
    Avec `--site`, l'exploitant a nommé ses sites : une source qui ne sert pas
    son référentiel ne doit pas l'empêcher de rattraper une journée, puisque
    le seed a déjà posé les sites courants. L'échec est journalisé, et c'est
    la clé étrangère qui tranchera s'il manquait vraiment quelque chose.
    """
    try:
        referential = client.fetch_sites()
    except SourceError:
        if not requested:
            raise
        logger.warning(
            "référentiel indisponible : collecte des sites demandés sans"
            " synchronisation"
        )
        return list(requested)
    sync_sites(engine, referential, batch_size)
    if requested:
        return list(requested)
    return [str(entry["site_id"]) for entry in referential if "site_id" in entry]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande du collecteur."""
    parser = argparse.ArgumentParser(
        prog="collector",
        description="Collecte des mesures EnerVision vers TimescaleDB.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Première journée collectée, incluse, au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Dernière journée collectée, incluse. Défaut : --start.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Dernière journée collectée. Forme courte, avec --days.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Nombre de journées remontées depuis --date. Défaut : 1.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site à collecter. Répétable. Par défaut : tout le référentiel.",
    )
    parser.add_argument(
        "--catch-up",
        action="store_true",
        help=(
            "Rattrape ce qui manque en base, sans période à fournir : la"
            " reprise part de la dernière mesure de chaque site. Destiné au"
            " redémarrage du collecteur."
        ),
    )
    parser.add_argument(
        "--catch-up-days",
        type=int,
        default=None,
        help=(
            "Profondeur maximale du rattrapage automatique, en journées."
            f" Défaut : collector.catch_up_days, sinon {DEFAULT_CATCH_UP_DAYS}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"Mesures demandées par requête, au plus {MAX_PAGE_SIZE}."
            " Défaut : source.page_size."
        ),
    )
    return parser.parse_args(argv)


def check_period_arguments(args: argparse.Namespace) -> None:
    """Refuse une période donnée deux fois, ou donnée à qui la déduit.

    Vérifiée dans les deux modes, et depuis `main` plutôt que depuis
    `requested_days` : cette dernière n'est pas appelée en mode rattrapage,
    et le contrôle n'y serait donc jamais exécuté dans le seul cas où il
    compte.
    """
    if not args.catch_up:
        return
    given = [
        name
        for name, value in (
            ("--start", args.start),
            ("--end", args.end),
            ("--date", args.date),
        )
        if value is not None
    ]
    if given:
        raise ValueError(
            "--catch-up déduit sa période de la base :"
            f" {', '.join(given)} n'a alors aucun effet et prête à confusion."
        )


def catch_up_days(
    engine,
    sites: Sequence[str],
    depth_days: int,
    today: date | None = None,
) -> list[date]:
    """Retourne les journées à rattraper, déduites des TROUS de la base.

    C'est le mode du redéploiement : personne ne sait ce qui manque, et
    demander une période reviendrait à le faire deviner à l'exploitant.

    La détection porte sur les trous et non sur la dernière mesure, et c'est
    la seule chose qui compte ici. Un serveur où le poller tourne déjà a une
    dernière mesure à « maintenant » quelle que soit l'ampleur de ce qui
    manque derrière : un repère de reprise ne verrait rien à combler, alors
    qu'il peut manquer un mois entier plus tôt dans la fenêtre.

    Une journée est retenue dès qu'UN site ne la couvre pas entièrement. Trois
    cas la rendent incomplète, et le troisième est celui qu'un simple comptage
    manquerait :

    - aucune ligne — la journée n'a jamais été collectée ;
    - la première mesure arrive trop tard — le poller a démarré en cours de
      journée, la matinée manque ;
    - la dernière arrive trop tôt — le poller s'est arrêté en cours de
      journée.

    La journée courante est toujours retenue : elle est incomplète par
    construction, puisqu'elle n'est pas finie.

    La fenêtre est commune à tous les sites plutôt que découpée par site :
    `ON CONFLICT DO NOTHING` rend le recouvrement gratuit en base, et une
    journée coûte UNE requête par site à la source, `/readings` servant 48
    points là où `limit` en autorise 1000.
    """
    end = today or datetime.now(UTC).date()
    first = end - timedelta(days=max(depth_days, 1) - 1)
    covered = {
        (entry.site_id, entry.day)
        for entry in day_coverage(engine, sites, first, end)
        if covers_full_day(entry, end)
    }
    return [
        day
        for day in date_range(first, end)
        if any((site_id, day) not in covered for site_id in sites)
    ]


def covers_full_day(entry: DayCoverage, today: date) -> bool:
    """Dit si ce que la base porte couvre la journée d'un bout à l'autre.

    La tolérance absorbe le pas de la source sans avoir à le connaître :
    `/readings` sert 48 points par journée, de 00:00 à 23:30, quand le poller
    en dépose 1440, de 00:00 à 23:59. Les deux couvrent la journée ; exiger
    une dernière mesure à 23:59 ferait rattraper indéfiniment toutes les
    journées venues du seul rattrapage.

    La journée courante n'est jamais couverte : elle n'est pas finie.
    """
    if entry.day >= today:
        return False
    day_start = datetime.combine(entry.day, time.min, tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return (
        entry.first_at <= day_start + COVERAGE_TOLERANCE
        and entry.last_at >= day_end - COVERAGE_TOLERANCE
    )


def requested_days(args: argparse.Namespace) -> list[date]:
    """Retourne les journées à collecter, dans l'ordre chronologique.

    Les deux formes s'excluent : les mélanger laisserait deux périodes
    possibles pour un même appel, et le run partirait sur l'une des deux sans
    que rien ne dise laquelle.
    """
    borne = args.start is not None or args.end is not None
    if borne and args.date is not None:
        raise ValueError(
            "--date et --start/--end désignent tous deux la période :"
            " en choisir une seule."
        )
    if borne:
        if args.start is None:
            raise ValueError("--end demande --start.")
        # Une borne haute absente vaut la borne basse : `--start` seul collecte
        # cette journée, ce qui est la lecture naturelle.
        first = parse_date(args.start)
        last = parse_date(args.end) if args.end else first
        return date_range(first, last)
    if args.date is None:
        raise ValueError(
            "Période absente : donner --start/--end, ou --date avec --days."
        )
    if args.days < 1:
        raise ValueError("--days doit valoir au moins 1.")
    last = parse_date(args.date)
    return date_range(last - timedelta(days=args.days - 1), last)


def _configure_logging() -> None:
    """Arme le journal, et met la sortie standard à l'abri de l'encodage local.

    MLflow imprime des emoji quand il rend la main ; une console Windows en
    cp1252 lève alors une UnicodeEncodeError au beau milieu d'un run qui, lui,
    s'est bien passé. On ne peut pas demander à MLflow de se taire, mais on
    peut faire en sorte qu'un caractère non représentable dégrade l'affichage
    au lieu d'interrompre le traitement.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée du conteneur de collecte."""
    _configure_logging()
    args = parse_args(argv)
    try:
        config = load_config()
        # En mode rattrapage la période vient de la base, pas de la ligne de
        # commande : elle est calculée plus bas, une fois le moteur ouvert et
        # le référentiel résolu.
        check_period_arguments(args)
        days = [] if args.catch_up else requested_days(args)
        settings = SourceSettings.from_config(config)
        if args.limit is not None:
            settings = settings.with_page_size(args.limit)
        engine = open_engine(config.get_optional_str("database.url"))
        # Avant la première page : un rattrapage d'un mois qui échouerait
        # au chargement aurait relu la source pour rien.
        verify_schema(engine, (mesure, site, ingestion_etat))
        batch_size = config.get_int("database.batch_size")
    except (ConfigError, DatabaseError, PathError, ValueError) as exc:
        logger.error("configuration invalide : %s", exc)
        return EXIT_FAILED

    total = 0
    try:
        with SourceClient(settings) as client:
            sites = resolve_sites(client, engine, batch_size, args.sites)
            if args.catch_up:
                days = catch_up_days(
                    engine,
                    sites,
                    args.catch_up_days
                    or config.get_int(
                        "collector.catch_up_days", DEFAULT_CATCH_UP_DAYS
                    ),
                )
            logger.info(
                "collecte de %d site(s) du %s au %s, %d mesure(s) par requête",
                len(sites),
                days[0],
                days[-1],
                settings.page_size,
            )
            for day in days:
                total += collect_day(client, engine, batch_size, day, sites)
    except (SourceError, SQLAlchemyError) as exc:
        logger.error("collecte interrompue : %s", exc)
        return EXIT_FAILED
    finally:
        engine.dispose()
    logger.info("collecte terminée : %d mesure(s) soumise(s)", total)
    return EXIT_OK



if __name__ == "__main__":
    raise SystemExit(main())
