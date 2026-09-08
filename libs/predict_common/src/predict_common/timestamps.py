"""Lecture des horodatages servis par la source, fuseau compris.

Un horodatage déclare son fuseau, ou ne le déclare pas. Les deux formes
viennent de la même source, et la seconde est apparue en cours d'exploitation :
le 8 septembre 2026 vers 07:51 UTC, `/current` a cessé de suffixer ses
horodatages d'un `Z` et s'est mise à servir l'heure locale de sa machine.
`/readings`, interrogée avec des bornes datées, répond toujours en UTC — le
changement ne portait donc pas sur toute la source, et un correctif qui aurait
décalé la chaîne entière aurait cassé le rattrapage pour réparer la collecte.

Prendre un horodatage sans fuseau pour de l'UTC — ce que fait pandas dès qu'on
lui passe `utc=True` — décale la collecte de l'écart entre les deux fuseaux,
deux heures en été. Ce décalage ne s'annonce pas : la mesure est écrite dans le
futur, le retard d'ingestion passe en négatif, et le dashboard, qui borne ses
lectures à l'instant présent, cesse d'afficher une collecte qui n'a pourtant
jamais été interrompue. Une panne muette qui se présente comme un arrêt.

Ce module tient une règle, et une seule : un horodatage qui déclare son fuseau
est cru sur parole ; un horodatage qui n'en déclare aucun est lu dans le fuseau
que la configuration prête à la source, puis ramené en UTC. Le reste de la
chaîne ne voit que de l'UTC, exactement comme avant.

L'hypothèse est journalisée à chaque lot où elle sert, et le restera tant que
la source ne datera pas ce qu'elle envoie. Elle est raisonnable, elle n'est pas
vérifiable : seule la source sait ce qu'elle a voulu dire, et une supposition
tue serait indiscernable d'une donnée juste.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

# Fuseau prêté aux horodatages nus quand la configuration n'en nomme aucun.
# Neutre à dessein : sans consigne, le code ne déplace rien et se comporte
# comme avant ce module. C'est le déploiement qui décrit sa source, dans
# `conf/base.yaml`.
DEFAULT_SOURCE_TIMEZONE = "UTC"

# `Z`, `+02:00`, `+0200` ou `-05:00`, en fin de chaîne seulement : l'ancre
# empêche de prendre les tirets d'une date pour un décalage négatif.
_OFFSET = re.compile(r"(?:[Zz]|[+-]\d{2}:?\d{2})$")

logger = logging.getLogger(__name__)


def declares_offset(raw: object) -> bool:
    """Vrai si l'horodatage textuel porte son fuseau.

    Une valeur qui n'est pas du texte n'en porte pas : ni un `None`, ni un
    `NaN`, ni un nombre. Aucune n'est un horodatage lisible, et toutes finiront
    écartées à l'écriture — les compter ici comme nues n'ajoute donc rien à
    leur sort et évite un cas particulier de plus.
    """
    return isinstance(raw, str) and _OFFSET.search(raw.strip()) is not None


def parse_timestamp(
    raw: object,
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> datetime | None:
    """Lit un horodatage de la source, ou rend `None` s'il est illisible.

    Le résultat est toujours conscient de son fuseau et exprimé en UTC : un
    appelant qui compare deux horodatages n'a pas à se demander lequel des deux
    est nu, comparaison que Python refuse de trancher en levant.

    Un `datetime` déjà construit est accepté tel quel : les réponses de la
    source ne portent que du texte, mais les appelants n'ont pas tous la même
    origine et aucun n'a à savoir laquelle il tient.

    Les deux heures pathologiques d'un changement d'heure ne sont pas traitées
    ici. `fold=0` désigne la première occurrence d'une heure jouée deux fois et
    reste une lecture défendable là où cette fonction sert : dater une alerte
    ou une panne capteur à l'heure près.
    """
    if isinstance(raw, datetime):
        parsed = raw
    else:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(naive_timezone))
    return parsed.astimezone(UTC)


def to_utc(
    column: pd.Series,
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> pd.Series:
    """Ramène une colonne d'horodatages de la source en UTC.

    Le format ISO est imposé et non deviné, comme partout ailleurs dans la
    chaîne : laisser pandas interpréter au cas par cas ferait accepter un
    `03/09/2026` dont personne ne saurait dire si c'est mars ou septembre.

    Un horodatage impossible — la nuit du passage à l'heure d'été — ou ambigu
    — celle du passage à l'heure d'hiver — n'est pas arbitré : il devient NaT,
    donc une mesure sans horodatage plaçable, que l'écriture écarte et compte.
    Deux heures par an, la source doit dire son fuseau pour être crue ; le
    reste du temps, la supposition suffit.
    """
    parsed = pd.to_datetime(column, errors="coerce", format="ISO8601", utc=True)
    if parsed.empty:
        return parsed

    # `utc=True` a déjà localisé les horodatages nus en UTC. Les distinguer
    # demande donc de regarder la valeur d'origine, seule à savoir ce que la
    # source avait écrit.
    naive = ~column.map(declares_offset)
    if not naive.any():
        return parsed

    logger.warning(
        "%d horodatage(s) sans fuseau, lus en %s : la source ne déclare pas"
        " le sien",
        int(naive.sum()),
        naive_timezone,
    )
    parsed = parsed.copy()
    parsed.loc[naive] = (
        parsed[naive]
        .dt.tz_localize(None)
        .dt.tz_localize(naive_timezone, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert(UTC)
    )
    return parsed
