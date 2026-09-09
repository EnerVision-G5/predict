# **********************************************************************
# * Nom     : timestamps.py                                            *
# * Type    : Module                                                   *
# * Sujet   : Lecture des horodatages de la source et passage en UTC   *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

# Fuseau prêté à un horodatage qui n'en déclare aucun.
DEFAULT_SOURCE_TIMEZONE = "UTC"

# Reconnaît un décalage horaire en fin d'horodatage.
_OFFSET = re.compile(r"(?:[Zz]|[+-]\d{2}:?\d{2})$")

logger = logging.getLogger(__name__)


def declares_offset(raw: object) -> bool:
    """Méthode : declares_offset
    Description : Dit si un horodatage brut porte un décalage explicite, Z ou
      +hh:mm.
    """
    return isinstance(raw, str) and _OFFSET.search(raw.strip()) is not None


def parse_timestamp(
    raw: object,
    naive_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> datetime | None:
    """Méthode : parse_timestamp
    Description : Convertit un horodatage isolé en datetime UTC, ou rend None
      s'il est illisible.
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
    """Méthode : to_utc
    Description : Convertit une colonne d'horodatages en UTC, en signalant ceux
      qui n'avaient pas de fuseau.
    """
    parsed = pd.to_datetime(column, errors="coerce", format="ISO8601", utc=True)
    if parsed.empty:
        return parsed

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
