"""Fixtures du service de transformation.

Les lots d'entrée sont construits ici, à partir du seul schéma partagé, et non
en appelant le collecteur : l'ETL ne l'importe pas, et un test qui le ferait
prouverait le contraire de ce que l'architecture affirme. Ce que ces fixtures
fabriquent, c'est exactement ce que `MEASURE_SCHEMA` décrit, c'est-à-dire ce
qu'une lecture de `mesure` rend — le contrat se suffit à lui-même.

Aucun test ne joint PostgreSQL. Le moteur factice mémorise les instructions au
lieu de les exécuter : ce qui compte n'est pas que la base les accepte, c'est
que l'ETL produise la bonne — celle qui repose les colonnes déduites sans
toucher à celles de la source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import pytest

from predict_common.db import SOURCE_COLUMNS
from predict_common.schemas import (
    NUMERIC_COLUMNS,
    SITE_COLUMN,
    TIMESTAMP_COLUMN,
)


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées."""

    def __init__(self, executed: list[Any]) -> None:
        self._executed = executed

    def execute(self, statement: Any) -> None:
        self._executed.append(statement)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self) -> None:
        self.executed: list[Any] = []

    def begin(self) -> FakeConnection:
        return FakeConnection(self.executed)

    def dispose(self) -> None:
        """Rien à rendre : il n'y a pas de connexion derrière."""


@pytest.fixture
def make_reading() -> Callable[..., dict[str, Any]]:
    """Fabrique une ligne telle qu'une lecture de `mesure` la rend."""

    def build(
        timestamp: str,
        site_id: str = "SITE001",
        **overrides: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            TIMESTAMP_COLUMN: timestamp,
            SITE_COLUMN: site_id,
            "consumption_kw": 87.34,
            "consumption_kwh": 1.45,
            "voltage_v": 230.1,
            "current_a": 12.5,
            "power_factor": 0.95,
            "temperature_celsius": 21.3,
            "humidity_percent": 48.0,
            "null_reasons": [],
            "data_quality": "good",
        }
        record.update(overrides)
        return record

    return build


@pytest.fixture
def make_raw() -> Callable[[Iterable[dict[str, Any]]], pd.DataFrame]:
    """Assemble des lignes en lot, typé comme le driver le rend."""

    def build(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(list(records), columns=list(SOURCE_COLUMNS))
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN], utc=True, errors="coerce"
        )
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    return build
