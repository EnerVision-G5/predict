"""Fixtures partagées par les tests ETL et entraînement."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from etl.config import EtlConfig


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées."""

    def __init__(self, executed):
        self._executed = executed

    def execute(self, statement):
        self._executed.append(statement)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self):
        self.executed = []

    def begin(self):
        return FakeConnection(self.executed)


@pytest.fixture
def config() -> EtlConfig:
    """Configuration inoffensive : aucun test ne joint réellement ces hôtes."""
    return EtlConfig(
        database_url="postgresql+psycopg://u:p@localhost:5432/db",
        mock_api_url="http://mock.invalid",
        batch_size=2,
    )


@pytest.fixture
def make_reading() -> Callable[..., dict[str, Any]]:
    """Fabrique une mesure au format de l'API Mock IoT."""

    def build(
        timestamp: str,
        site_id: str = "SITE001",
        **overrides: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "site_id": site_id,
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
