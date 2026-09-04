"""Fixtures du service de collecte.

Aucun test ne joint l'API Mock ni PostgreSQL.

Le transport HTTP est remplacé par un `httpx.MockTransport`, qui laisse le
client réel — sa pagination, ses reprises, sa limite de débit — s'exécuter tel
qu'il s'exécutera en production. Remplacer le client lui-même par un faux ne
testerait plus que le faux.

La base est remplacée par un moteur qui mémorise les instructions au lieu de
les exécuter. Ce qui compte n'est pas que PostgreSQL les accepte — c'est son
métier, et le schéma figé le garantit — mais que le collecteur produise la
bonne instruction : celle qui n'écrase rien.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from predict_common.source import SourceClient, SourceSettings


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées.

    Elle rend aussi des lignes : le collecteur lit `capteur_etat` avant de
    l'écraser, pour dater les débuts et les fins de panne. Les mêmes lignes
    sont rendues à chaque appel — aucun test n'a besoin de plus, et un
    séquenceur de résultats rendrait ces fixtures illisibles.
    """

    def __init__(self, executed: list[Any], rows: list[Any]) -> None:
        self._executed = executed
        self._rows = rows

    def execute(self, statement: Any) -> list[Any]:
        self._executed.append(statement)
        return list(self._rows)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.executed: list[Any] = []
        self.rows: list[Any] = list(rows or ())

    def begin(self) -> FakeConnection:
        return FakeConnection(self.executed, self.rows)

    def dispose(self) -> None:
        """Rien à rendre : il n'y a pas de connexion derrière."""


@pytest.fixture
def settings() -> SourceSettings:
    """Réglages inoffensifs : aucun test ne joint réellement cet hôte."""
    return SourceSettings(
        base_url="http://mock.invalid",
        sites_path="/api/v1/sites",
        readings_path="/api/v1/readings",
        current_path="/api/v1/sites/{site_id}/current",
        simulate_spike_path="/api/v1/simulate/spike/{site_id}",
        alerts_path="/api/v1/alerts",
        sensors_status_path="/api/v1/sensors/status",
        page_size=2,
        timeout_s=1.0,
        poll_timeout_s=0.5,
        retries=1,
        backoff_s=0.0,
        rate_limit_rps=0.0,
    )


@pytest.fixture
def make_client(settings: SourceSettings) -> Callable[..., SourceClient]:
    """Fabrique un client branché sur un gestionnaire de requêtes local."""

    def build(
        handler: Callable[[httpx.Request], httpx.Response],
        overrides: dict[str, Any] | None = None,
    ) -> SourceClient:
        import dataclasses

        applied = dataclasses.replace(settings, **(overrides or {}))
        transport = httpx.MockTransport(handler)
        client = httpx.Client(
            base_url=applied.base_url,
            transport=transport,
            timeout=applied.timeout_s,
        )
        return SourceClient(applied, client=client, sleep=lambda _: None)

    return build


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
