"""Extraction paginée des mesures depuis l'API Mock IoT.

Les tests injectent une session factice : aucun appel réseau n'est fait, et la
pagination est vérifiée sur les paramètres réellement envoyés.
"""

from datetime import UTC, datetime

import pytest

from etl.extract import (
    ExtractionError,
    build_session,
    fetch_readings,
    fetch_sites,
)

START = datetime(2026, 1, 15, tzinfo=UTC)
END = datetime(2026, 1, 16, tzinfo=UTC)


class FakeResponse:
    """Réponse minimale exposant ce que le module d'extraction consomme."""

    def __init__(self, payload, status_code: int = 200, url: str = "http://x"):
        self._payload = payload
        self.status_code = status_code
        self.url = url
        self.request = type("Request", (), {"method": "GET"})()

    def json(self):
        return self._payload


class FakeSession:
    """Session HTTP factice qui rejoue une liste de réponses préparées."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._responses.pop(0)


def test_build_session_announces_json() -> None:
    session = build_session()
    try:
        assert session.headers["Accept"] == "application/json"
    finally:
        session.close()


def test_fetch_sites_returns_the_referential(config) -> None:
    session = FakeSession([FakeResponse([{"site_id": "SITE001"}])])
    assert fetch_sites(config, session=session) == [{"site_id": "SITE001"}]


def test_fetch_sites_rejects_a_payload_that_is_not_a_list(config) -> None:
    session = FakeSession([FakeResponse({"items": []})])
    with pytest.raises(ExtractionError):
        fetch_sites(config, session=session)


def test_fetch_readings_walks_every_page(config, make_reading) -> None:
    first = [make_reading("2026-01-15T08:00:00Z")] * config.batch_size
    second = [make_reading("2026-01-15T09:00:00Z")]
    session = FakeSession(
        [
            FakeResponse({"items": first}),
            FakeResponse({"items": second}),
            FakeResponse({"items": []}),
        ]
    )
    items = list(fetch_readings(config, "SITE001", START, END, session=session))
    assert len(items) == config.batch_size + 1
    assert [call["params"]["offset"] for call in session.calls] == [
        0,
        config.batch_size,
        config.batch_size + 1,
    ]


def test_fetch_readings_stops_on_an_empty_page(config) -> None:
    session = FakeSession([FakeResponse({"items": []})])
    assert list(fetch_readings(config, "SITE001", START, END, session=session)) == []


def test_fetch_readings_raises_on_an_http_error(config) -> None:
    session = FakeSession([FakeResponse({}, status_code=503)])
    with pytest.raises(ExtractionError):
        list(fetch_readings(config, "SITE001", START, END, session=session))
