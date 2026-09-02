"""Client de la source : pagination, reprises, débit borné.

Aucun test ne sort sur le réseau. Ce qui compte ici est le comportement du
client face à ce que la source lui répond — une page pleine, une page vide,
une coupure, un statut d'erreur — parce que c'est ce comportement, et non la
bibliothèque HTTP, qui décide si une journée est collectée entière.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from collector.client import (
    RateLimiter,
    RetryExhausted,
    SourceError,
    SourceSettings,
)

WINDOW = (datetime(2026, 9, 2, tzinfo=UTC), datetime(2026, 9, 3, tzinfo=UTC))


def json_response(payload) -> httpx.Response:
    """Réponse 200 portant le corps demandé."""
    return httpx.Response(200, json=payload)


def test_fetch_sites_returns_the_reference_list(make_client) -> None:
    client = make_client(lambda _: json_response([{"site_id": "SITE001"}]))
    assert client.fetch_sites() == [{"site_id": "SITE001"}]


def test_site_ids_keeps_only_the_identifiers(make_client) -> None:
    payload = [{"site_id": "SITE001", "nom": "Usine"}, {"nom": "sans identifiant"}]
    client = make_client(lambda _: json_response(payload))
    assert client.site_ids() == ["SITE001"]


def test_fetch_sites_refuses_a_payload_that_is_not_a_list(make_client) -> None:
    # Une forme inattendue n'est pas retentée : insister ne changerait pas ce
    # que la source répond, seulement le temps qu'elle met à le répondre.
    client = make_client(lambda _: json_response({"items": []}))
    with pytest.raises(SourceError):
        client.fetch_sites()


def test_iter_readings_walks_every_page(make_client, make_reading) -> None:
    pages = [
        {"items": [make_reading("2026-09-02T00:00:00Z")] * 2},
        {"items": [make_reading("2026-09-02T02:00:00Z")]},
        {"items": []},
    ]
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        seen.append(offset)
        return json_response(pages[len(seen) - 1])

    client = make_client(handler)
    assert len(list(client.iter_readings("SITE001", *WINDOW))) == 3
    # L'API pagine par offset : sans avance stricte, une page pleine
    # relancerait indéfiniment la même requête.
    assert seen == [0, 2, 3]


def test_iter_readings_stops_on_the_first_empty_page(make_client) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return json_response({"items": []})

    client = make_client(handler)
    assert list(client.iter_readings("SITE001", *WINDOW)) == []
    assert len(calls) == 1


def test_iter_readings_passes_the_window_to_the_source(make_client) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return json_response({"items": []})

    client = make_client(handler)
    list(client.iter_readings("SITE001", *WINDOW))
    assert seen["start_time"].startswith("2026-09-02")
    assert seen["limit"] == "2"


def test_fetch_current_wraps_a_lone_measure_in_a_list(
    make_client, make_reading
) -> None:
    # Le reste de la chaîne travaille par lots : distinguer les deux formes
    # serait à la charge de chaque appelant.
    client = make_client(lambda _: json_response(make_reading("2026-09-02T08:00:00Z")))
    assert len(client.fetch_current("SITE001")) == 1


def test_fetch_current_unwraps_an_items_envelope(make_client, make_reading) -> None:
    payload = {"items": [make_reading("2026-09-02T08:00:00Z")]}
    client = make_client(lambda _: json_response(payload))
    assert len(client.fetch_current("SITE001")) == 1


def test_fetch_current_refuses_an_unusable_payload(make_client) -> None:
    client = make_client(lambda _: json_response(["pas un objet"]))
    with pytest.raises(SourceError):
        client.fetch_current("SITE001")


def test_an_error_status_is_not_taken_for_data(make_client) -> None:
    client = make_client(lambda _: httpx.Response(500, json={"detail": "boum"}))
    with pytest.raises(RetryExhausted):
        client.fetch_sites()


def test_a_transient_failure_is_retried(make_client) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("réseau coupé")
        return json_response([{"site_id": "SITE001"}])

    client = make_client(handler, {"retries": 2})
    assert client.site_ids() == ["SITE001"]
    assert len(attempts) == 2


def test_a_lasting_outage_gives_up_after_its_attempts(make_client) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("réseau coupé")

    client = make_client(handler, {"retries": 2})
    with pytest.raises(RetryExhausted):
        client.fetch_sites()
    # Trois tentatives : la première, plus les deux reprises demandées.
    assert len(attempts) == 3


def test_the_backoff_grows_with_the_attempt(make_client) -> None:
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("réseau coupé")

    client = make_client(handler, {"retries": 2, "backoff_s": 2.0})
    client.sleep = delays.append
    with pytest.raises(RetryExhausted):
        client.fetch_sites()
    # Une coupure qui dure ne se règle pas en insistant à la même cadence.
    assert delays == [2.0, 4.0]


def test_settings_read_the_configuration_blocks() -> None:
    from predict_common.config import Config

    config = Config(
        values={
            "source": {
                "base_url": "http://mock:8000/",
                "sites_path": "/sites",
                "readings_path": "/sites/{site_id}/readings",
                "current_path": "/sites/{site_id}/current",
                "page_size": 500,
                "timeout_s": 30,
                "retries": 3,
                "backoff_s": 2,
                "rate_limit_rps": 10,
            },
            "collector": {"poll_timeout_s": 10},
        }
    )
    settings = SourceSettings.from_config(config)
    # La barre finale est retirée : elle donnerait une URL à double séparateur.
    assert settings.base_url == "http://mock:8000"
    assert settings.page_size == 500


class TestRateLimiter:
    """La limite protège la source, pas le collecteur."""

    def test_a_null_rate_never_waits(self) -> None:
        delays: list[float] = []
        limiter = RateLimiter(0.0, sleep=delays.append, clock=lambda: 0.0)
        limiter.wait()
        limiter.wait()
        assert delays == []

    def test_the_first_call_goes_through_immediately(self) -> None:
        delays: list[float] = []
        limiter = RateLimiter(2.0, sleep=delays.append, clock=lambda: 100.0)
        limiter.wait()
        assert delays == []

    def test_the_next_call_waits_for_its_slot(self) -> None:
        delays: list[float] = []
        limiter = RateLimiter(2.0, sleep=delays.append, clock=lambda: 100.0)
        limiter.wait()
        limiter.wait()
        # Deux requêtes par seconde : une demi-seconde entre deux appels.
        assert delays == [0.5]
