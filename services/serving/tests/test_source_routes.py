from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from predict_common.source import SourceClient, SourceSettings
from serving import api

SITE = "SITE002"

SITES_PAYLOAD = [
    {
        "site_id": "SITE001",
        "site_type": "office",
        "site_name": "Bureau Paris La Défense",
        "location": "Paris, France",
        "capacity_kw": 200,
        "status": "active",
    },
    {
        "site_id": SITE,
        "site_type": "factory",
        "site_name": "Usine Lyon Vénissieux",
        "location": "Lyon, France",
        "capacity_kw": 1000,
        "status": "active",
    },
]

SPIKE_PAYLOAD = {
    "status": "simulated",
    "site_id": SITE,
    "event": "consumption_spike",
    "duration_minutes": 60,
    "message": "Pic de consommation simulé sur Usine Lyon Vénissieux pour 60 minutes.",
}

CURRENT_PAYLOAD = {
    "timestamp": "2026-09-04T14:32:00",
    "site_id": SITE,
    "site_type": "factory",
    "consumption_kw": 812.5,
    "consumption_kwh": 812.5,
    "voltage_v": 398.5,
    "current_a": 826.4,
    "power_factor": 0.921,
    "temperature_celsius": 18.3,
    "humidity_percent": 62.1,
    "null_reasons": [],
    "data_quality": "good",
}


def settings() -> SourceSettings:
    return SourceSettings(
        base_url="http://mock.invalid",
        sites_path="/api/v1/sites",
        readings_path="/api/v1/readings",
        current_path="/api/v1/sites/{site_id}/current",
        simulate_spike_path="/api/v1/simulate/spike/{site_id}",
        alerts_path="/api/v1/alerts",
        sensors_status_path="/api/v1/sensors/status",
        page_size=10,
        timeout_s=1.0,
        poll_timeout_s=0.5,
        retries=0,
        backoff_s=0.0,
        rate_limit_rps=0.0,
    )


@pytest.fixture
def client(monkeypatch):
    def build(handler):
        source = SourceClient(
            settings(),
            client=httpx.Client(
                base_url="http://mock.invalid",
                transport=httpx.MockTransport(handler),
            ),
            sleep=lambda _seconds: None,
        )

        def configure() -> None:
            api.state["registry"] = None
            api.state["spec"] = None
            api.state["source"] = source

        monkeypatch.setattr(api, "configure", configure)
        return TestClient(api.app)

    return build


def route(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/v1/sites":
        return httpx.Response(200, json=SITES_PAYLOAD)
    if request.url.path.startswith("/api/v1/simulate/spike/"):
        return httpx.Response(200, json=SPIKE_PAYLOAD)
    if request.url.path.endswith("/current"):
        return httpx.Response(200, json=CURRENT_PAYLOAD)
    return httpx.Response(404, json={"detail": "route inconnue"})


def test_le_referentiel_est_relaye_tel_que_la_source_le_sert(client) -> None:
    with client(route) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 200
    body = response.json()
    assert [site["site_id"] for site in body] == ["SITE001", SITE]
    assert body[1]["capacity_kw"] == 1000
    assert body[1]["site_name"] == "Usine Lyon Vénissieux"


def test_une_source_muette_donne_502_et_non_503(client) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "source en carafe"})

    with client(broken) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 502


def test_un_pic_rend_la_confirmation_et_la_mesure_qui_suit(client) -> None:
    with client(route) as http:
        response = http.post(
            f"/api/v1/simulate/spike/{SITE}", params={"duration_minutes": 60}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["site_id"] == SITE
    assert body["status"] == "simulated"
    assert body["duration_minutes"] == 60
    assert body["reading"]["consumption_kw"] == 812.5
    assert body["reading"]["data_quality"] == "good"


def test_un_pic_declenche_reste_un_succes_si_la_relecture_echoue(client) -> None:
    def spike_only(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/simulate/spike/"):
            return httpx.Response(200, json=SPIKE_PAYLOAD)
        return httpx.Response(503, json={"detail": "mesure indisponible"})

    with client(spike_only) as http:
        response = http.post(f"/api/v1/simulate/spike/{SITE}")

    assert response.status_code == 200
    assert response.json()["reading"] is None


def test_une_duree_hors_bornes_est_refusee_avant_le_reseau(client) -> None:
    calls: list[str] = []

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json=SPIKE_PAYLOAD)

    with client(counting) as http:
        response = http.post(
            f"/api/v1/simulate/spike/{SITE}", params={"duration_minutes": 241}
        )

    assert response.status_code == 422
    assert calls == []


def test_un_service_sans_source_le_dit_en_503(monkeypatch) -> None:
    monkeypatch.setattr(api, "configure", lambda: None)
    api.state["source"] = None
    with TestClient(api.app) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 503


def test_un_site_id_porteur_de_query_est_refuse_avant_tout_appel(client) -> None:
    seen: list[str] = []

    def spying(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return route(request)

    with client(spying) as http:
        injected = http.post("/api/v1/simulate/spike/SITE001%3Fadmin=1")
        truncated = http.post("/api/v1/simulate/spike/SITE001%23frag")

    assert injected.status_code == 422
    assert truncated.status_code == 422
    assert "site_id" in injected.json()["detail"]
    assert seen == []


def test_un_site_id_du_referentiel_passe(client) -> None:
    with client(route) as http:
        response = http.post(f"/api/v1/simulate/spike/{SITE}")

    assert response.status_code == 200
    assert response.json()["site_id"] == SITE


def test_un_corps_illisible_de_la_source_donne_502_et_non_500(client) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>portail captif</html>")

    with client(broken) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 502
