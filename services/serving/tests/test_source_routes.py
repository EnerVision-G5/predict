"""Les deux routes qui relaient la source vers l'API métier.

Le service d'inférence est le seul point d'entrée HTTP de predict, et l'API
métier ne connaît pas l'API Mock. Ces deux routes sont donc le chemin par
lequel un dashboard peut lister les sites de la source et y déclencher un pic.

Ce qui est vérifié ici est la frontière, pas le calcul : une source qui tombe
donne 502 et non 503, parce que la panne est en amont du service et non en
lui ; et un pic déclenché reste un succès même si la relecture échoue, parce
que rejouer l'appel superposerait deux pics.

Aucun réseau : la source est un httpx.MockTransport, comme dans les tests du
client partagé.
"""

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
    """Réglages inoffensifs : aucun test ne joint réellement cet hôte."""
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
    """Client HTTP dont la source est un transport local."""

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
    """Source nominale : référentiel, pic accepté, mesure relue."""
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
    # Relayé, pas traduit : les noms de champs restent ceux de la source, et
    # c'est l'API métier qui décide d'en faire un référentiel en base.
    assert body[1]["capacity_kw"] == 1000
    assert body[1]["site_name"] == "Usine Lyon Vénissieux"


def test_une_source_muette_donne_502_et_non_503(client) -> None:
    # 503 dirait que le service d'inférence est tombé, et l'exploitant
    # chercherait la panne du mauvais côté : c'est la source qui est muette.
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
    # La confirmation seule ne prouve rien à qui regarde un dashboard : c'est
    # la mesure relue qui montre le pic.
    assert body["reading"]["consumption_kw"] == 812.5
    assert body["reading"]["data_quality"] == "good"


def test_un_pic_declenche_reste_un_succes_si_la_relecture_echoue(client) -> None:
    """Le pic a eu lieu : le rendre en erreur inviterait à le rejouer."""

    def spike_only(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/v1/simulate/spike/"):
            return httpx.Response(200, json=SPIKE_PAYLOAD)
        return httpx.Response(503, json={"detail": "mesure indisponible"})

    with client(spike_only) as http:
        response = http.post(f"/api/v1/simulate/spike/{SITE}")

    assert response.status_code == 200
    assert response.json()["reading"] is None


def test_une_duree_hors_bornes_est_refusee_avant_le_reseau(client) -> None:
    # La source répondrait 422 elle-même, mais l'appel serait parti : une
    # borne tenue ici évite de la solliciter pour rien.
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
    # Ici le service EST en cause : il n'a pas terminé son démarrage.
    monkeypatch.setattr(api, "configure", lambda: None)
    api.state["source"] = None
    with TestClient(api.app) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 503


def test_un_site_id_porteur_de_query_est_refuse_avant_tout_appel(client) -> None:
    """Un identifiant hors forme n'atteint jamais la source.

    L'identifiant repartait tel quel dans le CHEMIN de l'appel sortant
    (`simulate_spike_path.format(...)`). Starlette décode le paramètre avant
    de le passer : `SITE001%3Fadmin=1` devient `SITE001?admin=1`, et httpx
    lit alors le `?` comme le début d'une chaîne de requête. L'appelant
    choisissait donc les paramètres de la requête que le service émet vers la
    source, en plus de ceux que le service y met lui-même.

    `%23` est le second vecteur, et il est plus sournois : le `#` tronque
    l'URL, `duration_minutes` disparaît, et la source applique sa durée par
    défaut sans que personne ne l'ait demandée.

    La traversée de chemin, elle, n'a jamais été possible : le routeur de
    Starlette ne fait pas correspondre `%2F` à un paramètre de segment.

    Le refus est un 422, que le contrat documente déjà sur cette route : la
    spécification gelée n'a pas à bouger pour que le trou soit fermé.
    """
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
    # Le point du test : rien n'est parti vers la source.
    assert seen == []


def test_un_site_id_du_referentiel_passe(client) -> None:
    """La borne refuse ce qui n'est pas un identifiant, et rien d'autre."""
    with client(route) as http:
        response = http.post(f"/api/v1/simulate/spike/{SITE}")

    assert response.status_code == 200
    assert response.json()["site_id"] == SITE


def test_un_corps_illisible_de_la_source_donne_502_et_non_500(client) -> None:
    """Une source qui répond 200 avec du HTML est une panne de la source.

    `response.json()` lève alors une JSONDecodeError, qui dérive de
    ValueError et que le client ne traduit pas en SourceError : elle
    traversait jusqu'à FastAPI, qui rendait 500. L'API métier lisait ce 500
    comme « le service d'inférence est tombé » et envoyait chercher
    l'incident du mauvais côté.
    """

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>portail captif</html>")

    with client(broken) as http:
        response = http.get("/api/v1/sites")

    assert response.status_code == 502
