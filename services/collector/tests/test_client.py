"""Client de la source : découpage des fenêtres, reprises, débit borné.

Aucun test ne sort sur le réseau. Ce qui compte ici est le comportement du
client face à ce que la source lui répond — une page pleine, une page vide,
une coupure, un statut d'erreur — parce que c'est ce comportement, et non la
bibliothèque HTTP, qui décide si une journée est collectée entière.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from predict_common.source import (
    MAX_SPIKE_MINUTES,
    RateLimiter,
    RetryExhausted,
    SourceError,
    SourceSettings,
)

# Fenêtres courtes : le découpage se lit au nombre de tranches, et une journée
# entière en ferait 720 avec le `page_size` de deux minutes des réglages.
FIVE_MINUTES = (
    datetime(2026, 9, 2, tzinfo=UTC),
    datetime(2026, 9, 2, 0, 5, tzinfo=UTC),
)
ONE_HOUR = (
    datetime(2026, 9, 2, tzinfo=UTC),
    datetime(2026, 9, 2, 1, 0, tzinfo=UTC),
)


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


def test_iter_readings_cuts_the_window_into_slices(make_client, make_reading) -> None:
    # `page_size` vaut 2 : des tranches de deux minutes. Cinq minutes en font
    # donc trois — deux pleines, et un reliquat d'une minute.
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        seen.append((params["start_time"], params["limit"]))
        return json_response([make_reading("2026-09-02T00:00:00Z")])

    client = make_client(handler)
    list(client.iter_readings("SITE001", *FIVE_MINUTES))
    assert [limit for _, limit in seen] == ["2", "2", "1"]


def test_iter_readings_asks_one_result_per_minute(make_client) -> None:
    # Le coeur du contrat de la source : `limit` est un NOMBRE DE RESULTATS
    # reparti sur la fenetre, donc une resolution. Autant de resultats que de
    # minutes, et pas un de plus, sinon deux points tombent dans la meme
    # minute et la serie n'est plus a la minute.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return json_response([])

    client = make_client(handler, {"page_size": 60})
    list(client.iter_readings("SITE001", *ONE_HOUR))
    assert seen["limit"] == "60"
    assert seen["start_time"].startswith("2026-09-02T00:00:00")
    assert seen["end_time"].startswith("2026-09-02T01:00:00")


def test_iter_readings_tiles_the_window_without_overlap(make_client) -> None:
    # La source pose son premier point sur `start_time` : une tranche qui
    # repartirait de la fin de la precedente moins une minute redemanderait
    # cette minute-la, et une qui sauterait une minute la perdrait.
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        seen.append((params["start_time"], params["end_time"]))
        return json_response([])

    client = make_client(handler)
    list(client.iter_readings("SITE001", *FIVE_MINUTES))
    for (_, fin), (debut, _) in zip(seen, seen[1:], strict=False):
        assert fin == debut


def test_iter_readings_never_reasks_the_same_window(make_client, make_reading) -> None:
    # La source rend TOUJOURS `limit` resultats : « page pleine, donc il en
    # reste » ne devient jamais faux. Une pagination par curseur redemandait
    # ici la meme fenetre sans fin. Le decoupage borne le nombre d'appels a
    # celui des tranches, quoi que la source reponde.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["start_time"])
        stamp = "2026-09-02T00:00:00Z"
        return json_response([make_reading(stamp), make_reading(stamp)])

    client = make_client(handler)
    list(client.iter_readings("SITE001", *FIVE_MINUTES))
    assert len(calls) == 3
    assert len(set(calls)) == 3


def test_iter_readings_reports_a_short_answer(
    make_client, make_reading, caplog
) -> None:
    # Moins de resultats que de minutes demandees laisse un trou. Ce n'est pas
    # fatal, mais le taire ferait passer une serie amputee pour complete.
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response([make_reading("2026-09-02T00:00:00Z")])

    client = make_client(handler)
    with caplog.at_level("WARNING"):
        list(client.iter_readings("SITE001", *ONE_HOUR))
    assert "1 mesure(s) recue(s)" in caplog.text.replace("ç", "c")


def test_iter_readings_drops_a_trailing_partial_minute(make_client) -> None:
    # Une source qui ne sert que des minutes ne sait rien faire de trente
    # secondes, et `limit=0` lui vaudrait un 422.
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return json_response([])

    client = make_client(handler)
    window = (
        datetime(2026, 9, 2, tzinfo=UTC),
        datetime(2026, 9, 2, 0, 0, 30, tzinfo=UTC),
    )
    assert list(client.iter_readings("SITE001", *window)) == []
    assert calls == []


def test_iter_readings_passes_the_site_as_a_query_parameter(
    make_client, make_reading
) -> None:
    # Le site n'est pas un segment de chemin : la source expose une seule
    # route d'historique pour tous les sites.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return json_response([])

    client = make_client(handler)
    list(client.iter_readings("SITE009", *FIVE_MINUTES))
    assert seen["site_id"] == "SITE009"


def test_iter_readings_yields_an_empty_answer_without_failing(make_client) -> None:
    # Une fenetre sans donnee est un cas normal au demarrage de la chaine, pas
    # une panne : elle ne rend rien et laisse les tranches suivantes essayer.
    client = make_client(lambda _: json_response([]))
    assert list(client.iter_readings("SITE001", *FIVE_MINUTES)) == []


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
                "readings_path": "/readings",
                "current_path": "/sites/{site_id}/current",
                "simulate_spike_path": "/simulate/spike/{site_id}",
                "alerts_path": "/alerts",
                "sensors_status_path": "/sensors/status",
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


class TestSimulateSpike:
    """La seule route en écriture de la source, et la seule qu'on ne rejoue pas."""

    def test_le_pic_est_demande_en_post_avec_la_duree(self, make_client) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response({"status": "simulated", "site_id": "SITE002"})

        client = make_client(handler)
        payload = client.simulate_spike("SITE002", 60)

        assert payload["status"] == "simulated"
        assert seen[0].method == "POST"
        assert seen[0].url.path == "/api/v1/simulate/spike/SITE002"
        assert seen[0].url.params["duration_minutes"] == "60"

    def test_une_duree_hors_bornes_ne_part_pas_sur_le_reseau(self, make_client) -> None:
        # La source répondrait 422 : la refuser ici évite de la solliciter, et
        # dit à l'appelant ce qu'elle attend sans qu'il ait à le découvrir.
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return json_response({})

        client = make_client(handler)
        with pytest.raises(ValueError):
            client.simulate_spike("SITE002", MAX_SPIKE_MINUTES + 1)
        assert calls == []

    def test_un_echec_n_est_jamais_rejoue(self, make_client) -> None:
        """Rejouer un POST déclencherait un second pic par-dessus le premier."""
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(500, json={"detail": "source en carafe"})

        client = make_client(handler)
        with pytest.raises(SourceError):
            client.simulate_spike("SITE002", 30)
        # Une lecture aurait été retentée ; une écriture, jamais.
        assert len(calls) == 1

    def test_une_reponse_qui_n_est_pas_un_objet_est_refusee(self, make_client) -> None:
        client = make_client(lambda _: json_response(["pas un objet"]))
        with pytest.raises(SourceError):
            client.simulate_spike("SITE002", 30)


class TestAlertesEtCapteurs:
    """Les deux lectures annexes : ce que `mesure` ne peut pas dire."""

    def test_les_alertes_sont_rendues_telles_que_la_source_les_sert(
        self, make_client
    ) -> None:
        payload = [{"alert_id": "ALR-1", "site_id": "SITE001"}]
        client = make_client(lambda _: json_response(payload))
        assert client.fetch_alerts() == payload

    def test_les_filtres_partent_en_parametres(self, make_client) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response([])

        client = make_client(handler)
        client.fetch_alerts(site_id="SITE002", severity="critical")

        assert seen[0].url.params["site_id"] == "SITE002"
        assert seen[0].url.params["severity"] == "critical"

    def test_aucune_alerte_est_une_reponse_valable(self, make_client) -> None:
        # La source ne sert que les alertes actives : une liste vide dit qu'il
        # n'y en a aucune, pas que la route est en panne.
        client = make_client(lambda _: json_response([]))
        assert client.fetch_alerts() == []

    def test_une_reponse_qui_n_est_pas_une_liste_est_refusee(self, make_client) -> None:
        client = make_client(lambda _: json_response({"alert_id": "ALR-1"}))
        with pytest.raises(SourceError):
            client.fetch_alerts()

    def test_l_etat_des_capteurs_reste_indexe_par_site(self, make_client) -> None:
        # Un objet et non une liste : remettre à plat ici obligerait
        # l'appelant à refaire le lien entre le site et ses capteurs.
        payload = {"SITE001": {"sensors": {}, "overall": "ok"}}
        client = make_client(lambda _: json_response(payload))
        assert client.fetch_sensors_status() == payload

    def test_un_etat_des_capteurs_en_liste_est_refuse(self, make_client) -> None:
        client = make_client(lambda _: json_response([]))
        with pytest.raises(SourceError):
            client.fetch_sensors_status()
