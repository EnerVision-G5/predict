"""Collecte continue : cadence, tolérance aux pannes, journal du retard.

Le poller est un processus long. Ce qui compte n'est donc pas seulement qu'il
collecte, mais qu'il survive — à un site injoignable, à un tick qui déborde de
la cadence, à un arrêt demandé au milieu d'une attente. Chaque test ci-dessous
porte sur l'une de ces trois situations, et aucun n'attend réellement : la
cadence et l'horloge sont injectées.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from collector_fakes import FakeEngine
from sqlalchemy.exc import SQLAlchemyError

from collector.poller import (
    SCHEDULE_SKEW_WARNING_S,
    PollContext,
    PollSettings,
    Schedule,
    TickReport,
    ingestion_lag_s,
    poll_forever,
    poll_site,
    quality_summary,
    record_tick,
    resolve_targets,
    run_tick,
)
from collector.sink import IngestionState, to_measures
from predict_common.source import SourceClient, SourceError

NOW = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)


@pytest.fixture
def poll_settings() -> PollSettings:
    """Réglages d'une boucle dont la cadence est injectée, jamais attendue."""
    return PollSettings(interval_s=60.0, lag_warning_s=180.0, batch_size=10)


@pytest.fixture
def make_context(poll_settings, make_client):
    """Fabrique un contexte de boucle branché sur une source simulée."""

    def build(handler, engine: FakeEngine | None = None) -> PollContext:
        return PollContext(
            settings=poll_settings,
            client=make_client(handler),
            engine=engine or FakeEngine(),
            stop=threading.Event(),
        )

    return build


def current(make_reading, **overrides):
    """Gestionnaire qui sert une mesure courante."""
    reading = make_reading("2026-09-02T07:59:50Z", **overrides)
    return lambda _: httpx.Response(200, json=reading)


class TestSchedule:
    """La cadence est ancrée sur des instants absolus."""

    def test_a_tick_on_time_advances_by_one_interval(self) -> None:
        schedule = Schedule(interval_s=60.0, due_at=NOW)
        schedule.advance(NOW + timedelta(seconds=1))
        assert schedule.due_at == NOW + timedelta(seconds=60)
        assert schedule.missed == 0

    def test_a_slow_tick_does_not_shift_the_following_ones(self) -> None:
        # Une attente de la durée de l'intervalle décalerait toute la suite, et
        # le retard disparaîtrait en se fondant dans la cadence.
        schedule = Schedule(interval_s=60.0, due_at=NOW)
        schedule.advance(NOW + timedelta(seconds=30))
        assert schedule.due_at == NOW + timedelta(seconds=60)

    def test_overrunning_the_cadence_skips_the_missed_ticks(self) -> None:
        # /current ne sert que la mesure du moment : rattraper relirait
        # plusieurs fois la même valeur.
        schedule = Schedule(interval_s=60.0, due_at=NOW)
        schedule.advance(NOW + timedelta(seconds=200))
        # Les ticks de 60, 120 et 180 s sont passés pendant le tick lent.
        assert schedule.missed == 3
        assert schedule.due_at == NOW + timedelta(seconds=240)


class TestLag:
    """Le retard de données mesure l'âge de ce que sert la source."""

    def test_the_lag_is_the_age_of_the_oldest_measure(self, make_reading) -> None:
        frame = to_measures([make_reading("2026-09-02T07:59:50Z")])
        assert ingestion_lag_s(frame, NOW) == pytest.approx(10.0)

    def test_a_source_clock_ahead_of_ours_is_not_hidden(self, make_reading) -> None:
        # Un retard négatif signale une horloge en avance : le masquer serait
        # une faute, c'est une panne d'infrastructure.
        frame = to_measures([make_reading("2026-09-02T08:00:10Z")])
        assert ingestion_lag_s(frame, NOW) < 0

    def test_a_silent_site_has_no_lag(self) -> None:
        assert ingestion_lag_s(to_measures([]), NOW) is None


def test_quality_summary_counts_each_qualification(make_reading) -> None:
    frame = to_measures(
        [
            make_reading("2026-09-02T07:59:00Z"),
            make_reading("2026-09-02T07:59:30Z", data_quality="critical"),
        ]
    )
    assert quality_summary(frame) == "critical=1 good=1"


def test_quality_summary_survives_an_empty_batch() -> None:
    assert quality_summary(to_measures([])) == "aucune"


def test_poll_site_writes_the_measure(make_context, make_reading) -> None:
    engine = FakeEngine()
    context = make_context(current(make_reading), engine=engine)
    tick = poll_site(context, "SITE001", NOW)
    assert tick.rows == 1
    assert tick.lag_s == pytest.approx(10.0)
    assert len(engine.executed) == 1


def test_two_ticks_write_two_measures(make_context, make_reading) -> None:
    engine = FakeEngine()
    context = make_context(current(make_reading), engine=engine)
    poll_site(context, "SITE001", NOW)
    poll_site(context, "SITE002", NOW)
    assert len(engine.executed) == 2


def test_the_poller_never_overwrites_what_the_etl_deduced(
    make_context, make_reading
) -> None:
    # Le tick suivant repasse sur des minutes déjà transformées : un
    # DO UPDATE y effacerait la qualification et l'imputation.
    from sqlalchemy.dialects import postgresql

    engine = FakeEngine()
    context = make_context(current(make_reading), engine=engine)
    poll_site(context, "SITE001", NOW)
    compiled = str(engine.executed[0].compile(dialect=postgresql.dialect()))
    assert "DO NOTHING" in compiled
    assert "DO UPDATE" not in compiled


def test_a_failing_site_does_not_stop_the_others(
    make_context, make_reading, monkeypatch
) -> None:
    # La vraie relance d'un site en panne, c'est le tick suivant.
    def poll(context, site_id, now):
        if site_id == "SITE002":
            raise SourceError("site muet")
        return poll_site(context, site_id, now)

    monkeypatch.setattr("collector.poller.poll_site", poll)
    context = make_context(current(make_reading))
    report = run_tick(context, ["SITE001", "SITE002", "SITE003"], NOW)
    assert report.failed_sites == ("SITE002",)
    assert report.rows == 2


def test_the_tick_reports_the_worst_lag(make_context, make_reading) -> None:
    context = make_context(current(make_reading))
    report = run_tick(context, ["SITE001", "SITE002"], NOW)
    assert report.max_lag_s == pytest.approx(10.0)


def test_a_tick_where_every_site_is_silent_has_no_lag(make_context) -> None:
    context = make_context(lambda _: httpx.Response(503))
    report = run_tick(context, ["SITE001"], NOW)
    assert report.max_lag_s is None
    assert report.failed_sites == ("SITE001",)


def test_an_excessive_lag_is_logged_as_a_warning(
    make_context, make_reading, caplog
) -> None:
    context = make_context(current(make_reading))
    with caplog.at_level(logging.WARNING, logger="collector.poller"):
        poll_site(context, "SITE001", NOW + timedelta(seconds=600))
    assert "au-delà du seuil" in caplog.text


def test_a_late_tick_is_logged_as_a_warning(
    make_context, make_reading, caplog
) -> None:
    context = make_context(current(make_reading))
    late = SCHEDULE_SKEW_WARNING_S + 5.0
    # Ancrage, attente, démarrage réel du tick, fin du tick. L'arrêt n'est
    # demandé qu'après le dernier, pour que la boucle exécute un tour entier.
    instants = [NOW, NOW, NOW + timedelta(seconds=late), NOW + timedelta(seconds=late)]
    calls: list[int] = []

    def clock() -> datetime:
        calls.append(1)
        if len(calls) >= len(instants):
            context.stop.set()
        return instants[min(len(calls), len(instants)) - 1]

    with caplog.at_level(logging.WARNING, logger="collector.poller"):
        poll_forever(context, ["SITE001"], clock=clock)
    assert "retard d'ordonnancement" in caplog.text


def test_the_loop_stops_as_soon_as_the_stop_is_requested(
    make_context, make_reading
) -> None:
    # Un conteneur qu'on stoppe rend la main tout de suite, il n'use pas la
    # minute en cours.
    context = make_context(current(make_reading))
    context.stop.set()
    assert poll_forever(context, ["SITE001"], clock=lambda: NOW) == 0


def test_resolve_targets_keeps_the_requested_sites(make_context) -> None:
    # L'exploitant a nommé ses sites : une source qui ne sert pas son
    # référentiel ne doit pas empêcher la boucle de tourner. Le seed a déjà
    # posé les sites courants, et la clé étrangère tranchera s'il manquait
    # vraiment quelque chose.
    context = make_context(lambda _: httpx.Response(500))
    assert resolve_targets(context, ["SITE009"]) == ["SITE009"]


def test_resolve_targets_synchronises_the_referential(make_context) -> None:
    payload = [
        {
            "site_id": "SITE001",
            "site_type": "office",
            "site_name": "Bureau",
            "capacity_kw": 200,
        }
    ]
    engine = FakeEngine()
    context = make_context(lambda _: httpx.Response(200, json=payload), engine=engine)
    resolve_targets(context, None)
    # `site` est entretenue avant toute mesure : la clé étrangère l'exige.
    assert len(engine.executed) == 1


def test_resolve_targets_reads_the_reference_list(make_context) -> None:
    payload = [{"site_id": "SITE001"}, {"site_id": "SITE002"}]
    context = make_context(lambda _: httpx.Response(200, json=payload))
    assert resolve_targets(context, None) == ["SITE001", "SITE002"]


def test_a_startup_without_the_reference_list_fails(make_context) -> None:
    # Sans référentiel, la boucle n'a rien à interroger : le processus sort et
    # c'est la politique de redémarrage du conteneur qui reprend la main.
    context = make_context(lambda _: httpx.Response(500))
    with pytest.raises(SourceError):
        resolve_targets(context, None)


def test_a_tick_across_midnight_writes_its_own_instant(
    make_context, make_reading
) -> None:
    # Le poller ne range rien par journée : c'est l'horodatage de la mesure qui
    # la place, et la clé primaire qui arbitre.
    engine = FakeEngine()
    context = make_context(
        lambda _: httpx.Response(200, json=make_reading("2026-09-03T00:00:05Z")),
        engine=engine,
    )
    tick = poll_site(context, "SITE001", datetime(2026, 9, 3, 0, 0, 10, tzinfo=UTC))
    assert tick.rows == 1


def test_settings_read_the_configuration_block() -> None:
    from predict_common.config import Config

    config = Config(
        values={
            "collector": {"poll_interval_s": 60, "lag_warning_s": 180},
            "database": {"batch_size": 1000},
        }
    )
    settings = PollSettings.from_config(config)
    assert settings.interval_s == 60.0
    assert settings.batch_size == 1000
    # Le bloc `source` est absent de cette configuration : le fuseau retombe
    # sur un défaut neutre plutôt que de faire échouer le démarrage.
    assert settings.source_timezone == "UTC"


def test_the_client_is_closed_when_the_context_ends(make_client) -> None:
    client: SourceClient = make_client(lambda _: httpx.Response(200, json=[]))
    with client:
        pass
    assert client._client.is_closed


def test_a_batch_the_source_cannot_place_is_not_written(
    make_context, make_reading
) -> None:
    engine = FakeEngine()
    context = make_context(
        lambda _: httpx.Response(200, json=make_reading("pas une date")), engine=engine
    )
    tick = poll_site(context, "SITE001", NOW)
    assert tick.rows == 0
    assert engine.executed == []


class TestIngestionState:
    """Ce que le tick repose en base, et que `mesure` ne peut pas dire.

    Un site en échec n'écrit aucune mesure. Sans cette table, il serait
    indiscernable d'un site dont la source n'avait rien de neuf — et c'est
    justement la panne qu'on cherche à voir.
    """

    @staticmethod
    def _state_statements(engine: FakeEngine) -> list[str]:
        """Rend les instructions visant `ingestion_etat`, compilées."""
        from sqlalchemy.dialects import postgresql

        compiled = [
            str(statement.compile(dialect=postgresql.dialect()))
            for statement in engine.executed
        ]
        return [text for text in compiled if "ingestion_etat" in text]

    def test_a_successful_tick_records_a_success(
        self, make_context, make_reading
    ) -> None:
        engine = FakeEngine()
        context = make_context(current(make_reading), engine=engine)
        report = run_tick(context, ["SITE001"], NOW)
        record_tick(context, report)

        statements = self._state_statements(engine)
        assert len(statements) == 1
        # Le présent, pas un historique : la ligne du site est reposée.
        assert "DO UPDATE" in statements[0]
        assert "last_success_at" in statements[0]

    def test_a_failed_tick_records_the_failure(self, make_context) -> None:
        engine = FakeEngine()
        context = make_context(lambda _: httpx.Response(503), engine=engine)
        report = run_tick(context, ["SITE001"], NOW)
        record_tick(context, report)

        assert report.states[0].succeeded is False
        statements = self._state_statements(engine)
        assert len(statements) == 1
        # Ni la date du dernier succès, ni le nombre de lignes, ni le retard
        # ne sont touchés : un échec n'a rien à en dire, et les écraser
        # effacerait la seule chose vraie qu'on sache encore du site.
        assert "last_success_at" not in statements[0]
        assert "consecutive_failures" in statements[0]

    def test_a_mixed_tick_records_both_shapes(
        self, make_context, make_reading, monkeypatch
    ) -> None:
        def poll(context, site_id, now):
            if site_id == "SITE002":
                raise SourceError("site muet")
            return poll_site(context, site_id, now)

        monkeypatch.setattr("collector.poller.poll_site", poll)
        engine = FakeEngine()
        context = make_context(current(make_reading), engine=engine)
        record_tick(context, run_tick(context, ["SITE001", "SITE002"], NOW))

        # Deux instructions et non une : succès et échecs ne posent pas les
        # mêmes colonnes, un lot mixte devrait choisir une forme pour les deux.
        assert len(self._state_statements(engine)) == 2

    def test_a_database_failure_on_the_state_write_does_not_kill_the_loop(
        self, make_context, make_reading, caplog
    ) -> None:
        # Le tick dont la base vient de refuser les mesures ne pourra pas non
        # plus y écrire son échec. Mourir là serait mourir au moment précis où
        # le processus a le plus de raisons de continuer à essayer.
        class BrokenEngine(FakeEngine):
            def begin(self):
                raise SQLAlchemyError("base injoignable")

        engine = BrokenEngine()
        context = make_context(current(make_reading), engine=engine)
        report = TickReport(
            rows=0,
            lags_s=(),
            states=(IngestionState(site_id="SITE001", attempted_at=NOW),),
        )
        with caplog.at_level(logging.ERROR, logger="collector.poller"):
            record_tick(context, report)
        assert "état d'ingestion non enregistré" in caplog.text
