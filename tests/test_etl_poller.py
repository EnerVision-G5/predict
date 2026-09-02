"""Ingestion continue des mesures courantes.

Aucun test ne joint le réseau ni une base : la session HTTP, le moteur
SQLAlchemy et l'horloge sont injectés. C'est ce qui permet de vérifier la
cadence et le retard d'ingestion sans attendre une minute réelle.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import requests

from etl.exclude import to_exclusions
from etl.extract import ExtractionError
from etl.impute import IMPUTED_COLUMN, METHOD_COLUMN, METHOD_NONE
from etl.poller import (
    EXIT_STARTUP_FAILED,
    PollContext,
    PollError,
    Schedule,
    SiteTick,
    TickReport,
    call_with_retry,
    ingestion_lag_s,
    install_signal_handlers,
    parse_args,
    poll_forever,
    poll_site,
    quality_summary,
    resolve_targets,
    run_tick,
)

NOW = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)


class FakeEngine:
    """Moteur factice : le poller n'en attend que `dispose`."""

    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class FakeSession:
    """Session factice : le poller n'en attend que `close`."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeStop:
    """Événement d'arrêt dont l'attente est instantanée.

    Un threading.Event réel dormirait la cadence complète entre deux ticks :
    la suite mettrait des minutes là où l'horloge factice suffit.
    """

    def __init__(self) -> None:
        self.waits: list[float] = []
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, timeout: float | None = None) -> bool:
        if timeout is not None:
            self.waits.append(timeout)
        return self._set


@pytest.fixture
def poll_config(config):
    """Configuration de polling déterministe, sans attente réelle."""
    return dataclasses.replace(
        config,
        poll_interval_s=60.0,
        poll_retries=1,
        poll_backoff_s=0.0,
        poll_timeout_s=1.0,
        lag_warning_s=180.0,
    )


@pytest.fixture
def context(poll_config) -> PollContext:
    """Contexte de polling branché sur des ressources factices."""
    return PollContext(
        config=poll_config,
        engine=FakeEngine(),
        session=FakeSession(),
        stop=FakeStop(),
    )


class TestCallWithRetry:
    """La reprise sur échec réseau, socle de la relance automatique."""

    def test_a_successful_call_is_not_retried(self) -> None:
        calls = []

        assert call_with_retry(
            lambda: calls.append(1) or "ok",
            attempts=3,
            backoff_s=1.0,
            sleep=lambda _delay: None,
            label="site SITE001",
        ) == "ok"
        assert len(calls) == 1

    def test_a_transient_failure_is_retried(self) -> None:
        attempts = []

        def flaky() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise requests.ConnectionError("réseau coupé")
            return "ok"

        result = call_with_retry(
            flaky,
            attempts=3,
            backoff_s=0.0,
            sleep=lambda _delay: None,
            label="site SITE001",
        )
        assert result == "ok"
        assert len(attempts) == 3

    def test_the_budget_spent_raises_a_poll_error(self) -> None:
        def always_fails() -> None:
            raise ExtractionError("503")

        with pytest.raises(PollError, match="2 tentative"):
            call_with_retry(
                always_fails,
                attempts=2,
                backoff_s=0.0,
                sleep=lambda _delay: None,
                label="site SITE001",
            )

    def test_the_original_error_is_chained(self) -> None:
        def always_fails() -> None:
            raise ExtractionError("503")

        with pytest.raises(PollError) as excinfo:
            call_with_retry(
                always_fails,
                attempts=1,
                backoff_s=0.0,
                sleep=lambda _delay: None,
                label="site SITE001",
            )
        assert isinstance(excinfo.value.__cause__, ExtractionError)

    def test_the_wait_grows_with_the_attempt(self) -> None:
        delays: list[float] = []

        def always_fails() -> None:
            raise requests.ConnectionError("réseau coupé")

        with pytest.raises(PollError):
            call_with_retry(
                always_fails,
                attempts=3,
                backoff_s=2.0,
                sleep=delays.append,
                label="site SITE001",
            )
        assert delays == [2.0, 4.0]

    def test_an_unexpected_error_is_not_swallowed(self) -> None:
        def broken() -> None:
            raise ZeroDivisionError("bug de programmation")

        with pytest.raises(ZeroDivisionError):
            call_with_retry(
                broken,
                attempts=3,
                backoff_s=0.0,
                sleep=lambda _delay: None,
                label="site SITE001",
            )


class TestIngestionLag:
    """Le retard de données, celui que l'exploitant lit dans les logs."""

    def test_an_empty_batch_has_no_lag(self) -> None:
        assert ingestion_lag_s(pd.DataFrame({"ts": []}), NOW) is None

    def test_the_lag_is_the_age_of_the_oldest_reading(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    ["2026-01-15T07:58:00Z", "2026-01-15T07:59:00Z"],
                    utc=True,
                )
            }
        )
        assert ingestion_lag_s(frame, NOW) == 120.0

    def test_a_source_clock_ahead_of_ours_stays_visible(self) -> None:
        frame = pd.DataFrame(
            {"ts": pd.to_datetime(["2026-01-15T08:00:30Z"], utc=True)}
        )
        assert ingestion_lag_s(frame, NOW) == -30.0


class TestQualitySummary:
    def test_an_empty_batch_is_reported_as_such(self) -> None:
        assert quality_summary(pd.DataFrame({"data_quality": []})) == "aucune"

    def test_qualities_are_counted_and_sorted(self) -> None:
        frame = pd.DataFrame({"data_quality": ["good", "partial", "good"]})
        assert quality_summary(frame) == "good=2 partial=1"


class TestSchedule:
    """La cadence : ancrée sur des instants absolus, sans dérive."""

    def test_the_next_tick_is_one_interval_later(self) -> None:
        schedule = Schedule(interval_s=60.0, due_at=NOW)

        schedule.advance(NOW + timedelta(seconds=2))

        assert schedule.due_at == NOW + timedelta(seconds=60)
        assert schedule.missed == 0

    def test_a_slow_tick_does_not_shift_the_cadence(self) -> None:
        """Un tick de 5 s ne décale pas l'échéance : elle reste sur la minute."""
        schedule = Schedule(interval_s=60.0, due_at=NOW)

        schedule.advance(NOW + timedelta(seconds=5))
        schedule.advance(NOW + timedelta(seconds=65))

        assert schedule.due_at == NOW + timedelta(seconds=120)

    def test_ticks_overrun_are_skipped_not_replayed(self) -> None:
        """/current ne sert que l'instant : rattraper relirait la même valeur."""
        schedule = Schedule(interval_s=60.0, due_at=NOW)

        schedule.advance(NOW + timedelta(seconds=200))

        assert schedule.missed == 3
        assert schedule.due_at == NOW + timedelta(seconds=240)

    def test_the_skipped_count_is_reset_on_a_healthy_tick(self) -> None:
        schedule = Schedule(interval_s=60.0, due_at=NOW)

        schedule.advance(NOW + timedelta(seconds=200))
        schedule.advance(NOW + timedelta(seconds=241))

        assert schedule.missed == 0


class TestPollSite:
    """Un site : lecture courante, écriture, retard journalisé."""

    @pytest.fixture(autouse=True)
    def _stub_io(self, monkeypatch, make_reading):
        """Remplace la source et la base par des doubles observables."""
        self.loaded: list[pd.DataFrame] = []
        self.excluded: list[pd.DataFrame] = []
        self.records = [make_reading("2026-01-15T07:59:00Z")]

        monkeypatch.setattr(
            "etl.poller.fetch_current",
            lambda config, site_id, session=None: self.records,
        )
        monkeypatch.setattr(
            "etl.poller.load_frame",
            lambda engine, frame, batch_size: self.loaded.append(frame)
            or len(frame),
        )
        monkeypatch.setattr(
            "etl.poller.load_exclusions",
            lambda engine, frame, batch_size: self.excluded.append(frame)
            or len(to_exclusions(frame)),
        )

    def test_the_current_reading_is_written(self, context) -> None:
        tick = poll_site(context, "SITE001", NOW)

        assert tick == SiteTick(site_id="SITE001", rows=1, lag_s=60.0)
        assert len(self.loaded) == 1
        assert self.loaded[0]["site_id"].tolist() == ["SITE001"]

    def test_the_written_batch_carries_its_imputation_columns(
        self, context
    ) -> None:
        """Le mode continu passe par les mêmes étages que le rattrapage."""
        poll_site(context, "SITE001", NOW)

        written = self.loaded[0]
        assert written.loc[0, IMPUTED_COLUMN] == 87.34
        assert written.loc[0, METHOD_COLUMN] == METHOD_NONE

    def test_an_outage_alone_on_its_tick_is_filed_and_logged(
        self, context, make_reading, caplog
    ) -> None:
        """Un tick ne porte qu'une mesure : rien ne permet de l'imputer."""
        self.records = [
            make_reading(
                "2026-01-15T07:59:00Z",
                consumption_kw=None,
                null_reasons=["sensor_failure"],
            )
        ]

        with caplog.at_level(logging.WARNING):
            tick = poll_site(context, "SITE001", NOW)

        assert tick.excluded == 1
        assert "1 mesure(s) écartée(s)" in caplog.text
        assert pd.isna(self.loaded[0].loc[0, "consumption_kw"])

    def test_a_silent_site_reports_no_lag(self, context) -> None:
        self.records = []

        assert poll_site(context, "SITE001", NOW) == SiteTick(
            site_id="SITE001", rows=0, lag_s=None
        )

    def test_a_duplicate_reading_is_written_once(
        self, context, make_reading
    ) -> None:
        """La PK (site_id, ts) refuserait le lot entier sur un doublon."""
        self.records = [make_reading("2026-01-15T07:59:00Z")] * 2

        assert poll_site(context, "SITE001", NOW).rows == 1

    def test_a_reading_without_timestamp_is_dropped_and_logged(
        self, context, make_reading, caplog
    ) -> None:
        self.records = [
            make_reading("2026-01-15T07:59:00Z"),
            make_reading("pas une date"),
        ]

        with caplog.at_level(logging.WARNING):
            tick = poll_site(context, "SITE001", NOW)

        assert tick.rows == 1
        assert "1 mesure(s) écartée(s)" in caplog.text

    def test_an_excessive_lag_is_logged_as_a_warning(
        self, context, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING):
            poll_site(context, "SITE001", NOW + timedelta(seconds=600))

        assert "au-delà du seuil" in caplog.text

    def test_a_normal_lag_stays_informational(self, context, caplog) -> None:
        with caplog.at_level(logging.INFO):
            poll_site(context, "SITE001", NOW)

        assert "retard 60.0 s" in caplog.text
        assert "au-delà du seuil" not in caplog.text


class TestRunTick:
    """Le tick : un site en panne ne prive pas les six autres."""

    def test_every_site_is_polled(self, context, monkeypatch) -> None:
        monkeypatch.setattr(
            "etl.poller.poll_site",
            lambda ctx, site_id, now: SiteTick(site_id, rows=1, lag_s=12.0),
        )

        report = run_tick(context, ["SITE001", "SITE002"], NOW)

        assert report == TickReport(
            rows=2, lags_s=(12.0, 12.0), failed_sites=()
        )

    def test_an_unreachable_site_does_not_stop_the_others(
        self, context, monkeypatch, caplog
    ) -> None:
        def poll(ctx, site_id, now):
            if site_id == "SITE001":
                raise PollError("site SITE001 : 2 tentative(s) échouée(s).")
            return SiteTick(site_id, rows=1, lag_s=12.0)

        monkeypatch.setattr("etl.poller.poll_site", poll)

        with caplog.at_level(logging.ERROR):
            report = run_tick(context, ["SITE001", "SITE002"], NOW)

        assert report.failed_sites == ("SITE001",)
        assert report.rows == 1
        assert "SITE001 : tick abandonné" in caplog.text

    def test_a_database_failure_is_survivable(
        self, context, monkeypatch
    ) -> None:
        from sqlalchemy.exc import OperationalError

        def poll(ctx, site_id, now):
            raise OperationalError("INSERT", {}, Exception("base coupée"))

        monkeypatch.setattr("etl.poller.poll_site", poll)

        assert run_tick(context, ["SITE001"], NOW).failed_sites == ("SITE001",)

    def test_a_programming_error_is_not_absorbed(
        self, context, monkeypatch
    ) -> None:
        def poll(ctx, site_id, now):
            raise AttributeError("bug de programmation")

        monkeypatch.setattr("etl.poller.poll_site", poll)

        with pytest.raises(AttributeError):
            run_tick(context, ["SITE001"], NOW)

    def test_a_silent_site_is_excluded_from_the_lag(
        self, context, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "etl.poller.poll_site",
            lambda ctx, site_id, now: SiteTick(site_id, rows=0, lag_s=None),
        )

        assert run_tick(context, ["SITE001"], NOW).max_lag_s is None


class TestPollForever:
    """La boucle : cadencée, interruptible, sans dérive."""

    def test_the_loop_stops_on_the_stop_event(
        self, context, monkeypatch
    ) -> None:
        ticks: list[datetime] = []

        def run(ctx, sites, now):
            ticks.append(now)
            if len(ticks) == 3:
                ctx.stop.set()
            return TickReport(rows=1, lags_s=(1.0,), failed_sites=())

        monkeypatch.setattr("etl.poller.run_tick", run)

        assert poll_forever(context, ["SITE001"], clock=_fake_clock()) == 3

    def test_a_stop_requested_during_the_wait_skips_the_tick(
        self, context, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "etl.poller.run_tick",
            lambda ctx, sites, now: pytest.fail("aucun tick attendu"),
        )
        context.stop.set()

        assert poll_forever(context, ["SITE001"], clock=_fake_clock()) == 0

    def test_the_schedule_skew_is_logged(
        self, context, monkeypatch, caplog
    ) -> None:
        """Le retard d'ordonnancement est la seconde forme de retard visible."""

        def run(ctx, sites, now):
            ctx.stop.set()
            return TickReport(rows=0, lags_s=(), failed_sites=())

        monkeypatch.setattr("etl.poller.run_tick", run)
        # L'horloge saute d'une minute par lecture : le tick démarre donc
        # très au-delà de son échéance.
        clock = _fake_clock(step_s=60.0)

        with caplog.at_level(logging.WARNING):
            poll_forever(context, ["SITE001"], clock=clock)

        assert "retard d'ordonnancement" in caplog.text


class TestResolveTargets:
    def test_the_requested_sites_win(self, context) -> None:
        assert resolve_targets(context, ["SITE003"]) == ["SITE003"]

    def test_the_referential_is_used_by_default(
        self, context, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "etl.poller.resolve_sites",
            lambda config, requested, session=None: ["SITE001", "SITE002"],
        )

        assert resolve_targets(context, None) == ["SITE001", "SITE002"]

    def test_an_unreachable_referential_raises_a_poll_error(
        self, context, monkeypatch
    ) -> None:
        def unreachable(config, requested, session=None):
            raise requests.ConnectionError("réseau coupé")

        monkeypatch.setattr("etl.poller.resolve_sites", unreachable)
        monkeypatch.setattr("etl.poller.STARTUP_BACKOFF_S", 0.0)

        with pytest.raises(PollError):
            resolve_targets(context, None)


class TestSignalHandling:
    def test_sigterm_requests_a_clean_stop(self, monkeypatch) -> None:
        handlers: dict[int, object] = {}
        monkeypatch.setattr(
            "signal.signal",
            lambda signum, handler: handlers.__setitem__(signum, handler),
        )
        stop = threading.Event()

        install_signal_handlers(stop)
        import signal as signal_module

        handlers[signal_module.SIGTERM](signal_module.SIGTERM, None)

        assert stop.is_set()


class TestParseArgs:
    def test_the_interval_defaults_to_the_configuration(self) -> None:
        assert parse_args([]).interval is None

    def test_the_interval_can_be_overridden(self) -> None:
        assert parse_args(["--interval", "5"]).interval == 5.0

    def test_sites_are_repeatable(self) -> None:
        args = parse_args(["--site", "SITE001", "--site", "SITE002"])
        assert args.sites == ["SITE001", "SITE002"]


class TestMain:
    """Le point d'entrée du conteneur, et son code de sortie."""

    def test_a_failed_startup_exits_non_zero(
        self, monkeypatch, poll_config, caplog
    ) -> None:
        """Sortie non nulle : c'est Docker qui relance sur un process neuf."""
        from etl import poller

        engine = FakeEngine()
        session = FakeSession()
        monkeypatch.setattr(poller, "load_config", lambda: poll_config)
        monkeypatch.setattr(poller, "_build_engine", lambda config: engine)
        monkeypatch.setattr(poller, "build_session", lambda: session)
        monkeypatch.setattr(poller, "install_signal_handlers", lambda stop: None)
        monkeypatch.setattr(
            poller,
            "resolve_targets",
            lambda context, requested: (_ for _ in ()).throw(
                PollError("référentiel des sites : 3 tentative(s) échouée(s).")
            ),
        )

        with caplog.at_level(logging.ERROR):
            assert poller.main([]) == EXIT_STARTUP_FAILED

        assert "démarrage impossible" in caplog.text
        assert engine.disposed and session.closed

    def test_resources_are_released_after_the_loop(
        self, monkeypatch, poll_config
    ) -> None:
        from etl import poller

        engine = FakeEngine()
        session = FakeSession()
        monkeypatch.setattr(poller, "load_config", lambda: poll_config)
        monkeypatch.setattr(poller, "_build_engine", lambda config: engine)
        monkeypatch.setattr(poller, "build_session", lambda: session)
        monkeypatch.setattr(poller, "install_signal_handlers", lambda stop: None)
        monkeypatch.setattr(
            poller, "resolve_targets", lambda context, requested: ["SITE001"]
        )
        monkeypatch.setattr(
            poller, "poll_forever", lambda context, sites: 0
        )

        assert poller.main([]) == 0
        assert engine.disposed and session.closed


def _fake_clock(step_s: float = 1.0):
    """Horloge factice qui avance d'un pas fixe à chaque lecture.

    Le poller lit l'horloge plusieurs fois par tick : une horloge figée
    ferait boucler l'attente, une horloge réelle ferait durer le test une
    minute par tick.
    """
    state = {"now": NOW}

    def clock() -> datetime:
        current = state["now"]
        state["now"] = current + timedelta(seconds=step_s)
        return current

    return clock
