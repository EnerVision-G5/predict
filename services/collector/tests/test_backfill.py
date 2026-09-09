from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from collector_fakes import FakeEngine
from sqlalchemy.dialects import postgresql

from collector.__main__ import (
    catch_up_days,
    check_period_arguments,
    collect_day,
    parse_args,
    requested_days,
)
from collector.sink import build_insert, to_measures, to_records, write
from predict_common.source import MAX_PAGE_SIZE, SourceError, SourceSettings

SITES = tuple(f"SITE{index:03d}" for index in range(1, 8))


def days_for(*argv: str) -> list[date]:
    return requested_days(parse_args(list(argv)))


class TestPeriode:
    def test_two_bounds_give_the_whole_range(self) -> None:
        days = days_for("--start", "2026-08-30", "--end", "2026-09-02")
        assert days == [
            date(2026, 8, 30),
            date(2026, 8, 31),
            date(2026, 9, 1),
            date(2026, 9, 2),
        ]

    def test_a_lone_start_collects_that_day(self) -> None:
        assert days_for("--start", "2026-09-02") == [date(2026, 9, 2)]

    def test_the_short_form_counts_backwards(self) -> None:
        days = days_for("--date", "2026-09-02", "--days", "3")
        assert days == [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2)]

    def test_a_single_day_is_the_default_depth(self) -> None:
        assert days_for("--date", "2026-09-02") == [date(2026, 9, 2)]

    def test_the_two_forms_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="une seule"):
            days_for("--date", "2026-09-02", "--start", "2026-08-01")

    def test_an_end_without_a_start_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--start"):
            days_for("--end", "2026-09-02")

    def test_an_absent_period_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Période absente"):
            days_for()

    def test_an_inverted_range_is_refused(self) -> None:
        from predict_common.paths import PathError

        with pytest.raises(PathError):
            days_for("--start", "2026-09-02", "--end", "2026-08-01")

    def test_a_null_depth_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--days"):
            days_for("--date", "2026-09-02", "--days", "0")


class TestLimite:
    def test_the_source_bound_is_a_thousand(self) -> None:
        assert MAX_PAGE_SIZE == 1000

    def test_the_command_line_overrides_the_configuration(self) -> None:
        settings = _settings().with_page_size(500)
        assert settings.page_size == 500

    def test_the_bound_is_accepted(self) -> None:
        assert _settings().with_page_size(MAX_PAGE_SIZE).page_size == 1000

    def test_beyond_the_bound_is_refused_here_and_not_by_a_422(self) -> None:
        with pytest.raises(ValueError, match="1000"):
            _settings().with_page_size(MAX_PAGE_SIZE + 1)

    def test_a_null_limit_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _settings().with_page_size(0)

    def test_the_original_settings_are_left_alone(self) -> None:
        settings = _settings()
        settings.with_page_size(500)
        assert settings.page_size == 1000

    def test_the_command_line_reads_the_limit(self) -> None:
        assert parse_args(["--date", "2026-09-02", "--limit", "250"]).limit == 250


class TestIdempotence:
    def test_the_insert_ignores_what_is_already_there(self, make_reading) -> None:
        frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
        statement = build_insert(to_records(frame))
        compiled = str(statement.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT (site_id, ts) DO NOTHING" in compiled

    def test_a_replay_never_overwrites_what_the_etl_deduced(
        self, make_reading
    ) -> None:
        frame = to_measures([make_reading("2026-09-02T08:00:00Z")])
        compiled = str(
            build_insert(to_records(frame)).compile(dialect=postgresql.dialect())
        )
        assert "DO UPDATE" not in compiled

    def test_a_duplicated_key_within_one_batch_is_settled_first(
        self, make_reading
    ) -> None:
        engine = FakeEngine()
        frame = to_measures(
            [
                make_reading("2026-09-02T08:00:00Z", consumption_kw=10.0),
                make_reading("2026-09-02T08:00:00Z", consumption_kw=20.0),
            ]
        )
        assert write(engine, frame, batch_size=10).rows == 1


class TestVolume:
    def test_two_days_are_expressible_in_both_forms(self) -> None:
        assert len(days_for("--start", "2026-09-01", "--end", "2026-09-02")) == 2
        assert len(days_for("--date", "2026-09-02", "--days", "2")) == 2

    def test_every_site_is_collected_by_default(self) -> None:
        assert parse_args(["--date", "2026-09-02"]).sites is None

    def test_the_sites_can_be_narrowed(self) -> None:
        argv = ["--date", "2026-09-02"]
        for site in SITES[:3]:
            argv += ["--site", site]
        assert parse_args(argv).sites == list(SITES[:3])

    def test_a_two_day_batch_of_seven_sites_is_submitted_whole(
        self, make_reading
    ) -> None:
        readings = [
            make_reading(f"2026-09-0{day}T{hour:02d}:{minute}0:00Z", site_id=site)
            for site in SITES
            for day in (1, 2)
            for hour in range(24)
            for minute in (0, 3)
        ]
        engine = FakeEngine()
        report = write(engine, to_measures(readings), batch_size=1000)
        assert report.rows == len(SITES) * 2 * 24 * 2
        assert report.dropped == 0


def _settings() -> SourceSettings:
    return SourceSettings(
        base_url="http://mock.invalid",
        sites_path="/api/v1/sites",
        readings_path="/api/v1/readings",
        current_path="/api/v1/sites/{site_id}/current",
        simulate_spike_path="/api/v1/simulate/spike/{site_id}",
        alerts_path="/api/v1/alerts",
        sensors_status_path="/api/v1/sensors/status",
        page_size=1000,
        timeout_s=30.0,
        poll_timeout_s=10.0,
        retries=1,
        backoff_s=0.0,
        rate_limit_rps=0.0,
    )


class _StubSource:
    def __init__(self, readings: list[dict], failing: str | None = None) -> None:
        self._readings = readings
        self._failing = failing

    def iter_readings(self, site_id: str, start_time, end_time):
        if site_id == self._failing:
            raise SourceError(f"{site_id} muet")
        return [dict(reading, site_id=site_id) for reading in self._readings]


def _state_params(engine: FakeEngine) -> list[dict]:
    compiled = [
        statement.compile(dialect=postgresql.dialect())
        for statement in engine.executed
    ]
    return [
        instruction.params
        for instruction in compiled
        if "ingestion_etat" in str(instruction)
    ]


class TestBackfillState:
    def test_a_replayed_day_is_recorded_as_a_backfill(self, make_reading) -> None:
        engine = FakeEngine()
        source = _StubSource([make_reading("2026-09-02T07:59:00Z")])
        collect_day(source, engine, 1000, date(2026, 9, 2), ["SITE001"])

        params = _state_params(engine)
        assert len(params) == 1
        assert "backfill" in params[0].values()

    def test_a_source_failure_is_recorded_before_it_propagates(
        self, make_reading
    ) -> None:
        engine = FakeEngine()
        source = _StubSource(
            [make_reading("2026-09-02T07:59:00Z")], failing="SITE002"
        )
        with pytest.raises(SourceError):
            collect_day(
                source, engine, 1000, date(2026, 9, 2), ["SITE001", "SITE002"]
            )

        params = _state_params(engine)
        assert len(params) == 2


TODAY = date(2026, 9, 5)


def covered(site_id: str, day: str, first: str = "00:00", last: str = "23:30"):
    stamp = date.fromisoformat(day)
    return (
        site_id,
        datetime.combine(stamp, datetime.min.time(), tzinfo=UTC),
        datetime.fromisoformat(f"{day}T{first}:00+00:00"),
        datetime.fromisoformat(f"{day}T{last}:00+00:00"),
    )


def engine_covering(*rows) -> FakeEngine:
    return FakeEngine(rows=list(rows))


def test_une_base_vide_est_entierement_rattrapee() -> None:
    days = catch_up_days(engine_covering(), ["SITE001"], depth_days=35, today=TODAY)

    assert days[0] == date(2026, 8, 2)
    assert days[-1] == TODAY
    assert len(days) == 35


def test_un_trou_ancien_est_vu_alors_que_le_poller_tourne() -> None:
    engine = engine_covering(
        covered("SITE001", "2026-09-04"),
        covered("SITE001", "2026-09-05"),
    )

    days = catch_up_days(engine, ["SITE001"], depth_days=35, today=TODAY)

    assert date(2026, 8, 2) in days
    assert date(2026, 9, 3) in days
    assert date(2026, 9, 4) not in days


def test_la_journee_courante_est_toujours_reprise() -> None:
    engine = engine_covering(covered("SITE001", "2026-09-05", last="23:30"))

    assert TODAY in catch_up_days(engine, ["SITE001"], depth_days=1, today=TODAY)


def test_une_journee_commencee_en_retard_est_reprise() -> None:
    engine = engine_covering(covered("SITE001", "2026-09-04", first="14:00"))

    assert date(2026, 9, 4) in catch_up_days(
        engine, ["SITE001"], depth_days=2, today=TODAY
    )


def test_une_journee_interrompue_est_reprise() -> None:
    engine = engine_covering(covered("SITE001", "2026-09-04", last="09:00"))

    assert date(2026, 9, 4) in catch_up_days(
        engine, ["SITE001"], depth_days=2, today=TODAY
    )


def test_une_journee_venue_du_seul_rattrapage_est_tenue_pour_complete() -> None:
    engine = engine_covering(covered("SITE001", "2026-09-04", last="23:30"))

    assert date(2026, 9, 4) not in catch_up_days(
        engine, ["SITE001"], depth_days=2, today=TODAY
    )


def test_un_seul_site_decouvert_suffit_a_reprendre_la_journee() -> None:
    engine = engine_covering(covered("SITE001", "2026-09-04"))

    days = catch_up_days(engine, ["SITE001", "SITE002"], depth_days=2, today=TODAY)

    assert date(2026, 9, 4) in days


def test_la_profondeur_borne_la_fenetre_cherchee() -> None:
    days = catch_up_days(engine_covering(), ["SITE001"], depth_days=7, today=TODAY)

    assert days[0] == date(2026, 8, 30)
    assert len(days) == 7


def test_le_rattrapage_refuse_une_periode_donnee_en_plus() -> None:
    args = parse_args(["--catch-up", "--start", "2026-08-01"])

    with pytest.raises(ValueError, match="--start"):
        check_period_arguments(args)


def test_le_rattrapage_seul_est_accepte() -> None:
    check_period_arguments(parse_args(["--catch-up"]))


def test_une_periode_explicite_reste_acceptee() -> None:
    check_period_arguments(parse_args(["--start", "2026-08-01"]))


def test_une_journee_passee_est_rattrapee_en_entier() -> None:
    from datetime import UTC, date, datetime

    from collector.__main__ import day_window

    now = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    debut, fin = day_window(date(2026, 9, 5), now=now)
    assert debut == datetime(2026, 9, 5, tzinfo=UTC)
    assert fin == datetime(2026, 9, 6, tzinfo=UTC)


def test_la_journee_en_cours_s_arrete_a_maintenant() -> None:
    from datetime import UTC, date, datetime

    from collector.__main__ import day_window

    now = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    debut, fin = day_window(date(2026, 9, 7), now=now)
    assert debut == datetime(2026, 9, 7, tzinfo=UTC)
    assert fin == now


def test_une_journee_future_donne_une_fenetre_vide() -> None:
    from datetime import UTC, date, datetime

    from collector.__main__ import day_window

    now = datetime(2026, 9, 7, 13, 0, tzinfo=UTC)
    debut, fin = day_window(date(2026, 9, 8), now=now)
    assert fin <= debut
