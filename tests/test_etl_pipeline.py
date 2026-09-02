"""Orchestration du run d'ingestion."""

from datetime import UTC, datetime

import pytest

from etl.pipeline import (
    DEFAULT_WINDOW_HOURS,
    RunReport,
    parse_args,
    resolve_sites,
    window_from_hours,
)

NOW = datetime(2026, 1, 16, 12, tzinfo=UTC)


class FakeSession:
    """Session factice rendant un référentiel de sites figé."""

    def __init__(self, sites):
        self._sites = sites

    def get(self, url, params=None, timeout=None):
        payload = self._sites

        class Response:
            status_code = 200
            url = "http://mock.invalid/api/v1/sites"
            request = type("Request", (), {"method": "GET"})()

            def json(self):
                return payload

        return Response()


def test_window_from_hours_ends_on_the_reference_instant() -> None:
    start_time, end_time = window_from_hours(6, now=NOW)
    assert end_time == NOW
    assert (end_time - start_time).total_seconds() == 6 * 3600


def test_window_from_hours_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError):
        window_from_hours(0, now=NOW)


def test_resolve_sites_prefers_the_requested_list(config) -> None:
    session = FakeSession([{"site_id": "SITE009"}])
    assert resolve_sites(config, ["SITE001"], session=session) == ["SITE001"]


def test_resolve_sites_falls_back_on_the_referential(config) -> None:
    session = FakeSession([{"site_id": "SITE001"}, {"site_id": "SITE002"}])
    assert resolve_sites(config, None, session=session) == ["SITE001", "SITE002"]


def test_parse_args_defaults_to_the_full_referential() -> None:
    args = parse_args([])
    assert args.hours == DEFAULT_WINDOW_HOURS
    assert args.sites is None


def test_parse_args_accepts_repeated_sites() -> None:
    args = parse_args(["--hours", "6", "--site", "SITE001", "--site", "SITE002"])
    assert args.hours == 6
    assert args.sites == ["SITE001", "SITE002"]


def test_run_report_totals_every_site() -> None:
    report = RunReport(rows_per_site={"SITE001": 10, "SITE002": 5})
    assert report.total_rows == 15


def test_run_report_counts_exclusions_apart_from_rows() -> None:
    """Une mesure écartée est chargée : la retrancher fausserait le suivi."""
    report = RunReport(
        rows_per_site={"SITE001": 10, "SITE002": 5},
        excluded_per_site={"SITE001": 2},
    )
    assert report.total_rows == 15
    assert report.total_excluded == 2
