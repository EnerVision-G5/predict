"""Lecture des horodatages de la source : ce qui est daté, et ce qui ne l'est pas.

Un seul comportement est vérifié ici, sous ses deux faces. Un horodatage qui
déclare son fuseau n'est jamais déplacé — c'est ce qui protège le rattrapage,
que `/readings` sert en UTC. Un horodatage nu est lu dans le fuseau prêté à la
source — c'est ce qui répare la collecte continue, que `/current` sert en heure
locale depuis le 8 septembre 2026.

Les deux formes arrivent de la même source et parfois dans le même lot : les
traiter uniformément casserait l'une pour réparer l'autre.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd
import pytest

from predict_common.timestamps import declares_offset, parse_timestamp, to_utc

PARIS = "Europe/Paris"


def test_a_dated_timestamp_is_taken_at_its_word() -> None:
    parsed = parse_timestamp("2026-09-08T08:53:13Z", PARIS)

    assert parsed == datetime(2026, 9, 8, 8, 53, 13, tzinfo=UTC)


def test_an_offset_is_honoured_rather_than_replaced() -> None:
    parsed = parse_timestamp("2026-09-08T10:53:13+02:00", PARIS)

    assert parsed == datetime(2026, 9, 8, 8, 53, 13, tzinfo=UTC)


def test_a_bare_timestamp_is_read_in_the_lent_timezone() -> None:
    parsed = parse_timestamp("2026-09-08T10:53:13.874698", PARIS)

    assert parsed == datetime(2026, 9, 8, 8, 53, 13, 874698, tzinfo=UTC)


def test_the_lent_timezone_defaults_to_moving_nothing() -> None:
    parsed = parse_timestamp("2026-09-08T10:53:13")

    assert parsed == datetime(2026, 9, 8, 10, 53, 13, tzinfo=UTC)


def test_the_shift_follows_the_season_rather_than_a_fixed_offset() -> None:
    summer = parse_timestamp("2026-09-08T10:00:00", PARIS)
    winter = parse_timestamp("2026-12-08T10:00:00", PARIS)

    assert summer == datetime(2026, 9, 8, 8, tzinfo=UTC)
    assert winter == datetime(2026, 12, 8, 9, tzinfo=UTC)


def test_a_datetime_is_read_on_the_same_terms_as_its_text() -> None:
    assert parse_timestamp(datetime(2026, 9, 8, 10, 53, 13), PARIS) == datetime(
        2026, 9, 8, 8, 53, 13, tzinfo=UTC
    )
    already_dated = datetime(2026, 9, 8, 8, 53, 13, tzinfo=UTC)
    assert parse_timestamp(already_dated, PARIS) == already_dated


def test_an_unreadable_timestamp_is_not_a_resumption_point() -> None:
    assert parse_timestamp("03/09/2026", PARIS) is None
    assert parse_timestamp(None, PARIS) is None


def test_a_dated_column_is_left_where_it_was() -> None:
    column = pd.Series(["2026-09-08T08:00:00Z", "2026-09-08T09:00:00Z"])

    stamps = to_utc(column, PARIS)

    assert list(stamps) == [
        pd.Timestamp("2026-09-08T08:00:00Z"),
        pd.Timestamp("2026-09-08T09:00:00Z"),
    ]


def test_a_bare_column_is_brought_back_to_utc() -> None:
    column = pd.Series(["2026-09-08T10:00:00", "2026-09-08T11:00:00"])

    stamps = to_utc(column, PARIS)

    assert list(stamps) == [
        pd.Timestamp("2026-09-08T08:00:00Z"),
        pd.Timestamp("2026-09-08T09:00:00Z"),
    ]


def test_each_timestamp_of_a_mixed_batch_is_read_on_its_own_terms() -> None:
    column = pd.Series(["2026-09-08T08:00:00Z", "2026-09-08T10:00:00"])

    stamps = to_utc(column, PARIS)

    assert list(stamps) == [
        pd.Timestamp("2026-09-08T08:00:00Z"),
        pd.Timestamp("2026-09-08T08:00:00Z"),
    ]


def test_an_hour_that_never_existed_is_not_placed() -> None:
    stamps = to_utc(pd.Series(["2026-03-29T02:30:00"]), PARIS)

    assert stamps.isna().all()


def test_an_hour_played_twice_is_not_arbitrated() -> None:
    stamps = to_utc(pd.Series(["2026-10-25T02:30:00"]), PARIS)

    assert stamps.isna().all()


def test_an_unreadable_value_becomes_an_unplaceable_measure() -> None:
    stamps = to_utc(pd.Series(["03/09/2026", None]), PARIS)

    assert stamps.isna().all()


def test_an_empty_batch_has_no_degenerate_case() -> None:
    stamps = to_utc(pd.Series([], dtype=object), PARIS)

    assert stamps.empty


def test_the_assumption_is_never_made_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        to_utc(pd.Series(["2026-09-08T10:00:00"]), PARIS)

    assert "sans fuseau" in caplog.text
    assert PARIS in caplog.text


def test_nothing_is_reported_when_the_source_dates_its_answers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        to_utc(pd.Series(["2026-09-08T08:00:00Z"]), PARIS)

    assert caplog.text == ""


def test_an_offset_is_recognised_in_each_of_its_written_forms() -> None:
    assert declares_offset("2026-09-08T08:00:00Z")
    assert declares_offset("2026-09-08T10:00:00+02:00")
    assert declares_offset("2026-09-08T10:00:00+0200")
    assert declares_offset("2026-09-08T03:00:00-05:00")
    assert not declares_offset("2026-09-08T10:00:00")
    assert not declares_offset("2026-09-08")
    assert not declares_offset(None)
