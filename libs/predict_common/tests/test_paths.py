"""Construction des chemins de partition.

Ce module est la seule chose que les quatre services partagent pour se
comprendre : deux services qui ne construiraient pas le même chemin pour la
même journée ne se parleraient plus. Les tests fixent donc la forme exacte des
chemins, et pas seulement le fait qu'ils soient produits.
"""

from __future__ import annotations

from datetime import date

import pytest

from predict_common.paths import (
    PathError,
    date_range,
    features_partition,
    format_date,
    join,
    lookback_range,
    normalize,
    parse_date,
    part_file,
    partition_segment,
    temporary_sibling,
)

DAY = date(2026, 9, 2)


def test_parse_date_reads_the_command_line_format() -> None:
    assert parse_date("2026-09-02") == DAY


def test_parse_date_refuses_an_approximate_date() -> None:
    with pytest.raises(PathError):
        parse_date("02/09/2026")


def test_format_date_round_trips() -> None:
    assert parse_date(format_date(DAY)) == DAY


def test_partition_segment_uses_hive_partitioning() -> None:
    # `dt=` n'est pas décoratif : c'est ce que pyarrow, DuckDB et Spark savent
    # élaguer sans lire les fichiers.
    assert partition_segment(DAY) == "dt=2026-09-02"


def test_features_partition_places_the_version_before_the_day() -> None:
    # La version est dans le chemin : c'est ce qui permet à v1 et v2 de
    # coexister sans se recouvrir.
    assert features_partition("data", "v1", DAY) == "data/features/v1/dt=2026-09-02"


def test_partitions_keep_an_s3_uri_intact() -> None:
    assert features_partition("s3://enervision/lac", "v1", DAY) == (
        "s3://enervision/lac/features/v1/dt=2026-09-02"
    )


def test_partitions_normalize_a_windows_root() -> None:
    # Un chemin système avec des antislashs donnerait deux formes du même
    # chemin selon l'hôte, et un stockage objet n'en lirait aucune.
    built = features_partition("D:\\lac", "v1", DAY)
    assert built == "D:/lac/features/v1/dt=2026-09-02"


def test_a_version_with_a_slash_is_refused() -> None:
    # Une barre oblique déplacerait la partition d'un niveau sans rien dire.
    with pytest.raises(PathError):
        features_partition("data", "v1/essai", DAY)


def test_a_version_climbing_out_of_the_root_is_refused() -> None:
    with pytest.raises(PathError):
        features_partition("data", "..", DAY)


def test_date_range_covers_both_bounds() -> None:
    days = date_range(date(2026, 9, 1), date(2026, 9, 3))
    assert days == [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]


def test_date_range_refuses_an_inverted_window() -> None:
    with pytest.raises(PathError):
        date_range(date(2026, 9, 3), date(2026, 9, 1))


def test_lookback_range_ends_on_the_requested_day() -> None:
    days = lookback_range(DAY, 3)
    assert days == [date(2026, 8, 31), date(2026, 9, 1), DAY]


def test_lookback_range_refuses_an_empty_window() -> None:
    with pytest.raises(PathError):
        lookback_range(DAY, 0)


def test_part_file_is_unique_between_two_writes() -> None:
    # Deux écritures concurrentes de la même partition ne doivent pas se
    # recouvrir : des noms égaux feraient perdre la première.
    names = {part_file() for _ in range(100)}
    assert len(names) == 100


def test_temporary_sibling_stays_outside_the_partition() -> None:
    # Un répertoire de travail à l'intérieur de la partition serait vu d'un
    # lecteur qui la liste pendant l'écriture.
    staging = temporary_sibling("data/features/v1/dt=2026-09-02")
    assert staging.startswith("data/features/v1/_tmp-dt=2026-09-02-")


def test_join_never_doubles_a_separator() -> None:
    assert join("data/", "/raw/", "dt=2026-09-02") == "data/raw/dt=2026-09-02"


def test_join_preserves_a_uri_scheme() -> None:
    assert join("s3://bucket", "raw") == "s3://bucket/raw"


def test_join_refuses_an_empty_path() -> None:
    with pytest.raises(PathError):
        join()


def test_normalize_turns_backslashes_into_separators() -> None:
    assert normalize("a\\b\\c") == "a/b/c"
