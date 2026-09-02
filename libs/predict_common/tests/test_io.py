"""Écriture et lecture des partitions parquet.

La propriété testée ici porte toute l'exploitation de la chaîne : l'écriture
d'une journée remplace la journée. C'est ce qui rend un rejeu après incident
sans effet de bord, et le rejeu est le mode d'exploitation normal. Un mode qui
ajouterait rendrait le résultat dépendant du nombre de fois qu'on a lancé la
commande, ce qu'aucun compte en aval ne saurait rattraper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow
import pytest

from predict_common import io
from predict_common.paths import join

SCHEMA = pyarrow.schema(
    [
        ("site_id", pyarrow.string()),
        ("valeur", pyarrow.float64()),
    ]
)


def frame(*values: float, site_id: str = "SITE001") -> pd.DataFrame:
    """Construit un lot minimal au schéma des tests."""
    return pd.DataFrame({"site_id": [site_id] * len(values), "valeur": list(values)})


def partition(tmp_path: Path) -> str:
    """Retourne le chemin d'une partition de test."""
    return join(str(tmp_path), "features/v1/dt=2026-09-02")


def test_write_frame_creates_a_readable_partition(tmp_path: Path) -> None:
    target = partition(tmp_path)
    io.write_frame(frame(1.0, 2.0), target, schema=SCHEMA)
    assert list(io.read_frames([target])["valeur"]) == [1.0, 2.0]


def test_write_frame_replaces_instead_of_appending(tmp_path: Path) -> None:
    # Le rejeu est le mode d'exploitation normal : relancer --date reproduit
    # la journée, il ne la double pas.
    target = partition(tmp_path)
    io.write_frame(frame(1.0, 2.0), target, schema=SCHEMA)
    io.write_frame(frame(3.0), target, schema=SCHEMA)
    assert list(io.read_frames([target])["valeur"]) == [3.0]


def test_write_frame_leaves_no_working_directory_behind(tmp_path: Path) -> None:
    target = partition(tmp_path)
    io.write_frame(frame(1.0), target, schema=SCHEMA)
    leftovers = [entry.name for entry in Path(target).parent.iterdir()]
    assert leftovers == ["dt=2026-09-02"]


def test_write_frame_accepts_an_empty_partition(tmp_path: Path) -> None:
    # Une journée sans mesure exploitable est un fait d'exploitation, pas une
    # panne : la partition existe et elle est vide.
    target = partition(tmp_path)
    io.write_frame(frame(), target, schema=SCHEMA)
    assert io.read_frames([target]).empty


def test_read_frames_concatenates_several_partitions(tmp_path: Path) -> None:
    first = join(str(tmp_path), "features/v1/dt=2026-09-01")
    second = join(str(tmp_path), "features/v1/dt=2026-09-02")
    io.write_frame(frame(1.0), first, schema=SCHEMA)
    io.write_frame(frame(2.0), second, schema=SCHEMA)
    assert sorted(io.read_frames([first, second])["valeur"]) == [1.0, 2.0]


def test_read_frames_skips_an_absent_partition(tmp_path: Path) -> None:
    # Au démarrage de la chaîne, la semaine qui précède le premier jour
    # collecté n'existe pas : ce n'est pas une panne.
    present = join(str(tmp_path), "features/v1/dt=2026-09-02")
    io.write_frame(frame(1.0), present, schema=SCHEMA)
    absent = join(str(tmp_path), "features/v1/dt=2026-08-01")
    assert len(io.read_frames([absent, present])) == 1


def test_read_frames_can_refuse_an_absent_partition(tmp_path: Path) -> None:
    absent = join(str(tmp_path), "features/v1/dt=2026-09-02")
    with pytest.raises(io.StorageError):
        io.read_frames([absent], missing_ok=False)


def test_read_frames_returns_an_empty_frame_when_nothing_exists(
    tmp_path: Path,
) -> None:
    absent = join(str(tmp_path), "features/v1/dt=2026-09-02")
    assert io.read_frames([absent], columns=["valeur"]).empty


def test_read_frames_projects_the_requested_columns(tmp_path: Path) -> None:
    target = partition(tmp_path)
    io.write_frame(frame(1.0), target, schema=SCHEMA)
    assert list(io.read_frames([target], columns=["valeur"]).columns) == ["valeur"]


def test_the_imposed_schema_survives_an_all_null_column(tmp_path: Path) -> None:
    # Sans schéma imposé, une colonne entièrement nulle s'écrirait en type
    # `null`, et la lecture conjointe des deux journées échouerait.
    empty_day = join(str(tmp_path), "features/v1/dt=2026-09-01")
    full_day = join(str(tmp_path), "features/v1/dt=2026-09-02")
    io.write_frame(
        pd.DataFrame({"site_id": ["SITE001"], "valeur": [None]}),
        empty_day,
        schema=SCHEMA,
    )
    io.write_frame(frame(2.0), full_day, schema=SCHEMA)
    read = io.read_frames([empty_day, full_day])
    assert str(read["valeur"].dtype) == "float64"
    assert len(read) == 2


def test_exists_ignores_an_empty_directory(tmp_path: Path) -> None:
    target = Path(partition(tmp_path))
    target.mkdir(parents=True)
    assert not io.exists(str(target))


def test_metadata_travels_with_the_partition(tmp_path: Path) -> None:
    import pyarrow.parquet

    target = partition(tmp_path)
    io.write_frame(
        frame(1.0), target, schema=SCHEMA, metadata={"feature_version": "v1"}
    )
    written = next(Path(target).glob("*.parquet"))
    stored = pyarrow.parquet.read_schema(written).metadata
    assert stored[b"feature_version"] == b"v1"


def test_resolve_makes_a_relative_path_absolute() -> None:
    # Le répertoire courant d'un conteneur n'est pas celui d'un poste.
    _, path = io.resolve("data/features")
    assert Path(path).is_absolute()


def test_resolve_refuses_an_unknown_scheme() -> None:
    with pytest.raises(io.StorageError):
        io.resolve("carrier-pigeon://bucket/features")
