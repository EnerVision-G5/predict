from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SERVICES = ("collector", "etl", "training", "serving")
SHARED = "predict_common"

ROOT = Path(__file__).resolve().parent.parent


def service_sources(name: str) -> list[Path]:
    return sorted((ROOT / "services" / name / "src").rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("service", SERVICES)
def test_a_service_never_imports_another_service(service: str) -> None:
    forbidden = set(SERVICES) - {service}
    offenders = {
        path.relative_to(ROOT).as_posix(): sorted(imported_roots(path) & forbidden)
        for path in service_sources(service)
    }
    breaches = {path: names for path, names in offenders.items() if names}
    assert not breaches, (
        f"{service} importe le code d'un autre service : {breaches}."
        " Les frontières passent par les artefacts — un chemin et un schéma —"
        " jamais par l'espace de noms Python."
    )


@pytest.mark.parametrize("service", SERVICES)
def test_a_service_has_at_least_one_entry_point(service: str) -> None:
    package = ROOT / "services" / service / "src" / service
    entry_points = [package / "__main__.py", package / "api.py"]
    assert any(path.is_file() for path in entry_points), (
        f"{service} n'a ni __main__.py ni api.py."
    )


@pytest.mark.parametrize("service", SERVICES)
def test_a_service_declares_no_other_service_as_a_dependency(service: str) -> None:
    manifest = (ROOT / "services" / service / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    dependencies = manifest.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = {
        line.strip().strip('",').split("=")[0].split("[")[0].strip().lower()
        for line in dependencies.splitlines()
        if line.strip()
    }
    assert not declared & (set(SERVICES) - {service})


@pytest.mark.parametrize("service", SERVICES)
def test_a_service_has_a_dockerfile(service: str) -> None:
    assert (ROOT / "services" / service / "Dockerfile").is_file()


def test_the_shared_library_knows_no_service() -> None:
    sources = sorted((ROOT / "libs" / SHARED / "src").rglob("*.py"))
    breaches = {
        path.relative_to(ROOT).as_posix(): sorted(imported_roots(path) & set(SERVICES))
        for path in sources
    }
    assert not any(breaches.values()), breaches


def test_the_shared_library_carries_only_its_declared_modules() -> None:
    modules = {
        path.stem
        for path in (ROOT / "libs" / SHARED / "src" / SHARED).glob("*.py")
        if path.stem != "__init__"
    }
    assert modules == {
        "config",
        "paths",
        "schemas",
        "io",
        "db",
        "source",
        "timestamps",
    }


def test_every_partition_path_is_built_by_the_shared_module() -> None:
    offenders: list[str] = []
    for service in SERVICES:
        for path in service_sources(service):
            text = path.read_text(encoding="utf-8")
            if 'f"dt=' in text or '"dt=" +' in text:
                offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, (
        f"Chemin de partition composé à la main : {offenders}."
        " Utiliser predict_common.paths."
    )


def test_the_measure_table_is_declared_once() -> None:
    declared_table = re.compile(r"Table\(\s*[\"'](mesure|mesure_exclu)[\"']")
    declaring = [
        path.relative_to(ROOT).as_posix()
        for service in SERVICES
        for path in service_sources(service)
        if declared_table.search(path.read_text(encoding="utf-8"))
    ]
    assert not declaring, (
        f"La table `mesure` est redéclarée dans {declaring}."
        " Sa définition appartient à predict_common.db."
    )


def test_the_serving_service_never_reaches_the_database() -> None:
    reaching = {
        service
        for service in SERVICES
        for path in service_sources(service)
        if {"sqlalchemy", "psycopg"} & imported_roots(path)
    }
    assert reaching == {"collector", "etl", "training"}


def test_the_training_service_reaches_the_database_only_to_promote() -> None:
    touching = {
        path.name
        for path in service_sources("training")
        if {"sqlalchemy", "psycopg"} & imported_roots(path)
    }
    assert touching == {"registry.py", "__main__.py"}
