"""Porte de sécurité : ce qui bloque une fusion, ce qui se contente d'alerter.

Le script analysé ici ne calcule rien — c'est Grype qui trouve les failles. Il
décide, et c'est justement la décision qui mérite des tests : un scanner qui se
trompe se corrige en montant sa base, une porte qui se trompe laisse passer ce
qu'elle devait arrêter, ou arrête tout et finit désactivée.

Le module vit dans `.github/scripts`, qui n'est pas un paquet importable — le
point du nom l'interdit. Il est donc chargé par son chemin, ce qui est aussi la
façon dont le runner l'exécute.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / ".github" / "scripts" / "grype_verdict.py"


def load_module():
    """Charge le script de verdict par son chemin, comme le fait le runner."""
    spec = importlib.util.spec_from_file_location("grype_verdict", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verdict = load_module()


def match(
    severity: str,
    name: str = "starlette",
    version: str = "0.49.3",
    fixed: list[str] | None = None,
    identifier: str = "GHSA-0000",
) -> dict[str, Any]:
    """Compose une correspondance Grype, réduite à ce que le script lit."""
    return {
        "artifact": {"name": name, "version": version},
        "vulnerability": {
            "severity": severity,
            "id": identifier,
            "fix": {"versions": fixed or []},
        },
    }


def report(tmp_path: Path, matches: list[dict[str, Any]]) -> Path:
    """Écrit un rapport Grype minimal sur disque."""
    path = tmp_path / "grype.json"
    path.write_text(json.dumps({"matches": matches}), encoding="utf-8")
    return path


class TestVerdict:
    """Le code de sortie est le verdict : un ordonnanceur ne lit pas un journal."""

    def test_a_clean_lock_passes(self, tmp_path: Path, capsys) -> None:
        assert verdict.main(["", str(report(tmp_path, []))]) == 0

    def test_a_critical_blocks_the_merge(self, tmp_path: Path, capsys) -> None:
        path = report(tmp_path, [match("Critical")])
        assert verdict.main(["", str(path)]) == 1

    def test_a_high_does_not_block(self, tmp_path: Path, capsys) -> None:
        # Le verrou porte des failles hautes sans correctif publié : les rendre
        # bloquantes laisserait la CI rouge en permanence, sans qu'aucune PR ne
        # puisse y remédier — et une porte toujours rouge est contournée.
        path = report(tmp_path, [match("High"), match("Medium"), match("Low")])
        assert verdict.main(["", str(path)]) == 0

    def test_a_critical_blocks_even_among_lesser_findings(
        self, tmp_path: Path, capsys
    ) -> None:
        path = report(tmp_path, [match("Low"), match("Critical"), match("High")])
        assert verdict.main(["", str(path)]) == 1

    def test_a_missing_report_is_not_a_silent_pass(self, tmp_path: Path) -> None:
        # Un scan qui n'a rien écrit n'est pas un scan qui n'a rien trouvé.
        with pytest.raises(SystemExit):
            verdict.main(["", str(tmp_path / "absent.json")])


class TestSummary:
    """Le résumé dit ce qu'il faut corriger, et par quelle version."""

    def test_an_empty_report_says_so(self) -> None:
        assert "Aucune vulnérabilité connue." in "\n".join(verdict.summary_lines([]))

    def test_the_table_lists_critical_and_high_only(self) -> None:
        lines = verdict.summary_lines(
            [
                match("Critical", name="paquet-critique"),
                match("High", name="paquet-haut"),
                match("Medium", name="paquet-moyen"),
            ]
        )
        text = "\n".join(lines)
        assert "paquet-critique" in text
        assert "paquet-haut" in text
        assert "paquet-moyen" not in text

    def test_the_most_severe_comes_first(self) -> None:
        lines = verdict.summary_lines(
            [match("High", name="haut"), match("Critical", name="critique")]
        )
        text = "\n".join(lines)
        assert text.index("critique") < text.index("haut")

    def test_the_tally_counts_every_severity(self) -> None:
        lines = verdict.summary_lines([match("Low"), match("Low"), match("High")])
        assert "**Low** 2" in "\n".join(lines)


class TestFixedIn:
    """Une faille sans correctif ne se traite pas en montant une version."""

    def test_a_published_fix_is_named(self) -> None:
        assert verdict.fixed_in({"fix": {"versions": ["50.0.0"]}}) == "50.0.0"

    def test_several_fixes_are_all_named(self) -> None:
        assert verdict.fixed_in({"fix": {"versions": ["1.1.0", "1.3.1"]}}) == (
            "1.1.0, 1.3.1"
        )

    def test_no_fix_is_a_dash_not_an_empty_string(self) -> None:
        # Une chaîne vide dans le tableau se lirait comme une colonne oubliée.
        assert verdict.fixed_in({"fix": {"versions": []}}) == "—"

    def test_an_absent_fix_key_is_tolerated(self) -> None:
        # Grype omet la clé quand aucun correctif n'est connu de sa base.
        assert verdict.fixed_in({}) == "—"


class TestPublish:
    """Le résumé va où GitHub l'attend, sans quoi il va sur la sortie standard."""

    def test_the_summary_is_appended_to_the_github_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        destination = tmp_path / "summary.md"
        destination.write_text("déjà là\n", encoding="utf-8")
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
        verdict.publish(["ajouté"])
        content = destination.read_text(encoding="utf-8")
        assert "déjà là" in content
        assert "ajouté" in content

    def test_without_the_variable_it_prints(self, monkeypatch, capsys) -> None:
        # C'est ce qui rend le script rejouable sur un poste, sur un rapport
        # déjà produit, sans reproduire l'environnement du runner.
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        verdict.publish(["sur la sortie standard"])
        assert "sur la sortie standard" in capsys.readouterr().out


class TestAnnotate:
    """Une annotation par faille rapportée, visible sur la PR."""

    def test_high_and_critical_are_annotated(self, capsys) -> None:
        verdict.annotate([match("Critical"), match("High")])
        assert capsys.readouterr().out.count("::warning") == 2

    def test_lesser_findings_are_not(self, capsys) -> None:
        # Annoter les moyennes noierait les hautes, que personne ne lirait plus.
        verdict.annotate([match("Medium"), match("Low")])
        assert capsys.readouterr().out == ""

    def test_the_annotation_names_the_fix(self, capsys) -> None:
        verdict.annotate([match("High", name="pyarrow", fixed=["23.0.1"])])
        assert "23.0.1" in capsys.readouterr().out


class TestLoadMatches:
    """La lecture du rapport."""

    def test_matches_are_returned(self, tmp_path: Path) -> None:
        path = report(tmp_path, [match("High"), match("Low")])
        assert len(verdict.load_matches(path)) == 2

    def test_an_absent_report_stops_the_job(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="introuvable"):
            verdict.load_matches(tmp_path / "absent.json")
