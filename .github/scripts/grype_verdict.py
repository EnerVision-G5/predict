"""Verdict de l'analyse Grype : ce qui bloque, ce qui alerte.

Un scanner rend une liste ; une porte qualité rend une décision. Ce script
fait la seconde à partir de la première, et la sépare en deux niveaux qui
n'ont pas le même effet.

**Critique bloque.** Une faille de ce rang sur une dépendance embarquée dans
les images n'a pas à traverser une revue : elle arrête la fusion.

**Haut alerte.** Le verrou porte aujourd'hui des failles hautes dont l'une n'a
aucun correctif publié, et une autre demanderait de monter `fastapi`, épinglé
à la version exacte parce que c'est lui qui produit la spécification OpenAPI
gelée. Les traiter comme bloquantes rendrait la CI rouge en permanence, sans
qu'aucune PR ne puisse y remédier — et une porte toujours rouge est une porte
que l'équipe apprend à contourner. Elles sont donc rendues visibles : un
tableau dans le résumé du job, une annotation par faille sur la PR.

Le script vit hors du YAML pour qu'il soit exécutable à la main sur un rapport
déjà produit, et lisible par ruff comme le reste du dépôt. Une logique de
décision enfouie dans un `run: |` n'est ni l'un ni l'autre.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

SEVERITIES = ("Critical", "High", "Medium", "Low", "Negligible", "Unknown")

BLOCKING = ("Critical",)
REPORTED = ("Critical", "High")


def load_matches(report: Path) -> list[dict[str, Any]]:
    """Lit les correspondances d'un rapport Grype au format JSON."""
    if not report.is_file():
        raise SystemExit(f"Rapport Grype introuvable : {report}.")
    return json.loads(report.read_text(encoding="utf-8"))["matches"]


def fixed_in(vulnerability: dict[str, Any]) -> str:
    """Retourne les versions qui corrigent la faille, ou un tiret.

    Le tiret n'est pas un détail d'affichage : une faille sans correctif ne se
    traite pas en montant une version, et confondre les deux cas ferait
    chercher une mise à jour qui n'existe pas.
    """
    versions = (vulnerability.get("fix") or {}).get("versions") or []
    return ", ".join(versions) if versions else "—"


def summary_lines(matches: list[dict[str, Any]]) -> list[str]:
    """Compose le résumé Markdown affiché sur la page du job."""
    counts = collections.Counter(
        match["vulnerability"]["severity"] for match in matches
    )
    tally = " · ".join(
        f"**{name}** {counts[name]}" for name in SEVERITIES if counts[name]
    )
    lines = [
        "## Grype — analyse du verrou de dépendances",
        "",
        tally or "Aucune vulnérabilité connue.",
    ]

    shown = [m for m in matches if m["vulnerability"]["severity"] in REPORTED]
    if not shown:
        return lines

    lines += [
        "",
        "| Gravité | Paquet | Version | Corrigé en | Avis |",
        "|---|---|---|---|---|",
    ]
    for match in sorted(
        shown, key=lambda m: SEVERITIES.index(m["vulnerability"]["severity"])
    ):
        artifact, vulnerability = match["artifact"], match["vulnerability"]
        lines.append(
            f"| {vulnerability['severity']} | {artifact['name']}"
            f" | {artifact['version']} | {fixed_in(vulnerability)}"
            f" | {vulnerability['id']} |"
        )
    return lines


def publish(lines: list[str]) -> None:
    """Écrit le résumé là où GitHub l'attend, ou sur la sortie standard.

    Hors CI, `GITHUB_STEP_SUMMARY` n'existe pas. Le script reste alors
    utilisable : c'est ce qui permet de rejouer un rapport sur un poste sans
    reproduire l'environnement du runner.
    """
    text = "\n".join(lines) + "\n"
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination is None:
        sys.stdout.write(text)
        return
    with open(destination, "a", encoding="utf-8") as out:
        out.write(text)


def annotate(matches: list[dict[str, Any]]) -> None:
    """Pose une annotation par faille rapportée, visible sur la PR."""
    for match in matches:
        if match["vulnerability"]["severity"] not in REPORTED:
            continue
        artifact, vulnerability = match["artifact"], match["vulnerability"]
        print(
            f"::warning title=Grype {vulnerability['severity']}::"
            f"{artifact['name']} {artifact['version']} — {vulnerability['id']}"
            f" (corrigé en {fixed_in(vulnerability)})"
        )


def main(argv: list[str]) -> int:
    """Publie le résumé, annote la PR, et rend le verdict par le code de sortie."""
    report = Path(argv[1] if len(argv) > 1 else "grype.json")
    matches = load_matches(report)
    publish(summary_lines(matches))
    annotate(matches)

    blocking = [m for m in matches if m["vulnerability"]["severity"] in BLOCKING]
    if blocking:
        names = ", ".join(
            sorted(
                f"{m['artifact']['name']} {m['vulnerability']['id']}"
                for m in blocking
            )
        )
        print(f"::error title=Grype::vulnérabilité(s) critique(s) : {names}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
