# **********************************************************************
# * Nom     : grype_verdict.py                                         *
# * Type    : Script                                                   *
# * Sujet   : Verdict de l'analyse de vulnérabilités : résumé,         *
# *   annotations, code de sortie                                      *
# * Service : intégration continue                                     *
# **********************************************************************

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path
from typing import Any

# Gravités connues de Grype, de la plus grave à la moindre.
SEVERITIES = ("Critical", "High", "Medium", "Low", "Negligible", "Unknown")

# Gravités qui font échouer la CI.
BLOCKING = ("Critical",)
# Gravités annotées dans la revue, sans forcément bloquer.
REPORTED = ("Critical", "High")


def load_matches(report: Path) -> list[dict[str, Any]]:
    """Méthode : load_matches
    Description : Lit le rapport Grype et rend ses correspondances.
    """
    if not report.is_file():
        raise SystemExit(f"Rapport Grype introuvable : {report}.")
    return json.loads(report.read_text(encoding="utf-8"))["matches"]


def fixed_in(vulnerability: dict[str, Any]) -> str:
    """Méthode : fixed_in
    Description : Versions qui corrigent une vulnérabilité, tiret s'il n'y en a
      pas.
    """
    versions = (vulnerability.get("fix") or {}).get("versions") or []
    return ", ".join(versions) if versions else "—"


def summary_lines(matches: list[dict[str, Any]]) -> list[str]:
    """Méthode : summary_lines
    Description : Compose le résumé affiché dans le récapitulatif du job.
    """
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
    """Méthode : publish
    Description : Écrit le résumé dans le récapitulatif GitHub, ou sur la
      sortie standard.
    """
    text = "\n".join(lines) + "\n"
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination is None:
        sys.stdout.write(text)
        return
    with open(destination, "a", encoding="utf-8") as out:
        out.write(text)


def annotate(matches: list[dict[str, Any]]) -> None:
    """Méthode : annotate
    Description : Annote la revue pour chaque vulnérabilité assez grave.
    """
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
    """Méthode : main
    Description : Point d'entrée : publie le verdict et échoue sur une
      vulnérabilité bloquante.
    """
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
