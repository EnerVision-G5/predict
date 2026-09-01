"""Exporte la spécification OpenAPI du service d'inférence.

Usage :
    python scripts/export_openapi.py                 # écrit sur stdout
    python scripts/export_openapi.py chemin.json     # écrit dans le fichier

Le JSON est produit avec les clés triées : c'est ce qui rend la comparaison
avec le contrat gelé déterministe, donc utilisable par le job CI de dérive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet l'exécution directe du script depuis la racine du repo : les
# modules vivent sous src/, qui n'est pas installé comme paquet.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference.app import CONTRACT_VERSION, app  # noqa: E402


def build_spec() -> dict:
    """Retourne la spécification OpenAPI avec la version de contrat forcée."""
    spec = app.openapi()
    spec["info"]["version"] = CONTRACT_VERSION
    return spec


def main(argv: list[str]) -> int:
    payload = json.dumps(build_spec(), indent=2, sort_keys=True, ensure_ascii=False)
    # Fins de ligne LF forcées : le contrat gelé est comparé par la CI Linux,
    # une génération Windows en CRLF ferait échouer le diff sans dérive réelle.
    encoded = (payload + "\n").encode("utf-8")
    if len(argv) > 1:
        destination = Path(argv[1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
    else:
        sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
