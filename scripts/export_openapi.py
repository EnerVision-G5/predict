# **********************************************************************
# * Nom     : export_openapi.py                                        *
# * Type    : Script                                                   *
# * Sujet   : Export de la spécification OpenAPI du service            *
# *   d'inférence                                                      *
# * Service : outillage                                                *
# **********************************************************************

from __future__ import annotations

import json
import sys
from pathlib import Path

from serving.api import CONTRACT_VERSION, app


def build_spec() -> dict:
    """Méthode : build_spec
    Description : Construit la spécification telle que le service la sert.
    """
    spec = app.openapi()
    spec["info"]["version"] = CONTRACT_VERSION
    return spec


def main(argv: list[str]) -> int:
    """Méthode : main
    Description : Point d'entrée : écrit la spécification sur la sortie
      standard.
    """
    payload = json.dumps(build_spec(), indent=2, sort_keys=True, ensure_ascii=False)
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
