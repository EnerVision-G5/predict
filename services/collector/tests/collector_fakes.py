"""Doublures de base du service collector.

Ces classes vivaient dans `conftest.py`, et les modules de test les en
importaient par `from conftest import FakeEngine`.

C'était une collision qui attendait son heure. Les répertoires de tests ne
sont pas des paquets : pytest les ajoute à `sys.path` et importe leurs
modules par leur nom de fichier. Trois services déclarent un `conftest.py`,
donc `sys.modules["conftest"]` est celui qui a été chargé EN PREMIER — et les
trois `FakeEngine` ne sont pas interchangeables.

Chacun passait donc seul, et lancer la suite entière en donnait neuf en
échec, sur une erreur qui ne nommait pas la cause :
    TypeError: FakeEngine.__init__() got an unexpected keyword argument 'rows'

Un nom de module unique par service supprime la collision à sa racine. Les
fixtures, elles, restent dans `conftest.py` : pytest les résout par
répertoire, et n'a jamais confondu celles-là.
"""

from __future__ import annotations

from typing import Any


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées.

    Elle rend aussi des lignes : le collecteur lit `capteur_etat` avant de
    l'écraser, pour dater les débuts et les fins de panne. Les mêmes lignes
    sont rendues à chaque appel — aucun test n'a besoin de plus, et un
    séquenceur de résultats rendrait ces fixtures illisibles.
    """

    def __init__(self, executed: list[Any], rows: list[Any]) -> None:
        self._executed = executed
        self._rows = rows

    def execute(self, statement: Any) -> list[Any]:
        self._executed.append(statement)
        return list(self._rows)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self, rows: list[Any] | None = None) -> None:
        self.executed: list[Any] = []
        self.rows: list[Any] = list(rows or ())

    def begin(self) -> FakeConnection:
        return FakeConnection(self.executed, self.rows)

    def dispose(self) -> None:
        """Rien à rendre : il n'y a pas de connexion derrière."""
