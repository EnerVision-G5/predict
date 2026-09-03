"""Fixtures du service d'entraînement.

Aucun test ne joint PostgreSQL. Le moteur factice mémorise les instructions au
lieu de les exécuter : ce qui compte n'est pas que la base les accepte, c'est
que la promotion produise les bonnes — celle qui inscrit la version en service
et celle qui éteint les précédentes, dans une seule transaction.

Le moteur est redéclaré ici plutôt qu'importé du service voisin. L'importer
ferait exactement l'import interdit que `tests/test_architecture.py` vérifie,
et la duplication de quinze lignes est le prix de la frontière.
"""

from __future__ import annotations

from typing import Any


class FakeConnection:
    """Connexion factice qui mémorise les instructions exécutées."""

    def __init__(self, executed: list[Any]) -> None:
        self._executed = executed

    def execute(self, statement: Any) -> None:
        self._executed.append(statement)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    """Moteur factice : begin() rend une transaction sans base derrière."""

    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.transactions = 0

    def begin(self) -> FakeConnection:
        self.transactions += 1
        return FakeConnection(self.executed)

    def dispose(self) -> None:
        """Rien à rendre : il n'y a pas de connexion derrière."""
