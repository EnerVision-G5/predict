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


