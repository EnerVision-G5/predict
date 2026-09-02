"""Socle partagé par le collecteur, l'ETL, l'entraînement et le service.

Quatre modules, et aucune logique métier. `config` charge la configuration,
`paths` construit les chemins de partition, `schemas` déclare ce que chaque
couche d'artefact doit contenir, `io` lit et écrit le parquet.

Tout le reste appartient au service qui l'utilise. La règle est vérifiable :
si une fonction ajoutée ici n'a qu'un appelant, elle est au mauvais endroit.
"""
