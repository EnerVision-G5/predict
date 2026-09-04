"""Socle partagé par le collecteur, l'ETL, l'entraînement et le service.

Six modules, et aucune logique métier. `config` charge la configuration,
`paths` construit les chemins de partition, `schemas` déclare ce que chaque
couche d'artefact doit contenir, `io` lit et écrit le parquet, `db` déclare
les tables de la couche brute, `source` parle à l'API Mock.

Rien n'est importé ici : un service qui a besoin de la source ne doit pas se
retrouver à charger SQLAlchemy pour autant. C'est ce qui permet au service
d'inférence de lire la source sans jamais voir la base.

Tout le reste appartient au service qui l'utilise. La règle est vérifiable :
si une fonction ajoutée ici n'a qu'un appelant, elle est au mauvais endroit.
"""
