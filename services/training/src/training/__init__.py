"""Entraînement du modèle de prévision, suivi par MLflow.

`dataset` lit les partitions de variables et les découpe dans l'ordre du
temps, `model` ajuste le régresseur et mesure sa qualité, `tracking` écrit le
run et enregistre le modèle. Le service d'inférence ne rejoue jamais ce code :
il charge un modèle déjà enregistré, désigné par un alias.

L'entraînement ne lit ni la base ni la source. Son amont est un ensemble de
partitions immuables, ce qui rend un run rejouable à l'identique.
"""
