"""Collecte des mesures depuis l'API Mock IoT vers la couche brute.

La couche brute est la table `mesure` de TimescaleDB. Deux points d'entrée y
mènent : `python -m collector` rattrape une journée passée par pagination,
`python -m collector.poller` interroge la mesure courante à cadence fixe. Les
deux écrivent le même schéma avec la même insertion idempotente.

Le service ne connaît aucun de ses consommateurs et n'importe le code d'aucun
autre service. Son contrat est une table et un schéma, déclarés dans
`predict_common.db` et `predict_common.schemas.MEASURE_SCHEMA`.

Il n'écrit jamais les colonnes déduites — `consumption_kw_imputed`,
`imputation_method` — ni ne recouvre celles que l'ETL a déjà reposées.
"""
