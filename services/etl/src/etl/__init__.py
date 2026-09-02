"""Transformation des mesures brutes en variables prêtes pour l'apprentissage.

Le service lit `raw/{source}/dt=.../`, écrit `features/{version}/dt=.../`, et
n'importe le code d'aucun autre service. Ses étages restent séparés et
testables un à un : `clean` normalise et déduplique, `quality` nomme la cause
de chaque valeur absente, `impute` reconstruit ce qui peut l'être dans une
colonne à part, `exclude` écarte ce qui ne peut pas l'être, `features` change
le pas de la série et calcule les décalages, `validate` refuse de publier un
artefact hors contrat.

`sink_db` est à part : c'est une sortie annexe vers TimescaleDB, désactivée
par défaut, que consomme l'API EnerVision et aucun des quatre services.
"""
