"""Service d'inférence : un alias de modèle, une partition de variables.

`loader` résout le modèle dans le registre MLflow, `forecast` reconstruit les
décalages depuis la dernière partition publiée par l'ETL et enchaîne la
prévision heure par heure, `api` porte le contrat HTTP, `schemas` porte les
DTO qui en sont la source de vérité.

Le service n'importe le code d'aucun autre. Il ne recalcule pas non plus les
variables depuis les mesures brutes : ce serait refaire le travail de l'ETL
avec un second jeu de règles, qui finirait par diverger du premier.
"""
