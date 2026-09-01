# notebooks

Espace d'exploration : analyse de la qualité des mesures, essais de variables,
comparaison de modèles avant de les figer dans `src/training`.

## Règles

- Un notebook n'est jamais une étape de production. Dès qu'un traitement doit
  tourner deux fois, il descend dans `src/etl` ou `src/training`, avec son test.
- Vider les sorties avant de committer. Une sortie contient des données de
  mesure et fait grossir le dépôt sans rien apporter à la relecture.
- Se connecter à MLflow par la variable `MLFLOW_TRACKING_URI` de `.env`, jamais
  par une URL écrite en dur dans une cellule.

## Démarrer

```bash
pip install -r requirements-dev.txt jupyterlab
jupyter lab
```

`jupyterlab` n'est pas dans les dépendances épinglées : c'est un outil de
poste, il n'entre ni dans l'image ETL ni dans la CI.
