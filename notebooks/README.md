# notebooks

Espace d'exploration : analyse de la qualité des mesures, essais de variables,
comparaison de modèles avant de les figer dans un service.

## Règles

- Un notebook n'est jamais une étape de production. Dès qu'un traitement doit
  tourner deux fois, il descend dans le service dont c'est le métier —
  `services/etl` pour une règle de transformation, `services/training` pour un
  choix de modèle — avec son test.
- Lire les artefacts, jamais la source. Un notebook qui appellerait l'API Mock
  se donnerait sa propre version de la collecte, qui divergerait de celle du
  collecteur. `predict_common.io.read_frames` lit les mêmes partitions que les
  services.
- Vider les sorties avant de committer. Une sortie contient des données de
  mesure et fait grossir le dépôt sans rien apporter à la relecture.
- Se connecter à MLflow par `conf/`, jamais par une URL écrite en dur dans une
  cellule.

## Démarrer

```bash
uv sync --group dev
uv run --with jupyterlab jupyter lab
```

```python
from predict_common.config import load_config
from predict_common.paths import features_partition, lookback_range
from predict_common import io

config = load_config()
root = config.get_str("storage.root")
version = config.get_str("etl.feature_version")

import datetime as dt
days = lookback_range(dt.date.today(), 7)
frame = io.read_frames([features_partition(root, version, day) for day in days])
```
