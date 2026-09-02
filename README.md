# predict

Repo Data d'EnerVision. Il porte trois choses qui partagent le même socle
Python : l'ETL qui alimente TimescaleDB depuis l'API Mock IoT, l'entraînement
des modèles de prévision suivi par MLflow, et le service d'inférence FastAPI
déployé sur Azure.

État d'avancement : le squelette et l'outillage sont en place (EV-40). Les
étages ETL et entraînement sont fonctionnels mais non branchés en production,
et `POST /api/v1/predict` renvoie toujours `501` : servir le modèle relève
d'EV-20 et des tickets suivants.

## Démarrer

```bash
python -m venv .venv
.venv/Scripts/activate      # Linux et macOS : source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # puis renseigner DATABASE_URL
```

Vérifier que tout est vert :

```bash
ruff check .
pytest
```

## Structure

| Chemin | Rôle |
|---|---|
| `src/etl/config.py` | Configuration lue dans l'environnement |
| `src/etl/extract.py` | Lecture paginée de l'API Mock IoT |
| `src/etl/transform.py` | Normalisation vers les colonnes de `mesure` |
| `src/etl/load.py` | Chargement idempotent (`ON CONFLICT DO NOTHING`) |
| `src/etl/pipeline.py` | Orchestration d'un run, point d'entrée du conteneur |
| `src/etl/poller.py` | Ingestion continue de `/current`, point d'entrée du conteneur `poller` |
| `src/training/dataset.py` | Jeu d'apprentissage et variables explicatives |
| `src/training/train.py` | Entraînement XGBoost et enregistrement MLflow |
| `src/inference/schemas.py` | `PredictionRequest`, `PredictionPoint`, `PredictionOut` |
| `src/inference/app.py` | Application FastAPI et `CONTRACT_VERSION` |
| `scripts/export_openapi.py` | Export de la spécification OpenAPI |
| `tests/` | Tests unitaires, sans base ni réseau |
| `notebooks/` | Exploration, jamais de production |

`src/` n'est pas un paquet installé. Il est rendu importable par `PYTHONPATH`
dans l'image Docker et par `pythonpath` de `pyproject.toml` sous pytest.

## Dépendances

Trois fichiers, tous épinglés à la version exacte.

| Fichier | Contenu | Qui l'installe |
|---|---|---|
| `requirements.txt` | fastapi, pydantic, uvicorn | Service d'inférence, job CI `contract-drift` |
| `requirements-ml.txt` | requests, numpy, pandas, SQLAlchemy, psycopg, scikit-learn, xgboost, prophet, mlflow | Image ETL |
| `requirements-dev.txt` | les deux précédents, plus pytest et ruff | Poste de développement, job CI `lint-test-build` |

Le découpage évite de faire installer xgboost et prophet au job qui régénère
une spécification OpenAPI. Les versions tiennent sur Python 3.11, version de la
CI : monter numpy au-delà de 2.4 ou xgboost au-delà de 3.2 impose de faire
passer la CI en 3.12 d'abord.

## MLflow

Le backend store est une base SQLite, hors Git, dans `mlruns/`. Une base et non
un simple file store : le Model Registry, qui donne son tag au modèle servi par
l'inférence, exige un backend relationnel.

En local, sans Docker :

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --artifacts-destination ./mlruns/artifacts \
  --host 127.0.0.1 --port 5000
```

L'interface est sur <http://127.0.0.1:5000>. `mlruns/` est dans `.gitignore` :
runs, métriques et artefacts sont versionnés par MLflow, jamais par Git.

## Pile Docker

```bash
cp .env.example .env
docker compose up -d mlflow      # serveur MLflow sur ${MLFLOW_PORT:-5000}
docker compose run --rm etl --hours 24
```

La base TimescaleDB n'est pas dans ce `docker-compose.yml` : elle est fournie
par `api/enervision-db/docker-compose.yml`, qui porte le schéma figé v1.0. La
dupliquer ici donnerait deux vérités sur la table `mesure`. Le service `etl`
la joint par `DATABASE_URL_CONTAINER`, et `host.docker.internal` résout
l'hôte depuis le conteneur, y compris sous Linux grâce à `extra_hosts`.

Le service `etl` est derrière le profil `etl` : un `docker compose up`
déclencherait sinon une ingestion à chaque démarrage de la pile.

`.env.example` porte deux jeux de valeurs pour la base et l'API source :
`DATABASE_URL` et `MOCK_API_URL` pour les commandes lancées sur le poste,
`DATABASE_URL_CONTAINER` et `MOCK_API_URL_CONTAINER` pour le service `etl`.
Un seul jeu ne peut pas convenir aux deux : depuis un conteneur, `localhost`
désigne le conteneur lui-même, et l'ETL chercherait la base chez lui.

Aucun secret n'est versionné. `.env.example` ne contient que des valeurs
d'exemple, et `.env` est dans `.gitignore`.

## Lancer l'ETL et l'entraînement en local

`src/` n'est pas installé : il faut le donner à Python. Sous Docker et sous
pytest c'est déjà fait, en ligne de commande non.

```bash
# Linux et macOS
PYTHONPATH=src python -m etl.pipeline --hours 24
PYTHONPATH=src python -m training.train --site SITE001 --history-days 90
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "src"
python -m etl.pipeline --hours 24
python -m training.train --site SITE001 --history-days 90
```

Chaque entraînement ouvre un run MLflow qui journalise les hyperparamètres, les
métriques (`mae`, `rmse`, `r2`) et le modèle, enregistré sous
`enervision_xgboost`. C'est l'identifiant de ce run que la table `modele`
référence dans `mlflow_run_id`, et le tag du modèle enregistré que
`PredictionOut.model_version` renverra. Un modèle entraîné hors de ce chemin
n'est donc pas déployable.

## Ingestion continue

`etl.pipeline` rattrape une fenêtre passée ; `etl.poller` alimente la base au
fil de l'eau. Il interroge `/current` sur chaque site à cadence fixe, une
minute par défaut, et écrit chaque lecture avec son horodatage, son
`data_quality` et ses `null_reasons`, par les mêmes étages de transformation
et de chargement que le rattrapage.

```bash
docker compose up -d poller
docker compose logs -f poller
```

Le service porte `restart: unless-stopped` : si le processus sort, Docker le
relance. Une coupure réseau ne le fait pas sortir pour autant, elle est
absorbée à trois niveaux, du plus fin au plus grossier.

| Niveau | Mécanisme | Ce qu'il couvre |
|---|---|---|
| Appel | `ETL_POLL_RETRIES` reprises, attente croissante | Micro-coupure |
| Tick | Le site en échec est journalisé, les autres sont ingérés | Site indisponible |
| Conteneur | `restart: unless-stopped` | Sortie du processus |

Un site injoignable ne fait donc perdre qu'un tick, et le tick suivant repart
une minute plus tard. Seul un démarrage sans référentiel des sites fait sortir
le processus, avec le code 1, et c'est alors Docker qui reprend la main.

En ligne de commande, hors conteneur :

```bash
PYTHONPATH=src python -m etl.poller
PYTHONPATH=src python -m etl.poller --interval 30 --site SITE001
```

### Lire le retard dans les journaux

Deux retards sont journalisés, parce qu'ils ne désignent pas la même panne.

Le **retard de données** est l'âge de la mesure servie par la source. Il
apparaît par site, et consolidé par tick :

```text
INFO etl.poller site SITE001 : 1 mesure(s), retard 12.4 s, qualité good=1
INFO etl.poller tick : 7/7 site(s), 7 mesure(s), retard données max 12.4 s, durée 0.83 s
```

Au-delà de `ETL_LAG_WARNING_S`, la ligne du site passe en `WARNING` :

```text
WARNING etl.poller site SITE003 : retard 240.0 s au-delà du seuil 180.0 s, qualité partial=1
```

Le **retard d'ordonnancement** est celui que le poller a pris lui-même. Il
n'apparaît que s'il dépasse cinq secondes, et signale un tick qui déborde de
la cadence :

```text
WARNING etl.poller retard d'ordonnancement : tick démarré 7.2 s trop tard
WARNING etl.poller 1 tick(s) sauté(s) : le tick précédent a dépassé la cadence
```

Un tick sauté n'est pas rejoué : `/current` ne sert que la mesure du moment,
un rattrapage relirait la même valeur. La cadence reste ancrée sur des
instants absolus, donc un tick lent ne décale pas les suivants.

## Contrat OpenAPI

La source de vérité du contrat, ce sont les DTO de `src/inference/schemas.py`.
Le fichier gelé qui fait foi entre les équipes est
`enervision/docs/contracts/openapi-predict.json`.

Régénérer le contrat gelé après une évolution volontaire :

```bash
python scripts/export_openapi.py ../docs/contracts/openapi-predict.json
```

La CI exécute le job `contract-drift` à chaque push. Il compare la
spécification générée depuis le code au fichier gelé et fait échouer le build
en cas d'écart. La procédure d'évolution est décrite dans
`enervision/docs/contracts/README.md`.

Les docstrings de `src/inference/schemas.py` alimentent les descriptions du
contrat : les reformater change le JSON gelé. C'est la raison pour laquelle
ruff est configuré à 88 colonnes et non 80.

## Endpoints du service d'inférence

```bash
uvicorn inference.app:app --reload --app-dir src
```

| Méthode | Route | Réponse |
|---|---|---|
| GET | `/health` | `HealthOut` |
| POST | `/api/v1/predict` | `PredictionOut`, erreurs 404 et 422 |

`/health` n'est volontairement pas préfixé par `/api/v1` : c'est la sonde de
disponibilité utilisée par l'hébergeur.
