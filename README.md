# predict

[![ci](https://github.com/EnerVision-G5/predict/actions/workflows/ci.yml/badge.svg)](https://github.com/EnerVision-G5/predict/actions/workflows/ci.yml)

Service d'inférence EnerVision, déployé sur Azure. Sert uniquement les
prédictions de consommation produites par le modèle XGBoost.

À ce stade (ticket EV-06) le repo ne contient que le **contrat d'interface**.
`POST /api/v1/predict` renvoie `501 Not Implemented` : le chargement du modèle
et la prédiction relèvent d'EV-20 et des tickets suivants.

## Démarrer

```bash
python -m venv .venv
.venv/Scripts/activate      # Linux et macOS : source .venv/bin/activate
pip install -r requirements.txt
uvicorn inference.app:app --reload
```

La documentation interactive est sur <http://127.0.0.1:8000/docs>.

## Structure

| Chemin | Rôle |
|---|---|
| `inference/schemas.py` | `PredictionRequest`, `PredictionPoint`, `PredictionOut` |
| `inference/app.py` | Application FastAPI et `CONTRACT_VERSION` |
| `scripts/export_openapi.py` | Export de la spécification OpenAPI |

## Endpoints

| Méthode | Route | Réponse |
|---|---|---|
| GET | `/health` | `HealthOut` |
| POST | `/api/v1/predict` | `PredictionOut`, erreurs 404 et 422 |

`/health` n'est volontairement pas préfixé par `/api/v1` : c'est la sonde de
disponibilité utilisée par l'hébergeur.

## Contrat OpenAPI

La source de vérité du contrat, ce sont les DTO de `inference/schemas.py`. Le
fichier gelé qui fait foi entre les équipes est
`enervision/docs/contracts/openapi-predict.json`.

Régénérer le contrat gelé après une évolution volontaire :

```bash
python scripts/export_openapi.py ../docs/contracts/openapi-predict.json
```

La CI exécute le job `contract-drift` à chaque push. Il compare la
spécification générée depuis le code au fichier gelé et fait échouer le build
en cas d'écart. La procédure d'évolution est décrite dans
`enervision/docs/contracts/README.md`.
