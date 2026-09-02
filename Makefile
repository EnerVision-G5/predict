# Orchestrateur minimal de la chaîne.
#
# Les services ne s'appellent jamais entre eux : c'est un ordonnanceur externe
# qui les enchaîne, et ici c'est ce fichier. Commencer par un Makefile plutôt
# que par Airflow n'est pas un pis-aller — tant que l'enchaînement tient en
# trois commandes séquentielles, un DAG n'apporterait qu'une infrastructure de
# plus à exploiter. Le jour où il faudra des reprises partielles, des
# dépendances entre journées ou un calendrier, ces trois cibles se
# transposeront telles quelles.
#
# GNU make est attendu. Sous Windows, `make` peut désigner un homonyme livré
# avec un autre outillage : vérifier avec `make --version`.

DATE ?= $(shell date -u +%F)
FV   ?= v1
DAYS ?= 1
HISTORY_DAYS ?= 90

# Chaque cible est un processus indépendant, lançable seul. Les enchaîner ici
# ne crée aucun couplage : le seul état partagé, ce sont les partitions et le
# registre MLflow.
.PHONY: help install lint test check collect poll etl train serve run-day \
        backfill openapi up down clean

help:  ## Liste les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Installe le workspace (bibliothèque + quatre services)
	uv sync

lint:  ## Vérifie le style et les imports
	uv run ruff check .

test:  ## Exécute la suite complète
	uv run pytest

check: lint test  ## Ce que la CI exécute

collect:  ## Collecte une journée vers `mesure` : make collect DATE=2026-09-02
	uv run python -m collector --date $(DATE) --days $(DAYS)

poll:  ## Démarre la collecte continue (processus long)
	uv run python -m collector.poller

etl:  ## `mesure` -> variables d'une journée : make etl DATE=... FV=v1
	uv run python -m etl --date $(DATE) --feature-version $(FV)

train:  ## Entraîne un modèle : make train FV=v1 HISTORY_DAYS=90
	uv run python -m training --feature-version $(FV) \
		--history-days $(HISTORY_DAYS) --until $(DATE)

serve:  ## Démarre le service d'inférence en local
	uv run uvicorn serving.api:app --reload

run-day: collect etl train  ## Chaîne complète sur une journée
	@echo "journée $(DATE) traitée en $(FV)"

backfill:  ## Rattrape N journées : make backfill DATE=2026-09-02 DAYS=30
	uv run python -m collector --date $(DATE) --days $(DAYS)
	@for day in $$(seq $$(($(DAYS) - 1)) -1 0); do \
		target=$$(date -u -d "$(DATE) -$$day day" +%F); \
		echo "--- etl $$target"; \
		uv run python -m etl --date $$target --feature-version $(FV) || exit 1; \
	done

openapi:  ## Régénère la spécification OpenAPI sur stdout
	uv run python scripts/export_openapi.py

up:  ## Démarre MLflow et le service d'inférence
	docker compose up -d mlflow serving

storage:  ## Démarre Garage et prépare le seau (relève la clé dans les logs)
	docker compose --profile garage up -d garage garage-init
	docker compose logs garage-init

down:  ## Arrête la pile
	docker compose down

clean:  ## Retire les partitions locales, jamais la base ni les runs MLflow
	rm -rf data/features
