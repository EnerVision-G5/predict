# Image du service collector.
#
# Elle ne copie que ce que ce service a besoin de connaître : la bibliothèque
# partagée, son propre code, et la configuration. Le code des trois autres
# services n'est pas dans l'image, ce qui rend la frontière physique plutôt que
# conventionnelle — un import interdit ne se contenterait pas d'être mal vu, il
# ne démarrerait pas.
#
# Le contexte de build est la racine du workspace : uv a besoin de voir les
# manifestes des membres pour résoudre predict-common. C'est aussi ce qui
# garantit que les deux paquets sont figés par le même uv.lock, donc que le
# service tourne avec les versions exactes que la CI a testées.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     UV_COMPILE_BYTECODE=1     UV_LINK_MODE=copy     PREDICT_CONF_DIR=/app/conf     PATH=/app/.venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /app

# Les manifestes d'abord, le code ensuite : tant que les dépendances ne
# changent pas, la couche d'installation reste en cache et un changement de
# code ne réinstalle pas la pile scientifique.
COPY pyproject.toml uv.lock ./
COPY libs/predict_common/pyproject.toml libs/predict_common/
COPY services/collector/pyproject.toml services/collector/
RUN uv sync --frozen --no-install-project --package collector

COPY libs/predict_common/ libs/predict_common/
COPY services/collector/src/ services/collector/src/
COPY conf/ conf/
RUN uv sync --frozen --package collector

# Utilisateur non privilégié : le service n'a aucune raison d'écrire dans
# /app. Le répertoire des artefacts est monté ou distant, jamais dans l'image.
RUN useradd --create-home --uid 10001 predict && chown -R predict:predict /app
USER predict

# ENTRYPOINT et non CMD : les arguments passés à `docker compose run` viennent
# alors s'ajouter à la commande au lieu de la remplacer, ce qui rend
# `docker compose run --rm collector --date 2026-09-02` possible. Le service
# poller, qui partage cette image, redéclare son propre entrypoint.
ENTRYPOINT ["python", "-m", "collector"]
