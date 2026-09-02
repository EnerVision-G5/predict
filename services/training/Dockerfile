# Image du service training.
#
# Elle ne copie que ce que ce service a besoin de connaître : la bibliothèque
# partagée, son propre code, et la configuration. Le code des trois autres
# services n'est pas dans l'image, ce qui rend la frontière physique plutôt que
# conventionnelle — un import interdit ne se contenterait pas d'être mal vu, il
# ne démarrerait pas.
#
# Deux étages, et ce n'est pas cosmétique. Un build en une seule passe empile
# les couches : le premier `uv sync` écrit un environnement complet, le second
# en écrit un autre par-dessus, et l'image porte les deux — plus le binaire uv
# et son cache de roues. L'étage de construction est jeté ; seul le venv fini
# traverse. Sur ce service, la différence se compte en gigaoctets, et un
# runner d'intégration continue n'a que quelques dizaines de gigaoctets.
#
# Le contexte de build est la racine du workspace : uv a besoin de voir les
# manifestes des membres pour résoudre predict-common. C'est aussi ce qui
# garantit que les deux paquets sont figés par le même uv.lock, donc que le
# service tourne avec les versions exactes que la CI a testées.

FROM python:3.11-slim AS builder

ENV UV_COMPILE_BYTECODE=1     UV_LINK_MODE=copy     UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /app

# Les manifestes d'abord, le code ensuite : tant que les dépendances ne
# changent pas, la couche d'installation reste en cache et un changement de
# code ne réinstalle pas la pile scientifique.
COPY pyproject.toml uv.lock ./
COPY libs/predict_common/pyproject.toml libs/predict_common/
COPY services/training/pyproject.toml services/training/
# Le cache d'uv est monté et non copié : les roues téléchargées accélèrent les
# builds suivants sans peser dans aucune couche.
RUN --mount=type=cache,target=/root/.cache/uv     uv sync --frozen --no-dev --no-editable --no-install-project --package training

COPY libs/predict_common/ libs/predict_common/
COPY services/training/src/ services/training/src/
# --no-editable : les paquets du workspace sont réellement installés dans le
# venv, et non pointés par un chemin. L'étage d'exécution n'a donc pas besoin
# des sources, seulement du venv.
RUN --mount=type=cache,target=/root/.cache/uv     uv sync --frozen --no-dev --no-editable --package training


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PREDICT_CONF_DIR=/app/conf     PATH=/app/.venv/bin:$PATH

# Utilisateur non privilégié : le service n'a aucune raison d'écrire dans
# /app. Le répertoire des artefacts est monté ou distant, jamais dans l'image.
#
# Créé AVANT la copie, et les droits posés par --chown. Un `chown -R` après
# coup réécrit chaque fichier du venv, ce qu'overlayfs enregistre comme une
# couche entière de plus : l'image porterait deux fois son environnement.
RUN useradd --create-home --uid 10001 predict

# /mlruns est créé et donné à predict AVANT tout montage : docker initialise un
# volume nommé vide avec les droits du répertoire présent dans l'image, et un
# /mlruns appartenant à root rendrait le serveur MLflow incapable d'écrire son
# backend store — il boucle alors sur « unable to open database file ».
RUN mkdir -p /mlruns && chown predict:predict /mlruns

WORKDIR /app

COPY --from=builder --chown=predict:predict /app/.venv /app/.venv
COPY --chown=predict:predict conf/ conf/

USER predict

# Cette image sert aussi de serveur MLflow dans docker-compose, qui redéclare
# son entrypoint. Une seule image pour les deux garantit que le mlflow qui
# écrit les runs et celui qui sert l'UI sont exactement la même version : un
# écart entre les deux produit des runs illisibles dans l'interface.
ENTRYPOINT ["python", "-m", "training"]
