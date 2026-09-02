# Image commune à l'ETL et au serveur MLflow.
#
# Une seule image pour les deux services garantit que le mlflow qui sert l'UI
# et le mlflow qui écrit les runs sont exactement la même version : un écart
# entre les deux produit des runs illisibles dans l'interface.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Les requirements sont copiés seuls : tant qu'ils ne changent pas, la couche
# d'installation reste en cache et un changement de code ne réinstalle pas
# xgboost et prophet.
COPY requirements-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-ml.txt

COPY src/ ./src/

# Utilisateur non privilégié : le conteneur n'a aucune raison d'écrire ailleurs
# que dans /mlruns. Ce répertoire est créé et donné à etl AVANT le montage,
# parce que docker initialise un volume nommé vide avec les droits du
# répertoire présent dans l'image : un /mlruns appartenant à root rendrait le
# serveur MLflow incapable d'écrire son backend store.
RUN useradd --create-home --uid 10001 etl \
    && mkdir -p /mlruns \
    && chown -R etl:etl /app /mlruns
USER etl

# ENTRYPOINT et non CMD : les arguments passés à docker compose run
# s'ajoutent alors à la commande au lieu de la remplacer, ce qui rend
# "docker compose run --rm etl --hours 24" possible. Le service mlflow,
# qui partage cette image, redéclare son propre entrypoint.
ENTRYPOINT ["python", "-m", "etl.pipeline"]
