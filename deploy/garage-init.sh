#!/bin/sh
# Prépare un nœud Garage neuf : disposition, seau, clé d'accès.
#
# Un nœud qui vient de démarrer n'a pas de disposition ("layout") : il connaît
# son disque mais ne s'est pas encore déclaré prêt à servir, et toute écriture
# S3 échouerait sur une erreur qui ne dit pas pourquoi. Le script pose donc la
# disposition d'abord, le seau ensuite.
#
# Chaque étape est idempotente : après un `docker compose down` qui a gardé les
# volumes, la relance ne casse rien et ne recrée rien.
set -e

BUCKET="${GARAGE_BUCKET:-enervision}"
# Seau de l'historique de référence, séparé de celui des partitions : un rejeu
# de l'ETL réécrit les partitions, et deux années de mesures que rien ne sait
# régénérer n'ont pas à partager un préfixe avec ce qui s'efface.
DATASETS_BUCKET="${GARAGE_DATASETS_BUCKET:-enervision-datasets}"
KEY_NAME="${GARAGE_KEY_NAME:-predict}"

NODE=$(/garage node id -q 2>/dev/null | cut -d@ -f1)

if ! /garage layout show 2>/dev/null | grep -q "$NODE"; then
    echo "disposition du nœud $NODE"
    /garage layout assign -z local -c 1G "$NODE"
    /garage layout apply --version 1
fi

/garage key create "$KEY_NAME" 2>/dev/null || echo "clé $KEY_NAME déjà présente"

for bucket in "$BUCKET" "$DATASETS_BUCKET"; do
    /garage bucket create "$bucket" 2>/dev/null || echo "seau $bucket déjà présent"
    /garage bucket allow --read --write --owner "$bucket" --key "$KEY_NAME"
done

echo "--- seaux $BUCKET et $DATASETS_BUCKET prêts."
echo "--- Identifiants à reporter dans .env : ---"
/garage key info "$KEY_NAME" --show-secret
