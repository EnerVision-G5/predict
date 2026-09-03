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
KEY_NAME="${GARAGE_KEY_NAME:-predict}"

NODE=$(/garage node id -q 2>/dev/null | cut -d@ -f1)

if ! /garage layout show 2>/dev/null | grep -q "$NODE"; then
    echo "disposition du nœud $NODE"
    /garage layout assign -z local -c 1G "$NODE"
    /garage layout apply --version 1
fi

/garage bucket create "$BUCKET" 2>/dev/null || echo "seau $BUCKET déjà présent"
/garage key create "$KEY_NAME" 2>/dev/null || echo "clé $KEY_NAME déjà présente"
/garage bucket allow --read --write --owner "$BUCKET" --key "$KEY_NAME"

echo "--- seau $BUCKET prêt. Identifiants à reporter dans .env : ---"
/garage key info "$KEY_NAME" --show-secret
