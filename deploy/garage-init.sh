#!/bin/sh
# Prépare un nœud Garage neuf : disposition, seaux, clé d'accès.
#
# Un nœud qui vient de démarrer n'a pas de disposition ("layout") : il connaît
# son disque mais ne s'est pas encore déclaré prêt à servir, et toute écriture
# S3 échoue alors sur « Could not reach quorum of 1 (0 of 0) », une erreur qui
# ne dit pas ce qui manque. Le script pose donc la disposition d'abord, les
# seaux et la clé ensuite.
#
# Il s'exécute SUR L'HÔTE et pilote le nœud par `docker compose exec`, au lieu
# d'être l'entrypoint d'un conteneur d'init. Trois raisons, découvertes en
# essayant l'inverse :
#
#  - l'image `dxflrs/garage` ne porte que son binaire. Ni /bin/sh ni busybox :
#    un entrypoint `/bin/sh /garage-init.sh` ne crée même pas le conteneur ;
#  - `garage node id` lit la clé du nœud dans le répertoire de métadonnées.
#    Un conteneur d'init séparé ne l'a pas, et échoue sur « Unable to read
#    node key » ;
#  - la CLI jointe depuis un autre conteneur devrait porter le secret RPC et
#    l'hôte du nœud, soit deux valeurs de plus à tenir en phase avec
#    garage.toml.
#
# Dans le conteneur du nœud, aucun de ces trois problèmes n'existe.
#
# Chaque étape est idempotente : après un `docker compose down` qui a gardé
# les volumes, la relance ne casse rien et ne recrée rien.
set -e

# Git Bash (MSYS) réécrit tout argument qui ressemble à un chemin absolu avant
# de lancer un binaire natif : `/garage` devenait `C:/Program Files/Git/garage`
# et docker répondait « stat ... no such file or directory ». Sans effet sur un
# shell POSIX, indispensable sur un poste Windows.
MSYS_NO_PATHCONV=1
export MSYS_NO_PATHCONV

SERVICE="${GARAGE_SERVICE:-garage}"
BUCKET="${GARAGE_BUCKET:-enervision}"
# Seau de l'historique de référence, séparé de celui des partitions : un rejeu
# de l'ETL réécrit les partitions, et deux années de mesures que rien ne sait
# régénérer n'ont pas à partager un préfixe avec ce qui s'efface.
DATASETS_BUCKET="${GARAGE_DATASETS_BUCKET:-enervision-datasets}"
KEY_NAME="${GARAGE_KEY_NAME:-predict}"

# -T : pas de pseudo-terminal. Sans lui, la sortie revient avec des retours
# chariot qui se retrouvent dans l'identifiant du nœud.
garage() {
    docker compose exec -T "$SERVICE" /garage "$@"
}

# L'identifiant est vérifié et non seulement testé non vide : quand la
# commande échoue, sa sortie d'erreur se retrouve dans la variable et le
# script poursuivait sur un nœud fantôme, pour finir sur des « déjà présent »
# qui laissaient croire à une base déjà prête.
NODE=$(garage node id -q 2>/dev/null | cut -d@ -f1 | tr -d '\r')
if ! echo "$NODE" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "nœud injoignable ou réponse inattendue : $NODE" >&2
    echo "démarrer d'abord : docker compose --profile garage up -d $SERVICE" >&2
    exit 1
fi

# La présence d'une disposition se lit sur son numéro de version, et non en
# cherchant l'identifiant du nœud dans la sortie : `layout show` n'affiche que
# les seize premiers caractères de l'identifiant, jamais les soixante-quatre
# que rend `node id`. La recherche échouait donc toujours, et chaque relance
# reposait une disposition déjà en place — ce que Garage refuse.
current=$(garage layout show 2>/dev/null \
    | sed -n 's/^Current cluster layout version: \([0-9][0-9]*\).*/\1/p')
if [ "${current:-0}" -eq 0 ]; then
    echo "disposition du nœud $NODE"
    garage layout assign -z local -c 1G "$NODE"
    # Le numéro de version est repris de la suggestion que Garage imprime
    # lui-même, et non calculé ici. `--version 1` en dur ne vaut que pour un
    # nœud vierge, et une version déduite du « Current cluster layout
    # version » se trompe dès qu'un changement est en attente : les deux
    # échouent sur un « Invalid new layout version » qui ne dit pas quel
    # numéro était attendu.
    version=$(garage layout show 2>/dev/null \
        | sed -n 's/.*layout apply --version \([0-9][0-9]*\).*/\1/p' | tail -1)
    garage layout apply --version "${version:-1}"
fi

# L'existence est TESTÉE avant création, et non déduite d'un échec. Garage
# accepte deux clés du même nom sans broncher : une deuxième exécution en
# créait une seconde, après quoi `--key predict` devenait ambigu et toutes les
# permissions échouaient. Un seau, lui, refuse le doublon — mais le test rend
# la sortie lisible dans les deux cas.
if garage key info "$KEY_NAME" >/dev/null 2>&1; then
    echo "clé $KEY_NAME déjà présente"
else
    garage key create "$KEY_NAME" >/dev/null
    echo "clé $KEY_NAME créée"
fi

for bucket in "$BUCKET" "$DATASETS_BUCKET"; do
    if garage bucket info "$bucket" >/dev/null 2>&1; then
        echo "seau $bucket déjà présent"
    else
        garage bucket create "$bucket" >/dev/null
        echo "seau $bucket créé"
    fi
    # Reposée à chaque passage : la permission est la seule chose que le
    # script doit garantir, et elle ne coûte rien à réaffirmer.
    garage bucket allow --read --write --owner "$bucket" --key "$KEY_NAME" >/dev/null
done

echo "--- Identifiants à reporter dans .env : ---"
garage key info "$KEY_NAME" --show-secret
