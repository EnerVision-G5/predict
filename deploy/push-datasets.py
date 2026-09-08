"""Dépose l'historique de référence sur le stockage objet.

Le pendant de la lecture faite par `collector.datasets` : les CSV ne sont ni
dans le dépôt ni dans l'image, il faut donc les poser une fois là où le
collecteur ira les chercher.

    python deploy/push-datasets.py datasets s3://enervision-datasets

Pourquoi ce script plutôt qu'un `aws s3 cp`. Parce qu'il n'exige rien de plus
que l'environnement du projet : la même `predict_common.io` qui résout une
racine pour l'ETL et pour le collecteur résout la destination ici, et un poste
qui sait lancer la chaîne sait donc lancer l'envoi. Installer un client S3
pour un transfert que sept fichiers épuisent serait une dépendance de plus à
tenir à jour sur chaque poste et dans chaque image de CI.

Les identifiants ne sont pas lus ici. `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`
et `AWS_SECRET_ACCESS_KEY` sont pris dans l'environnement par la bibliothèque
cliente, comme pour tous les autres accès au stockage de la chaîne.

L'envoi ÉCRASE ce qui porte le même nom, et c'est voulu : le jeu de données
est figé, un renvoi n'a donc rien à fusionner. Ce qui n'a pas de source ne
disparaît pas pour autant — le script ne supprime rien à la destination.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from fnmatch import fnmatch

import pyarrow.fs

from predict_common import io
from predict_common.paths import join

SITE_PATTERN = "SITE*.csv"
ALL_PATTERN = "*"

CHUNK_BYTES = 1 << 20

EXIT_OK = 0
EXIT_FAILED = 1

logger = logging.getLogger("push-datasets")


def source_files(root: str, pattern: str) -> list[str]:
    """Liste les fichiers à envoyer, triés, et refuse une source muette.

    Une source vide est une erreur et non un envoi de zéro fichier : la
    commande aurait l'air d'avoir réussi, et le manque ne se verrait qu'au
    premier import, du côté du serveur.
    """
    filesystem, path = io.resolve(root)
    selector = pyarrow.fs.FileSelector(path, recursive=False, allow_not_found=True)
    files = sorted(
        entry.path
        for entry in filesystem.get_file_info(selector)
        if entry.type == pyarrow.fs.FileType.File
        and fnmatch(entry.base_name, pattern)
    )
    if not files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {root}.")
    return files


def copy_file(source: str, destination: str) -> int:
    """Copie un fichier d'un stockage à l'autre et retourne sa taille.

    Les deux extrémités sont ouvertes par `predict_common.io`, ce qui rend le
    sens du transfert indifférent : disque vers seau, seau vers disque, ou
    seau vers seau.
    """
    source_fs, source_path = io.resolve(source)
    target_fs, target_path = io.resolve(destination)
    written = 0
    with (
        source_fs.open_input_stream(source_path) as reader,
        target_fs.open_output_stream(target_path) as writer,
    ):
        while chunk := reader.read(CHUNK_BYTES):
            writer.write(chunk)
            written += len(chunk)
    return written


def push(root: str, destination: str, pattern: str) -> int:
    """Envoie les fichiers de `root` vers `destination` et retourne le total."""
    total = 0
    for source in source_files(root, pattern):
        name = source.replace("\\", "/").rsplit("/", 1)[-1]
        size = copy_file(source, join(destination, name))
        logger.info("%s : %.1f Mo déposés", name, size / (1 << 20))
        total += size
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="push-datasets",
        description="Dépose l'historique de référence sur le stockage objet.",
    )
    parser.add_argument(
        "root",
        help="Racine source des fichiers, chemin ou URI. Exemple : datasets.",
    )
    parser.add_argument(
        "destination",
        help="Racine cible. Exemple : s3://enervision-datasets.",
    )
    parser.add_argument(
        "--tout",
        dest="everything",
        action="store_true",
        help=(
            "Envoie aussi all_sites_combined.csv et les métadonnées. Le"
            " collecteur ne les lit pas ; ils ne coûtent que du stockage."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    pattern = ALL_PATTERN if args.everything else SITE_PATTERN
    try:
        total = push(args.root, args.destination, pattern)
    except (OSError, io.StorageError) as exc:
        logger.error("envoi impossible : %s", exc)
        return EXIT_FAILED
    logger.info(
        "envoi terminé : %.1f Mo vers %s", total / (1 << 20), args.destination
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
