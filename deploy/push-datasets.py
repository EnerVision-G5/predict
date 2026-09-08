# **********************************************************************
# * Nom     : push-datasets.py                                         *
# * Type    : Script                                                   *
# * Sujet   : Dépôt de l'historique de référence sur le stockage objet *
# * Service : outillage de déploiement                                 *
# **********************************************************************

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from fnmatch import fnmatch

import pyarrow.fs

from predict_common import io
from predict_common.paths import join

# Fichiers envoyés par défaut : un par site, et rien d'autre.
SITE_PATTERN = "SITE*.csv"
# Tout le contenu de la racine, demandé par --tout.
ALL_PATTERN = "*"

# Taille des blocs de transfert, pour ne pas tout charger en mémoire.
CHUNK_BYTES = 1 << 20

# Code de sortie d'un envoi abouti.
EXIT_OK = 0
# Code de sortie d'un envoi interrompu.
EXIT_FAILED = 1

logger = logging.getLogger("push-datasets")


def source_files(root: str, pattern: str) -> list[str]:
    """Méthode : source_files
    Description : Liste les fichiers à envoyer, triés, et refuse une source
      muette.
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
    """Méthode : copy_file
    Description : Copie un fichier d'un stockage à l'autre et retourne sa
      taille.
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
    """Méthode : push
    Description : Envoie les fichiers de la racine vers la destination.
    """
    total = 0
    for source in source_files(root, pattern):
        name = source.replace("\\", "/").rsplit("/", 1)[-1]
        size = copy_file(source, join(destination, name))
        logger.info("%s : %.1f Mo déposés", name, size / (1 << 20))
        total += size
    return total


def build_parser() -> argparse.ArgumentParser:
    """Méthode : build_parser
    Description : Analyse la ligne de commande de l'envoi.
    """
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
    """Méthode : main
    Description : Point d'entrée : dépose l'historique et rend un code de
      sortie.
    """
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
