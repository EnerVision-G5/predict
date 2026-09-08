# **********************************************************************
# * Nom     : io.py                                                    *
# * Type    : Module                                                   *
# * Sujet   : Lecture et écriture atomiques des partitions parquet,    *
# *   sur disque comme sur S3                                          *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import pyarrow
import pyarrow.dataset
import pyarrow.fs
import pyarrow.parquet

from predict_common.paths import join, normalize, part_file, temporary_sibling

# Codec de compression des fichiers parquet écrits.
COMPRESSION = "zstd"

# Préfixes des entrées ignorées à la lecture, dont les répertoires de travail.
IGNORED_PREFIXES = ("_", ".")

# Schéma d'URI qui désigne le stockage objet.
S3_SCHEME = "s3"
# Variable d'environnement qui détourne le client S3 vers Garage.
ENDPOINT_VARIABLE = "AWS_ENDPOINT_URL"

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Classe : StorageError
    Description : Le stockage n'a pas pu être joint, lu ou écrit.
    """


def resolve(uri: str) -> tuple[pyarrow.fs.FileSystem, str]:
    """Méthode : resolve
    Description : Retourne le système de fichiers d'une URI et le chemin qu'il
      attend.
    """
    path = normalize(uri)
    if "://" not in path:
        return pyarrow.fs.LocalFileSystem(), os.path.abspath(path)
    scheme = path.split("://", 1)[0]
    if scheme == S3_SCHEME:
        return _s3_filesystem(), path.split("://", 1)[1]
    try:
        filesystem, resolved = pyarrow.fs.FileSystem.from_uri(path)
    except (pyarrow.ArrowInvalid, ValueError) as exc:
        raise StorageError(f"Stockage inconnu pour l'URI {uri!r}.") from exc
    return filesystem, resolved


def write_frame(
    frame: pd.DataFrame,
    partition: str,
    schema: pyarrow.Schema | None = None,
    metadata: Mapping[str, str] | None = None,
) -> str:
    """Méthode : write_frame
    Description : Écrit une partition en entier, en remplaçant celle qui
      existait.
    """
    filesystem, target = resolve(partition)
    staging = temporary_sibling(target)
    table = _to_table(frame, schema, metadata)
    filesystem.create_dir(staging, recursive=True)
    try:
        pyarrow.parquet.write_table(
            table,
            join(staging, part_file()),
            filesystem=filesystem,
            compression=COMPRESSION,
        )
        _replace(filesystem, staging, target)
    except Exception:
        _discard(filesystem, staging)
        raise
    logger.info("partition écrite : %s (%d ligne(s))", partition, len(frame))
    return partition


def read_frames(
    partitions: Iterable[str],
    columns: Sequence[str] | None = None,
    missing_ok: bool = True,
) -> pd.DataFrame:
    """Méthode : read_frames
    Description : Lit plusieurs partitions et les assemble en un seul tableau.
    """
    present: list[str] = []
    for partition in partitions:
        if exists(partition):
            present.append(partition)
        elif not missing_ok:
            raise StorageError(f"Partition absente : {partition}.")
        else:
            logger.debug("partition absente, ignorée : %s", partition)
    if not present:
        return pd.DataFrame(columns=list(columns or []))
    return _read_dataset(present, columns)


def exists(partition: str) -> bool:
    """Méthode : exists
    Description : Dit si une partition existe et porte au moins un fichier.
    """
    filesystem, path = resolve(partition)
    info = filesystem.get_file_info(path)
    if info.type != pyarrow.fs.FileType.Directory:
        return False
    return bool(_data_files(filesystem, path))


def _read_dataset(
    partitions: Sequence[str],
    columns: Sequence[str] | None,
) -> pd.DataFrame:
    """Méthode : _read_dataset
    Description : Assemble les fichiers de plusieurs partitions en un tableau.
    """
    frames: list[pd.DataFrame] = []
    for partition in partitions:
        filesystem, path = resolve(partition)
        files = _data_files(filesystem, path)
        if not files:
            continue
        dataset = pyarrow.dataset.dataset(
            files, filesystem=filesystem, format="parquet"
        )
        selected = list(columns) if columns else None
        frames.append(dataset.to_table(columns=selected).to_pandas())
    if not frames:
        return pd.DataFrame(columns=list(columns or []))
    return pd.concat(frames, ignore_index=True)


def _data_files(filesystem: pyarrow.fs.FileSystem, path: str) -> list[str]:
    """Méthode : _data_files
    Description : Liste les fichiers de données d'une partition, entrées
      techniques exclues.
    """
    selector = pyarrow.fs.FileSelector(path, recursive=False, allow_not_found=True)
    return sorted(
        info.path
        for info in filesystem.get_file_info(selector)
        if info.type == pyarrow.fs.FileType.File
        and not info.base_name.startswith(IGNORED_PREFIXES)
    )


def _to_table(
    frame: pd.DataFrame,
    schema: pyarrow.Schema | None,
    metadata: Mapping[str, str] | None,
) -> pyarrow.Table:
    """Méthode : _to_table
    Description : Projette un tableau pandas sur le schéma attendu et y attache
      ses métadonnées.
    """
    if schema is None:
        table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    else:
        projected = frame.reindex(columns=[field.name for field in schema])
        table = pyarrow.Table.from_pandas(
            projected, schema=schema, preserve_index=False
        )
    if not metadata:
        return table
    existing = table.schema.metadata or {}
    encoded = {key.encode(): str(value).encode() for key, value in metadata.items()}
    return table.replace_schema_metadata({**existing, **encoded})


def _replace(filesystem: pyarrow.fs.FileSystem, staging: str, target: str) -> None:
    """Méthode : _replace
    Description : Substitue la partition cible par le contenu du répertoire de
      travail.
    """
    _delete_directory(filesystem, target)
    filesystem.create_dir(target, recursive=True)
    for source in _data_files(filesystem, staging):
        name = source.rsplit("/", 1)[-1]
        filesystem.move(source, join(target, name))
    _discard(filesystem, staging)


def _delete_directory(filesystem: pyarrow.fs.FileSystem, path: str) -> None:
    """Méthode : _delete_directory
    Description : Supprime un répertoire, fichier par fichier si le stockage
      refuse en bloc.
    """
    info = filesystem.get_file_info(path)
    if info.type != pyarrow.fs.FileType.Directory:
        return
    try:
        filesystem.delete_dir(path)
    except OSError as exc:
        logger.debug("suppression groupée refusée sur %s (%s)", path, exc)
        _delete_files(filesystem, path)


def _delete_files(filesystem: pyarrow.fs.FileSystem, path: str) -> None:
    """Méthode : _delete_files
    Description : Supprime récursivement les fichiers d'un répertoire.
    """
    selector = pyarrow.fs.FileSelector(path, recursive=True, allow_not_found=True)
    for entry in filesystem.get_file_info(selector):
        if entry.type == pyarrow.fs.FileType.File:
            filesystem.delete_file(entry.path)


def _discard(filesystem: pyarrow.fs.FileSystem, staging: str) -> None:
    """Méthode : _discard
    Description : Jette un répertoire de travail sans faire échouer l'appelant.
    """
    try:
        _delete_directory(filesystem, staging)
    except OSError as exc:
        logger.warning("répertoire de travail non nettoyé : %s (%s)", staging, exc)


def _s3_filesystem() -> pyarrow.fs.S3FileSystem:
    """Méthode : _s3_filesystem
    Description : Ouvre le système de fichiers S3, en suivant le point d'accès
      déclaré dans l'environnement.
    """
    endpoint = os.environ.get(ENDPOINT_VARIABLE, "").strip()
    if not endpoint:
        return pyarrow.fs.S3FileSystem()
    scheme, _, host = endpoint.rpartition("://")
    return pyarrow.fs.S3FileSystem(
        endpoint_override=host,
        scheme=scheme or "https",
    )
