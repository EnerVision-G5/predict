"""Lecture et écriture des partitions parquet, en local comme sur S3.

Deux propriétés à tenir, et tout le module est là pour elles.

L'écriture d'une partition est atomique et remplaçante. Les fichiers partent
dans un répertoire de travail voisin, puis la partition cible est supprimée et
le contenu déplacé. Un lecteur ne voit donc jamais une partition à moitié
écrite, et relancer `--date 2026-09-02` reproduit la partition du jour au lieu
de la doubler. C'est cette propriété qui rend un rejeu après incident sans
effet de bord, et le rejeu est le mode d'exploitation normal, pas l'exception.

Le stockage est interchangeable. Un chemin nu désigne le disque, une URI
`s3://` désigne Garage ou S3, et le reste du code ne fait pas la différence :
il manipule des chaînes construites par `paths`. Passer du poste au
déploiement ne change qu'une valeur de `conf/`.

Il n'y a volontairement pas d'ajout. Une partition se produit en entier ou pas
du tout : un mode qui déposerait un fichier de plus rendrait le résultat
dépendant du nombre de fois qu'on a lancé la commande, ce qu'aucun compte en
aval ne saurait rattraper. Le fil de l'eau, lui, va en base, où la clé primaire
arbitre.
"""

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

COMPRESSION = "zstd"

IGNORED_PREFIXES = ("_", ".")

S3_SCHEME = "s3"
ENDPOINT_VARIABLE = "AWS_ENDPOINT_URL"

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Le stockage n'a pas pu être joint, lu ou écrit."""


def resolve(uri: str) -> tuple[pyarrow.fs.FileSystem, str]:
    """Retourne le système de fichiers d'une URI et le chemin qu'il attend.

    Un chemin sans schéma désigne le disque local et est rendu absolu : un
    chemin relatif dépendrait du répertoire courant du processus, qui n'est
    pas le même dans un conteneur et sur un poste.
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
    """Écrit la partition en entier, en remplaçant celle qui existait.

    C'est l'opération d'un traitement daté : `--date 2026-09-02` produit la
    partition du 2 septembre, qu'elle ait déjà été produite ou non. Un ajout
    ferait grossir la partition à chaque rejeu et fausserait tout compte en
    aval sans qu'aucune erreur ne le signale.
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
    """Lit une ou plusieurs partitions et retourne un seul tableau.

    Une partition absente est ignorée par défaut : lire une fenêtre de sept
    jours dont le premier n'a pas encore été collecté est un cas normal au
    démarrage de la chaîne, pas une panne. Passer `missing_ok=False` en fait
    une erreur, ce que fait l'ETL pour la partition du jour qu'il traite —
    celle-là, si elle manque, il n'a rien à faire.
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
    """Dit si la partition existe et porte au moins un fichier."""
    filesystem, path = resolve(partition)
    info = filesystem.get_file_info(path)
    if info.type != pyarrow.fs.FileType.Directory:
        return False
    return bool(_data_files(filesystem, path))


def _read_dataset(
    partitions: Sequence[str],
    columns: Sequence[str] | None,
) -> pd.DataFrame:
    """Assemble les fichiers de plusieurs partitions en un tableau."""
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
    """Liste les fichiers de données d'une partition, travaux exclus."""
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
    """Convertit le tableau pandas en table arrow au schéma imposé.

    Le schéma est imposé et non déduit : déduit, une colonne entièrement nulle
    sur une journée s'écrirait en type `null`, et la lecture conjointe de deux
    partitions dont l'une a des valeurs échouerait à la concaténation.
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
    """Substitue le contenu du répertoire de travail à celui de la cible.

    Le déplacement se fait fichier par fichier et non répertoire par
    répertoire : un stockage objet n'a pas de répertoires à renommer, il n'a
    que des clés. Le coût est une fenêtre courte pendant laquelle la partition
    est vide — acceptable ici, où les producteurs sont des traitements datés
    et non des écritures concurrentes sur la même partition.
    """
    _delete_directory(filesystem, target)
    filesystem.create_dir(target, recursive=True)
    for source in _data_files(filesystem, staging):
        name = source.rsplit("/", 1)[-1]
        filesystem.move(source, join(target, name))
    _discard(filesystem, staging)


def _delete_directory(filesystem: pyarrow.fs.FileSystem, path: str) -> None:
    """Supprime un répertoire s'il existe, sans échouer s'il est absent.

    Le repli sur la suppression une par une n'est pas un rattrapage d'erreur
    mais une détection de capacité : tous les stockages objet n'implémentent
    pas la suppression groupée du protocole S3, et la seule façon de le savoir
    est de l'essayer. Voir `_delete_files` pour ce que le repli coûte.
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
    """Supprime les fichiers d'un répertoire un par un.

    `delete_dir` émet un `DeleteObjects`, la suppression groupée du protocole
    S3. Garage la refuse (`Invalid delete XML query`), et l'ETL échouait alors
    au moment de remplacer la partition — après avoir tout calculé. Le
    protocole a une seconde forme, `DeleteObject`, qui ne porte qu'une clé et
    que `delete_file` émet : une requête par fichier au lieu d'une par lot,
    pour des partitions qui en comptent une poignée.

    Ce que le repli laisse derrière lui, ce sont les objets vides qui figurent
    les répertoires. Ils ne portent aucune donnée, `_data_files` les ignore et
    `exists` ne les compte pas : une partition ainsi vidée est bien vue comme
    absente.
    """
    selector = pyarrow.fs.FileSelector(path, recursive=True, allow_not_found=True)
    for entry in filesystem.get_file_info(selector):
        if entry.type == pyarrow.fs.FileType.File:
            filesystem.delete_file(entry.path)


def _discard(filesystem: pyarrow.fs.FileSystem, staging: str) -> None:
    """Retire le répertoire de travail, y compris après un échec d'écriture.

    L'échec du nettoyage n'est pas propagé : il masquerait l'erreur d'origine,
    qui est la seule à expliquer pourquoi l'écriture n'a pas abouti.
    """
    try:
        _delete_directory(filesystem, staging)
    except OSError as exc:
        logger.warning("répertoire de travail non nettoyé : %s (%s)", staging, exc)


def _s3_filesystem() -> pyarrow.fs.S3FileSystem:
    """Ouvre le système de fichiers S3, Garage compris.

    Garage est un S3 servi ailleurs que chez AWS et souvent en clair : c'est
    `AWS_ENDPOINT_URL` qui l'indique, et son schéma qui décide du chiffrement.
    Les identifiants ne sont jamais lus ici, la bibliothèque AWS les prend
    dans l'environnement ou dans le rôle de la machine.
    """
    endpoint = os.environ.get(ENDPOINT_VARIABLE, "").strip()
    if not endpoint:
        return pyarrow.fs.S3FileSystem()
    scheme, _, host = endpoint.rpartition("://")
    return pyarrow.fs.S3FileSystem(
        endpoint_override=host,
        scheme=scheme or "https",
    )
