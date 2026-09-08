# **********************************************************************
# * Nom     : config.py                                                *
# * Type    : Module                                                   *
# * Sujet   : Chargement de la configuration en couches YAML, puis     *
# *   résolution depuis l'environnement                                *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Environnement retenu quand aucun n'est déclaré.
DEFAULT_ENV = "local"
# Variable qui désigne la couche de configuration à superposer.
ENV_VARIABLE = "PREDICT_ENV"
# Variable qui désigne le répertoire des fichiers de configuration.
CONF_DIRECTORY_VARIABLE = "PREDICT_CONF_DIR"
# Nom de la couche de base, toujours chargée et obligatoire.
BASE_LAYER = "base"

# Sentinelle : distingue « pas de défaut » d'un défaut à None.
_REQUIRED = object()

# Reconnaît ${NOM} et ${NOM:-repli} dans une valeur.
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Écritures admises d'un booléen vrai venu de l'environnement.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
# Écritures admises d'un booléen faux venu de l'environnement.
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


class ConfigError(RuntimeError):
    """Classe : ConfigError
    Description : La configuration est absente, illisible, ou incomplète.
    """


@dataclass(frozen=True)
class Config:
    """Classe : Config
    Description : Arbre de configuration résolu, lu par chemin pointé.
    """
    values: Mapping[str, Any]
    env_name: str = DEFAULT_ENV

    def get(self, path: str, default: Any = _REQUIRED) -> Any:
        """Méthode : get
        Description : Descend l'arbre par un chemin pointé et rend la valeur
          trouvée.
        """
        node: Any = self.values
        for key in path.split("."):
            if not isinstance(node, Mapping) or key not in node:
                if default is _REQUIRED:
                    raise ConfigError(f"Clé de configuration absente : {path}.")
                return default
            node = node[key]
        return node

    def get_str(self, path: str, default: Any = _REQUIRED) -> str:
        """Méthode : get_str
        Description : Retourne une chaîne non vide, en refusant le vide
          silencieux.
        """
        value = self.get(path, default)
        text = "" if value is None else str(value).strip()
        if not text:
            raise ConfigError(f"{path} doit porter une valeur non vide.")
        return text

    def get_optional_str(self, path: str, default: str = "") -> str:
        """Méthode : get_optional_str
        Description : Retourne une chaîne, le vide valant une absence assumée.
        """
        value = self.get(path, default)
        return "" if value is None else str(value).strip()

    def get_int(self, path: str, default: Any = _REQUIRED) -> int:
        """Méthode : get_int
        Description : Retourne un entier, en signalant une saisie illisible.
        """
        return _parse_number(path, self.get(path, default), int)

    def get_float(self, path: str, default: Any = _REQUIRED) -> float:
        """Méthode : get_float
        Description : Retourne un flottant, en signalant une saisie illisible.
        """
        return _parse_number(path, self.get(path, default), float)

    def get_bool(self, path: str, default: Any = _REQUIRED) -> bool:
        """Méthode : get_bool
        Description : Retourne un booléen, en n'admettant que des écritures
          connues.
        """
        value = self.get(path, default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        raise ConfigError(
            f"{path} doit être un booléen, reçu {value!r}."
            f" Valeurs admises : {', '.join(sorted(_TRUE_WORDS | _FALSE_WORDS))}."
        )

    def get_int_list(self, path: str, default: Any = _REQUIRED) -> list[int]:
        """Méthode : get_int_list
        Description : Retourne une liste d'entiers, en découpant une chaîne à
          virgules venue de l'environnement.
        """
        value = self.get(path, default)
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
            if not value:
                raise ConfigError(f"{path} est vide : au moins un entier.")
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ConfigError(f"{path} doit porter une liste d'entiers.")
        return [_parse_number(path, item, int) for item in value]

    def section(self, path: str) -> Mapping[str, Any]:
        """Méthode : section
        Description : Retourne un sous-arbre entier, pour les blocs passés tels
          quels.
        """
        value = self.get(path)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} doit porter un bloc de clés.")
        return value


def config_directory(env: Mapping[str, str] | None = None) -> Path:
    """Méthode : config_directory
    Description : Retourne le répertoire des fichiers de configuration.
    """
    source = os.environ if env is None else env
    override = source.get(CONF_DIRECTORY_VARIABLE, "").strip()
    if override:
        return Path(override)
    return _repository_root() / "conf"


def read_dotenv(path: Path) -> dict[str, str]:
    """Méthode : read_dotenv
    Description : Lit un fichier .env et rend ses paires clé-valeur.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_config(
    env: Mapping[str, str] | None = None,
    directory: Path | None = None,
    dotenv: Path | None = None,
) -> Config:
    """Méthode : load_config
    Description : Superpose les couches YAML puis résout leurs variables
      d'environnement.
    """
    real = dict(os.environ if env is None else env)
    root = directory or config_directory(real)
    file_values = read_dotenv(dotenv if dotenv is not None else root.parent / ".env")
    source = {**file_values, **{k: v for k, v in real.items() if v.strip()}}
    env_name = source.get(ENV_VARIABLE, "").strip() or DEFAULT_ENV
    base = _read_layer(root / f"{BASE_LAYER}.yaml", required=True)
    overlay = _read_layer(root / f"{env_name}.yaml", required=False)
    merged = _merge(base, overlay)
    return Config(values=_expand(merged, source), env_name=env_name)


def _repository_root() -> Path:
    """Méthode : _repository_root
    Description : Remonte les répertoires jusqu'à trouver conf/base.yaml.
    """
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "conf" / f"{BASE_LAYER}.yaml").is_file():
            return candidate
    raise ConfigError(
        "conf/base.yaml est introuvable depuis le répertoire courant."
        " Lancer la commande depuis la racine du repo, ou fixer"
        f" {CONF_DIRECTORY_VARIABLE}."
    )


def _read_layer(path: Path, required: bool) -> Mapping[str, Any]:
    """Méthode : _read_layer
    Description : Lit une couche YAML, en refusant tout ce qui n'est pas un
      bloc de clés.
    """
    if not path.is_file():
        if required:
            raise ConfigError(f"Fichier de configuration absent : {path}.")
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{path} doit porter un bloc de clés à la racine.")
    return loaded


def _merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Méthode : _merge
    Description : Superpose deux couches clé par clé, sans écraser un bloc
      entier.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge(current, value)
        else:
            merged[key] = value
    return merged


def _expand(node: Any, env: Mapping[str, str]) -> Any:
    """Méthode : _expand
    Description : Résout les variables d'environnement dans tout l'arbre.
    """
    if isinstance(node, Mapping):
        return {key: _expand(value, env) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand(item, env) for item in node]
    if isinstance(node, str):
        return _expand_text(node, env)
    return node


def _expand_text(text: str, env: Mapping[str, str]) -> str:
    """Méthode : _expand_text
    Description : Résout les ${NOM} d'une chaîne, en exigeant celles sans
      repli.
    """
    def resolve(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        value = env.get(name, "").strip()
        if value:
            return value
        if fallback is None:
            raise ConfigError(
                f"La variable d'environnement {name} est obligatoire :"
                " la configuration la référence sans valeur de repli."
            )
        return fallback

    return _PLACEHOLDER.sub(resolve, text)


def _parse_number(path: str, value: Any, parse: type) -> Any:
    """Méthode : _parse_number
    Description : Convertit une valeur en nombre, en refusant les booléens.
    """
    if isinstance(value, bool):
        raise ConfigError(f"{path} doit être un nombre, reçu {value!r}.")
    try:
        return parse(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} doit être un nombre, reçu {value!r}.") from exc
