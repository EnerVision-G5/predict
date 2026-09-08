"""Chargement de la configuration : YAML en couches, puis environnement.

Trois principes.

La configuration est un fichier, pas une constante Python. `conf/base.yaml`
décrit la chaîne, `conf/{PREDICT_ENV}.yaml` décrit ce qu'une machine en
change. La superposition se fait clé par clé et non section par section :
surcharger `training.params.n_estimators` ne fait pas disparaître les autres
hyperparamètres.

Aucun secret n'entre dans un fichier versionné. Les valeurs sensibles sont
écrites `${VARIABLE}` et résolues dans l'environnement au chargement. Un
`${VARIABLE}` sans repli est obligatoire : mieux vaut un démarrage qui échoue
en nommant la variable qu'un run qui part avec une chaîne vide et écrit ses
artefacts au mauvais endroit.

La lecture est typée et explicite. `get_int` sur une valeur illisible échoue
au lieu de retomber sur un défaut que personne n'a demandé.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ENV = "local"
ENV_VARIABLE = "PREDICT_ENV"
CONF_DIRECTORY_VARIABLE = "PREDICT_CONF_DIR"
BASE_LAYER = "base"

# Absence de défaut, distincte de None : une clé peut légitimement porter
# None, et un défaut None doit alors rester un défaut fourni.
_REQUIRED = object()

# ${NOM} ou ${NOM:-valeur de repli}. Le repli s'arrête à l'accolade fermante,
# il peut donc contenir n'importe quoi d'autre, y compris deux-points et
# barres obliques d'une URL.
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Écritures admises d'un booléen venu de l'environnement. La liste est fermée :
# tout ce qui n'y figure pas est une faute de frappe, et une faute de frappe
# sur un drapeau de sécurité doit s'entendre.
_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


class ConfigError(RuntimeError):
    """La configuration est absente, illisible, ou incomplète."""


@dataclass(frozen=True)
class Config:
    """Arbre de configuration résolu, lu par chemin pointé.

    Les services ne se passent jamais de sous-arbres nus : ils lisent
    `config.get_str("source.base_url")`. Le chemin pointé est ce qui rend
    l'erreur lisible quand une clé manque, là où un `dict["source"]["url"]`
    ne dirait que `KeyError: 'url'`.
    """

    values: Mapping[str, Any]
    env_name: str = DEFAULT_ENV

    def get(self, path: str, default: Any = _REQUIRED) -> Any:
        """Retourne la valeur au chemin pointé, ou `default` s'il est fourni."""
        node: Any = self.values
        for key in path.split("."):
            if not isinstance(node, Mapping) or key not in node:
                if default is _REQUIRED:
                    raise ConfigError(f"Clé de configuration absente : {path}.")
                return default
            node = node[key]
        return node

    def get_str(self, path: str, default: Any = _REQUIRED) -> str:
        """Retourne une chaîne non vide, en refusant le vide silencieux."""
        value = self.get(path, default)
        text = "" if value is None else str(value).strip()
        if not text:
            raise ConfigError(f"{path} doit porter une valeur non vide.")
        return text

    def get_optional_str(self, path: str, default: str = "") -> str:
        """Retourne une chaîne, le vide valant une absence assumée."""
        value = self.get(path, default)
        return "" if value is None else str(value).strip()

    def get_int(self, path: str, default: Any = _REQUIRED) -> int:
        """Retourne un entier, en signalant une saisie illisible."""
        return _parse_number(path, self.get(path, default), int)

    def get_float(self, path: str, default: Any = _REQUIRED) -> float:
        """Retourne un décimal, en signalant une saisie illisible."""
        return _parse_number(path, self.get(path, default), float)

    def get_bool(self, path: str, default: Any = _REQUIRED) -> bool:
        """Retourne un booléen, en refusant une valeur qu'on ne sait pas lire.

        Nécessaire parce qu'une valeur venue de l'environnement est TOUJOURS
        une chaîne : `${SERVING_AUTH_ENABLED:-true}` produit `"true"`, jamais
        `True`, et `if config.get(...)` serait alors vrai pour `"false"` comme
        pour `"true"`. C'est le genre d'inversion silencieuse qui ouvre un
        contrôle d'accès en croyant le fermer.

        Une valeur illisible échoue plutôt que de retomber sur un défaut :
        `SERVING_AUTH_ENABLED=oui` doit se voir, pas se deviner.
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
        """Retourne une liste d'entiers, en refusant une valeur seule.

        Une valeur seule là où une liste est attendue est presque toujours une
        faute de frappe dans le YAML, pas une intention. Un entier nu reste
        donc refusé.

        Une chaîne séparée par des virgules, elle, est acceptée et découpée :
        une valeur venue de l'environnement est TOUJOURS une chaîne, et
        `${ETL_LAG_HOURS:-1,24,168}` ne peut pas produire une liste YAML. Sans
        ce découpage, la liste serait la seule valeur de la configuration
        qu'un déploiement ne pourrait pas surcharger.
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
        """Retourne un sous-arbre, pour les blocs passés tels quels."""
        value = self.get(path)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} doit porter un bloc de clés.")
        return value


def config_directory(env: Mapping[str, str] | None = None) -> Path:
    """Retourne le répertoire des fichiers de configuration.

    Le répertoire est déductible du disque en développement, où `conf/` est à
    la racine du repo, mais pas dans une image où seul le service est copié.
    `PREDICT_CONF_DIR` sert alors à le désigner.
    """
    source = os.environ if env is None else env
    override = source.get(CONF_DIRECTORY_VARIABLE, "").strip()
    if override:
        return Path(override)
    return _repository_root() / "conf"


def read_dotenv(path: Path) -> dict[str, str]:
    """Lit un fichier `.env`, sans rien exporter dans l'environnement réel.

    Docker Compose lit `.env` tout seul ; une commande lancée sur le poste,
    non. Sans cette lecture, le même `.env` réglerait les conteneurs et
    laisserait `python -m collector` sur les valeurs de repli — l'écart le plus
    pénible à diagnostiquer, parce que les deux marchent, mais pas sur la même
    source.

    Le format admis est volontairement pauvre : `CLE=valeur`, une par ligne,
    `#` en commentaire. Ni substitution, ni guillemets multilignes, ni `export`
    — un `.env` qui aurait besoin de plus est un script, et devrait le dire.
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
    """Charge la base, superpose la couche d'environnement, résout les ${}.

    L'environnement réel l'emporte sur le `.env` : c'est lui que docker compose
    et la CI renseignent, et une valeur laissée dans un `.env` oublié ne doit
    pas recouvrir ce qu'un déploiement a explicitement posé.

    `env` est injectable pour que les tests n'aient pas à écrire dans
    os.environ, qui est un état global partagé entre tests.
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
    """Remonte jusqu'au répertoire qui porte `conf/`.

    La bibliothèque est installée dans un site-packages, pas dans l'arbre du
    repo : on part du répertoire courant, seul point commun aux commandes
    lancées depuis la racine.
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
    """Lit une couche YAML, un fichier absent valant une couche vide."""
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
    """Superpose deux arbres clé par clé, en descendant dans les blocs.

    Remplacer un bloc en entier obligerait chaque surcharge à recopier les
    clés qu'elle ne change pas, et une clé ajoutée à la base disparaîtrait
    silencieusement des environnements qui surchargent son bloc.
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
    """Résout les ${VARIABLE} de l'arbre dans l'environnement."""
    if isinstance(node, Mapping):
        return {key: _expand(value, env) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand(item, env) for item in node]
    if isinstance(node, str):
        return _expand_text(node, env)
    return node


def _expand_text(text: str, env: Mapping[str, str]) -> str:
    """Remplace les motifs d'une chaîne, en exigeant ceux sans repli."""

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
    """Convertit une valeur numérique, en nommant la clé fautive."""
    if isinstance(value, bool):
        raise ConfigError(f"{path} doit être un nombre, reçu {value!r}.")
    try:
        return parse(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} doit être un nombre, reçu {value!r}.") from exc
