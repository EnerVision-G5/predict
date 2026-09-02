"""Extraction des mesures depuis l'API Mock IoT.

L'API source est la seule vérité amont. Aucun renommage de champ n'est fait
ici : les noms voyagent tels quels jusqu'à la table `mesure`, c'est ce qui
permet de relire une ligne en base et de la comparer à la source sans table de
correspondance.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import requests

from etl.config import EtlConfig

READINGS_PATH = "/api/v1/sites/{site_id}/readings"
SITES_PATH = "/api/v1/sites"


class ExtractionError(RuntimeError):
    """L'API source a répondu autre chose qu'une page de mesures."""


def build_session() -> requests.Session:
    """Retourne la session HTTP utilisée par les appels d'extraction.

    Une session réutilise la connexion TCP entre les pages : sur un rattrapage
    de plusieurs milliers de mesures, c'est la différence entre une poignée de
    handshakes et un par page.
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


def fetch_sites(
    config: EtlConfig,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Retourne le référentiel des sites exposé par l'API source."""
    http = session or build_session()
    response = http.get(
        f"{config.mock_api_url}{SITES_PATH}",
        timeout=config.request_timeout_s,
    )
    _raise_for_status(response)
    payload = response.json()
    if not isinstance(payload, list):
        raise ExtractionError(f"{SITES_PATH} devait renvoyer une liste.")
    return payload


def fetch_readings(
    config: EtlConfig,
    site_id: str,
    start_time: datetime,
    end_time: datetime,
    session: requests.Session | None = None,
) -> Iterator[dict[str, Any]]:
    """Itère les mesures d'un site sur une fenêtre, page par page.

    Le générateur évite de matérialiser tout le rattrapage en mémoire : une
    fenêtre large sur 7 sites à la minute dépasse vite le million de lignes.
    """
    http = session or build_session()
    url = f"{config.mock_api_url}{READINGS_PATH.format(site_id=site_id)}"
    offset = 0
    while True:
        response = http.get(
            url,
            params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "limit": config.batch_size,
                "offset": offset,
            },
            timeout=config.request_timeout_s,
        )
        _raise_for_status(response)
        items = response.json().get("items", [])
        if not items:
            return
        yield from items
        # L'API pagine par offset : sans avance stricte, une page pleine
        # relancerait indéfiniment la même requête.
        offset += len(items)


def fetch_current(
    config: EtlConfig,
    site_id: str,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Retourne la mesure courante d'un site, telle que servie par la source.

    Le résultat est une liste, jamais un objet seul : le reste de la chaîne
    travaille par lots, et une source qui répondrait plusieurs mesures d'un
    coup ne doit pas obliger l'appelant à distinguer les deux cas. Le timeout
    est celui du mode continu, plus court que celui du rattrapage, pour qu'un
    site muet ne mange pas la cadence des six autres.
    """
    http = session or build_session()
    path = config.current_path.format(site_id=site_id)
    response = http.get(
        f"{config.mock_api_url}{path}",
        timeout=config.poll_timeout_s,
    )
    _raise_for_status(response)
    return _as_readings(response.json(), path)


def _raise_for_status(response: requests.Response) -> None:
    """Transforme une réponse HTTP en échec explicite du run."""
    if response.status_code >= 400:
        raise ExtractionError(
            f"{response.request.method} {response.url} a répondu"
            f" {response.status_code}."
        )


def _as_readings(payload: Any, path: str) -> list[dict[str, Any]]:
    """Ramène les formes acceptables de réponse à une liste de mesures."""
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return _as_readings(items, path)
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ExtractionError(f"{path} a renvoyé une liste non exploitable.")
        return payload
    raise ExtractionError(
        f"{path} devait renvoyer une mesure ou une liste de mesures."
    )
