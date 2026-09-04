"""Client de l'API Mock IoT : pagination, reprises, débit borné.

Le collecteur est le seul service qui parle à la source. Tout ce qui relève du
réseau est donc ici, et rien d'autre : pas de conversion en tableau, pas de
règle de qualité, pas d'écriture. Ce module rend des dictionnaires tels que la
source les sert.

Trois mécanismes, et une raison pour chacun.

La pagination est un générateur. Une fenêtre de rattrapage de trois mois sur
sept sites à la minute dépasse le million de lignes : les matérialiser toutes
avant d'en écrire une seule ferait sortir le processus sur un défaut de
mémoire, très loin de la ligne qui l'a causé.

Elle avance par le temps, et non par un rang. La source ne connaît pas
d'`offset` : elle rend au plus `limit` mesures à partir de `start_time`.
Réclamer la page suivante consiste donc à redemander la même fenêtre à partir
du dernier horodatage reçu. Une pagination par rang, ici, redemanderait
indéfiniment la même page — la boucle ne s'arrêterait jamais, et rien dans le
journal ne dirait pourquoi.

Les reprises absorbent la micro-coupure, et elle seule. L'attente croît avec le
rang de la tentative : une coupure qui dure ne se règle pas en insistant à la
même cadence. Au-delà, l'échec remonte — c'est à l'appelant, qui sait s'il
rattrape une journée ou s'il tient une cadence, de décider quoi en faire.

Le débit est borné en sortie. La limite protège la source, pas le collecteur :
un rattrapage de trois mois lui envoie des milliers de pages, et rien du côté
de l'API Mock ne l'en empêcherait.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

import httpx

from predict_common.config import Config

# Plafond de `limit` imposé par la source : au-delà, elle répond 422. Le
# refuser ici plutôt que de le découvrir en réponse évite de partir sur un
# rattrapage de trois mois qui échouera à la première page.
MAX_PAGE_SIZE = 1000

# Nom de l'horodatage tel que la source le sert. Le collecteur le traduit à
# l'écriture ; ici, il ne sert qu'à savoir où reprendre la pagination.
SOURCE_TIMESTAMP_FIELD = "timestamp"

logger = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """La source a répondu autre chose que ce que le contrat annonce."""


class RetryExhausted(SourceError):
    """Toutes les tentatives d'un appel ont échoué."""


@dataclass
class RateLimiter:
    """Espace les appels sortants d'un intervalle minimal.

    Un débit nul lève la limite, ce qui est le réglage du poste : brider les
    appels vers une API Mock qui tourne sur la même machine ne protégerait
    personne et allongerait chaque test.
    """

    requests_per_second: float
    sleep: Callable[[float], Any] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _next_at: float = field(default=0.0, init=False)

    def wait(self) -> None:
        """Attend, si nécessaire, avant de laisser passer l'appel suivant."""
        if self.requests_per_second <= 0:
            return
        interval_s = 1.0 / self.requests_per_second
        now = self.clock()
        delay_s = self._next_at - now
        if delay_s > 0:
            self.sleep(delay_s)
            now += delay_s
        self._next_at = now + interval_s


@dataclass(frozen=True)
class SourceSettings:
    """Ce que le collecteur doit savoir de la source pour l'interroger."""

    base_url: str
    sites_path: str
    readings_path: str
    current_path: str
    page_size: int
    timeout_s: float
    poll_timeout_s: float
    retries: int
    backoff_s: float
    rate_limit_rps: float

    def with_page_size(self, page_size: int) -> SourceSettings:
        """Retourne les mêmes réglages avec la taille de page demandée.

        La borne est celle de la source, pas une préférence : une valeur plus
        grande ferait répondre 422 à chaque page, et l'exploitant chercherait
        la panne du côté du réseau.
        """
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(
                f"--limit doit être compris entre 1 et {MAX_PAGE_SIZE},"
                f" reçu {page_size}."
            )
        return replace(self, page_size=page_size)

    @classmethod
    def from_config(cls, config: Config) -> SourceSettings:
        """Lit le bloc `source` et la cadence du collecteur."""
        return cls(
            base_url=config.get_str("source.base_url").rstrip("/"),
            sites_path=config.get_str("source.sites_path"),
            readings_path=config.get_str("source.readings_path"),
            current_path=config.get_str("source.current_path"),
            page_size=config.get_int("source.page_size"),
            timeout_s=config.get_float("source.timeout_s"),
            poll_timeout_s=config.get_float("collector.poll_timeout_s"),
            retries=config.get_int("source.retries"),
            backoff_s=config.get_float("source.backoff_s"),
            rate_limit_rps=config.get_float("source.rate_limit_rps"),
        )


class SourceClient:
    """Accès à l'API Mock IoT, partagé par le rattrapage et le poller.

    Le client tient une connexion ouverte pour toute sa durée de vie. Sur un
    rattrapage de plusieurs milliers de pages, c'est la différence entre une
    poignée de poignées de main TLS et une par page.
    """

    def __init__(
        self,
        settings: SourceSettings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        self.settings = settings
        self.sleep = sleep
        self.limiter = RateLimiter(settings.rate_limit_rps, sleep=sleep)
        self._client = client or httpx.Client(
            base_url=settings.base_url,
            headers={"Accept": "application/json"},
            timeout=settings.timeout_s,
        )

    def close(self) -> None:
        """Rend la connexion. Un processus long n'en ouvre qu'une."""
        self._client.close()

    def __enter__(self) -> SourceClient:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False

    def fetch_sites(self) -> list[dict[str, Any]]:
        """Retourne le référentiel des sites exposé par la source."""
        path = self.settings.sites_path
        payload = self._get_json(path, label="référentiel des sites")
        if not isinstance(payload, list):
            raise SourceError(f"{path} devait renvoyer une liste.")
        return payload

    def site_ids(self) -> list[str]:
        """Retourne les seuls identifiants du référentiel."""
        return [
            str(site["site_id"]) for site in self.fetch_sites() if "site_id" in site
        ]

    def iter_readings(
        self,
        site_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Iterator[dict[str, Any]]:
        """Itère les mesures d'un site sur une fenêtre, page par page.

        Une page pleine signifie que la source a tronqué : la suivante repart
        du dernier horodatage reçu. Une page incomplète signifie qu'il n'y a
        plus rien, et la boucle s'arrête là.
        """
        path = self.settings.readings_path
        limit = self.settings.page_size
        cursor = start_time
        previous: datetime | None = None
        while cursor <= end_time:
            payload = self._get_json(
                path,
                params={
                    "site_id": site_id,
                    "start_time": cursor.isoformat(),
                    "end_time": end_time.isoformat(),
                    "limit": limit,
                },
                label=f"site {site_id} depuis {cursor.isoformat()}",
            )
            items = _as_readings(payload, path)
            if not items:
                return
            yield from items
            if len(items) < limit:
                return
            latest = _latest_timestamp(items)
            if latest is None or (previous is not None and latest <= previous):
                # Sans avance stricte, la même fenêtre serait redemandée sans
                # fin. Mieux vaut une journée incomplète, et le dire, qu'un
                # processus qui tourne indéfiniment sans rien produire.
                logger.warning(
                    "site %s : la source ne progresse plus à %s, page"
                    " abandonnée",
                    site_id,
                    cursor.isoformat(),
                )
                return
            previous = latest
            # Une microseconde après la dernière mesure reçue : la source
            # borne inclusivement, et repartir d'elle la renverrait en double.
            # Le doublon serait absorbé plus loin, mais autant ne pas le créer.
            cursor = latest + timedelta(microseconds=1)

    def fetch_current(self, site_id: str) -> list[dict[str, Any]]:
        """Retourne la mesure courante d'un site, toujours sous forme de liste.

        Une liste et jamais un objet seul : le reste de la chaîne travaille par
        lots, et une source qui répondrait plusieurs mesures d'un coup ne doit
        pas obliger l'appelant à distinguer les deux cas.
        """
        path = self.settings.current_path.format(site_id=site_id)
        payload = self._get_json(
            path,
            label=f"site {site_id} mesure courante",
            timeout_s=self.settings.poll_timeout_s,
        )
        return _as_readings(payload, path)

    def _get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        label: str = "",
        timeout_s: float | None = None,
    ) -> Any:
        """Exécute un GET, en retentant les échecs dont on peut se remettre."""
        attempts = max(self.settings.retries, 0) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._request(path, params, timeout_s)
            except (httpx.HTTPError, SourceError) as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                delay_s = self.settings.backoff_s * attempt
                logger.warning(
                    "%s : tentative %d/%d échouée (%s), nouvel essai dans %.1f s",
                    label or path,
                    attempt,
                    attempts,
                    exc,
                    delay_s,
                )
                self.sleep(delay_s)
        raise RetryExhausted(
            f"{label or path} : {attempts} tentative(s) échouée(s)."
        ) from last_error

    def _request(
        self,
        path: str,
        params: dict[str, Any] | None,
        timeout_s: float | None,
    ) -> Any:
        """Envoie une requête et rend son corps, un statut d'erreur exclu."""
        self.limiter.wait()
        response = self._client.get(
            path,
            params=params,
            timeout=timeout_s if timeout_s is not None else self.settings.timeout_s,
        )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise SourceError(
                f"GET {response.url} a répondu {response.status_code}."
            )
        return response.json()


def _latest_timestamp(items: list[dict[str, Any]]) -> datetime | None:
    """Retourne l'horodatage le plus récent d'une page, pour la reprendre.

    Une mesure sans horodatage lisible n'est pas un point de reprise : elle
    est ignorée ici, et c'est l'écriture qui la comptera comme écartée.
    """
    stamps: list[datetime] = []
    for item in items:
        raw = item.get(SOURCE_TIMESTAMP_FIELD)
        if not raw:
            continue
        try:
            stamps.append(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
        except ValueError:
            continue
    return max(stamps) if stamps else None


def _as_readings(payload: Any, path: str) -> list[dict[str, Any]]:
    """Ramène les formes acceptables de réponse à une liste de mesures."""
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return _as_readings(items, path)
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise SourceError(f"{path} a renvoyé une liste non exploitable.")
        return payload
    raise SourceError(f"{path} devait renvoyer une mesure ou une liste de mesures.")
