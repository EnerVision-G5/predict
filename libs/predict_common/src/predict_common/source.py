"""Client de l'API Mock IoT : pagination, reprises, débit borné.

Deux services parlent à la source : le collecteur, qui en tire les mesures, et
le service d'inférence, qui relaie le référentiel et la simulation de pic pour
le compte de l'API métier — laquelle ne connaît pas la source. D'où la place de
ce module dans la bibliothèque partagée plutôt que dans l'un des deux : deux
clients donneraient deux façons de lire la même API, et un service ne peut pas
importer le code d'un autre.

Tout ce qui relève du réseau est ici, et rien d'autre : pas de conversion en
tableau, pas de règle de qualité, pas d'écriture. Ce module rend des
dictionnaires tels que la source les sert.

Trois mécanismes, et une raison pour chacun.

La lecture est un générateur. Une fenêtre de rattrapage de trois mois sur
sept sites à la minute dépasse le million de lignes : les matérialiser toutes
avant d'en écrire une seule ferait sortir le processus sur un défaut de
mémoire, très loin de la ligne qui l'a causé.

Elle découpe le temps, et ne pagine pas. `/api/v1/readings` n'est pas une API
paginée : sa documentation définit `limit` comme le NOMBRE DE RÉSULTATS
(1–1000), et la source rend toujours ce nombre-là, réparti uniformément entre
`start_time` et `end_time` puis arrondi à la minute. `limit` est donc une
RÉSOLUTION, pas une taille de page.

Deux conséquences, et le code découle des deux.

Demander une journée avec `limit=1000` ne rend pas les mille premières minutes,
mais mille points étalés sur 1440 minutes — une série trouée, à un point toutes
les 86 secondes, qu'aucune erreur ne signale. Une fenêtre ne peut donc pas être
plus large que `limit` minutes, et `limit` doit valoir exactement le nombre de
minutes demandées.

Et comme la source rend toujours `limit` résultats, une réponse n'est JAMAIS
incomplète : « page pleine, donc il en reste » est une condition qui ne devient
jamais fausse. Une pagination par curseur redemandait alors la même fenêtre en
la divisant par mille à chaque tour, jusqu'à ramper d'une microseconde par
requête sans plus rien collecter.

Le découpage supprime les deux problèmes : chaque fenêtre est demandée une
fois, à sa résolution native, et les fenêtres pavent la période sans se
recouvrir.

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
from predict_common.timestamps import DEFAULT_SOURCE_TIMEZONE

# Plafond de `limit` imposé par la source : au-delà, elle répond 422. Le
# refuser ici plutôt que de le découvrir en réponse évite de partir sur un
# rattrapage de trois mois qui échouera à la première page.
MAX_PAGE_SIZE = 1000

# Bornes de `duration_minutes` déclarées par la source pour la simulation de
# pic. Hors de cet intervalle elle répond 422.
MIN_SPIKE_MINUTES = 1
MAX_SPIKE_MINUTES = 240
DEFAULT_SPIKE_MINUTES = 30

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
    simulate_spike_path: str
    alerts_path: str
    sensors_status_path: str
    page_size: int
    timeout_s: float
    poll_timeout_s: float
    retries: int
    backoff_s: float
    rate_limit_rps: float
    # Fuseau des horodatages que la source envoie sans le leur. Il décrit la
    # source, pas une préférence d'affichage : `/current` sert l'heure locale
    # de sa machine depuis le 8 septembre 2026, et rien dans la réponse ne le
    # dit. Voir `predict_common.timestamps`.
    timezone: str = DEFAULT_SOURCE_TIMEZONE

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
            simulate_spike_path=config.get_str("source.simulate_spike_path"),
            alerts_path=config.get_str("source.alerts_path"),
            sensors_status_path=config.get_str("source.sensors_status_path"),
            page_size=config.get_int("source.page_size"),
            timeout_s=config.get_float("source.timeout_s"),
            poll_timeout_s=config.get_float("collector.poll_timeout_s"),
            retries=config.get_int("source.retries"),
            backoff_s=config.get_float("source.backoff_s"),
            rate_limit_rps=config.get_float("source.rate_limit_rps"),
            timezone=config.get_str("source.timezone", DEFAULT_SOURCE_TIMEZONE),
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
        """Itère les mesures d'un site, fenêtre par fenêtre.

        La période est découpée en tranches d'au plus `page_size` minutes, et
        chaque tranche est demandée avec `limit` égal à son nombre de minutes.
        C'est ce qui rend la série à sa résolution native : la source répartit
        `limit` points sur la fenêtre, donc `limit` minutes sur une fenêtre de
        `limit` minutes font exactement un point par minute.

        Les tranches pavent la période sans se recouvrir. La source pose son
        premier point sur `start_time` et espace les suivants de
        (fin - début) / limit : la dernière minute d'une tranche est donc celle
        qui précède le début de la suivante.
        """
        path = self.settings.readings_path
        span = timedelta(minutes=self.settings.page_size)
        cursor = start_time
        while cursor < end_time:
            window_end = min(cursor + span, end_time)
            minutes = _minutes_between(cursor, window_end)
            if minutes < 1:
                # Reliquat de moins d'une minute : la source ne sait pas le
                # servir, et `limit=0` lui vaudrait un 422.
                return
            payload = self._get_json(
                path,
                params={
                    "site_id": site_id,
                    "start_time": cursor.isoformat(),
                    "end_time": window_end.isoformat(),
                    "limit": minutes,
                },
                label=f"site {site_id} de {cursor.isoformat()}",
            )
            items = _as_readings(payload, path)
            if len(items) < minutes:
                # La source rend normalement autant de résultats que demandé.
                # En rendre moins n'est pas fatal — la journée sera simplement
                # trouée — mais doit se voir, sans quoi un changement de
                # comportement de la source produirait une série amputée que
                # seul un compte en aval finirait par trahir.
                logger.warning(
                    "site %s : %d mesure(s) reçue(s) pour %d minute(s)"
                    " demandée(s) à partir de %s",
                    site_id,
                    len(items),
                    minutes,
                    cursor.isoformat(),
                )
            yield from items
            cursor = window_end

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

    def fetch_alerts(
        self,
        site_id: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retourne les alertes actives, filtrées à la demande.

        La source ne sert que ce qui est en cours : une réponse vide est une
        réponse valable, et non une panne. C'est l'appelant qui décide d'en
        faire un journal, parce que c'est lui qui écrit.
        """
        params = {
            name: value
            for name, value in (("site_id", site_id), ("severity", severity))
            if value is not None
        }
        payload = self._get_json(
            self.settings.alerts_path,
            params=params or None,
            label="alertes actives",
        )
        if not isinstance(payload, list):
            raise SourceError(f"{self.settings.alerts_path} devait renvoyer une liste.")
        return [item for item in payload if isinstance(item, dict)]

    def fetch_sensors_status(self) -> dict[str, Any]:
        """Retourne l'état des capteurs, indexé par site tel que la source le sert.

        Un objet et non une liste : la source indexe par identifiant de site,
        et le remettre à plat ici obligerait l'appelant à refaire le lien.
        """
        path = self.settings.sensors_status_path
        payload = self._get_json(path, label="état des capteurs")
        if not isinstance(payload, dict):
            raise SourceError(f"{path} devait renvoyer un objet.")
        return payload

    def simulate_spike(
        self,
        site_id: str,
        duration_minutes: int,
    ) -> dict[str, Any]:
        """Demande à la source de simuler un pic de consommation sur un site.

        Seule écriture de tout ce module, et la seule que la source expose.
        Elle n'est pas retentée comme l'est une lecture : rejouer un POST
        déclencherait un second pic, et deux pics qui se recouvrent ne sont
        pas ce qu'on a demandé. Un échec remonte donc au premier essai.

        La borne sur la durée est celle de la source : au-delà elle répond
        422, et le dire ici évite d'aller le découvrir sur le réseau.
        """
        if not MIN_SPIKE_MINUTES <= duration_minutes <= MAX_SPIKE_MINUTES:
            raise ValueError(
                "duration_minutes doit être compris entre"
                f" {MIN_SPIKE_MINUTES} et {MAX_SPIKE_MINUTES},"
                f" reçu {duration_minutes}."
            )
        path = self.settings.simulate_spike_path.format(site_id=site_id)
        payload = self._request(
            path,
            {"duration_minutes": duration_minutes},
            None,
            method="POST",
        )
        if not isinstance(payload, dict):
            raise SourceError(f"{path} devait renvoyer un objet.")
        return payload

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
        method: str = "GET",
    ) -> Any:
        """Envoie une requête et rend son corps, un statut d'erreur exclu.

        Le débit reste borné quelle que soit la méthode : la limite protège la
        source, et un POST la sollicite autant qu'un GET.
        """
        self.limiter.wait()
        response = self._client.request(
            method,
            path,
            params=params,
            timeout=timeout_s if timeout_s is not None else self.settings.timeout_s,
        )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise SourceError(
                f"{method} {response.url} a répondu {response.status_code}."
            )
        return response.json()


def _minutes_between(start: datetime, end: datetime) -> int:
    """Nombre de minutes ENTIÈRES d'une fenêtre, soit le `limit` à demander.

    Tronqué et non arrondi : demander une minute de plus que la fenêtre n'en
    porte resserrerait l'espacement sous la minute, et la source rendrait deux
    points dans la même minute plutôt qu'un par minute.
    """
    return int((end - start).total_seconds() // 60)


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
