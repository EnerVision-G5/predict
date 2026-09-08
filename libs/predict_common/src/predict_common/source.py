# **********************************************************************
# * Nom     : source.py                                                *
# * Type    : Module                                                   *
# * Sujet   : Client HTTP de la source amont : pagination, cadence,    *
# *   reprise sur échec                                                *
# * Service : predict_common (bibliothèque partagée)                   *
# **********************************************************************

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

# Nombre maximal de mesures demandées en une requête.
MAX_PAGE_SIZE = 1000

# Durée minimale acceptée pour une simulation de pic.
MIN_SPIKE_MINUTES = 1
# Durée maximale acceptée pour une simulation de pic.
MAX_SPIKE_MINUTES = 240
# Durée retenue quand l'appelant n'en propose aucune.
DEFAULT_SPIKE_MINUTES = 30

logger = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """Classe : SourceError
    Description : La source n'a pas répondu, ou a répondu autre chose que ce
      qui était attendu.
    """


class RetryExhausted(SourceError):
    """Classe : RetryExhausted
    Description : Toutes les tentatives ont échoué sur le même appel.
    """


@dataclass
class RateLimiter:
    """Classe : RateLimiter
    Description : Espace les requêtes pour tenir la cadence que la source
      accepte.
    """
    requests_per_second: float
    sleep: Callable[[float], Any] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _next_at: float = field(default=0.0, init=False)

    def wait(self) -> None:
        """Méthode : wait
        Description : Attend, si besoin, le temps qui reste avant la requête
          suivante.
        """
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
    """Classe : SourceSettings
    Description : Adresses, délais et cadence de la source, lus dans la
      configuration.
    """
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
    timezone: str = DEFAULT_SOURCE_TIMEZONE

    def with_page_size(self, page_size: int) -> SourceSettings:
        """Méthode : with_page_size
        Description : Rend les mêmes réglages avec une autre taille de page,
          bornée.
        """
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(
                f"--limit doit être compris entre 1 et {MAX_PAGE_SIZE},"
                f" reçu {page_size}."
            )
        return replace(self, page_size=page_size)

    @classmethod
    def from_config(cls, config: Config) -> SourceSettings:
        """Méthode : from_config
        Description : Construit les réglages depuis le bloc source de la
          configuration.
        """
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
    """Classe : SourceClient
    Description : Client de la source : une requête par appel, avec cadence et
      reprise.
    """
    def __init__(
        self,
        settings: SourceSettings,
        client: httpx.Client | None = None,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        """Méthode : __init__
        Description : Prépare le client HTTP, sa cadence et ses délais.
        """
        self.settings = settings
        self.sleep = sleep
        self.limiter = RateLimiter(settings.rate_limit_rps, sleep=sleep)
        self._client = client or httpx.Client(
            base_url=settings.base_url,
            headers={"Accept": "application/json"},
            timeout=settings.timeout_s,
        )

    def close(self) -> None:
        """Méthode : close
        Description : Ferme le client HTTP sous-jacent.
        """
        self._client.close()

    def __enter__(self) -> SourceClient:
        """Méthode : __enter__
        Description : Rend le client lui-même, pour un usage en bloc with.
        """
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Méthode : __exit__
        Description : Ferme le client à la sortie du bloc, sans avaler
          d'exception.
        """
        self.close()
        return False

    def fetch_sites(self) -> list[dict[str, Any]]:
        """Méthode : fetch_sites
        Description : Lit le référentiel des sites servi par la source.
        """
        path = self.settings.sites_path
        payload = self._get_json(path, label="référentiel des sites")
        if not isinstance(payload, list):
            raise SourceError(f"{path} devait renvoyer une liste.")
        return payload

    def site_ids(self) -> list[str]:
        """Méthode : site_ids
        Description : Ne retient que les identifiants du référentiel des sites.
        """
        return [
            str(site["site_id"]) for site in self.fetch_sites() if "site_id" in site
        ]

    def iter_readings(
        self,
        site_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Iterator[dict[str, Any]]:
        """Méthode : iter_readings
        Description : Parcourt les mesures d'un site par fenêtres, en signalant
          les pages incomplètes.
        """
        path = self.settings.readings_path
        span = timedelta(minutes=self.settings.page_size)
        cursor = start_time
        while cursor < end_time:
            window_end = min(cursor + span, end_time)
            minutes = _minutes_between(cursor, window_end)
            if minutes < 1:
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
        """Méthode : fetch_current
        Description : Lit la mesure courante d'un site, avec le délai court de
          la collecte.
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
        """Méthode : fetch_alerts
        Description : Lit les alertes actives, éventuellement filtrées par site
          ou gravité.
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
        """Méthode : fetch_sensors_status
        Description : Lit l'état des capteurs tel que la source le présente.
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
        """Méthode : simulate_spike
        Description : Demande à la source de simuler un pic sur un site, durée
          bornée.
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
        """Méthode : _get_json
        Description : Rejoue l'appel jusqu'à épuisement des tentatives, en
          espaçant les essais.
        """
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
        """Méthode : _request
        Description : Émet une requête après attente de cadence et refuse tout
          statut d'erreur.
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
    """Méthode : _minutes_between
    Description : Compte les minutes entières séparant deux instants.
    """
    return int((end - start).total_seconds() // 60)


def _as_readings(payload: Any, path: str) -> list[dict[str, Any]]:
    """Méthode : _as_readings
    Description : Ramène une réponse à une liste de mesures, quelle que soit
      son enveloppe.
    """
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
