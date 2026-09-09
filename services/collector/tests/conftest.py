from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from predict_common.source import SourceClient, SourceSettings


@pytest.fixture
def settings() -> SourceSettings:
    return SourceSettings(
        base_url="http://mock.invalid",
        sites_path="/api/v1/sites",
        readings_path="/api/v1/readings",
        current_path="/api/v1/sites/{site_id}/current",
        simulate_spike_path="/api/v1/simulate/spike/{site_id}",
        alerts_path="/api/v1/alerts",
        sensors_status_path="/api/v1/sensors/status",
        page_size=2,
        timeout_s=1.0,
        poll_timeout_s=0.5,
        retries=1,
        backoff_s=0.0,
        rate_limit_rps=0.0,
    )


@pytest.fixture
def make_client(settings: SourceSettings) -> Callable[..., SourceClient]:
    def build(
        handler: Callable[[httpx.Request], httpx.Response],
        overrides: dict[str, Any] | None = None,
    ) -> SourceClient:
        import dataclasses

        applied = dataclasses.replace(settings, **(overrides or {}))
        transport = httpx.MockTransport(handler)
        client = httpx.Client(
            base_url=applied.base_url,
            transport=transport,
            timeout=applied.timeout_s,
        )
        return SourceClient(applied, client=client, sleep=lambda _: None)

    return build


@pytest.fixture
def make_reading() -> Callable[..., dict[str, Any]]:
    def build(
        timestamp: str,
        site_id: str = "SITE001",
        **overrides: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": timestamp,
            "site_id": site_id,
            "consumption_kw": 87.34,
            "consumption_kwh": 1.45,
            "voltage_v": 230.1,
            "current_a": 12.5,
            "power_factor": 0.95,
            "temperature_celsius": 21.3,
            "humidity_percent": 48.0,
            "null_reasons": [],
            "data_quality": "good",
        }
        record.update(overrides)
        return record

    return build
