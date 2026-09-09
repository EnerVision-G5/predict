from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import pytest

from predict_common.db import SOURCE_COLUMNS
from predict_common.schemas import (
    NUMERIC_COLUMNS,
    SITE_COLUMN,
    TIMESTAMP_COLUMN,
)


@pytest.fixture
def make_reading() -> Callable[..., dict[str, Any]]:
    def build(
        timestamp: str,
        site_id: str = "SITE001",
        **overrides: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            TIMESTAMP_COLUMN: timestamp,
            SITE_COLUMN: site_id,
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


@pytest.fixture
def make_raw() -> Callable[[Iterable[dict[str, Any]]], pd.DataFrame]:
    def build(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(list(records), columns=list(SOURCE_COLUMNS))
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[TIMESTAMP_COLUMN], utc=True, errors="coerce"
        )
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    return build
