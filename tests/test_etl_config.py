"""Chargement de la configuration ETL depuis l'environnement."""

import pytest

from etl.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MOCK_API_URL,
    ConfigError,
    load_config,
    read_float,
    read_int,
    read_text,
)

MINIMAL_ENV = {"DATABASE_URL": "postgresql+psycopg://u:p@localhost:5432/db"}


def test_load_config_applies_defaults() -> None:
    config = load_config(dict(MINIMAL_ENV))
    assert config.database_url == MINIMAL_ENV["DATABASE_URL"]
    assert config.mock_api_url == DEFAULT_MOCK_API_URL
    assert config.batch_size == DEFAULT_BATCH_SIZE


def test_load_config_reads_overrides() -> None:
    env = dict(MINIMAL_ENV) | {
        "MOCK_API_URL": "http://mock:9000",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_EXPERIMENT": "essai",
        "ETL_BATCH_SIZE": "250",
        "ETL_REQUEST_TIMEOUT_S": "5",
    }
    config = load_config(env)
    assert config.mock_api_url == "http://mock:9000"
    assert config.mlflow_tracking_uri == "http://mlflow:5000"
    assert config.mlflow_experiment == "essai"
    assert config.batch_size == 250
    assert config.request_timeout_s == 5.0


def test_load_config_requires_database_url() -> None:
    with pytest.raises(ConfigError):
        load_config({"MOCK_API_URL": "http://mock:9000"})


def test_read_int_returns_default_when_absent() -> None:
    assert read_int({}, "ETL_BATCH_SIZE", 42) == 42
    assert read_int({"ETL_BATCH_SIZE": ""}, "ETL_BATCH_SIZE", 42) == 42


def test_read_int_rejects_unparsable_value() -> None:
    with pytest.raises(ConfigError):
        read_int({"ETL_BATCH_SIZE": "beaucoup"}, "ETL_BATCH_SIZE", 42)


def test_read_float_accepts_a_decimal_value() -> None:
    env = {"ETL_REQUEST_TIMEOUT_S": "2.5"}
    assert read_float(env, "ETL_REQUEST_TIMEOUT_S", 30.0) == 2.5


def test_read_float_rejects_an_unparsable_value() -> None:
    with pytest.raises(ConfigError):
        read_float({"ETL_REQUEST_TIMEOUT_S": "vite"}, "ETL_REQUEST_TIMEOUT_S", 1.0)


def test_read_text_falls_back_on_a_blank_value() -> None:
    assert read_text({"MOCK_API_URL": " "}, "MOCK_API_URL", "http://d") == "http://d"
    assert read_text({}, "MOCK_API_URL", "http://d") == "http://d"
