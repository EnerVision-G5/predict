from __future__ import annotations

from pathlib import Path

import pytest

from predict_common.config import (
    Config,
    ConfigError,
    config_directory,
    load_config,
)

BASE = """
storage:
  root: ${PREDICT_STORAGE_ROOT:-data}
source:
  base_url: ${MOCK_API_URL:-http://localhost:8000}
  page_size: 1000
  retries: 3
etl:
  lag_hours: [1, 24, 168]
  database_url: ${DATABASE_URL:-}
training:
  params:
    n_estimators: 600
    max_depth: 6
"""

LOCAL = """
source:
  retries: 1
training:
  params:
    n_estimators: 120
"""


@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    (tmp_path / "base.yaml").write_text(BASE, encoding="utf-8")
    (tmp_path / "local.yaml").write_text(LOCAL, encoding="utf-8")
    return tmp_path


def test_load_config_reads_the_base_layer(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get_int("source.page_size") == 1000


def test_load_config_applies_the_local_layer_by_default(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.env_name == "local"
    assert config.get_int("source.retries") == 1


def test_load_config_merges_blocks_key_by_key(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get_int("training.params.n_estimators") == 120
    assert config.get_int("training.params.max_depth") == 6


def test_load_config_ignores_a_layer_that_does_not_exist(conf_dir: Path) -> None:
    config = load_config(env={"PREDICT_ENV": "prod"}, directory=conf_dir)
    assert config.env_name == "prod"
    assert config.get_int("source.retries") == 3


def test_load_config_expands_an_environment_variable(conf_dir: Path) -> None:
    config = load_config(env={"MOCK_API_URL": "http://mock:9000"}, directory=conf_dir)
    assert config.get_str("source.base_url") == "http://mock:9000"


def test_load_config_falls_back_on_the_declared_default(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get_str("source.base_url") == "http://localhost:8000"


def test_load_config_treats_an_empty_variable_as_absent(conf_dir: Path) -> None:
    config = load_config(env={"MOCK_API_URL": "   "}, directory=conf_dir)
    assert config.get_str("source.base_url") == "http://localhost:8000"


def test_load_config_refuses_a_missing_required_variable(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "storage:\n  root: ${PREDICT_ROOT}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="PREDICT_ROOT"):
        load_config(env={}, directory=tmp_path)


def test_load_config_refuses_a_missing_base_layer(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(env={}, directory=tmp_path)


def test_get_raises_on_an_unknown_key(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    with pytest.raises(ConfigError, match="source.absente"):
        config.get("source.absente")


def test_get_returns_the_default_when_one_is_given(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get("source.absente", "repli") == "repli"


def test_get_str_refuses_an_empty_value() -> None:
    config = Config(values={"a": "   "})
    with pytest.raises(ConfigError):
        config.get_str("a")


def test_get_optional_str_accepts_an_assumed_absence(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get_optional_str("etl.database_url") == ""


def test_get_int_refuses_an_unreadable_number() -> None:
    config = Config(values={"a": "beaucoup"})
    with pytest.raises(ConfigError, match="nombre"):
        config.get_int("a")


def test_get_int_refuses_a_boolean() -> None:
    config = Config(values={"a": True})
    with pytest.raises(ConfigError):
        config.get_int("a")


def test_get_int_list_reads_a_list(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert config.get_int_list("etl.lag_hours") == [1, 24, 168]


def test_get_int_list_refuses_a_lone_value() -> None:
    config = Config(values={"a": 24})
    with pytest.raises(ConfigError, match="liste"):
        config.get_int_list("a")


def test_get_int_list_splits_a_value_from_the_environment() -> None:
    assert Config(values={"a": "1, 24, 168"}).get_int_list("a") == [1, 24, 168]
    assert Config(values={"a": "24"}).get_int_list("a") == [24]


def test_get_int_list_refuses_an_empty_environment_value() -> None:
    with pytest.raises(ConfigError, match="vide"):
        Config(values={"a": " , "}).get_int_list("a")


def test_section_returns_a_block(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    assert dict(config.section("training.params"))["max_depth"] == 6


def test_section_refuses_a_leaf(conf_dir: Path) -> None:
    config = load_config(env={}, directory=conf_dir)
    with pytest.raises(ConfigError):
        config.section("source.page_size")


def test_config_directory_honours_the_override(tmp_path: Path) -> None:
    assert config_directory({"PREDICT_CONF_DIR": str(tmp_path)}) == tmp_path


@pytest.mark.parametrize("word", ["true", "TRUE", "1", "yes", "on"])
def test_get_bool_lit_les_ecritures_vraies(word: str) -> None:
    assert Config(values={"a": {"b": word}}).get_bool("a.b") is True


@pytest.mark.parametrize("word", ["false", "False", "0", "no", "off"])
def test_get_bool_lit_les_ecritures_fausses(word: str) -> None:
    assert Config(values={"a": {"b": word}}).get_bool("a.b") is False


def test_get_bool_accepte_un_booleen_deja_type() -> None:
    assert Config(values={"a": {"b": False}}).get_bool("a.b") is False


def test_get_bool_refuse_ce_qu_il_ne_sait_pas_lire() -> None:
    with pytest.raises(ConfigError, match="booléen"):
        Config(values={"a": {"b": "oui"}}).get_bool("a.b")


def test_get_bool_rend_le_defaut_quand_la_cle_manque() -> None:
    assert Config(values={}).get_bool("serving.auth_enabled", True) is True
