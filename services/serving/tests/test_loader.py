from __future__ import annotations

from types import SimpleNamespace

import numpy
import pandas as pd
import pytest

from serving import loader
from serving.loader import (
    LoadedModel,
    ModelRegistry,
    ModelUnavailable,
    UnservableModel,
)

COLUMNS = ("hour", "lag_1h")


class FakeSignatureInputs:
    def __init__(self, names, types):
        self._names = list(names)
        self._types = list(types)

    def input_names(self):
        return list(self._names)

    def numpy_types(self):
        return list(self._types)


class FakeModel:
    def __init__(self, signed: bool = True) -> None:
        inputs = (
            FakeSignatureInputs(COLUMNS, [numpy.dtype("int32"), numpy.dtype("float64")])
            if signed
            else None
        )
        self.metadata = SimpleNamespace(
            signature=SimpleNamespace(inputs=inputs) if signed else None,
            model_uuid="uuid-1",
        )
        self.seen: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame):
        self.seen.append(frame)
        return [42.0] * len(frame)


def loaded(model: FakeModel) -> LoadedModel:
    inputs = model.metadata.signature.inputs
    return LoadedModel(
        model=model,
        uri="models:/enervision_xgboost@champion",
        version="3",
        columns=tuple(inputs.input_names()),
        dtypes=dict(zip(inputs.input_names(), inputs.numpy_types(), strict=True)),
    )


class FakeVersion:
    def __init__(self, version: str = "3", tags: dict[str, str] | None = None) -> None:
        self.version = version
        self.tags = dict(tags or {})


SERVED_ENTRY = FakeVersion("3", {"residual_std": "2.5"})


def registry_serving(
    monkeypatch,
    model: FakeModel | None,
    entry: FakeVersion | None = SERVED_ENTRY,
) -> ModelRegistry:
    def load_model(uri: str):
        if model is None:
            raise RuntimeError("registre injoignable")
        return model

    monkeypatch.setattr(loader.mlflow.pyfunc, "load_model", load_model)
    monkeypatch.setattr(loader.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(loader, "_registry_entry", lambda uri: entry)
    return ModelRegistry("http://mlflow.invalid", "models:/enervision_xgboost@champion")


def test_a_resolvable_alias_makes_the_service_ready(monkeypatch) -> None:
    registry = registry_serving(monkeypatch, FakeModel())
    assert registry.load() is not None
    assert registry.is_ready


def test_an_unreachable_registry_does_not_raise(monkeypatch) -> None:
    registry = registry_serving(monkeypatch, None)
    assert registry.load() is None
    assert not registry.is_ready


def test_serving_without_a_model_is_refused_explicitly(monkeypatch) -> None:
    registry = registry_serving(monkeypatch, None)
    registry.load()
    with pytest.raises(ModelUnavailable, match="models:/"):
        registry.current()


def test_a_model_without_a_signature_is_not_served(monkeypatch) -> None:
    registry = registry_serving(monkeypatch, FakeModel(signed=False))
    assert registry.load() is None


def test_the_input_columns_come_from_the_signature(monkeypatch) -> None:
    registry = registry_serving(monkeypatch, FakeModel())
    registry.load()
    assert registry.input_columns() == list(COLUMNS)


class TestConform:
    def test_the_columns_are_reordered(self) -> None:
        model = FakeModel()
        frame = pd.DataFrame({"lag_1h": [50.0], "hour": [8]})
        assert list(loaded(model).conform(frame).columns) == list(COLUMNS)

    def test_the_types_follow_the_signature(self) -> None:
        model = FakeModel()
        frame = pd.DataFrame({"hour": [8], "lag_1h": [50.0]})
        conformed = loaded(model).conform(frame)
        assert conformed["hour"].dtype == numpy.dtype("int32")

    def test_a_missing_variable_is_refused(self) -> None:
        model = FakeModel()
        with pytest.raises(UnservableModel, match="lag_1h"):
            loaded(model).conform(pd.DataFrame({"hour": [8]}))

    def test_predict_presents_the_conformed_frame(self) -> None:
        model = FakeModel()
        loaded(model).predict(pd.DataFrame({"lag_1h": [50.0], "hour": [8]}))
        assert list(model.seen[0].columns) == list(COLUMNS)


class TestAliasParsing:
    def test_an_alias_uri_is_decomposed(self) -> None:
        assert loader._parse_alias("models:/enervision_xgboost@champion") == (
            "enervision_xgboost",
            "champion",
        )

    def test_a_plain_path_has_no_alias(self) -> None:
        assert loader._parse_alias("runs:/abc/model") == ("", "")

    def test_a_registry_without_an_alias_has_none(self) -> None:
        assert loader._parse_alias("models:/enervision_xgboost/3") == ("", "")

    def test_the_internal_identifier_takes_over_without_a_registry(self) -> None:
        assert loader._version_of(None, FakeModel(), "runs:/abc/model") == "uuid-1"

    def test_the_registry_version_wins_over_the_identifier(self) -> None:
        entry = FakeVersion(version="7")
        assert loader._version_of(entry, FakeModel(), "models:/m@champion") == "7"


class TestResidualStd:
    def test_the_tag_is_read(self, monkeypatch) -> None:
        registry = registry_serving(
            monkeypatch, FakeModel(), FakeVersion("3", {"residual_std": "4.25"})
        )
        assert registry.load().residual_std == 4.25

    def test_a_version_without_the_tag_serves_without_bounds(self, monkeypatch) -> None:
        registry = registry_serving(monkeypatch, FakeModel(), FakeVersion("3", {}))
        assert registry.load().residual_std is None

    def test_an_unreadable_tag_serves_without_bounds(self, monkeypatch) -> None:
        registry = registry_serving(
            monkeypatch, FakeModel(), FakeVersion("3", {"residual_std": "large"})
        )
        assert registry.load().residual_std is None

    def test_a_null_spread_is_refused(self, monkeypatch) -> None:
        registry = registry_serving(
            monkeypatch, FakeModel(), FakeVersion("3", {"residual_std": "0"})
        )
        assert registry.load().residual_std is None
