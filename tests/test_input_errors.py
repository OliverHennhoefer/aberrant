"""Every component rejects a malformed event with ``ValidationError``."""

import math
from collections.abc import Callable

import pytest

from aberrant.base import ValidationError
from aberrant.catalog import (
    MODEL_CATALOG,
    TRANSFORMER_CATALOG,
    create_model,
    create_transformer,
)
from aberrant.drift import ADWIN, KSWIN, PageHinkley
from aberrant.forecast import HoltWinters, SeasonalNaive

_MODEL_PARAMS: dict[str, dict[str, object]] = {
    "threshold": {"ceiling": 1.0},
    "residual_detector": {
        "target": "a",
        "model": {"id": "seasonal_naive", "params": {"season_length": 4}},
    },
    "multivariate_rolling_matrix_profile": {"subsequence_length": 4},
    "rolling_matrix_profile": {"subsequence_length": 4},
    "x_lag_damp": {"subsequence_length": 4},
}
_TRANSFORMER_PARAMS: dict[str, dict[str, object]] = {
    "feature_schema_guard": {"features": ["a", "b"]},
    "select": {"features": ["a"]},
    "rename": {"mapping": {"a": "c"}},
    "incremental_pca": {"n_components": 1, "n0": 5},
    "random_projection": {"n_components": 1},
}
_EDGE_MODELS = {"ano_edge_l", "isconna", "midas", "signed_graph_sketch_detector"}
_MULTIVARIATE_STATISTICS = {
    "moving_correlation_coefficient",
    "moving_covariance",
    "moving_mahalanobis_distance",
}
_UNIVARIATE_MODELS = {
    "rolling_matrix_profile",
    "x_lag_damp",
}


def _event_factory(component_id: str) -> Callable[[int], dict[str, object]]:
    if component_id in _EDGE_MODELS:

        def edge(index: int) -> dict[str, object]:
            event: dict[str, object] = {
                "src": index % 5,
                "dst": index * 7 % 5,
                "t": index,
            }
            if component_id == "signed_graph_sketch_detector":
                event["graph"] = 1
            return event

        return edge
    if component_id == "quantile_threshold":
        return lambda index: {"score": float(index % 10)}
    univariate = component_id in _UNIVARIATE_MODELS or (
        component_id.startswith("moving_")
        and component_id not in _MULTIVARIATE_STATISTICS
    )
    if univariate:
        return lambda index: {"a": index % 7 + 0.1}
    return lambda index: {"a": index % 7 + 0.1, "b": index % 3 + 0.2}


def _malformed(event: dict[str, object]) -> dict[str, dict[str, object]]:
    first = next(iter(event))
    cases = {
        "empty": {},
        "nan": {**event, first: math.nan},
        "infinite": {**event, first: math.inf},
        "text": {**event, first: "x"},
        "none": {**event, first: None},
        "extra": {**event, "unexpected": 1.0},
    }
    if len(event) > 1:
        cases["missing"] = {key: value for key, value in event.items() if key != first}
    if "t" in event:
        cases["backwards"] = {**event, "t": -5}
    return cases


def _warmed(component: object, make_event: Callable[[int], dict[str, object]]):
    for index in range(80):
        event = make_event(index)
        if hasattr(component, "score_one"):
            component.score_one(event)
        component.learn_one(event)
    return component


def _rejections(component: object, event: dict[str, object], methods: tuple[str, ...]):
    for case, malformed in _malformed(event).items():
        for method in methods:
            call = getattr(component, method)
            try:
                call(malformed)
            except ValidationError:
                continue
            except Exception as error:  # noqa: BLE001 - reported below
                yield f"{case}/{method}: {type(error).__name__}: {error}"


_MODEL_IDS = [
    component_id
    for component_id, spec in MODEL_CATALOG.items()
    if spec.available and spec.declarative and component_id != "knn"
]
_TRANSFORMER_IDS = [
    component_id
    for component_id, spec in TRANSFORMER_CATALOG.items()
    if spec.available and spec.declarative
]


@pytest.mark.parametrize("component_id", _MODEL_IDS)
def test_models_reject_malformed_events_with_validation_error(component_id):
    if component_id.startswith("moving_"):
        params: dict[str, object] = {"window_size": 8}
    else:
        params = _MODEL_PARAMS.get(component_id, {})
    make_event = _event_factory(component_id)
    model = _warmed(create_model(component_id, params), make_event)
    failures = list(_rejections(model, make_event(81), ("score_one", "learn_one")))
    assert failures == []


@pytest.mark.parametrize("component_id", _TRANSFORMER_IDS)
def test_transformers_reject_malformed_events_with_validation_error(component_id):
    params = _TRANSFORMER_PARAMS.get(component_id, {})
    make_event = _event_factory(component_id)
    transformer = _warmed(create_transformer(component_id, params), make_event)
    failures = list(
        _rejections(transformer, make_event(81), ("transform_one", "learn_one"))
    )
    assert failures == []


@pytest.mark.parametrize("detector", [ADWIN, KSWIN, PageHinkley])
@pytest.mark.parametrize("value", [math.nan, math.inf, "x", None])
def test_drift_detectors_reject_malformed_values(detector, value):
    with pytest.raises(ValidationError):
        detector().update(value)


@pytest.mark.parametrize(
    "forecaster",
    [lambda: SeasonalNaive(season_length=2), lambda: HoltWinters(season_length=2)],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, "x", None])
def test_forecasters_reject_malformed_values(forecaster, value):
    with pytest.raises(ValidationError):
        forecaster().learn_one(value)


def test_validation_error_is_a_value_error():
    assert issubclass(ValidationError, ValueError)
