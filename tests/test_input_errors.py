"""Every component rejects a malformed event with ``ValidationError``."""

import math
from collections.abc import Callable

import pytest

from aberrant.catalog import create_model, create_transformer
from aberrant.drift import ADWIN, CUSUM, KSWIN, PageHinkley
from aberrant.exceptions import ValidationError
from aberrant.forecast import HoltWinters, SeasonalNaive
from tests.catalog_cases import (
    MODEL_IDS,
    TRANSFORMER_IDS,
    TRANSFORMER_PARAMS,
    event_factory,
    model_params,
)


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


@pytest.mark.parametrize("component_id", MODEL_IDS)
def test_models_reject_malformed_events_with_validation_error(component_id):
    make_event = event_factory(component_id)
    model = _warmed(create_model(component_id, model_params(component_id)), make_event)
    failures = list(_rejections(model, make_event(81), ("score_one", "learn_one")))
    assert failures == []


@pytest.mark.parametrize("component_id", TRANSFORMER_IDS)
def test_transformers_reject_malformed_events_with_validation_error(component_id):
    params = TRANSFORMER_PARAMS.get(component_id, {})
    make_event = event_factory(component_id)
    transformer = _warmed(create_transformer(component_id, params), make_event)
    failures = list(
        _rejections(transformer, make_event(81), ("transform_one", "learn_one"))
    )
    assert failures == []


@pytest.mark.parametrize("detector", [ADWIN, CUSUM, KSWIN, PageHinkley])
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
