"""Tests for the sliding-window z-score and robust z-score."""

import math

import numpy as np
import pytest

from aberrant.exceptions import ValidationError
from aberrant.model.stat import MovingMAD, MovingZScore


def _zscore(window: np.ndarray, value: float, floor: float) -> float:
    return abs(value - window.mean()) / max(window.std(), floor)


def _robust(window: np.ndarray, value: float, floor: float) -> float:
    median = np.median(window)
    mad = 1.482602218505602 * np.median(np.abs(window - median))
    return abs(value - median) / max(mad, floor)


@pytest.mark.parametrize(
    ("model_type", "reference"), [(MovingZScore, _zscore), (MovingMAD, _robust)]
)
def test_scores_match_a_recomputed_window(model_type, reference):
    rng = np.random.default_rng(4)
    values = np.concatenate([rng.normal(5.0, 2.0, 3000), rng.normal(1e6, 1.0, 200)])
    model = model_type(window_size=50, warm_up=10)
    history: list[float] = []
    for value in values:
        window = np.array(history[-50:])
        score = model.score_one({"x": float(value)})
        if len(window) < 10:
            assert score == 0.0
        else:
            assert score == pytest.approx(
                reference(window, float(value), 1e-12), rel=1e-9, abs=1e-9
            )
        model.learn_one({"x": float(value)})
        history.append(float(value))


@pytest.mark.parametrize("model_type", [MovingZScore, MovingMAD])
def test_warm_up_defaults_to_the_window(model_type):
    model = model_type(window_size=4)
    assert model.warm_up == 4
    for value in (1.0, 2.0, 3.0):
        model.learn_one({"x": value})
        assert model.score_one({"x": 10.0}) == 0.0
    model.learn_one({"x": 4.0})
    assert model.score_one({"x": 10.0}) > 0.0


@pytest.mark.parametrize(
    ("model_type", "floor"), [(MovingZScore, "min_std"), (MovingMAD, "min_mad")]
)
def test_a_constant_window_uses_the_floor(model_type, floor):
    model = model_type(window_size=5, **{floor: 0.5})
    for _ in range(5):
        model.learn_one({"x": 2.0})
    assert model.score_one({"x": 2.0}) == 0.0
    assert model.score_one({"x": 3.0}) == 2.0


def test_mad_ignores_a_few_extreme_values():
    robust = MovingMAD(window_size=21)
    plain = MovingZScore(window_size=21)
    for value in [*range(19), 1e9, -1e9]:
        robust.learn_one({"x": float(value)})
        plain.learn_one({"x": float(value)})
    assert robust.score_one({"x": 200.0}) > 10.0
    assert plain.score_one({"x": 200.0}) < 1.0


@pytest.mark.parametrize("model_type", [MovingZScore, MovingMAD])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_size": 1}, "window_size"),
        ({"window_size": 4, "warm_up": 5}, "warm_up must not exceed window_size"),
        ({"warm_up": 1}, "warm_up"),
    ],
)
def test_rejects_invalid_configuration(model_type, kwargs, message):
    with pytest.raises(ValueError, match=message):
        model_type(**kwargs)


@pytest.mark.parametrize("model_type", [MovingZScore, MovingMAD])
def test_events_must_carry_one_stable_feature(model_type):
    model = model_type(window_size=4)
    for event in ({}, {"a": 1.0, "b": 2.0}, {"a": math.nan}, {"a": "x"}):
        with pytest.raises(ValidationError):
            model.learn_one(event)
    model.learn_one({"a": 1.0})
    with pytest.raises(ValidationError, match="'a'"):
        model.score_one({"b": 1.0})
    assert model.n_samples_seen == 1
