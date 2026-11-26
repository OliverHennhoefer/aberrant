"""Imputer: missing values become values, and only observed ones are learned."""

import math
import pickle

import pytest

from aberrant.base import Readiness
from aberrant.base.pipeline import Pipeline
from aberrant.exceptions import ValidationError
from aberrant.model.stat import MovingZScore
from aberrant.transform import Imputer


def run(imputer, events):
    outputs = []
    for event in events:
        imputer.learn_one(event)
        outputs.append(imputer.transform_one(event))
    return outputs


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("last", [1.0, 3.0, 3.0, 3.0]), ("mean", [1.0, 3.0, 2.0, 2.0])],
)
def test_fills_absent_none_and_nan_from_observed_values(strategy, expected) -> None:
    imputer = Imputer(strategy=strategy)
    events = [{"a": 1.0}, {"a": 3.0}, {"a": None}, {"a": math.nan}]
    assert [row["a"] for row in run(imputer, events)] == expected
    assert imputer.transform_one({}) == {"a": expected[-1]}


def test_a_constant_fills_every_missing_value() -> None:
    imputer = Imputer(strategy="constant", value=-1.0, features=["a", "b"])
    assert imputer.readiness == Readiness(True, 1, 1)
    assert imputer.transform_one({"a": 2.0}) == {"a": 2.0, "b": -1.0}


def test_waits_until_every_feature_was_observed() -> None:
    imputer = Imputer()
    assert not imputer.readiness.ready
    rows = run(imputer, [{"a": 1.0, "b": None}])
    assert rows == [{"a": 1.0, "b": 0.0}]
    assert not imputer.readiness.ready
    run(imputer, [{"a": None, "b": 5.0}])
    assert imputer.readiness == Readiness(True, 1, 1)


def test_indicators_follow_the_schema() -> None:
    imputer = Imputer(indicate=True, features=["b", "a"])
    imputer.learn_one({"a": 1.0, "b": 2.0})
    assert list(imputer.transform_one({"a": None, "b": 4.0}).items()) == [
        ("b", 4.0),
        ("a", 1.0),
        ("b_missing", 0.0),
        ("a_missing", 1.0),
    ]
    with pytest.raises(ValidationError, match="already exist"):
        Imputer(indicate=True).learn_one({"a": 1.0, "a_missing": 0.0})


def test_rejects_what_is_not_missing_but_wrong() -> None:
    imputer = Imputer()
    imputer.learn_one({"a": 1.0})
    for value in (math.inf, "x"):
        with pytest.raises(ValidationError):
            imputer.learn_one({"a": value})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="Unexpected"):
        imputer.learn_one({"a": 1.0, "b": 2.0})


def test_lets_a_detector_take_events_with_gaps() -> None:
    detector = Pipeline(Imputer(), MovingZScore(window_size=5))
    events = [{"x": float(t % 3)} for t in range(10)]
    events[6] = {"x": None}
    for event in events:
        detector.score_one(event)
        detector.learn_one(event)
    assert math.isfinite(detector.score_one({"x": math.nan}))


def test_a_failed_downstream_update_restores_the_learned_values() -> None:
    class Failing(MovingZScore):
        def learn_one(self, x):
            if x["x"] == 99.0:
                raise RuntimeError("refused")
            super().learn_one(x)

    imputer = Imputer()
    pipeline = Pipeline(imputer, Failing(window_size=5))
    pipeline.learn_one({"x": 1.0})
    before = pickle.dumps(imputer)
    with pytest.raises(RuntimeError):
        pipeline.learn_one({"x": 99.0})
    assert pickle.dumps(imputer) == before


@pytest.mark.parametrize(
    "params",
    [
        {"strategy": "median"},
        {"value": math.nan},
        {"indicate": 1},
        {"features": "a"},
        {"features": []},
        {"features": ["a", "a"]},
    ],
)
def test_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        Imputer(**params)
