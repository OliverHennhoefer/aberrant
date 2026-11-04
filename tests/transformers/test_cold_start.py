"""Scalers return neutral values before their first learned event."""

import math
import pickle

import pytest

from aberrant.model.iforest import HalfSpaceTrees, OnlineIsolationForest
from aberrant.transform import MinMaxScaler, RollingRobustScaler, StandardScaler


@pytest.mark.parametrize(
    ("scaler", "neutral"),
    [
        (StandardScaler(), 0.0),
        (StandardScaler(with_std=False), 0.0),
        (MinMaxScaler(), 0.0),
        (MinMaxScaler(feature_range=(-1.0, 1.0)), -1.0),
        (RollingRobustScaler(window_size=8), 0.0),
    ],
)
def test_cold_scaler_returns_neutral_values_without_learning(
    scaler: StandardScaler | MinMaxScaler | RollingRobustScaler, neutral: float
) -> None:
    before = pickle.dumps(scaler)

    assert scaler.transform_one({"a": 5.0, "b": -3.0}) == {"a": neutral, "b": neutral}
    assert pickle.dumps(scaler) == before
    assert not scaler.readiness.ready


@pytest.mark.parametrize(
    "scaler",
    [StandardScaler(), MinMaxScaler(), RollingRobustScaler(window_size=8)],
)
def test_cold_scaler_still_validates_input(
    scaler: StandardScaler | MinMaxScaler | RollingRobustScaler,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        scaler.transform_one({"a": math.nan})


@pytest.mark.parametrize(
    "scaler",
    [StandardScaler(), MinMaxScaler(), RollingRobustScaler(window_size=8)],
)
def test_unseen_feature_after_learning_still_raises(
    scaler: StandardScaler | MinMaxScaler | RollingRobustScaler,
) -> None:
    scaler.learn_one({"a": 1.0})

    with pytest.raises(ValueError, match="has not been seen during learning"):
        scaler.transform_one({"a": 1.0, "b": 2.0})


def test_score_before_learn_pipeline_starts_on_the_first_event() -> None:
    pipelines = [
        StandardScaler() | OnlineIsolationForest(num_trees=5, window_size=16, seed=0),
        MinMaxScaler() | HalfSpaceTrees(n_trees=2, height=3, window_size=4, seed=0),
        RollingRobustScaler(window_size=8)
        | OnlineIsolationForest(num_trees=5, window_size=16, seed=0),
    ]
    events = [{"x": float(i % 5), "y": float(i % 3)} for i in range(20)]

    for pipeline in pipelines:
        for event in events:
            score = pipeline.score_one(event)
            assert math.isfinite(score)
            pipeline.learn_one(event)
