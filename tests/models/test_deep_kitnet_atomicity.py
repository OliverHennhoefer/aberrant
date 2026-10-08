"""Rejected KitNET events preserve the complete learned state."""

from __future__ import annotations

import math
import pickle
from copy import deepcopy

import numpy as np
import pytest

from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble


@pytest.mark.parametrize("adaptive", [False, True])
def test_later_child_overflow_does_not_publish_earlier_updates(adaptive: bool) -> None:
    model = OnlineAutoencoderEnsemble(
        max_ae_size=1,
        feature_map_grace=1,
        ad_grace=0 if adaptive else 1,
        adaptive_after_warmup=adaptive,
        seed=2,
    )
    model.learn_one({"x": 0.0, "y": 0.0})
    control = deepcopy(model)
    before = pickle.dumps(model)

    # The first child's update is finite; the second child's gradient overflows.
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(OverflowError):
        model.learn_one({"x": 0.2, "y": 1.7e308})
    assert pickle.dumps(model) == before

    point = {"x": 0.2, "y": 0.3}
    model.learn_one(point)
    control.learn_one(point)
    assert pickle.dumps(model) == pickle.dumps(control)
    assert math.isfinite(model.score_one(point))


def test_output_overflow_does_not_publish_child_updates() -> None:
    model = OnlineAutoencoderEnsemble(
        max_ae_size=1, feature_map_grace=1, ad_grace=1, seed=7
    )
    model.learn_one({"x": 0.0, "y": 0.0})
    assert model._output_ae is not None
    model._output_ae.learning_rate = 1e308
    control = deepcopy(model)
    before = pickle.dumps(model)

    # Both children accept this sample, but the output network rejects its step.
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(OverflowError):
        model.learn_one({"x": 100.0, "y": 50.0})
    assert pickle.dumps(model) == before

    assert control._output_ae is not None
    model._output_ae.learning_rate = control._output_ae.learning_rate = 0.1
    point = {"x": 0.2, "y": 0.3}
    model.learn_one(point)
    control.learn_one(point)
    assert pickle.dumps(model) == pickle.dumps(control)


@pytest.mark.parametrize("grace", [1, 2])
@pytest.mark.parametrize("seed", [7, None])
def test_rejected_feature_map_transition_preserves_moments_schema_and_rng(
    grace: int, seed: int | None
) -> None:
    model = OnlineAutoencoderEnsemble(
        max_ae_size=1,
        feature_map_grace=grace,
        ad_grace=0,
        learning_rate=1e308,
        seed=seed,
    )
    if grace == 2:
        model.learn_one({"x": 0.1, "y": 0.2})
    control = deepcopy(model)
    before = pickle.dumps(model)

    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(OverflowError):
        model.learn_one({"x": 100.0, "y": 50.0})
    assert pickle.dumps(model) == before

    # A rejected first event must still allow a different feature dimension.
    point = {"other": 0.0} if grace == 1 else {"x": 0.0, "y": 0.0}
    model.learn_one(point)
    control.learn_one(point)
    assert model.is_ready
    assert pickle.dumps(model) == pickle.dumps(control)
    assert math.isfinite(model.score_one(point))
