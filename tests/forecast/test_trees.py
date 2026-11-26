"""Hoeffding trees and adaptive random forests of standardized inputs."""

import pickle

import numpy as np
import pytest

from aberrant.forecast import (
    AdaptiveRandomForestRegressor,
    HoeffdingTreeRegressor,
    LagForecaster,
    RLSRegressor,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HoeffdingTreeRegressor(grace_period=0),
        lambda: HoeffdingTreeRegressor(delta=0.0),
        lambda: HoeffdingTreeRegressor(delta=1.0),
        lambda: HoeffdingTreeRegressor(tau=-0.1),
        lambda: HoeffdingTreeRegressor(n_bins=1),
        lambda: HoeffdingTreeRegressor(max_depth=-1),
        lambda: HoeffdingTreeRegressor(leaf_prediction="model"),  # type: ignore[arg-type]
        lambda: AdaptiveRandomForestRegressor(n_models=0),
        lambda: AdaptiveRandomForestRegressor(max_features=0),
        lambda: AdaptiveRandomForestRegressor(lambda_value=0.0),
        lambda: AdaptiveRandomForestRegressor(drift_delta=1.0),
        lambda: AdaptiveRandomForestRegressor(warning_delta=0.0),
        lambda: AdaptiveRandomForestRegressor(seed=-1),
    ],
)
def test_rejects_invalid_configuration(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def _step(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = rng.uniform(-1, 1, size=2)
        yield {"a": float(a), "b": float(b)}, (1.0 if a > 0.3 else -1.0)


def test_a_tree_splits_where_the_target_jumps() -> None:
    model = HoeffdingTreeRegressor(grace_period=50, leaf_prediction="mean")
    for x, y in _step(1000):
        model.learn_one(x, y)
    split = model._nodes[0]
    assert split.feature == model.feature_names.index("a")
    a = model._center[split.feature] + split.threshold * model._spread_x[split.feature]
    assert a == pytest.approx(0.3, abs=0.07)
    assert model.predict_one({"a": 0.8, "b": 0.0}).mean[0] == pytest.approx(1.0)
    assert model.predict_one({"a": -0.5, "b": 0.0}).mean[0] == pytest.approx(-1.0)


def test_a_tree_of_depth_zero_is_one_leaf() -> None:
    model = HoeffdingTreeRegressor(grace_period=50, max_depth=0)
    for x, y in _step(1000):
        model.learn_one(x, y)
    assert model.n_leaves == 1


def test_linear_leaves_fit_a_piecewise_linear_target() -> None:
    errors = {}
    for mode in ("mean", "linear", "adaptive"):
        model = HoeffdingTreeRegressor(grace_period=100, leaf_prediction=mode)
        rng = np.random.default_rng(2)
        for _ in range(3000):
            a = float(rng.uniform(-2, 2))
            model.learn_one({"a": a}, 3 * abs(a))
        grid = np.linspace(-1.8, 1.8, 19)
        errors[mode] = max(
            abs(model.predict_one({"a": float(a)}).mean[0] - 3 * abs(a)) for a in grid
        )
    assert errors["linear"] < 0.2
    assert errors["adaptive"] < 0.2
    assert errors["mean"] > 2 * errors["linear"]


def test_the_tree_forecasts_a_threshold_autoregression() -> None:
    rng = np.random.default_rng(0)
    values = [0.0]
    for _ in range(5000):
        y = values[-1]
        values.append((0.7 * y + 1.0 if y < 0 else -0.5 * y - 0.5) + rng.normal())
    models = {
        "linear": LagForecaster(RLSRegressor(), order=1),
        "tree": LagForecaster(HoeffdingTreeRegressor(), order=1),
    }
    errors = dict.fromkeys(models, 0.0)
    for t, y in enumerate(values):
        for name, model in models.items():
            if t >= 2000:
                errors[name] += abs(y - model.forecast_one().point(1))
            model.learn_one(y)
    assert errors["tree"] < 0.9 * errors["linear"]


def test_the_forest_replaces_trees_after_a_drift() -> None:
    forest = AdaptiveRandomForestRegressor(leaf_prediction="mean", seed=0)
    tree = HoeffdingTreeRegressor(leaf_prediction="mean")
    rng = np.random.default_rng(1)
    errors = {"forest": 0.0, "tree": 0.0}
    for t in range(6000):
        a, b = (float(v) for v in rng.uniform(-1, 1, size=2))
        y = (1.0 if a > 0 else -1.0) * (1 if t < 3000 else -1)
        y += 0.1 * float(rng.normal())
        for name, model in (("forest", forest), ("tree", tree)):
            if t >= 4000:
                errors[name] += abs(y - model.predict_one({"a": a, "b": b}).mean[0])
            model.learn_one({"a": a, "b": b}, y)
    assert forest.n_drifts > 0
    assert errors["forest"] < 0.5 * errors["tree"]


def test_the_forest_is_seeded() -> None:
    first = AdaptiveRandomForestRegressor(seed=3, n_models=3)
    second = AdaptiveRandomForestRegressor(seed=3, n_models=3)
    for x, y in _step(500):
        first.learn_one(x, y)
        second.learn_one(x, y)
    point = {"a": 0.4, "b": -0.2}
    assert first.predict_one(point) == second.predict_one(point)


@pytest.mark.parametrize(
    "factory",
    [HoeffdingTreeRegressor, lambda: AdaptiveRandomForestRegressor(seed=2)],
)
def test_an_unrepresentable_event_changes_nothing(factory) -> None:
    model = factory()
    for x, y in _step(100):
        model.learn_one(x, y)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one({"a": 1e300, "b": 0.0}, 0.0)
    assert pickle.dumps(model) == before
