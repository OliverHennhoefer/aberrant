"""Online regressors with standardized inputs: kernel, PA, Huber SGD, KNN."""

import math
import pickle

import numpy as np
import pytest

from aberrant.base import ForecasterProtocol, RegressorProtocol
from aberrant.exceptions import ValidationError
from aberrant.forecast import (
    KNNRegressor,
    LagForecaster,
    PassiveAggressiveRegressor,
    RandomFourierRegressor,
    RLSRegressor,
    SGDRegressor,
)

REGRESSORS = {
    "fourier": lambda: RandomFourierRegressor(seed=1),
    "passive_aggressive": PassiveAggressiveRegressor,
    "sgd": SGDRegressor,
    "knn": KNNRegressor,
}


def _linear(n: int, seed: int = 0, noise: float = 0.0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, b = rng.normal(size=2)
        yield (
            {"a": float(a), "b": float(b)},
            3.0 * a - 2.0 * b + 0.5 + noise * rng.normal(),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomFourierRegressor(n_components=0),
        lambda: RandomFourierRegressor(length_scale=0.0),
        lambda: RandomFourierRegressor(forgetting=0.0),
        lambda: RandomFourierRegressor(delta=-1.0),
        lambda: RandomFourierRegressor(seed=-1),
        lambda: RandomFourierRegressor(warm_up=0),
        lambda: PassiveAggressiveRegressor(C=0.0),
        lambda: PassiveAggressiveRegressor(epsilon=-0.1),
        lambda: SGDRegressor(learning_rate=0.0),
        lambda: SGDRegressor(huber=0.0),
        lambda: SGDRegressor(l2=-1.0),
        lambda: KNNRegressor(k=0),
        lambda: KNNRegressor(k=6, window_size=5),
        lambda: KNNRegressor(weighted=1),  # type: ignore[arg-type]
        lambda: SGDRegressor(residual_alpha=0.0),
    ],
)
def test_rejects_invalid_configuration(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("name", REGRESSORS)
def test_is_a_regressor_ready_after_the_warm_up(name) -> None:
    model = REGRESSORS[name]()
    assert isinstance(model, RegressorProtocol)
    assert not isinstance(model, ForecasterProtocol)
    assert model.readiness.required == 33
    for learned, (x, y) in enumerate(_linear(40), start=1):
        assert (model.predict_one(x) is None) == (learned <= 33)
        model.learn_one(x, y)
        assert model.readiness.ready == (learned >= 33)
    forecast = model.predict_one({"a": 0.0, "b": 0.0})
    assert forecast.leads == (0,) and forecast.scale[0] > 0


@pytest.mark.parametrize("name", ["fourier", "passive_aggressive", "sgd"])
def test_learns_a_linear_target(name) -> None:
    model = REGRESSORS[name]()
    for x, y in _linear(3000):
        model.learn_one(x, y)
    assert model.predict_one({"a": 1.0, "b": -1.0}).mean[0] == pytest.approx(
        5.5, abs=0.1
    )


def test_random_features_fit_a_bend_that_least_squares_cannot() -> None:
    fourier, linear = RandomFourierRegressor(seed=0), RLSRegressor()
    rng = np.random.default_rng(3)
    for _ in range(2000):
        a = float(rng.uniform(-2, 2))
        for model in (fourier, linear):
            model.learn_one({"a": a}, math.sin(2 * a))
    grid = np.linspace(-1.8, 1.8, 19)
    errors = [
        [
            abs(model.predict_one({"a": float(a)}).mean[0] - math.sin(2 * a))
            for a in grid
        ]
        for model in (fourier, linear)
    ]
    assert max(errors[0]) < 0.05
    assert max(errors[1]) > 0.5


def test_random_features_are_seeded() -> None:
    first, second = RandomFourierRegressor(seed=4), RandomFourierRegressor(seed=4)
    for x, y in _linear(50, noise=0.3):
        first.learn_one(x, y)
        second.learn_one(x, y)
    assert first.predict_one({"a": 0.3, "b": 0.1}) == second.predict_one(
        {"a": 0.3, "b": 0.1}
    )


def test_passive_aggressive_leaves_errors_within_epsilon_alone() -> None:
    model = PassiveAggressiveRegressor(epsilon=0.5, warm_up=2)
    for a in (0.0, 1.0, 0.5):
        model.learn_one({"a": a}, a)
    weights = model._weights
    target = model.predict_one({"a": 2.0}).mean[0]
    model.learn_one({"a": 2.0}, target + 0.2)  # 0.2 target units is 0.4 sd
    assert np.array_equal(model._weights, weights)
    model.learn_one({"a": 2.0}, target + 2.0)
    assert not np.array_equal(model._weights, weights)


def test_huber_sgd_is_moved_little_by_an_outlier() -> None:
    robust, exact = SGDRegressor(), RLSRegressor()
    for x, y in _linear(2000, noise=0.1):
        robust.learn_one(x, y)
        exact.learn_one(x, y)
    point = {"a": 0.5, "b": 0.5}
    before = [model.predict_one(point).mean[0] for model in (robust, exact)]
    for model in (robust, exact):
        model.learn_one(point, 1000.0)
    shifts = [
        abs(model.predict_one(point).mean[0] - previous)
        for model, previous in zip((robust, exact), before, strict=True)
    ]
    assert shifts[0] < 0.3
    assert shifts[1] > 5 * shifts[0]


@pytest.mark.parametrize("weighted", [False, True])
def test_knn_averages_the_nearest_recent_targets(weighted) -> None:
    model = KNNRegressor(k=2, window_size=4, weighted=weighted, warm_up=2)
    for a, y in [(0.0, 0.0), (10.0, 100.0), (1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]:
        model.learn_one({"a": a}, y)
    # (0, 0) has left the window of four; the nearest to 1.4 are 1 and 2.
    mean = model.predict_one({"a": 1.4}).mean[0]
    if weighted:
        assert mean == pytest.approx((10 / 0.4 + 20 / 0.6) / (1 / 0.4 + 1 / 0.6))
    else:
        assert mean == pytest.approx(15.0)


@pytest.mark.parametrize("name", REGRESSORS)
def test_pickle_continues_and_reset_forgets(name) -> None:
    model = REGRESSORS[name]()
    for x, y in _linear(60, noise=0.2):
        model.learn_one(x, y)
    restored = pickle.loads(pickle.dumps(model))
    for x, y in _linear(10, seed=1):
        model.learn_one(x, y)
        restored.learn_one(x, y)
        assert restored.predict_one(x) == model.predict_one(x)
    model.reset()
    assert not model.readiness.ready
    assert model.feature_names is None


@pytest.mark.parametrize("name", REGRESSORS)
def test_an_unrepresentable_warm_up_changes_nothing(name) -> None:
    model = REGRESSORS[name]()
    for index in range(32 - 1):
        model.learn_one({"a": float(index)}, 1.7e308 * (-1) ** index)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one({"a": 0.0}, -1.7e308)
    assert pickle.dumps(model) == before


@pytest.mark.parametrize("name", REGRESSORS)
def test_rejects_malformed_values(name) -> None:
    model = REGRESSORS[name]()
    with pytest.raises(ValidationError):
        model.learn_one({"a": 1.0}, math.nan)
    with pytest.raises(ValidationError):
        model.learn_one({"a": math.inf}, 1.0)


def test_nonlinear_regressors_forecast_a_nonlinear_map() -> None:
    rng = np.random.default_rng(5)
    values = [0.3]
    for _ in range(3000):
        x = 3.7 * values[-1] * (1 - values[-1]) + 0.005 * float(rng.standard_normal())
        values.append(min(max(x, 0.001), 0.999))
    models = {
        "linear": LagForecaster(RLSRegressor(), order=2),
        "fourier": LagForecaster(RandomFourierRegressor(seed=0), order=2),
        "knn": LagForecaster(KNNRegressor(k=5, window_size=1000), order=2),
    }
    errors = dict.fromkeys(models, 0.0)
    for t, y in enumerate(values):
        for name, model in models.items():
            if t >= 1000:
                errors[name] += abs(y - model.forecast_one().point(1))
            model.learn_one(y)
    assert errors["fourier"] < 0.2 * errors["linear"]
    assert errors["knn"] < 0.3 * errors["linear"]
