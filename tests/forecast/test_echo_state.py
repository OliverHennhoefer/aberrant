"""EchoStateForecaster: a random reservoir with a readout learned online."""

import math
import pickle

import numpy as np
import pytest

from aberrant.forecast import AutoRegressive, EchoStateForecaster


@pytest.mark.parametrize(
    "params",
    [
        {"n_reservoir": 0},
        {"spectral_radius": 0.0},
        {"leak_rate": 0.0},
        {"leak_rate": 1.5},
        {"input_scale": -1.0},
        {"connectivity": 0.0},
        {"difference": 3},
        {"seasonal_difference": 1},
        {"seasonal_difference": 2, "season_length": 4},
        {"forgetting": 0.0},
        {"delta": 0.0},
        {"warm_up": 0},
        {"seed": -1},
        {"residual_alpha": 0.0},
    ],
)
def test_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        EchoStateForecaster(**params)


def test_the_reservoir_has_the_requested_spectral_radius() -> None:
    model = EchoStateForecaster(n_reservoir=50, spectral_radius=0.7, seed=0)
    radius = np.abs(np.linalg.eigvals(model._recurrent)).max()
    assert radius == pytest.approx(0.7)
    assert np.mean(model._recurrent != 0) == pytest.approx(0.1, abs=0.03)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"difference": 1},
        {"difference": 2, "season_length": 4, "seasonal_difference": 1},
    ],
)
def test_ready_exactly_when_required(params) -> None:
    model = EchoStateForecaster(horizon=3, warm_up=10, seed=0, **params)
    required = model.readiness.required
    assert required == 10 + 3 + params.get("difference", 0) + 4 * params.get(
        "seasonal_difference", 0
    )
    for index in range(1, 40):
        model.learn_one(math.sin(index / 2))
        assert model.readiness.ready == (index >= required)
        assert (model.forecast_one() is None) == (index < required)


def test_forecasts_a_sine_several_leads_ahead() -> None:
    model = EchoStateForecaster(horizon=6, seed=0)
    for t in range(1500):
        model.learn_one(math.sin(t / 3))
    expected = [math.sin(t / 3) for t in range(1500, 1506)]
    assert model.forecast_one().mean == pytest.approx(expected, abs=0.02)


def test_the_reservoir_forecasts_a_nonlinear_map_better_than_a_line() -> None:
    rng = np.random.default_rng(5)
    values = [0.3]
    for _ in range(3000):
        x = 3.7 * values[-1] * (1 - values[-1]) + 0.005 * float(rng.standard_normal())
        values.append(min(max(x, 0.001), 0.999))
    models = {
        "linear": AutoRegressive(order=2),
        "echo": EchoStateForecaster(input_scale=1.0, seed=0),
    }
    errors = dict.fromkeys(models, 0.0)
    for t, y in enumerate(values):
        for name, model in models.items():
            if t >= 1000:
                errors[name] += abs(y - model.forecast_one().point(1))
            model.learn_one(y)
    assert errors["echo"] < 0.3 * errors["linear"]


def test_the_reservoir_is_seeded() -> None:
    first, second = EchoStateForecaster(seed=4), EchoStateForecaster(seed=4)
    for t in range(80):
        first.learn_one(math.cos(t / 5))
        second.learn_one(math.cos(t / 5))
    assert first.forecast_one() == second.forecast_one()


def test_pickle_continues_and_reset_forgets() -> None:
    model = EchoStateForecaster(horizon=2, difference=1, seed=1)
    for t in range(60):
        model.learn_one(0.1 * t + math.sin(t))
    restored = pickle.loads(pickle.dumps(model))
    for y in [6.5, None, 7.0, 6.8]:
        model.learn_one(y)
        restored.learn_one(y)
        assert restored.forecast_one() == model.forecast_one()
    model.reset()
    assert not model.readiness.ready
    assert model.forecast_one() is None


@pytest.mark.parametrize("warmed", [False, True])
def test_an_unrepresentable_value_changes_nothing(warmed) -> None:
    model = EchoStateForecaster(warm_up=4, seed=0)
    values = [1.0, 2.0, 1.5, 1.7, 1.2] if warmed else [1.7e308, -1.7e308, 1.7e308]
    for y in values:
        model.learn_one(y)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one(1e308 if warmed else -1.7e308)
    assert pickle.dumps(model) == before
