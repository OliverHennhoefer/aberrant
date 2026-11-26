"""LagForecaster: any regressor of the series' own past, recursively forecast."""

import math
import pickle

import numpy as np
import pytest

from aberrant.forecast import (
    AutoRegressive,
    LagForecaster,
    RLSRegressor,
    SeasonalNaive,
)


def _series(n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 0.1 * t + 3 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.normal(0, 0.5, n))
    return [float(v) for v in values]


@pytest.mark.parametrize(
    "params",
    [
        {"order": -1},
        {"order": 0},
        {"difference": 3},
        {"seasonal_difference": 2, "season_length": 4},
        {"seasonal_order": 1},
        {"harmonics": 1},
        {"seasonal_difference": 1},
        {"residual_alpha": 0.0},
    ],
)
def test_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        LagForecaster(RLSRegressor(), **params)


def test_rejects_what_is_not_a_regressor() -> None:
    with pytest.raises(ValueError, match="regressor"):
        LagForecaster(SeasonalNaive())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "params",
    [
        {"order": 2},
        {"order": 2, "difference": 1},
        {"order": 3, "season_length": 4, "seasonal_difference": 1},
        {"order": 4, "season_length": 4, "difference": 2},
    ],
)
def test_with_least_squares_it_is_the_autoregression(params) -> None:
    lagged = LagForecaster(
        RLSRegressor(forgetting=0.99, delta=10.0), horizon=4, **params
    )
    reference = AutoRegressive(horizon=4, forgetting=0.99, delta=10.0, **params)
    for y in _series(200):
        lagged.learn_one(y)
        reference.learn_one(y)
        forecast = lagged.forecast_one()
        if forecast is not None:
            assert forecast.mean == pytest.approx(
                reference.forecast_one().mean, rel=1e-9, abs=1e-9
            )
    assert lagged.readiness.ready


@pytest.mark.parametrize(
    "params",
    [
        {"order": 2},
        {"order": 1, "season_length": 6, "seasonal_order": 2, "difference": 1},
        {"order": 0, "season_length": 5, "harmonics": 2},
    ],
)
def test_ready_exactly_when_required(params) -> None:
    model = LagForecaster(RLSRegressor(), horizon=3, **params)
    assert model.readiness.required is None
    for index, y in enumerate(_series(80), start=1):
        model.learn_one(y)
        required = model.readiness.required
        if required is not None:
            assert model.readiness.ready == (index >= required)
    assert model.readiness.ready


def test_phase_harmonics_forecast_a_fixed_season() -> None:
    model = LagForecaster(
        RLSRegressor(forgetting=1.0, delta=1e8),
        horizon=7,
        order=0,
        season_length=7,
        harmonics=3,
    )
    pattern = [
        math.sin(2 * math.pi * t / 7) + 0.5 * math.cos(4 * math.pi * t / 7)
        for t in range(7)
    ]
    for t in range(100):
        model.learn_one(2.0 + pattern[t % 7])
    expected = [2.0 + pattern[t % 7] for t in range(100, 107)]
    assert model.forecast_one().mean == pytest.approx(expected, abs=1e-6)


def test_pickle_continues_identically() -> None:
    model = LagForecaster(RLSRegressor(), horizon=2, order=2, difference=1)
    for y in _series(40):
        model.learn_one(y)
    restored = pickle.loads(pickle.dumps(model))
    for y in [12.0, None, 13.0, 12.5]:
        model.learn_one(y)
        restored.learn_one(y)
        assert restored.forecast_one() == model.forecast_one()
    model.reset()
    assert not model.readiness.ready
    assert not model.regressor.readiness.ready
