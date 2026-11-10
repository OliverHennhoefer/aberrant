"""SeasonalNaive and HoltWinters: forecasts per lead, spreads, state safety."""

import math
import pickle

import numpy as np
import pytest

from aberrant.forecast import HoltWinters, ResidualDetector, SeasonalNaive
from aberrant.model.timeseries import SeasonalResidualDetector


def _series(n: int, season: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 10 + 0.02 * t + 3 * np.sin(2 * np.pi * t / season) + rng.normal(0, 0.5, n)
    return [float(v) for v in values]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SeasonalNaive(season_length=0),
        lambda: SeasonalNaive(horizon=0),
        lambda: SeasonalNaive(residual_alpha=0.0),
        lambda: SeasonalNaive(min_scale=0.0),
        lambda: HoltWinters(1),
        lambda: HoltWinters(2.0),  # type: ignore[arg-type]
        lambda: HoltWinters(2, alpha=1.1),
        lambda: HoltWinters(2, alpha=0.9, gamma=0.2),
        lambda: HoltWinters(2, horizon=True),  # type: ignore[arg-type]
    ],
)
def test_rejects_invalid_configuration(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_seasonal_naive_repeats_the_last_cycle() -> None:
    model = SeasonalNaive(season_length=3, horizon=4)
    for y in (1.0, 5.0, 3.0, 1.5, 5.5):
        model.learn_one(y)
        assert model.forecast_one() is None
    model.learn_one(3.5)
    forecast = model.forecast_one()
    assert forecast is not None
    assert forecast.origin == 6
    assert forecast.leads == (1, 2, 3, 4)
    assert forecast.mean == (1.5, 5.5, 3.5, 1.5)
    # Warm-up spread: mean absolute seasonal-naive error within cycle two.
    assert forecast.scale == pytest.approx((0.5,) * 4)


def test_naive_spread_fades_per_lead_against_the_forecast_issued_that_far_back() -> (
    None
):
    model = SeasonalNaive(horizon=2, residual_alpha=0.5)
    for y in (0.0, 1.0):  # warm-up spread |1 - 0| = 1 for both leads
        model.learn_one(y)
    model.learn_one(4.0)  # lead 1 issued at origin 2 forecast 1.0: error 3
    assert model.forecast_one().scale == pytest.approx((2.0, 1.0))  # type: ignore[union-attr]
    model.learn_one(4.0)  # lead 1 from 4.0: error 0; lead 2 from origin 2: error 3
    assert model.forecast_one().scale == pytest.approx((1.0, 2.0))  # type: ignore[union-attr]


def test_holt_winters_multi_step_follows_component_equations() -> None:
    model = HoltWinters(2, horizon=3, alpha=0.25, beta=0.5, gamma=0.125)
    for y in (9.0, 13.0, 11.0, 15.0):
        model.learn_one(y)
    # Initialized: level 13.5, trend 1, seasons [-1.5, 1.5], next phase 0.
    assert model.forecast_one().mean == pytest.approx((13.0, 17.0, 15.0))  # type: ignore[union-attr]
    level, trend, seasons = 13.5, 1.0, [-1.5, 1.5]
    value = 17.0
    error = value - (level + trend + seasons[0])
    new_level = 0.25 * (value - seasons[0]) + 0.75 * (level + trend)
    new_trend = 0.5 * (new_level - level) + 0.5 * trend
    seasons[0] = 0.125 * (value - level - trend) + 0.875 * seasons[0]
    model.learn_one(value)
    assert error == 4.0
    expected = tuple(new_level + h * new_trend + seasons[h % 2] for h in (1, 2, 3))
    assert model.forecast_one().mean == pytest.approx(expected)  # type: ignore[union-attr]


@pytest.mark.parametrize("season", [2, 7, 24])
@pytest.mark.parametrize("horizon", [1, 5])
def test_holt_winters_lead_one_reproduces_seasonal_residual_detector(
    season: int, horizon: int
) -> None:
    params = {"alpha": 0.3, "beta": 0.1, "gamma": 0.2, "residual_alpha": 0.1}
    reference = SeasonalResidualDetector(season, **params)
    raw = SeasonalResidualDetector(season, normalize=False, **params)
    scaled = ResidualDetector(HoltWinters(season, horizon=horizon, **params), "v")
    absolute = ResidualDetector(
        HoltWinters(season, horizon=horizon, **params), "v", score="absolute"
    )
    for y in _series(30 * season, season, seed=season):
        x = {"v": y}
        assert scaled.score_one(x) == reference.score_one(x)
        assert absolute.score_one(x) == raw.score_one(x)
        for model in (reference, raw, scaled, absolute):
            model.learn_one(x)


def test_overflow_leaves_state_unchanged_and_pickle_continues() -> None:
    model = HoltWinters(2, horizon=3)
    for y in _series(20, 2):
        model.learn_one(y)
    model.learn_one(1.7e308)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one(-1.7e308)
    assert pickle.dumps(model) == before
    clone = pickle.loads(pickle.dumps(model))
    for y in (11.0, 12.0, 10.5):
        model.learn_one(y)
        clone.learn_one(y)
    assert clone.forecast_one() == model.forecast_one()


def test_invalid_values_and_reset() -> None:
    for model in (SeasonalNaive(), HoltWinters(2)):
        with pytest.raises(ValueError):
            model.learn_one(math.nan)
        with pytest.raises(ValueError):
            model.learn_one("1")  # type: ignore[arg-type]
        for y in _series(10, 2):
            model.learn_one(y)
        assert model.is_ready
        model.reset()
        assert not model.is_ready
        assert model.forecast_one() is None


def test_spreads_grow_with_the_lead_on_a_random_walk() -> None:
    rng = np.random.default_rng(3)
    model = SeasonalNaive(horizon=8, residual_alpha=0.01)
    for y in np.cumsum(rng.normal(size=5000)):
        model.learn_one(float(y))
    scale = model.forecast_one().scale  # type: ignore[union-attr]
    assert scale is not None
    assert all(a < b for a, b in zip(scale, scale[1:], strict=False))
    # Mean absolute error of a random walk at lead h is sqrt(2h/pi).
    assert scale[3] == pytest.approx(math.sqrt(8 / math.pi), rel=0.25)
