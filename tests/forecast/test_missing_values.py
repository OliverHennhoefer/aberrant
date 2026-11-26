"""Every forecaster learns ``None`` as a missing value of its series."""

import pickle

import numpy as np
import pytest

from aberrant.forecast import (
    AutoRegressive,
    HoltWinters,
    KalmanForecaster,
    SeasonalNaive,
)

HORIZON = 6
FORECASTERS = {
    "naive": lambda: SeasonalNaive(season_length=4, horizon=HORIZON),
    "holt_winters": lambda: HoltWinters(season_length=4, horizon=HORIZON),
    "kalman": lambda: KalmanForecaster(horizon=HORIZON, slope_ratio=0.01),
    "kalman_seasonal": lambda: KalmanForecaster(
        horizon=HORIZON, slope_ratio=0.01, season_length=4, season_ratio=0.01
    ),
    "autoregressive": lambda: AutoRegressive(
        horizon=HORIZON, season_length=4, seasonal_difference=1
    ),
}


def _series(n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 0.3, n)
    return [float(v) for v in values]


def _learned(name: str, values: list[float | None]):
    model = FORECASTERS[name]()
    for y in values:
        model.learn_one(y)
    return model


@pytest.mark.parametrize("name", FORECASTERS)
@pytest.mark.parametrize("gap", [1, 3])
def test_after_a_gap_the_forecast_is_the_longer_lead_issued_before_it(name, gap):
    before = _learned(name, _series(60)).forecast_one()
    after = _learned(name, [*_series(60), *[None] * gap]).forecast_one()

    assert after.origin == before.origin + gap
    np.testing.assert_allclose(after.mean[: HORIZON - gap], before.mean[gap:])
    np.testing.assert_allclose(after.scale[: HORIZON - gap], before.scale[gap:])
    # Beyond the horizon a forecast reports at least the longest lead's spread.
    assert min(after.scale[HORIZON - gap :]) >= before.scale[-1] - 1e-12


@pytest.mark.parametrize("name", ["naive", "holt_winters", "autoregressive"])
def test_a_forecast_issued_after_a_gap_is_not_scored_at_its_own_lead(name):
    series = _series(60)
    before = _learned(name, series).forecast_one()
    after = _learned(name, [*series, None, 50.0]).forecast_one()

    # The value after the gap was a lead-2 forecast, so lead 1 learned nothing.
    assert after.scale[0] == before.scale[0]
    assert after.scale[1] != before.scale[1]


@pytest.mark.parametrize("name", ["naive", "holt_winters"])
def test_missing_values_before_the_first_one_do_not_start_the_series(name):
    series = _series(8)
    plain = _learned(name, series)
    late = FORECASTERS[name]()
    for _ in range(3):
        late.learn_one(None)
        assert not late.readiness.ready
    for y in series[:-1]:
        late.learn_one(y)
        assert not late.readiness.ready
    late.learn_one(series[-1])

    assert late.readiness.ready and late.readiness.observed == 8
    assert late.forecast_one().origin == 11
    assert late.forecast_one().mean == plain.forecast_one().mean


def test_a_warm_up_gap_takes_the_value_one_cycle_earlier():
    model = SeasonalNaive(season_length=2, horizon=2)
    for y in [1.0, 2.0, None, 4.0]:
        model.learn_one(y)
    assert model.forecast_one().mean == (1.0, 4.0)
    # Only the second position was observed in both cycles.
    assert model.forecast_one().scale == (2.0, 2.0)

    model = SeasonalNaive(season_length=2, horizon=1)
    for y in [1.0, None, 3.0, 5.0]:
        model.learn_one(y)
    # A first-cycle gap repeats the last observed value.
    assert model.forecast_one().mean == (3.0,)


@pytest.mark.parametrize("name", FORECASTERS)
def test_a_pickled_forecaster_continues_identically_inside_a_gap(name):
    model = _learned(name, [*_series(40), None, None])
    restored = pickle.loads(pickle.dumps(model))
    rest = [None, 12.0, None, 13.0, 12.5]
    for y in rest:
        model.learn_one(y)
        restored.learn_one(y)
        assert restored.forecast_one() == model.forecast_one()
