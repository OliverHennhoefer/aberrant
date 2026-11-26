"""EnsembleForecaster: ML-Poly weights per lead, readiness, members left out."""

import math
import pickle

import numpy as np
import pytest

from aberrant.base import Readiness
from aberrant.forecast import (
    AutoRegressive,
    EnsembleForecaster,
    Forecast,
    HoltWinters,
    KalmanForecaster,
    RLSRegressor,
    SeasonalNaive,
)


def _series(n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 10 + 3 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.normal(0, 0.3, n))
    return [float(v) for v in values]


def _members(horizon: int = 2):
    return [
        SeasonalNaive(season_length=4, horizon=horizon),
        KalmanForecaster(horizon=horizon),
        AutoRegressive(horizon=horizon, order=2),
    ]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EnsembleForecaster([SeasonalNaive()]),
        lambda: EnsembleForecaster([]),
        lambda: EnsembleForecaster("naive"),  # type: ignore[arg-type]
        lambda: EnsembleForecaster([SeasonalNaive(), RLSRegressor()]),  # type: ignore[list-item]
        lambda: EnsembleForecaster([SeasonalNaive(), SeasonalNaive(horizon=2)]),
        lambda: EnsembleForecaster(_members(), forgetting=0.0),
        lambda: EnsembleForecaster(_members(), forgetting=1.5),
        lambda: EnsembleForecaster(_members(), residual_alpha=0.0),
    ],
)
def test_rejects_invalid_configuration(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_rejects_one_member_twice() -> None:
    member = SeasonalNaive()
    with pytest.raises(ValueError, match="distinct"):
        EnsembleForecaster([member, member])


@pytest.mark.parametrize("forgetting", [1.0, 0.9])
def test_combines_by_ml_poly_on_the_absolute_error(forgetting) -> None:
    horizon = 2
    ensemble = EnsembleForecaster(_members(horizon), forgetting=forgetting)
    singles = _members(horizon)
    regret = np.zeros((horizon, 3))
    squares = np.zeros((horizon, 3))
    issued: list[tuple[np.ndarray, list[float]] | None] = []
    compared = 0
    for y in _series(200):
        for lead in range(1, horizon + 1):
            if len(issued) >= lead and issued[-lead] is not None:
                members, combined = issued[-lead]
                p = combined[lead - 1]
                instant = math.copysign(1.0, p - y) * (p - members[:, lead - 1])
                regret[lead - 1] = forgetting * regret[lead - 1] + instant
                squares[lead - 1] = forgetting * squares[lead - 1] + instant**2
        ensemble.learn_one(y)
        forecasts = []
        for single in singles:
            single.learn_one(y)
            forecasts.append(single.forecast_one())
        if any(forecast is None for forecast in forecasts):
            issued.append(None)
            continue
        members = np.array([forecast.mean for forecast in forecasts])
        combined = []
        for lead in range(horizon):
            ratio = np.maximum(regret[lead], 0) / np.maximum(squares[lead], 1e-300)
            positive = np.where(squares[lead] > 0, ratio, 0)
            weights = (
                positive / positive.sum() if positive.sum() > 0 else np.ones(3) / 3
            )
            combined.append(float(weights @ members[:, lead]))
        issued.append((members, combined))
        forecast = ensemble.forecast_one()
        if forecast is not None:
            assert forecast.mean == pytest.approx(combined, rel=1e-12)
            compared += 1
    assert compared > 150


def test_weights_move_to_the_member_that_fits() -> None:
    pattern = [0.0, 4.0, 1.0, -3.0]
    ensemble = EnsembleForecaster(
        [SeasonalNaive(season_length=4), KalmanForecaster()], forgetting=1.0
    )
    rng = np.random.default_rng(1)
    for t in range(400):
        ensemble.learn_one(10.0 + pattern[t % 4] + 0.01 * float(rng.standard_normal()))
    assert ensemble.weights[0][0] > 0.99


def test_beats_every_member_across_a_change_of_regime() -> None:
    rng = np.random.default_rng(2)
    values, previous = [], [0.0, 0.0]
    for t in range(3000):
        if t < 1500:
            y = 0.6 * previous[-1] - 0.3 * previous[-2] + float(rng.standard_normal())
        else:
            y = 5 * math.sin(2 * math.pi * t / 12) + float(rng.standard_normal())
        previous.append(y)
        values.append(y)

    def members():
        return [
            HoltWinters(season_length=12),
            AutoRegressive(order=2),
            KalmanForecaster(),
        ]

    models = [EnsembleForecaster(members()), *members()]
    errors = np.zeros(len(models))
    for t, y in enumerate(values):
        for index, model in enumerate(models):
            forecast = model.forecast_one()
            if t >= 300:
                errors[index] += abs(y - forecast.point(1))
            model.learn_one(y)
    assert errors[0] < min(errors[1:])


def test_ready_once_every_member_is_and_the_longest_lead_is_scored() -> None:
    ensemble = EnsembleForecaster(
        [SeasonalNaive(season_length=4, horizon=3), KalmanForecaster(horizon=3)]
    )
    assert ensemble.readiness.required == 8 + 3
    for index, y in enumerate(_series(30), start=1):
        ensemble.learn_one(y)
        assert ensemble.readiness.ready == (index >= 11)


class _Flaky:
    """A forecaster that gives no forecast on every third origin."""

    horizon = 1

    def __init__(self, value: float) -> None:
        self.value = value
        self.seen = 0

    @property
    def readiness(self) -> Readiness:
        return Readiness.after(self.seen, 1)

    def learn_one(self, y: float | None) -> None:
        self.seen += 1

    def forecast_one(self) -> Forecast | None:
        if self.seen % 3 == 0:
            return None
        return Forecast(origin=self.seen, leads=(1,), mean=(self.value,), scale=(1.0,))


def test_a_member_without_a_forecast_is_left_out() -> None:
    ensemble = EnsembleForecaster([_Flaky(1.0), _Flaky(5.0), SeasonalNaive()])
    alone = []
    for seen in range(1, 13):
        ensemble.learn_one(3.0)
        if seen % 3 == 0 and ensemble._errors.issued:
            alone.append(ensemble._errors.issued[-1][0])
    # Both flaky members give none at the same origins; naive then speaks alone.
    assert alone == [3.0] * 4


def test_reset_and_pickle() -> None:
    ensemble = EnsembleForecaster(_members())
    for y in _series(60):
        ensemble.learn_one(y)
    restored = pickle.loads(pickle.dumps(ensemble))
    for y in [11.0, None, 12.0, 10.5]:
        ensemble.learn_one(y)
        restored.learn_one(y)
        assert restored.forecast_one() == ensemble.forecast_one()
    ensemble.reset()
    assert not ensemble.readiness.ready
    assert all(not member.readiness.ready for member in ensemble.forecasters)
