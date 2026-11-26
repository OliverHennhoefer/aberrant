"""AutoRegressive: exact faded least squares, recursive forecasts, spreads."""

import pickle

import numpy as np
import pytest

from aberrant.forecast import AutoRegressive


def _series(n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 0.1 * t + 3 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.normal(0, 0.5, n))
    return [float(v) for v in values]


@pytest.mark.parametrize(
    "params",
    [
        {"order": -1},
        {"difference": 3},
        {"seasonal_difference": 2, "season_length": 4},
        {"seasonal_order": 1},
        {"seasonal_difference": 1},
        {"forgetting": 0.0},
        {"forgetting": 1.1},
        {"delta": 0.0},
        {"residual_alpha": 0.0},
    ],
)
def test_rejects_invalid_configuration(params) -> None:
    with pytest.raises(ValueError):
        AutoRegressive(**params)


def _reference(values, lags, *, difference, seasonal, m, forgetting, delta, horizon):
    """Faded least squares with the RLS prior, then recursive forecasts."""
    y = np.asarray(values)
    w = y.copy()
    for _ in range(difference):
        w = w[1:] - w[:-1]
    for _ in range(seasonal):
        w = w[m:] - w[:-m]
    largest = max(lags)
    rows = np.array(
        [[*(w[t - lag] for lag in lags), 1.0] for t in range(largest, len(w))]
    )
    targets = w[largest:]
    n = len(targets)
    fade = forgetting ** np.arange(n - 1, -1, -1)
    gram = forgetting**n / delta * np.eye(len(lags) + 1) + rows.T @ (
        fade[:, None] * rows
    )
    weights = np.linalg.solve(gram, rows.T @ (fade * targets))

    history, differenced = list(y), list(w)
    means = []
    for _ in range(horizon):
        step = weights @ [*(differenced[-lag] for lag in lags), 1.0]
        # Undo (1 - B)^d (1 - B^m)^D on the levels.
        level = np.array([*history, 0.0])
        for _ in range(difference):
            level = level[1:] - level[:-1]
        for _ in range(seasonal):
            level = level[m:] - level[:-m]
        mean = step - level[-1]
        means.append(mean)
        history.append(mean)
        differenced.append(step)
    return means


@pytest.mark.parametrize(
    ("order", "seasonal_order", "difference", "seasonal", "forgetting"),
    [(2, 0, 0, 0, 1.0), (1, 1, 1, 1, 0.98), (3, 2, 2, 0, 0.995), (0, 1, 0, 1, 1.0)],
)
def test_matches_faded_least_squares_and_recursive_forecasts(
    order, seasonal_order, difference, seasonal, forgetting
) -> None:
    values = _series(300)
    model = AutoRegressive(
        horizon=6,
        order=order,
        season_length=4,
        seasonal_order=seasonal_order,
        difference=difference,
        seasonal_difference=seasonal,
        forgetting=forgetting,
        delta=10.0,
    )
    for y in values:
        model.learn_one(y)

    lags = sorted(
        {i + 4 * j for i in range(order + 1) for j in range(seasonal_order + 1)} - {0}
    )
    expected = _reference(
        values,
        lags,
        difference=difference,
        seasonal=seasonal,
        m=4,
        forgetting=forgetting,
        delta=10.0,
        horizon=6,
    )
    assert model.forecast_one().mean == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_docstring_example_converges_to_the_fixed_point() -> None:
    model = AutoRegressive(horizon=2, order=1)
    y = 0.0
    for _ in range(200):
        y = 0.8 * y + 1.0
        model.learn_one(y)
    assert model.forecast_one().mean == pytest.approx((5.0, 5.0), abs=1e-4)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"order": 0},
        {"horizon": 3, "difference": 2},
        {
            "horizon": 4,
            "season_length": 5,
            "seasonal_order": 2,
            "seasonal_difference": 1,
        },
    ],
)
def test_ready_exactly_when_the_longest_lead_is_first_scored(params) -> None:
    model = AutoRegressive(**params)
    required = model.readiness.required
    for index, y in enumerate(_series(60), start=1):
        model.learn_one(y)
        assert model.readiness.ready == (index >= required)
        assert (model.forecast_one() is None) == (index < required)


@pytest.mark.parametrize(("residual_alpha", "fades"), [(1e-9, False), (0.2, True)])
def test_spreads_are_the_mean_absolute_error_per_lead(residual_alpha, fades) -> None:
    values = _series(80, seed=4)
    model = AutoRegressive(
        horizon=3, order=2, forgetting=1.0, delta=10.0, residual_alpha=residual_alpha
    )
    for y in values:
        model.learn_one(y)
    # Forecasts are first issued once every weight has been updated once.
    errors = {lead: [] for lead in (1, 2, 3)}
    for origin in range(5, len(values)):
        means = _reference(
            values[:origin],
            [1, 2],
            difference=0,
            seasonal=0,
            m=4,
            forgetting=1.0,
            delta=10.0,
            horizon=3,
        )
        for lead, mean in enumerate(means, start=1):
            if origin + lead <= len(values):
                errors[lead].append(abs(values[origin + lead - 1] - mean))
    for lead, observed in errors.items():
        expected = observed[0]
        for count, error in enumerate(observed[1:], start=2):
            weight = max(residual_alpha, 1 / count) if fades else 1 / count
            expected += weight * (error - expected)
        assert model.forecast_one().scale[lead - 1] == pytest.approx(expected, rel=1e-6)


def test_seasonal_lags_capture_a_seasonal_autoregression() -> None:
    rng = np.random.default_rng(1)
    values = [0.0] * 12
    for _ in range(3000):
        values.append(0.9 * values[-12] + float(rng.standard_normal()))
    plain = AutoRegressive(order=2, residual_alpha=0.002)
    seasonal = AutoRegressive(
        order=2, season_length=12, seasonal_order=1, residual_alpha=0.002
    )
    for y in values:
        plain.learn_one(y)
        seasonal.learn_one(y)
    # The innovation has mean absolute value sqrt(2 / pi), about 0.8.
    assert seasonal.forecast_one().scale[0] == pytest.approx(0.8, rel=0.1)
    assert plain.forecast_one().scale[0] > 1.5


def test_overflow_leaves_state_unchanged_and_pickle_continues() -> None:
    model = AutoRegressive(horizon=3, order=2, difference=1)
    for y in _series(40):
        model.learn_one(y)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one(1.7e308)
    assert pickle.dumps(model) == before
    model.reset()
    clone = pickle.loads(pickle.dumps(model))
    for y in _series(40):
        model.learn_one(y)
        clone.learn_one(y)
    assert clone.forecast_one() == model.forecast_one()
