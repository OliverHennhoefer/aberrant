"""KalmanForecaster: exact diffuse filter, calibrated spreads, missing values."""

import math
import pickle

import numpy as np
import pytest

from aberrant.forecast import KalmanForecaster, ResidualDetector


def _matrix_filter(values, level_ratio, slope_ratio, horizon, prior=1e9):
    """A textbook Kalman filter with a large prior variance for the state."""
    order = 1 if slope_ratio is None else 2
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])[:order, :order]
    noise = np.diag([level_ratio, slope_ratio or 0.0])[:order, :order]
    observe = np.eye(order)[0]
    state, covariance = np.zeros(order), prior * np.eye(order)
    standardized = []
    observed = 0
    for value in values:
        if value is not None:
            spread = observe @ covariance @ observe + 1
            error = value - observe @ state
            gain = covariance @ observe / spread
            state = state + gain * error
            covariance = covariance - np.outer(gain, observe @ covariance)
            observed += 1
            if observed > order:
                standardized.append(error * error / spread)
        state = transition @ state
        covariance = transition @ covariance @ transition.T + noise
    variance = float(np.mean(standardized))
    means, scales = [], []
    for _ in range(horizon):
        means.append(observe @ state)
        scales.append(math.sqrt(variance * (observe @ covariance @ observe + 1)))
        state = transition @ state
        covariance = transition @ covariance @ transition.T + noise
    return means, scales


@pytest.mark.parametrize("slope_ratio", [None, 0.0, 0.01])
def test_matches_a_matrix_filter_with_a_vague_prior(slope_ratio):
    rng = np.random.default_rng(0)
    values: list[float | None] = list(
        np.cumsum(rng.standard_normal(80)) + rng.standard_normal(80)
    )
    for gap in (1, 6, 30, 31):
        values[gap] = None
    model = KalmanForecaster(
        horizon=4, level_ratio=0.3, slope_ratio=slope_ratio, residual_alpha=1e-9
    )
    for value in values:
        model.learn_one(value)

    means, scales = _matrix_filter(values, 0.3, slope_ratio, 4)
    forecast = model.forecast_one()
    assert forecast.mean == pytest.approx(means, rel=1e-7)
    assert forecast.scale == pytest.approx(scales, rel=1e-7)


def test_a_line_is_forecast_exactly_across_a_gap():
    model = KalmanForecaster(horizon=3, slope_ratio=0.0)
    for value in [1.0, 2.0, 3.0, None, 5.0, 6.0]:
        model.learn_one(value)
    assert model.forecast_one().mean == pytest.approx((7.0, 8.0, 9.0), abs=1e-12)


@pytest.mark.parametrize(("slope_ratio", "required"), [(None, 2), (0.01, 3)])
def test_readiness_counts_observed_values(slope_ratio, required):
    model = KalmanForecaster(slope_ratio=slope_ratio)
    model.learn_one(None)
    for _ in range(required - 1):
        model.learn_one(1.0)
        assert model.forecast_one() is None
    model.learn_one(2.0)
    assert model.readiness.ready
    assert model.readiness.observed == required
    assert model.forecast_one().origin == required + 1


@pytest.mark.parametrize("level_ratio", [0.05, 0.5, 4.0])
def test_a_local_level_settles_to_exponential_smoothing(level_ratio):
    rng = np.random.default_rng(1)
    model = KalmanForecaster(level_ratio=level_ratio)
    for value in rng.standard_normal(200):
        model.learn_one(value)
    before = model.forecast_one().point(1)
    model.learn_one(before + 1.0)

    weight = (math.sqrt(level_ratio**2 + 4 * level_ratio) - level_ratio) / 2
    assert model.forecast_one().point(1) - before == pytest.approx(weight, rel=1e-9)


@pytest.mark.parametrize("slope_ratio", [None, 0.001])
def test_spreads_are_calibrated_at_every_lead(slope_ratio):
    rng = np.random.default_rng(2)
    n, horizon = 20_000, 6
    level = np.cumsum(math.sqrt(0.1) * rng.standard_normal(n))
    if slope_ratio is not None:
        slope = np.cumsum(math.sqrt(slope_ratio) * rng.standard_normal(n))
        level += np.cumsum(slope)
    values = 3.0 * (level + rng.standard_normal(n))

    model = KalmanForecaster(
        horizon=horizon, level_ratio=0.1, slope_ratio=slope_ratio, residual_alpha=0.01
    )
    issued = []
    inside = np.zeros(horizon)
    counted = 0
    for t, value in enumerate(values):
        if t >= 500:
            counted += 1
            for lead in range(1, horizon + 1):
                forecast = issued[t - lead]
                z = (value - forecast.point(lead)) / forecast.spread(lead)
                inside[lead - 1] += abs(z) <= 1.645
        model.learn_one(value)
        issued.append(model.forecast_one())

    assert inside / counted == pytest.approx([0.9] * horizon, abs=0.015)


def test_a_missing_value_keeps_the_level_and_widens_the_spread():
    model = KalmanForecaster(horizon=2)
    for value in [1.0, 3.0, 2.0, 2.5]:
        model.learn_one(value)
    before = model.forecast_one()
    model.learn_one(None)
    after = model.forecast_one()

    assert after.origin == before.origin + 1
    assert after.mean == before.mean
    assert after.point(1) == before.point(1)
    assert after.spread(1) > before.spread(1)
    assert after.spread(1) == pytest.approx(before.spread(2))


def test_a_constant_series_reports_the_spread_floor():
    model = KalmanForecaster(horizon=2, min_scale=1e-3)
    for _ in range(10):
        model.learn_one(4.0)
    assert model.forecast_one().scale == (1e-3, 1e-3)


def test_residual_detector_scores_in_predictive_deviations():
    rng = np.random.default_rng(4)
    detector = ResidualDetector(KalmanForecaster(), target="y")
    scores = []
    for value in rng.standard_normal(3000):
        scores.append(detector.score_one({"y": value}))
        detector.learn_one({"y": value})
    assert np.mean(np.array(scores[100:]) <= 1.645) == pytest.approx(0.9, abs=0.02)


def test_overflow_leaves_state_unchanged_and_pickle_continues():
    model = KalmanForecaster(horizon=3, slope_ratio=0.01)
    for value in [1.0, 2.0, 2.5, 4.0]:
        model.learn_one(value)
    model.learn_one(1e150)
    before = pickle.dumps(model)
    with pytest.raises(OverflowError):
        model.learn_one(-1e200)
    assert pickle.dumps(model) == before

    clone = pickle.loads(before)
    for value in (11.0, None, 10.5):
        model.learn_one(value)
        clone.learn_one(value)
    assert clone.forecast_one() == model.forecast_one()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"level_ratio": -0.1},
        {"level_ratio": math.inf},
        {"slope_ratio": -1.0},
        {"slope_ratio": math.nan},
        {"residual_alpha": 0.0},
        {"min_scale": 0.0},
        {"horizon": 0},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        KalmanForecaster(**kwargs)


def test_reset_forgets_everything():
    model = KalmanForecaster()
    for value in [1.0, 2.0, 3.0]:
        model.learn_one(value)
    model.reset()
    assert not model.readiness.ready
    assert model.forecast_one() is None
    assert model.n_samples_seen == 0


def _seasonal_filter(values, ratios, season, horizon, prior=1e9):
    """A textbook filter of a level, a slope and a trigonometric season."""
    level_ratio, slope_ratio, season_ratio = ratios
    season_length, harmonics = season
    blocks = [np.array([[1.0, 1.0], [0.0, 1.0]])]
    observe, noise = [1.0, 0.0], [level_ratio, slope_ratio]
    for j in range(1, harmonics + 1):
        frequency = 2 * np.pi * j / season_length
        if 2 * j == season_length:
            blocks.append(np.array([[np.cos(frequency)]]))
            observe += [1.0]
            noise += [season_ratio]
        else:
            c, s = np.cos(frequency), np.sin(frequency)
            blocks.append(np.array([[c, s], [-s, c]]))
            observe += [1.0, 0.0]
            noise += [season_ratio, season_ratio]
    size = len(observe)
    transition = np.zeros((size, size))
    at = 0
    for block in blocks:
        transition[at : at + len(block), at : at + len(block)] = block
        at += len(block)
    observe, noise = np.array(observe), np.diag(noise)
    state, covariance = np.zeros(size), prior * np.eye(size)
    standardized = []
    for value in values:
        if value is not None:
            spread = observe @ covariance @ observe + 1
            error = value - observe @ state
            gain = covariance @ observe / spread
            state = state + gain * error
            covariance = covariance - np.outer(gain, observe @ covariance)
            # A value the prior dominates fits the state and gives no error.
            if spread < math.sqrt(prior):
                standardized.append(error * error / spread)
        state = transition @ state
        covariance = transition @ covariance @ transition.T + noise
    variance = float(np.mean(standardized))
    means, scales = [], []
    for _ in range(horizon):
        means.append(observe @ state)
        scales.append(math.sqrt(variance * (observe @ covariance @ observe + 1)))
        state = transition @ state
        covariance = transition @ covariance @ transition.T + noise
    return means, scales


@pytest.mark.parametrize(("season_length", "harmonics"), [(4, 2), (7, 3), (12, 2)])
def test_a_season_matches_a_matrix_filter_with_a_vague_prior(season_length, harmonics):
    rng = np.random.default_rng(5)
    t = np.arange(120)
    values: list[float | None] = list(
        np.cumsum(0.3 * rng.standard_normal(120))
        + 2 * np.sin(2 * np.pi * t / season_length)
        + rng.standard_normal(120)
    )
    for gap in (2, 40, 41, 77):
        values[gap] = None
    model = KalmanForecaster(
        horizon=2 * season_length,
        level_ratio=0.1,
        slope_ratio=0.001,
        season_length=season_length,
        season_ratio=0.01,
        harmonics=harmonics,
        residual_alpha=1e-9,
    )
    for value in values:
        model.learn_one(value)

    means, scales = _seasonal_filter(
        values, (0.1, 0.001, 0.01), (season_length, harmonics), 2 * season_length
    )
    forecast = model.forecast_one()
    assert forecast.mean == pytest.approx(means, rel=1e-6)
    assert forecast.scale == pytest.approx(scales, rel=1e-6)


def test_a_fixed_season_is_forecast_exactly():
    pattern = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0]
    model = KalmanForecaster(
        horizon=8, level_ratio=0.0, slope_ratio=0.0, season_length=6
    )
    for t in range(30):
        model.learn_one(10.0 + 0.5 * t + pattern[t % 6])
    expected = [10.0 + 0.5 * t + pattern[t % 6] for t in range(30, 38)]
    assert model.forecast_one().mean == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "required"),
    [
        ({"season_length": 12}, 13),
        ({"season_length": 12, "harmonics": 2, "slope_ratio": 0.0}, 7),
        ({"season_length": 7}, 8),
    ],
)
def test_a_season_is_ready_after_one_value_per_state(kwargs, required):
    model = KalmanForecaster(**kwargs)
    assert model.readiness.required == required
    rng = np.random.default_rng(6)
    for _ in range(required):
        assert model.forecast_one() is None
        model.learn_one(float(rng.standard_normal()))
    assert model.readiness.ready and model.readiness.observed == required


def test_seasonal_spreads_are_calibrated_at_every_lead():
    rng = np.random.default_rng(7)
    n, horizon, season = 20_000, 12, 12
    level = np.cumsum(math.sqrt(0.05) * rng.standard_normal(n))
    pattern = 3 * np.sin(2 * np.pi * np.arange(n) / season)
    values = 2.0 * (level + pattern + rng.standard_normal(n))

    model = KalmanForecaster(
        horizon=horizon, season_length=season, harmonics=1, residual_alpha=0.01
    )
    issued = []
    inside = np.zeros(horizon)
    counted = 0
    for t, value in enumerate(values):
        if t >= 500:
            counted += 1
            for lead in range(1, horizon + 1):
                forecast = issued[t - lead]
                z = (value - forecast.point(lead)) / forecast.spread(lead)
                inside[lead - 1] += abs(z) <= 1.645
        model.learn_one(value)
        issued.append(model.forecast_one())

    assert inside / counted == pytest.approx([0.9] * horizon, abs=0.015)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"season_length": 0},
        {"season_length": 2.0},
        {"season_ratio": -0.1},
        {"harmonics": 1},
        {"season_length": 12, "harmonics": 0},
        {"season_length": 12, "harmonics": 7},
    ],
)
def test_rejects_invalid_seasons(kwargs):
    with pytest.raises(ValueError):
        KalmanForecaster(**kwargs)
