"""SPOT and DSPOT: generalized Pareto tails over a window of scores."""

import math
import random

import numpy as np
import pytest

from aberrant.calibration import CALIBRATORS, ScoreCalibrator
from aberrant.calibration._extreme import MIN_SCORES, fit_gpd


@pytest.mark.parametrize(
    ("sample", "gamma"),
    [
        (lambda rng, n: rng.exponential(1.0, n), 0.0),
        (lambda rng, n: rng.pareto(3.0, n), 1 / 3),
        (lambda rng, n: (1 - rng.random(n) ** 0.2) / 0.2, -0.2),
    ],
    ids=["exponential", "heavy", "bounded"],
)
def test_the_fit_recovers_the_tail_shape(sample, gamma):
    fitted, sigma = fit_gpd(sample(np.random.default_rng(4), 4000))

    assert fitted == pytest.approx(gamma, abs=0.05)
    assert sigma > 0


def _run(method, scores, *, alpha, window, **params):
    calibrator = ScoreCalibrator(method, alpha=alpha, window=window, **params)
    thresholds = []
    for t, score in enumerate(scores):
        thresholds.append(calibrator.predict(t))
        calibrator.update(t, score)
    return np.asarray(thresholds)


def _exponential(n, seed=7, scale=1.0):
    rng = random.Random(seed)
    return [rng.expovariate(1 / scale) for _ in range(n)]


@pytest.mark.parametrize("method", ["spot", "dspot"])
def test_the_threshold_is_the_tail_quantile(method):
    alpha, window = 1e-3, 5000
    scores = np.asarray(_exponential(40_000))
    thresholds = _run(method, scores, alpha=alpha, window=window)[window:]

    assert thresholds.mean() == pytest.approx(-math.log(alpha), rel=0.05)
    rate = np.mean(scores[window:] > thresholds)
    assert 0.5 * alpha < rate < 2 * alpha


def test_spot_extrapolates_where_the_window_holds_no_such_score():
    # alpha * window = 0.2: the window's conformal quantile would be +inf.
    alpha, window = 1e-4, 2000
    scores = _exponential(30_000)
    thresholds = _run("spot", scores, alpha=alpha, window=window)[window:]

    assert np.mean(thresholds) == pytest.approx(-math.log(alpha), rel=0.1)


def test_spot_comes_down_after_the_scores_do_and_clipping_does_not():
    alpha, window = 1e-4, 2000
    scores = _exponential(20_000) + _exponential(20_000, seed=8, scale=0.5)
    spot = _run("spot", scores, alpha=alpha, window=window)
    split = _run("split", scores, alpha=alpha, window=window)

    later = slice(30_000, None)
    assert spot[later].mean() == pytest.approx(-0.5 * math.log(alpha), rel=0.15)
    # The split threshold is +inf there, clipped to the largest score so far.
    assert split[later].min() == pytest.approx(max(scores[:20_000]))


def test_dspot_follows_a_drifting_level():
    alpha, window = 1e-3, 1000
    rng = random.Random(3)
    scores = np.asarray([t / 250 + rng.expovariate(1.0) for t in range(10_000)])
    later = slice(2000, None)
    rates = {
        method: np.mean(
            scores[later] > _run(method, scores, alpha=alpha, window=window)[later]
        )
        for method in ("spot", "dspot")
    }

    assert rates["dspot"] < 2 * alpha < rates["spot"]


def test_before_a_fit_the_threshold_is_the_window_quantile():
    scores = _exponential(MIN_SCORES - 1)
    spot = ScoreCalibrator("spot", alpha=0.1, window=200)
    split = ScoreCalibrator("split", alpha=0.1, window=200)
    for t, score in enumerate(scores):
        assert spot.predict(t) == split.predict(t)
        spot.update(t, score)
        split.update(t, score)


def test_the_extreme_value_methods_are_registered_as_such():
    for method in ("spot", "dspot"):
        spec = CALIBRATORS[method]
        assert (spec.family, spec.coverage, spec.extrapolates) == (
            "extreme_value",
            "none",
            True,
        )
    assert CALIBRATORS["dspot"].defaults() == {"level": 0.98, "depth": 10}
