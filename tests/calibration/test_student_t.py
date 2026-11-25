"""The Student-t helpers behind the Bayesian calibrators, checked against SciPy."""

import math

import pytest

from aberrant.calibration._stats import (
    StudentT,
    betai,
    lgamma,
    norm_cdf,
    t_cdf_std,
    t_mixture_quantile,
)

special = pytest.importorskip("scipy.special")
stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize("x", [0.05, 0.3, 0.5, 1.0, 2.5, 30.0, 170.5])
def test_lgamma_matches_the_standard_library(x):
    assert lgamma(x) == pytest.approx(math.lgamma(x), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("a", "b", "x"),
    [
        (0.5, 0.5, 0.1),
        (2.0, 3.0, 0.4),
        (2.0, 3.0, 0.9),
        (30.0, 0.5, 0.97),
        (1e3, 1e3, 0.5),
    ],
)
def test_incomplete_beta_matches_scipy(a, b, x):
    assert betai(a, b, x) == pytest.approx(special.betainc(a, b, x), abs=1e-10)


@pytest.mark.parametrize(("x", "expected"), [(-0.5, 0.0), (0.0, 0.0), (1.0, 1.0)])
def test_incomplete_beta_is_clamped_outside_the_unit_interval(x, expected):
    assert betai(2.0, 3.0, x) == expected


@pytest.mark.parametrize("z", [-4.0, -1.0, 0.0, 0.7, 3.0])
def test_normal_cdf_is_accurate_to_its_approximation_error(z):
    assert norm_cdf(z) == pytest.approx(stats.norm.cdf(z), abs=2e-7)


@pytest.mark.parametrize(
    ("nu", "tolerance"),
    [(1, 1e-12), (2, 1e-12), (5, 1e-12), (12, 1e-12), (2.5, 1e-9), (80, 1e-4)],
    ids=["cauchy", "even", "odd", "even-large", "fractional", "normal-limit"],
)
@pytest.mark.parametrize("z", [-6.0, -1.5, 0.0, 0.3, 4.0])
def test_t_cdf_matches_scipy_on_each_branch(z, nu, tolerance):
    assert t_cdf_std(z, nu) == pytest.approx(stats.t.cdf(z, nu), abs=tolerance)


_MIXTURE = [StudentT(3.0, -2.0, 1.0), StudentT(7.0, 4.0, 0.5), StudentT(2.0, 0.0, 9.0)]


def _mixture_cdf(x, weights):
    return sum(
        w * stats.t.cdf(x, c.nu, loc=c.mu, scale=c.sigma)
        for c, w in zip(_MIXTURE, weights, strict=True)
    ) / sum(weights)


@pytest.mark.parametrize("p", [1e-4, 0.05, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("guess", [None, 1e6], ids=["mean", "far-guess"])
def test_mixture_quantile_inverts_the_mixture_cdf(p, guess):
    weights = [0.5, 0.3, 0.2]
    x = t_mixture_quantile(_MIXTURE, weights, p, guess=guess)
    assert _mixture_cdf(x, weights) == pytest.approx(p, abs=1e-6)


def test_components_without_weight_are_ignored():
    x = t_mixture_quantile(_MIXTURE, [0.0, 1.0, 0.0], 0.75)
    assert x == pytest.approx(stats.t.ppf(0.75, 7.0, loc=4.0, scale=0.5), abs=1e-6)


@pytest.mark.parametrize(("p", "expected"), [(0.0, -math.inf), (1.0, math.inf)])
def test_mixture_quantile_of_the_boundaries_is_infinite(p, expected):
    assert t_mixture_quantile(_MIXTURE, [0.5, 0.3, 0.2], p) == expected


def test_mixture_without_weight_has_no_quantile():
    assert math.isnan(t_mixture_quantile(_MIXTURE, [0.0, 0.0, 0.0], 0.5))
