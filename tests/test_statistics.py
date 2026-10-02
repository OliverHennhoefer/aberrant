"""Numerical contracts for shared linear quantile interpolation."""

import math
import sys
from fractions import Fraction

import numpy as np
import pytest

from aberrant.utils.statistics import linear_quantile


@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, -sys.float_info.max), (1.0, sys.float_info.max)],
)
def test_quantile_endpoints(probability: float, expected: float) -> None:
    values = (-sys.float_info.max, 0.0, sys.float_info.max)

    assert linear_quantile(values, probability) == expected


@pytest.mark.parametrize(
    "value", [0.0, 5e-324, -5e-324, sys.float_info.max, -sys.float_info.max]
)
@pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_singleton_and_constant_samples(value: float, probability: float) -> None:
    assert linear_quantile((value,), probability) == value
    assert linear_quantile((value,) * 4, probability) == value


@pytest.mark.parametrize("length", [2, 3, 4, 7, 16])
@pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ordinary_quantiles_match_numpy(length: int, probability: float) -> None:
    values = sorted(np.random.default_rng(42).normal(size=length).tolist())
    expected = float(np.quantile(values, probability, method="linear"))

    assert linear_quantile(values, probability) == pytest.approx(
        expected, rel=1e-14, abs=1e-15
    )


@pytest.mark.parametrize(
    "values",
    [
        (1.5000000000000004, 1.5000000000000007),
        (100.00000000000003, 100.00000000000004),
        (-1.5000000000000007, -1.5000000000000004),
        (5e-324, 1e-323),
        (1e-323, 1.5e-323),
        (-1.5e-323, -1e-323),
    ],
)
def test_adjacent_float_quartiles_are_correctly_rounded_and_ordered(
    values: tuple[float, float],
) -> None:
    lower, upper = values
    assert math.nextafter(lower, math.inf) == upper
    probabilities = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    expected = [
        float(Fraction(lower) * (1 - probability) + Fraction(upper) * probability)
        for probability in probabilities
    ]
    actual = [
        linear_quantile(values, float(probability)) for probability in probabilities
    ]

    assert actual == expected
    assert lower <= actual[0] <= actual[1] <= actual[2] <= upper


@pytest.mark.parametrize(
    "values",
    [
        (-sys.float_info.max, sys.float_info.max),
        (-1e308, 1e308),
        (-1e308, 8e307),
    ],
)
@pytest.mark.parametrize("probability", [0.25, 0.5, 0.75])
def test_finite_quantiles_when_endpoint_difference_overflows(
    values: tuple[float, float], probability: float
) -> None:
    lower, upper = values
    assert math.isinf(upper - lower)
    weight = Fraction(probability)
    expected = float(Fraction(lower) * (1 - weight) + Fraction(upper) * weight)
    actual = linear_quantile(values, probability)

    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-15, abs=0.0)
