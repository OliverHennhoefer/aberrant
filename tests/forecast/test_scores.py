"""Nonconformity scores and the intervals their thresholds imply."""

import math

import pytest

from aberrant.base import Forecast
from aberrant.forecast import interval, nonconformity

POINT = Forecast(origin=3, leads=(1, 2), mean=(10.0, 12.0), scale=(2.0, 4.0))
QUANTILES = Forecast(
    origin=3,
    leads=(1,),
    mean=(10.0,),
    levels=(0.05, 0.5, 0.95),
    quantiles=((8.0, 10.0, 13.0),),
)


def test_absolute_and_scaled() -> None:
    assert nonconformity(POINT, 1, 13.0) == 3.0
    assert nonconformity(POINT, 2, 4.0, "scaled") == 2.0
    assert interval(POINT, 1, 3.0) == (7.0, 13.0)
    assert interval(POINT, 2, 0.5, "scaled") == (10.0, 14.0)


def test_cqr_is_signed_and_needs_quantiles() -> None:
    assert nonconformity(QUANTILES, 1, 9.0, "cqr", alpha=0.1) == -1.0
    assert nonconformity(QUANTILES, 1, 15.0, "cqr", alpha=0.1) == 2.0
    assert interval(QUANTILES, 1, 1.0, "cqr", alpha=0.1) == (7.0, 14.0)
    assert interval(QUANTILES, 1, -10.0, "cqr", alpha=0.1) == (10.5, 10.5)
    with pytest.raises(ValueError, match="alpha"):
        nonconformity(QUANTILES, 1, 9.0, "cqr")
    with pytest.raises(ValueError, match="quantiles"):
        nonconformity(POINT, 1, 9.0, "cqr", alpha=0.1)


def test_score_inside_interval_iff_at_most_threshold() -> None:
    for score, forecast, alpha in (
        ("absolute", POINT, None),
        ("scaled", POINT, None),
        ("cqr", QUANTILES, 0.1),
    ):
        lo, hi = interval(forecast, 1, 1.5, score, alpha=alpha)  # type: ignore[arg-type]
        for y in (lo - 1e-9, lo, (lo + hi) / 2, hi, hi + 1e-9):
            inside = lo <= y <= hi
            s = nonconformity(forecast, 1, y, score, alpha=alpha)  # type: ignore[arg-type]
            assert (s <= 1.5 + 1e-12) == inside


def test_infinite_threshold_and_bad_input() -> None:
    assert interval(POINT, 1, math.inf) == (-math.inf, math.inf)
    with pytest.raises(ValueError):
        interval(POINT, 1, math.nan)
    with pytest.raises(ValueError):
        nonconformity(POINT, 1, 1.0, "squared")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spread"):
        nonconformity(Forecast(origin=0, leads=(1,), mean=(0.0,)), 1, 1.0, "scaled")
    with pytest.raises(ValueError):
        nonconformity(POINT, 1, math.inf)
