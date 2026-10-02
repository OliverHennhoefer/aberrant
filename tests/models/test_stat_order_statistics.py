"""Shared interpolation regressions through the moving-statistic models."""

import pytest

from aberrant.model.stat import (
    MovingInterquartileRange,
    MovingMedian,
    MovingQuantile,
)


@pytest.mark.parametrize("model_type", [MovingMedian, MovingQuantile])
def test_constant_subnormal_reference_has_no_median_change(model_type: type) -> None:
    model = model_type(window_size=2)
    model.learn_one({"x": 5e-324})
    model.learn_one({"x": 5e-324})

    assert model.score_one({"x": 5e-324}) == 0.0


def test_narrow_window_iqr_change_is_nonnegative() -> None:
    model = MovingInterquartileRange(window_size=2)
    model.learn_one({"x": 1.5000000000000004})
    model.learn_one({"x": 1.5000000000000007})

    assert model.score_one({"x": 1.500000000000001}) == 2.220446049250313e-16
