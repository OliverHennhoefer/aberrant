"""Shared interpolation regressions through the moving-statistic models."""

import numpy as np
import pytest

from aberrant._utils.statistics import linear_quantile
from aberrant.model.stat import MovingInterquartileRange, MovingMedian, MovingQuantile


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


def _reference_change(window: list[float], value: float, statistic) -> float:
    """Score like aberrant 1.5.0: sort the window and the candidate per score."""
    current = sorted(window)
    candidate = sorted([*current, value])
    return statistic(candidate) - statistic(current)


@pytest.mark.parametrize(
    ("model_type", "params", "statistic"),
    [
        (MovingMedian, {}, lambda o: linear_quantile(o, 0.5)),
        (MovingQuantile, {"quantile": 0.0}, lambda o: linear_quantile(o, 0.0)),
        (MovingQuantile, {"quantile": 0.9}, lambda o: linear_quantile(o, 0.9)),
        (MovingQuantile, {"quantile": 1.0}, lambda o: linear_quantile(o, 1.0)),
        (
            MovingInterquartileRange,
            {},
            lambda o: linear_quantile(o, 0.75) - linear_quantile(o, 0.25),
        ),
    ],
)
@pytest.mark.parametrize("window_size", [1, 2, 7, 64])
@pytest.mark.parametrize("abs_diff", [True, False])
def test_sorted_window_scores_match_sorting_per_score(
    model_type: type, params: dict, statistic, window_size: int, abs_diff: bool
) -> None:
    rng = np.random.default_rng(window_size)
    # Repeated levels, wide magnitudes and subnormals exercise ties and the
    # interpolation edge cases.
    values = np.concatenate(
        [
            rng.integers(-3, 4, size=150).astype(float),
            rng.normal(scale=1e300, size=50),
            rng.choice([5e-324, -5e-324, 0.0, 1e-310], size=50),
        ]
    )
    rng.shuffle(values)
    model = model_type(window_size=window_size, abs_diff=abs_diff, **params)
    window: list[float] = []
    for value in values.tolist():
        expected = 0.0
        if window:
            expected = _reference_change(window, value, statistic)
            expected = abs(expected) if abs_diff else expected
        assert model.score_one({"x": value}) == expected
        model.learn_one({"x": value})
        window = [*window, value][-window_size:]
    assert list(model.window) == window
