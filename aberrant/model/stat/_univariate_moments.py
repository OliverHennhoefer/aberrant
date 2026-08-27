"""Dispersion and standardized-moment moving-window detectors."""

from __future__ import annotations

from collections.abc import Sequence

from aberrant.model.stat._univariate_base import _BaseMovingUnivariate
from aberrant.model.stat._univariate_order import _linear_quantile


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _interquartile_range(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return _linear_quantile(ordered, 0.75) - _linear_quantile(ordered, 0.25)


def _average_absolute_deviation(values: Sequence[float]) -> float:
    mean = _mean(values)
    return sum(abs(value - mean) for value in values) / len(values)


def _standardized_moment_parts(
    values: Sequence[float], order: int
) -> tuple[float, float]:
    mean = _mean(values)
    central_moment = sum((value - mean) ** order for value in values) / len(values)
    variance = _variance(values)
    denominator = variance ** (order / 2)
    return central_moment, denominator


def _standardized_moment_change(
    model: _BaseMovingUnivariate,
    x: dict[str, float],
    order: int,
) -> float:
    if not model.window:
        return 0.0

    current = list(model.window)
    candidate = [*current, model._extract_value(x)]
    current_moment, current_denominator = _standardized_moment_parts(current, order)
    candidate_moment, candidate_denominator = _standardized_moment_parts(
        candidate, order
    )
    if current_denominator == 0 or candidate_denominator == 0:
        return 0.0
    return model._difference(
        candidate_moment / candidate_denominator,
        current_moment / current_denominator,
    )


class MovingVariance(_BaseMovingUnivariate):
    """Score the change in population variance."""

    def score_one(self, x: dict[str, float]) -> float:
        return self._score_statistic_change(x, _variance)

    def __repr__(self) -> str:
        return (
            f"MovingVariance(window_size={self.window_size}, abs_diff={self.abs_diff})"
        )


class MovingInterquartileRange(_BaseMovingUnivariate):
    """Score the change in linearly interpolated interquartile range."""

    def score_one(self, x: dict[str, float]) -> float:
        self.actual_window_length = len(self.window)
        return self._score_statistic_change(x, _interquartile_range)


class MovingAverageAbsoluteDeviation(_BaseMovingUnivariate):
    """Score the change in mean absolute deviation from the mean."""

    def score_one(self, x: dict[str, float]) -> float:
        return self._score_statistic_change(x, _average_absolute_deviation)


class MovingKurtosis(_BaseMovingUnivariate):
    """Score the change in Pearson kurtosis."""

    def score_one(self, x: dict[str, float]) -> float:
        return _standardized_moment_change(self, x, 4)


class MovingSkewness(_BaseMovingUnivariate):
    """Score the change in population skewness."""

    def score_one(self, x: dict[str, float]) -> float:
        return _standardized_moment_change(self, x, 3)
