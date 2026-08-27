"""Order-statistic moving-window detectors."""

from __future__ import annotations

from collections.abc import Sequence

from aberrant.model.stat._univariate_base import _BaseMovingUnivariate


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Match NumPy's default linear interpolation for an already sorted sample."""
    rank = (len(sorted_values) - 1) * quantile
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    return float(
        (1 - fraction) * sorted_values[lower_index]
        + fraction * sorted_values[upper_index]
    )


class MovingMedian(_BaseMovingUnivariate):
    """Score the change in median after adding a candidate value."""

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        current = sorted(self.window)
        candidate = sorted([*current, self._extract_value(x)])
        return self._difference(
            _linear_quantile(candidate, 0.5),
            _linear_quantile(current, 0.5),
        )

    def __repr__(self) -> str:
        return f"MovingMedian(window_size={self.window_size}, abs_diff={self.abs_diff})"


class MovingQuantile(_BaseMovingUnivariate):
    """Score the change in a configured linearly interpolated quantile."""

    def __init__(
        self,
        window_size: int,
        key: str | None = None,
        quantile: float = 0.5,
        abs_diff: bool = True,
    ) -> None:
        super().__init__(window_size, key, abs_diff)
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be between 0 and 1.")
        self.quantile = quantile

    def _quantile(self, sorted_list: list[float]) -> float:
        return _linear_quantile(sorted_list, self.quantile)

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        current = sorted(self.window)
        candidate = sorted([*current, self._extract_value(x)])
        return self._difference(self._quantile(candidate), self._quantile(current))
