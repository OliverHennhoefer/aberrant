"""Order-statistic moving-window detectors."""

from __future__ import annotations

from aberrant.model.stat._univariate_base import _BaseMovingUnivariate
from aberrant.utils.statistics import linear_quantile


class MovingMedian(_BaseMovingUnivariate):
    """Score the change in median after adding a candidate value."""

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        current = sorted(self.window)
        candidate = sorted([*current, self._extract_value(x)])
        return self._difference(
            linear_quantile(candidate, 0.5),
            linear_quantile(current, 0.5),
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

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        current = sorted(self.window)
        candidate = sorted([*current, self._extract_value(x)])
        return self._difference(
            linear_quantile(candidate, self.quantile),
            linear_quantile(current, self.quantile),
        )
