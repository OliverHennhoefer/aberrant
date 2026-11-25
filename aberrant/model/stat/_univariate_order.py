"""Order-statistic moving-window detectors."""

from __future__ import annotations

from collections.abc import Sequence

from aberrant.model.stat._univariate_base import _BaseMovingOrderStatistic
from aberrant.utils.statistics import linear_quantile


class MovingMedian(_BaseMovingOrderStatistic):
    """Score the change in median after adding a candidate value."""

    def _statistic(self, ordered: Sequence[float]) -> float:
        return linear_quantile(ordered, 0.5)

    def __repr__(self) -> str:
        return f"MovingMedian(window_size={self.window_size}, abs_diff={self.abs_diff})"


class MovingQuantile(_BaseMovingOrderStatistic):
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

    def _statistic(self, ordered: Sequence[float]) -> float:
        return linear_quantile(ordered, self.quantile)
