"""The score history that the calibrators of one stream share."""

from __future__ import annotations

import bisect
import math

from aberrant.utils.validation import coerce_integer_count


class ScoreHistory:
    """Scores in arrival order, and the last ``window`` of them sorted.

    Every calibrator of one score stream reads the same history. A calibrator
    learns a score in :meth:`~aberrant.calibration.OnlineCalibrator.update`
    before the score joins the history, so it adapts against the window it
    predicted from.

    Args:
        window: Number of recent scores in the calibration window.
    """

    def __init__(self, window: int) -> None:
        self.window = coerce_integer_count(window, label="window")
        self.all: list[float] = []
        self._sorted: list[float] = []
        self._max_abs = 0.0

    @property
    def count(self) -> int:
        """Number of scores seen."""
        return len(self.all)

    @property
    def window_sorted(self) -> list[float]:
        """The calibration window in ascending order. Do not mutate."""
        return self._sorted

    @property
    def max_abs_score(self) -> float:
        """Largest absolute score seen."""
        return self._max_abs

    def push(self, score: float) -> None:
        """Add a score, dropping the oldest one from the window when full."""
        self.all.append(score)
        bisect.insort_left(self._sorted, score)
        if len(self._sorted) > self.window:
            _remove_sorted(self._sorted, self.all[-1 - self.window])
        self._max_abs = max(self._max_abs, abs(score))

    def recent_range(self, n: int) -> float:
        """Largest absolute score among the last ``n`` scores."""
        return max((abs(s) for s in self.all[max(0, len(self.all) - n) :]), default=0.0)

    def spread(self, n: int) -> float:
        """Range (max - min) of the last ``n`` scores, 0 for fewer than two values."""
        recent = self.all[max(0, len(self.all) - n) :]
        if not recent:
            return 0.0
        low, high = min(recent), max(recent)
        return high - low if high > low else 0.0

    def cap(self, q: float) -> float:
        """Clip a threshold to the largest absolute score seen.

        This also makes infinite thresholds finite. A threshold is unchanged
        before the first score, and NaN becomes the largest absolute score.
        """
        if not self.all:
            return q
        if math.isnan(q):
            return self._max_abs
        return max(-self._max_abs, min(self._max_abs, q))

    def conformal_quantile(self, beta: float) -> float:
        """The ``ceil((n + 1) beta)``-th smallest score of the window.

        ``+inf`` when that rank exceeds the window, ``-inf`` when ``beta <= 0``.
        """
        return conformal_quantile(self._sorted, beta)


def conformal_quantile(sorted_scores: list[float], beta: float) -> float:
    """Conformal quantile of ascending scores at level ``beta``."""
    n = len(sorted_scores)
    if beta <= 0:
        return -math.inf
    if n == 0:
        return math.inf
    rank = math.ceil((n + 1) * beta - 1e-12)
    if rank > n:
        return math.inf
    return sorted_scores[max(rank, 1) - 1]


def pinball(y: float, q: float, tau: float) -> float:
    """Pinball (quantile) loss of predicting ``q`` for ``y`` at level ``tau``."""
    d = y - q
    return tau * d if d >= 0 else (tau - 1) * d


def _remove_sorted(values: list[float], value: float) -> None:
    index = bisect.bisect_left(values, value)
    if index < len(values) and values[index] == value:
        del values[index]
