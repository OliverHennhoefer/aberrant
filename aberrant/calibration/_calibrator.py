"""One calibrated threshold over a score stream."""

from __future__ import annotations

import math
from collections.abc import Hashable

from aberrant.calibration._registry import (
    CALIBRATORS,
    CalibratorSpec,
    OnlineCalibrator,
)
from aberrant.calibration._window import ScoreHistory
from aberrant.utils.validation import coerce_finite_number


class ScoreCalibrator:
    """A threshold on a stream of scores, learned online with delayed feedback.

    ``predict(key)`` issues the threshold for a prediction identified by
    ``key``; ``update(key, score)`` reports its score once known, which may be
    several steps later (a forecast's lead). Scores may be negative.

    Args:
        method: Identifier of a method in :data:`CALIBRATORS`.
        alpha: Target miscoverage, so ``1 - alpha`` of scores should fall at
            or below the threshold.
        window: Number of recent scores in the calibration window.
        **params: Method parameters; omitted ones take their defaults.

    Examples:
        ```python
        from aberrant.calibration import ScoreCalibrator

        calibrator = ScoreCalibrator("aci", alpha=0.1, window=100, gamma=0.01)
        for t, score in enumerate([0.2, 0.4, 0.1, 0.3]):
            threshold = calibrator.predict(t)
            calibrator.update(t, score)
        ```
    """

    def __init__(
        self, method: str, *, alpha: float, window: int = 100, **params: object
    ) -> None:
        spec = CALIBRATORS.get(method)
        if spec is None:
            known = ", ".join(sorted(CALIBRATORS))
            raise ValueError(
                f"Unknown calibration method {method!r}; use one of {known}"
            )
        alpha = coerce_finite_number(alpha, label="alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.method_id = method
        self.alpha = alpha
        self.params = spec.resolve(params)
        self.history = ScoreHistory(window)
        self.method: OnlineCalibrator = spec.factory(self.history, alpha, self.params)

    @property
    def spec(self) -> CalibratorSpec:
        """The method's description and parameters."""
        return CALIBRATORS[self.method_id]

    def predict(self, key: Hashable) -> float:
        """The threshold issued for ``key``.

        ``+inf`` before any score; otherwise clipped to the largest absolute
        score seen, as the method compares its feedback against that value.
        """
        q = self.method.predict(key)
        if self.history.count == 0:
            return math.inf
        return self.history.cap(q)

    def update(self, key: Hashable, score: float) -> None:
        """Learn the score of the prediction issued for ``key``."""
        score = coerce_finite_number(score, label="score")
        self.method.update(key, score)
        self.history.push(score)

    def __repr__(self) -> str:
        params = "".join(f", {k}={v:g}" for k, v in self.params.items())
        return (
            f"ScoreCalibrator({self.method_id!r}, alpha={self.alpha:g}, "
            f"window={self.history.window}{params})"
        )
