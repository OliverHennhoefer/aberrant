"""Quantile-based adaptive threshold for anomaly detection."""

import math
from collections import deque

import numpy as np

from aberrant.base.model import BaseModel


class QuantileThreshold(BaseModel):
    """
    Adaptive threshold model based on score distribution quantiles.

    This model maintains a sliding window of anomaly scores and computes
    an adaptive threshold based on a specified quantile. Points scoring
    above the threshold are classified as anomalies.

    Unlike static thresholds, QuantileThreshold adapts to the actual
    distribution of scores observed during streaming.

    Args:
        quantile: Quantile level for the threshold. Values closer to 1.0
            result in higher thresholds (fewer detections). Default is 0.95.
        window_size: Number of scores to keep for quantile computation.
            Default is 1000.
        score_key: Name of the score feature in the input dictionary.
            Default is "score".

    Examples:
        ```python
        from aberrant.model import QuantileThreshold

        threshold = QuantileThreshold(quantile=0.8, window_size=10)
        for score in range(10):
            threshold.learn_one({"score": float(score)})
        assert threshold.score_one({"score": 10.0}) == 1.0
        ```

    Note:
        The model expects input dictionaries with a score key (default ``"score"``).
        ``score_one`` returns:
        - 1.0 if the score is greater than or equal to the threshold (anomaly)
        - max(score/threshold, 0.0) if below threshold (normalized, in [0, 1))
        - 0.0 during warmup (insufficient data for threshold)
    """

    def __init__(
        self,
        quantile: float = 0.95,
        window_size: int = 1000,
        score_key: str = "score",
    ) -> None:
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not score_key:
            raise ValueError("score_key cannot be empty")

        self.quantile = quantile
        self.window_size = window_size
        self.score_key = score_key

        self._scores: deque[float] = deque(maxlen=window_size)
        self._threshold: float | None = None
        self._min_samples: int = min(
            window_size,
            max(10, int(window_size * 0.1)),
        )

    @property
    def threshold(self) -> float | None:
        """
        Current adaptive threshold.

        Returns None if insufficient data has been collected.
        """
        return self._threshold

    @property
    def n_scores(self) -> int:
        """Number of scores currently in the window."""
        return len(self._scores)

    @staticmethod
    def _validated_score(value: float) -> float:
        """Return a finite Python float without allowing persistent state poisoning."""
        try:
            score = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError("score must be numeric") from e
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        return score

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the threshold estimate with a new score.

        Args:
            x: Dictionary containing the score. Must have the score_key.
        """
        if self.score_key not in x:
            raise ValueError(f"Input must contain '{self.score_key}' key")

        score = self._validated_score(x[self.score_key])
        self._scores.append(score)

        # Update threshold if we have enough samples
        if len(self._scores) >= self._min_samples:
            self._update_threshold()

    def _update_threshold(self) -> None:
        """Recompute the quantile-based threshold."""
        scores_array = np.array(self._scores)
        self._threshold = float(np.quantile(scores_array, self.quantile))

    def score_one(self, x: dict[str, float]) -> float:
        """
        Evaluate a score against the adaptive threshold.

        Args:
            x: Dictionary containing the score. Must have the score_key.

        Returns:
            - 1.0 if score >= threshold (anomaly)
            - score/threshold if score < threshold (normalized)
            - 0.0 if threshold not yet computed (warmup period)
        """
        if self.score_key not in x:
            raise ValueError(f"Input must contain '{self.score_key}' key")

        score = self._validated_score(x[self.score_key])

        # During warmup, return 0 (no anomalies)
        if self._threshold is None:
            return 0.0

        # Avoid division by zero
        if self._threshold <= 0:
            return 1.0 if score > 0 else 0.0

        # Classify as anomaly if above threshold
        if score >= self._threshold:
            return 1.0

        # Return normalized score
        return max(score / self._threshold, 0.0)

    def reset(self) -> None:
        """Reset the model to its initial state."""
        self._scores.clear()
        self._threshold = None

    def __repr__(self) -> str:
        return (
            f"QuantileThreshold(quantile={self.quantile}, "
            f"window_size={self.window_size}, threshold={self._threshold})"
        )
