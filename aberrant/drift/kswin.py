"""KSWIN (Kolmogorov-Smirnov WINdowing) drift detector."""

import random
from bisect import bisect_right
from collections import deque
from itertools import chain

from aberrant.drift.base import BaseDriftDetector, _finite_observation
from aberrant.utils.validation import coerce_integer_count


class KSWIN(BaseDriftDetector):
    """
    KSWIN (Kolmogorov-Smirnov WINdowing) drift detector.

    KSWIN detects concept drift by comparing recent observations with
    historical data using the Kolmogorov-Smirnov two-sample test.
    This is a distribution-free test that makes no assumptions about
    the underlying data distribution.

    The detector maintains a sliding window and compares the most recent
    samples with a random sample from the earlier part of the window, using
    the exact two-sample KS p-value. After a drift, the most recent
    ``stat_size`` samples become the new window.

    Args:
        alpha: Significance level for the KS test. Lower values require
            stronger evidence to detect drift. Default is 0.005.
        window_size: Size of the sliding window. Default is 100.
        stat_size: Number of samples to use for comparison. Must be
            less than window_size / 2. Default is 30.
        seed: Random seed for reproducibility. Default is None.

    Examples:
        ```python
        from aberrant.drift import KSWIN

        detector = KSWIN(alpha=0.005)
        drift_points = []
        for index, value in enumerate([0.0] * 100 + [1.0] * 100):
            if detector.update(value):
                drift_points.append(index)
        ```

    References:
        Raab, C., Heusinger, M., & Schleif, F. M. (2020). Reactive Soft
        Prototype Computing for Concept Drift Streams.
        Neurocomputing, 416, 340-351.
        https://doi.org/10.1016/j.neucom.2019.11.111
        Reference implementation:
        https://github.com/scikit-multiflow/scikit-multiflow/blob/master/src/skmultiflow/drift_detection/kswin.py
    """

    def __init__(
        self,
        *,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
    ) -> None:
        window_size = coerce_integer_count(window_size, label="window_size", minimum=1)
        stat_size = coerce_integer_count(stat_size, label="stat_size", minimum=1)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if stat_size >= window_size / 2:
            raise ValueError("stat_size must be less than window_size / 2")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

        self._rng = random.Random(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        self._window: deque[float] = deque(maxlen=self.window_size)
        self._drift_detected: bool = False
        self._n_detections: int = 0
        self._p_value: float | None = None
        self._statistic: float | None = None

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._rng = random.Random(self.seed)
        self._reset_state()

    @property
    def drift_detected(self) -> bool:
        """Return True if drift was detected on the last update."""
        return self._drift_detected

    @property
    def n_detections(self) -> int:
        """Total number of drift detections."""
        return self._n_detections

    @property
    def p_value(self) -> float | None:
        """P-value from the last KS test, or None if not yet computed."""
        return self._p_value

    @property
    def statistic(self) -> float | None:
        """KS statistic from the last test, or None if not yet computed."""
        return self._statistic

    def update(self, x: float) -> bool:
        """
        Update the detector with a new observation.

        Args:
            x: The observed value.

        Returns:
            self: Returns self for method chaining.
        """
        x = _finite_observation(x)
        self._drift_detected = False
        self._window.append(x)

        # Only check for drift when we have enough data
        if len(self._window) >= self.window_size:
            self._check_drift()

        return self._drift_detected

    def _check_drift(self) -> None:
        """Perform the KS test to check for drift."""
        window_values = list(self._window)

        # Recent samples (most recent stat_size observations)
        recent = window_values[-self.stat_size :]

        # Historical samples (random sample from earlier window)
        historical_indices = range(len(window_values) - self.stat_size)
        sample_indices = self._rng.sample(
            historical_indices, min(self.stat_size, len(historical_indices))
        )
        historical = [window_values[i] for i in sample_indices]

        statistic, p_value = _ks_two_sample(recent, historical)
        self._statistic = statistic
        self._p_value = p_value

        # Detect drift if p-value is below threshold and statistic is significant
        if p_value <= self.alpha and statistic > 0.1:
            self._drift_detected = True
            self._n_detections += 1
            # The recent samples start the window of the new concept.
            self._window = deque(recent, maxlen=self.window_size)

    def __repr__(self) -> str:
        return (
            f"KSWIN(alpha={self.alpha}, window_size={self.window_size}, "
            f"stat_size={self.stat_size})"
        )


def _ks_two_sample(first: list[float], second: list[float]) -> tuple[float, float]:
    """Return the two-sided KS statistic and its exact p-value for equal sizes.

    The result equals ``scipy.stats.ks_2samp(first, second, method="exact")``.
    With equal sizes ``n`` the statistic is ``h / n`` for the largest count
    difference ``h`` of the two empirical distributions, and the p-value is the
    share of lattice paths that leave the band ``|x - y| < h``, summed in
    SciPy's nested form to avoid cancellation.
    """
    n = len(first)
    if n == 0 or len(second) != n:
        raise ValueError("KS samples must be non-empty and of equal size")
    first = sorted(first)
    second = sorted(second)
    h = max(
        abs(bisect_right(first, value) - bisect_right(second, value))
        for value in chain(first, second)
    )
    if h == 0:
        return 0.0, 1.0
    probability = 0.0
    for k in range(n // h, -1, -1):
        term = 1.0
        for j in range(h):
            term = (n - k * h - j) * term / (n + k * h + j + 1)
        probability = term * (1.0 - probability)
    return h / n, min(max(2 * probability, 0.0), 1.0)
