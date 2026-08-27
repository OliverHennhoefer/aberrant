"""KSWIN (Kolmogorov-Smirnov WINdowing) drift detector."""

import random
from collections import deque

from scipy.stats import ks_2samp

from aberrant.drift.base import BaseDriftDetector, _finite_observation


class KSWIN(BaseDriftDetector):
    """
    KSWIN (Kolmogorov-Smirnov WINdowing) drift detector.

    KSWIN detects concept drift by comparing recent observations with
    historical data using the Kolmogorov-Smirnov two-sample test.
    This is a distribution-free test that makes no assumptions about
    the underlying data distribution.

    The detector maintains a sliding window and compares the most recent
    samples with a random sample from the earlier part of the window.

    Args:
        alpha: Significance level for the KS test. Lower values require
            stronger evidence to detect drift. Default is 0.005.
        window_size: Size of the sliding window. Default is 100.
        stat_size: Number of samples to use for comparison. Must be
            less than window_size / 2. Default is 30.
        seed: Random seed for reproducibility. Default is None.

    Example:
        >>> detector = KSWIN(alpha=0.005)
        >>> for value in data_stream:
        ...     detector.update(value)
        ...     if detector.drift_detected:
        ...         print("Drift detected!")

    References:
        Raab, C., Heusinger, M., & Schleif, F. M. (2020). Reactive Soft
        Prototype Computing for Concept Drift Streams.
        Neurocomputing, 416, 340-351.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stat_size <= 0:
            raise ValueError("stat_size must be positive")
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

    def update(self, x: float) -> "KSWIN":
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

        return self

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

        # Perform KS test
        statistic, p_value = ks_2samp(recent, historical, method="asymp")
        self._statistic = statistic
        self._p_value = p_value

        # Detect drift if p-value is below threshold and statistic is significant
        if p_value <= self.alpha and statistic > 0.1:
            self._drift_detected = True
            self._n_detections += 1
            # Reset window and RNG on drift
            self._window.clear()
            self._rng = random.Random(self.seed)

    def __repr__(self) -> str:
        return (
            f"KSWIN(alpha={self.alpha}, window_size={self.window_size}, "
            f"stat_size={self.stat_size})"
        )
