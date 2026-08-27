"""Page-Hinkley drift detector."""

from typing import Literal

from aberrant.drift.base import BaseDriftDetector, _finite_observation


class PageHinkley(BaseDriftDetector):
    """
    Page-Hinkley drift detector.

    The Page-Hinkley test is a sequential analysis technique for detecting
    changes in the mean of a distribution. It is based on the cumulative
    sum (CUSUM) control chart method.

    The detector monitors the cumulative deviation from the running mean
    and triggers drift when this deviation exceeds a threshold.

    Args:
        min_instances: Minimum number of observations before detection
            starts. Default is 30.
        delta: Magnitude of changes to tolerate. Smaller values make the
            detector more sensitive. Default is 0.005.
        threshold: Detection threshold (lambda). When the test statistic
            exceeds this value, drift is detected. Default is 50.0.
        alpha: Forgetting factor for the cumulative sums. Values closer
            to 1 give more weight to historical data. Default is 0.9999.
        mode: Direction of change to detect:
            - "up": Detect increases in the mean
            - "down": Detect decreases in the mean
            - "both": Detect both increases and decreases (default)

    Example:
        >>> detector = PageHinkley(threshold=50.0)
        >>> for value in data_stream:
        ...     detector.update(value)
        ...     if detector.drift_detected:
        ...         print("Drift detected!")

    References:
        Page, E. S. (1954). Continuous inspection schemes.
        Biometrika, 41(1/2), 100-115.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: Literal["up", "down", "both"] = "both",
    ) -> None:
        if min_instances <= 0:
            raise ValueError("min_instances must be positive")
        if delta < 0:
            raise ValueError("delta must be non-negative")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if mode not in ("up", "down", "both"):
            raise ValueError("mode must be 'up', 'down', or 'both'")

        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        self._n: int = 0
        self._sum: float = 0.0
        self._mean: float = 0.0

        # Cumulative sums for detecting increases and decreases
        self._sum_up: float = 0.0
        self._sum_down: float = 0.0

        # Tracked extrema
        self._min_sum_up: float = float("inf")
        self._max_sum_down: float = float("-inf")

        self._drift_detected: bool = False
        self._n_detections: int = 0

    def reset(self) -> None:
        """Reset the detector to its initial state (clears all counters)."""
        self._reset_state()

    def _soft_reset(self) -> None:
        """Reset detector state but preserve n_detections counter."""
        n_detections = self._n_detections
        self._reset_state()
        self._n_detections = n_detections

    @property
    def drift_detected(self) -> bool:
        """Return True if drift was detected on the last update."""
        return self._drift_detected

    @property
    def n_detections(self) -> int:
        """Total number of drift detections."""
        return self._n_detections

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    def update(self, x: float) -> "PageHinkley":
        """
        Update the detector with a new observation.

        Args:
            x: The observed value.

        Returns:
            self: Returns self for method chaining.
        """
        x = _finite_observation(x)

        # Auto-reset after drift detection (following River's behavior)
        if self._drift_detected:
            self._soft_reset()
        self._drift_detected = False
        self._n += 1

        # Update running mean
        self._sum += x
        self._mean = self._sum / self._n

        # Compute deviation from mean
        deviation = x - self._mean - self.delta

        # Update cumulative sums with forgetting factor
        self._sum_up = self.alpha * self._sum_up + deviation
        self._sum_down = self.alpha * self._sum_down + (x - self._mean + self.delta)

        # Track extrema
        self._min_sum_up = min(self._min_sum_up, self._sum_up)
        self._max_sum_down = max(self._max_sum_down, self._sum_down)

        # Check for drift after minimum instances
        if self._n >= self.min_instances:
            self._check_drift()

        return self

    def _check_drift(self) -> None:
        """Check if drift has occurred based on the mode."""
        if self.mode == "up":
            if self._sum_up - self._min_sum_up > self.threshold:
                self._drift_detected = True
                self._n_detections += 1
        elif self.mode == "down":
            if self._max_sum_down - self._sum_down > self.threshold:
                self._drift_detected = True
                self._n_detections += 1
        elif (
            self._sum_up - self._min_sum_up > self.threshold
            or self._max_sum_down - self._sum_down > self.threshold
        ):
            self._drift_detected = True
            self._n_detections += 1

    def __repr__(self) -> str:
        return (
            f"PageHinkley(min_instances={self.min_instances}, delta={self.delta}, "
            f"threshold={self.threshold}, alpha={self.alpha}, mode='{self.mode}')"
        )
