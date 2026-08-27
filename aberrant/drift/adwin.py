"""ADWIN (ADaptive WINdowing) drift detector."""

import math

from aberrant.drift.base import BaseDriftDetector, _finite_observation


class ADWIN(BaseDriftDetector):
    """
    ADWIN (ADaptive WINdowing) drift detector.

    ADWIN maintains a variable-length window of recent data and detects
    concept drift by comparing the distributions of two subwindows.
    When drift is detected, it shrinks the window to remove old data.

    The algorithm uses an exponential histogram (bucket structure) for
    memory-efficient storage and the Hoeffding bound for statistical
    significance testing.

    Args:
        delta: Significance level for drift detection. Lower values make
            the detector more conservative. Default is 0.002.
        clock: How often to check for drift (every `clock` samples).
            Default is 32.
        max_buckets: Maximum number of buckets per level. Default is 5.
        min_window_length: Minimum subwindow size for comparison.
            Default is 5.
        grace_period: Number of samples before drift detection starts.
            Default is 10.

    Example:
        >>> detector = ADWIN(delta=0.002)
        >>> for value in data_stream:
        ...     detector.update(value)
        ...     if detector.drift_detected:
        ...         print("Drift detected!")

    References:
        Bifet, A., & Gavalda, R. (2007). Learning from time-changing data
        with adaptive windowing. In Proceedings of the 2007 SIAM
        International Conference on Data Mining (pp. 443-448).
        Reference implementation:
        https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/core/driftdetection/ADWIN.java
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if clock <= 0:
            raise ValueError("clock must be positive")
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive")
        if min_window_length <= 0:
            raise ValueError("min_window_length must be positive")
        if grace_period < 0:
            raise ValueError("grace_period must be non-negative")

        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period

        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        # Bucket structure: list of levels, each level has buckets
        # Each bucket stores (count, sum, variance_sum)
        self._bucket_count: list[int] = []  # Number of buckets at each level
        self._bucket_sum: list[list[float]] = []  # Sum at each bucket
        self._bucket_variance: list[list[float]] = []  # Variance contribution

        self._total: float = 0.0  # Total sum of all elements
        self._variance: float = 0.0  # Total variance
        self._width: int = 0  # Current window width
        self._n_detections: int = 0
        self._drift_detected: bool = False
        self._samples_since_reset: int = 0
        self._last_bucket_idx: int = 0

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
    def width(self) -> int:
        """Current window width (number of observations)."""
        return self._width

    @property
    def n_detections(self) -> int:
        """Total number of drift detections."""
        return self._n_detections

    @property
    def estimation(self) -> float:
        """Current mean estimate of the window."""
        if self._width == 0:
            return 0.0
        return self._total / self._width

    @property
    def variance(self) -> float:
        """Current variance estimate of the window."""
        if self._width <= 1:
            return 0.0
        return self._variance / self._width

    def update(self, x: float) -> "ADWIN":
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
        self._samples_since_reset += 1

        # Add new element to the window
        self._insert_element(x)

        # Check for drift periodically
        if self._samples_since_reset % self.clock == 0:
            self._detect_change()

        return self

    def _insert_element(self, x: float) -> None:
        """Insert a new element into the bucket structure."""
        self._width += 1

        # Update variance using Welford's online algorithm
        previous_total = self._total
        self._total += x
        if self._width > 1:
            old_mean = previous_total / (self._width - 1)
            new_mean = self._total / self._width
            self._variance += (x - old_mean) * (x - new_mean)

        # Add element as a new bucket at level 0
        if len(self._bucket_count) == 0:
            self._bucket_count.append(0)
            self._bucket_sum.append([])
            self._bucket_variance.append([])

        self._bucket_count[0] += 1
        self._bucket_sum[0].append(x)
        self._bucket_variance[0].append(0.0)

        # Compress buckets if needed
        self._compress_buckets()

    def _compress_buckets(self) -> None:
        """Compress buckets when a level exceeds max_buckets."""
        level = 0
        while level < len(self._bucket_count):
            if self._bucket_count[level] <= self.max_buckets:
                break

            # Ensure next level exists
            if level + 1 >= len(self._bucket_count):
                self._bucket_count.append(0)
                self._bucket_sum.append([])
                self._bucket_variance.append([])

            # Merge two oldest buckets into one at next level
            n1 = 2**level
            n2 = 2**level

            # Get the two oldest buckets at this level
            s1 = self._bucket_sum[level][0]
            s2 = self._bucket_sum[level][1]
            v1 = self._bucket_variance[level][0]
            v2 = self._bucket_variance[level][1]

            # Compute merged bucket statistics
            merged_sum = s1 + s2
            mean1 = s1 / n1
            mean2 = s2 / n2
            merged_var = v1 + v2 + n1 * n2 * (mean1 - mean2) ** 2 / (n1 + n2)

            # Remove two oldest buckets from current level
            self._bucket_sum[level] = self._bucket_sum[level][2:]
            self._bucket_variance[level] = self._bucket_variance[level][2:]
            self._bucket_count[level] -= 2

            # Add merged bucket to next level
            self._bucket_sum[level + 1].append(merged_sum)
            self._bucket_variance[level + 1].append(merged_var)
            self._bucket_count[level + 1] += 1

            level += 1

    def _detect_change(self) -> None:
        """Check for drift by comparing subwindows."""
        if self._width < 2 * self.min_window_length:
            return

        if self._samples_since_reset <= self.grace_period:
            return

        # Try different split points
        n0 = 0
        sum0 = 0.0
        var0 = 0.0

        # Iterate through buckets from newest to oldest
        for level in range(len(self._bucket_count)):
            bucket_size = 2**level
            for i in range(self._bucket_count[level] - 1, -1, -1):
                n0 += bucket_size
                sum0 += self._bucket_sum[level][i]
                var0 += self._bucket_variance[level][i]

                n1 = self._width - n0
                if n0 < self.min_window_length or n1 < self.min_window_length:
                    continue

                sum1 = self._total - sum0
                mean0 = sum0 / n0
                mean1 = sum1 / n1

                # Hoeffding bound for the difference of means
                m = 1.0 / n0 + 1.0 / n1
                delta_prime = self.delta / math.log(self._width)
                eps = math.sqrt(2 * m * self.variance * math.log(2 / delta_prime))
                eps += 2 / 3 * m * math.log(2 / delta_prime)

                if abs(mean0 - mean1) > eps:
                    # Shrink window by removing oldest elements
                    self._remove_oldest(n1)
                    self._drift_detected = True
                    self._n_detections += 1
                    return

    def _remove_oldest(self, count: int) -> None:
        """Remove the oldest `count` elements from the window."""
        original_total = self._total
        original_variance = self._variance
        original_width = self._width
        removed_sum = 0.0
        removed_variance = 0.0
        removed_count = 0

        # Remove from highest levels (oldest data) first
        for level in range(len(self._bucket_count) - 1, -1, -1):
            bucket_size = 2**level
            while (
                self._bucket_count[level] > 0 and removed_count + bucket_size <= count
            ):
                # Remove oldest bucket at this level
                bucket_sum = self._bucket_sum[level][0]
                bucket_variance = self._bucket_variance[level][0]
                if removed_count > 0:
                    removed_mean = removed_sum / removed_count
                    bucket_mean = bucket_sum / bucket_size
                    removed_variance += (
                        removed_count
                        * bucket_size
                        * (removed_mean - bucket_mean) ** 2
                        / (removed_count + bucket_size)
                    )
                removed_sum += bucket_sum
                removed_variance += bucket_variance
                removed_count += bucket_size

                self._bucket_sum[level] = self._bucket_sum[level][1:]
                self._bucket_variance[level] = self._bucket_variance[level][1:]
                self._bucket_count[level] -= 1

        self._total = original_total - removed_sum
        self._width = original_width - removed_count

        if self._width > 0 and removed_count > 0:
            removed_mean = removed_sum / removed_count
            remaining_mean = self._total / self._width
            between_groups = (
                removed_count
                * self._width
                * (removed_mean - remaining_mean) ** 2
                / original_width
            )
            self._variance = max(
                0.0,
                original_variance - removed_variance - between_groups,
            )
        else:
            self._variance = 0.0

    def __repr__(self) -> str:
        return (
            f"ADWIN(delta={self.delta}, clock={self.clock}, "
            f"max_buckets={self.max_buckets}, width={self._width})"
        )
