"""Tabular CUSUM drift detector."""

import math
from typing import Literal

from aberrant._utils.validation import coerce_integer_count
from aberrant.drift.base import BaseDriftDetector, _finite_observation


class CUSUM(BaseDriftDetector):
    """
    Two-sided tabular CUSUM on standardized observations.

    Each observation is standardized, ``z = (x - target) / scale``, and
    accumulated into an upper and a lower sum::

        S+ = max(0, S+ + z - k)
        S- = max(0, S- - z - k)

    Drift is signalled when a sum exceeds ``h``. Because both are in units
    of the standard deviation, ``k`` and ``h`` mean the same on every
    signal: ``k`` is half the shift the chart detects fastest, and ``h``
    trades false alarms against delay. On normal observations the defaults
    pass about 20,000 observations between false alarms and detect a
    one-deviation shift after about 20; the textbook ``h = 5`` with a known
    reference gives about 470 and 10.

    Unless ``target`` and ``scale`` fix them, the mean and sample standard
    deviation of the first ``warm_up`` observations are the in-control
    reference, and the sums start after them. On the update after a signal
    the sums restart, and an estimated reference is estimated again from
    the next ``warm_up`` observations, so the new level becomes the
    reference. An estimated deviation that comes out low raises the false
    alarm rate, so a shorter warm-up alarms more often.
    A warm-up that contains the change hides it.

    Page-Hinkley differs in two ways: it compares with a running mean, which
    follows a slow drift, and its threshold is in the signal's own units.

    Args:
        k: Allowance, in standard deviations. Default is 0.5.
        h: Decision threshold, in standard deviations. Default is 10.0.
        warm_up: Observations that estimate the reference. Default is 100.
        target: In-control mean. Estimated when ``None``.
        scale: In-control standard deviation. Estimated when ``None``.
        mode: Direction of change to detect:
            - "up": Detect increases in the mean
            - "down": Detect decreases in the mean
            - "both": Detect both increases and decreases (default)

    Examples:
        ```python
        from aberrant.drift import CUSUM

        detector = CUSUM(target=0.0, scale=1.0)
        for index, value in enumerate([0.0] * 64 + [2.0] * 64):
            if detector.update(value):
                start = index - detector.onset + 1
                print(f"{detector.direction} shift from {start}, seen at {index}")
                break
        ```

    References:
        Page, E. S. (1954). Continuous inspection schemes.
        Biometrika, 41(1/2), 100-115.
        https://doi.org/10.1093/biomet/41.1-2.100

        Montgomery, D. C. (2019). Introduction to Statistical Quality
        Control, 8th ed., chapter 9. Wiley.
    """

    def __init__(
        self,
        *,
        k: float = 0.5,
        h: float = 10.0,
        warm_up: int = 100,
        target: float | None = None,
        scale: float | None = None,
        mode: Literal["up", "down", "both"] = "both",
    ) -> None:
        warm_up = coerce_integer_count(warm_up, label="warm_up", minimum=2)
        if not (math.isfinite(k) and k >= 0):
            raise ValueError("k must be a finite, non-negative number")
        if not (math.isfinite(h) and h > 0):
            raise ValueError("h must be a finite, positive number")
        if target is not None and not math.isfinite(target):
            raise ValueError("target must be finite")
        if scale is not None and not (math.isfinite(scale) and scale > 0):
            raise ValueError("scale must be a finite, positive number")
        if mode not in ("up", "down", "both"):
            raise ValueError("mode must be 'up', 'down', or 'both'")

        self.k = k
        self.h = h
        self.warm_up = warm_up
        self.target = target
        self.scale = scale
        self.mode = mode
        self._n_detections = 0
        self._restart()

    def _restart(self) -> None:
        self._seen = 0
        self._mean = 0.0
        self._squares = 0.0
        self._upper = 0.0
        self._lower = 0.0
        self._upper_run = 0
        self._lower_run = 0
        self._drift_detected = False
        self._direction: Literal["up", "down"] | None = None
        self._onset = 0

    @property
    def drift_detected(self) -> bool:
        """Return True if drift was detected on the last update."""
        return self._drift_detected

    @property
    def n_detections(self) -> int:
        """Total number of drift detections."""
        return self._n_detections

    @property
    def direction(self) -> Literal["up", "down"] | None:
        """``"up"`` or ``"down"`` when the last update signalled, else ``None``."""
        return self._direction

    @property
    def onset(self) -> int:
        """Observations, this one included, since the signalling sum was 0.

        The change most likely began with the first of them, ``onset - 1``
        updates before the signal. ``0`` when the last update did not signal.
        """
        return self._onset

    @property
    def statistic(self) -> float:
        """The larger of the monitored sums, in standard deviations."""
        if self.mode == "up":
            return self._upper
        if self.mode == "down":
            return self._lower
        return max(self._upper, self._lower)

    def update(self, x: float) -> bool:
        """
        Update the detector with a new observation.

        Args:
            x: The observed value.

        Returns:
            Whether this observation signalled drift.
        """
        x = _finite_observation(x)
        if self._drift_detected:
            self._restart()

        if self._seen < self._estimated_count():
            self._seen += 1
            half = x / 2 - self._mean / 2
            self._mean += half * (2 / self._seen)
            self._squares += 4 * half * (x / 2 - self._mean / 2)
            return False

        z = _standardized(x / 2 - self._reference() / 2, self._deviation() / 2)
        self._upper = max(0.0, self._upper + z - self.k)
        self._lower = max(0.0, self._lower - z - self.k)
        self._upper_run = self._upper_run + 1 if self._upper > 0 else 0
        self._lower_run = self._lower_run + 1 if self._lower > 0 else 0

        if self.mode != "down" and self._upper > self.h:
            self._signal("up", self._upper_run)
        elif self.mode != "up" and self._lower > self.h:
            self._signal("down", self._lower_run)
        return self._drift_detected

    def _estimated_count(self) -> int:
        if self.target is not None and self.scale is not None:
            return 0
        return self.warm_up

    def _reference(self) -> float:
        return self._mean if self.target is None else self.target

    def _deviation(self) -> float:
        if self.scale is not None:
            return self.scale
        return math.sqrt(self._squares / (self._seen - 1))

    def _signal(self, direction: Literal["up", "down"], run: int) -> None:
        self._drift_detected = True
        self._direction = direction
        self._onset = run
        self._n_detections += 1


def _standardized(difference: float, deviation: float) -> float:
    """``difference / deviation``, with any change from a constant infinite.

    Both are halved by the caller, so neither overflows for finite inputs.
    """
    if deviation > 0:
        return difference / deviation
    if difference == 0:
        return 0.0
    return math.copysign(math.inf, difference)
