"""Base class for concept drift detectors."""

import abc
import math


def _finite_observation(x: float) -> float:
    """Validate an observation before it reaches persistent detector state."""
    try:
        value = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError("observation must be numeric") from e
    if not math.isfinite(value):
        raise ValueError("observation must be finite")
    return value


class BaseDriftDetector(abc.ABC):
    """Abstract base class for scalar stream-change detectors.

    ``update`` processes one finite scalar and returns the detector, while
    ``drift_detected`` describes the observation most recently processed. The
    monitored scalar can be an anomaly score, prediction error, residual,
    feature value, or another application-defined signal.

    Subclasses implement ``update``, ``drift_detected``, and ``reset``. A drift
    flag is evidence of change in the monitored signal; it does not diagnose
    the cause or prescribe a response for an anomaly model.
    """

    @abc.abstractmethod
    def update(self, x: float) -> "BaseDriftDetector":
        """
        Update the detector with a single observation.

        Args:
            x: The observed value.

        Returns:
            self: Returns self for method chaining.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def drift_detected(self) -> bool:
        """
        Return True if drift was detected on the last update.

        Returns:
            True if drift was detected, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the detector to its initial state."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
