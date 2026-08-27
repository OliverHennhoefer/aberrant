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
    """
    Abstract base class for concept drift detectors.

    Drift detectors monitor streaming data for distribution changes.
    Following River's design, they use `update(x: float)` with a single
    float value, ideal for monitoring:
    - Anomaly scores from models
    - Prediction errors
    - Individual feature values
    - Any streaming metric

    Subclasses must implement:
    - update(x): Process a single observation
    - drift_detected: Property indicating if drift was detected
    - reset(): Reset detector state
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
