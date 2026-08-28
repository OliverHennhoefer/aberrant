"""Base model interface for online anomaly detection."""

import abc


class BaseModel(abc.ABC):
    """
    Abstract base class for online anomaly detection models.

    Online models process one observation at a time. ``score_one`` evaluates an
    observation against the model's current reference state; ``learn_one``
    incorporates an observation into that state.

    Subclasses must implement ``learn_one`` and ``score_one``. Score ranges,
    warm-up behavior, and score orientation are model-specific.
    """

    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
                and the values are the corresponding feature values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute the anomaly score for a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
                and the values are the corresponding feature values.

        Returns:
            The model-specific anomaly score for the data point. Consult the
            concrete model for its range and orientation.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the model."""
        return self.__repr__()
