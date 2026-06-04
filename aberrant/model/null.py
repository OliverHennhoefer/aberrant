"""Null model for baseline anomaly detection."""

from aberrant.base.model import BaseModel


class NullModel(BaseModel):
    """
    Null model that serves as a baseline for anomaly detection.

    This model never learns from data and always returns a score of 0.0,
    indicating no anomalies are detected. It's useful as a baseline for
    comparing other anomaly detection models.

    Example:
        >>> model = NullModel()
        >>> score = model.score_one({"feature1": 1.0, "feature2": 2.0})
        >>> model.learn_one({"feature1": 1.0, "feature2": 2.0})
        >>> print(score)  # Always 0.0
    """

    def __init__(self) -> None:
        """Initialize the null model."""
        super().__init__()

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        This method does nothing as the null model never learns.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        pass

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Always returns 0.0 (no anomaly detected).
        """
        return 0.0

    def __repr__(self) -> str:
        """Return a string representation of the null model."""
        return "NullModel()"
