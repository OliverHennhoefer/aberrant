"""Random model for baseline anomaly detection."""

import numpy as np

from aberrant.base.model import BaseModel


class RandomModel(BaseModel):
    """
    Random model that generates random anomaly scores.

    This model never learns from data and returns uniformly distributed
    random scores between 0 and 1. It serves as a random baseline for
    comparing other anomaly detection models.

    Args:
        seed: Random seed for reproducibility.

    Example:
        >>> model = RandomModel(seed=42)
        >>> score = model.score_one({"feature1": 1.0, "feature2": 2.0})
        >>> model.learn_one({"feature1": 1.0, "feature2": 2.0})
        >>> print(f"Random score: {score:.3f}")
    """

    def __init__(self, seed: int = 1) -> None:
        """Initialize the random model with a specified seed."""
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        This method does nothing as the random model never learns.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        pass

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute a random anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Random float between 0.0 and 1.0.
        """
        return self.rng.uniform(0.0, 1.0)

    def __repr__(self) -> str:
        """Return a string representation of the random model."""
        return f"RandomModel(seed={self.seed})"
