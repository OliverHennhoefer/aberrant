"""Threshold model for simple boundary-based anomaly detection."""

from aberrant.base.model import BaseModel


class ThresholdModel(BaseModel):
    """
    Threshold model that detects anomalies based on boundary violations.

    This model never learns from data and returns a binary anomaly score
    based on whether feature values exceed specified thresholds. It supports
    both one-sided (ceiling or floor only) and two-sided (corridor) detection.

    Args:
        ceiling: Upper threshold(s). Can be a scalar (applies to all features)
            or a dict mapping feature names to their upper thresholds.
        floor: Lower threshold(s). Can be a scalar (applies to all features)
            or a dict mapping feature names to their lower thresholds.

    Note:
        At least one of ceiling or floor must be provided.

    Examples:
        ```python
        from aberrant.model import ThresholdModel

        ceiling_only = ThresholdModel(ceiling=10.0)
        assert ceiling_only.score_one({"temp": 15.0}) == 1.0
        assert ceiling_only.score_one({"temp": 5.0}) == 0.0

        corridor = ThresholdModel(ceiling=100.0, floor=0.0)
        assert corridor.score_one({"temp": 50.0}) == 0.0
        assert corridor.score_one({"temp": -5.0}) == 1.0

        per_feature = ThresholdModel(
            ceiling={"temp": 100.0, "pressure": 50.0},
            floor={"temp": 0.0, "pressure": 10.0},
        )
        assert per_feature.score_one({"temp": 50.0, "pressure": 30.0}) == 0.0
        assert per_feature.score_one({"temp": 150.0, "pressure": 30.0}) == 1.0
        ```
    """

    def __init__(
        self,
        ceiling: float | dict[str, float] | None = None,
        floor: float | dict[str, float] | None = None,
    ) -> None:
        """Initialize the threshold model."""
        super().__init__()

        if ceiling is None and floor is None:
            raise ValueError("At least one of 'ceiling' or 'floor' must be provided")

        self.ceiling = dict(ceiling) if isinstance(ceiling, dict) else ceiling
        self.floor = dict(floor) if isinstance(floor, dict) else floor

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        This method does nothing as the threshold model never learns.

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
            1.0 if any feature violates its threshold(s), 0.0 otherwise.
        """
        for feature, value in x.items():
            # Check ceiling violation
            if self.ceiling is not None:
                if isinstance(self.ceiling, dict):
                    if feature in self.ceiling and value > self.ceiling[feature]:
                        return 1.0
                elif value > self.ceiling:
                    return 1.0

            # Check floor violation
            if self.floor is not None:
                if isinstance(self.floor, dict):
                    if feature in self.floor and value < self.floor[feature]:
                        return 1.0
                elif value < self.floor:
                    return 1.0

        return 0.0

    def __repr__(self) -> str:
        """Return a string representation of the threshold model."""
        return f"ThresholdModel(ceiling={self.ceiling}, floor={self.floor})"
