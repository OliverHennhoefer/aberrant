"""Test utilities for ABERRANT test suite."""

import numpy as np


class DataGenerator:
    """Utility class for generating test data for anomaly detection models."""

    def __init__(self, seed: int = 42):
        """Initialize data generator with random seed."""
        self.rng = np.random.default_rng(seed)

    def generate_streaming_data(
        self, n: int = 1000, n_features: int = 3
    ) -> list[dict[str, float]]:
        """
        Generate synthetic streaming data for testing.

        Args:
            n: Number of data points to generate
            n_features: Number of features per data point

        Returns:
            List of feature dictionaries
        """
        data = []
        for i in range(n):
            point = {}
            for j in range(n_features):
                # Mix of different distributions
                if j == 0:
                    # Normal distribution
                    point[f"feature_{j}"] = float(self.rng.normal(0, 1))
                elif j == 1:
                    # Trending data
                    point[f"feature_{j}"] = float(i * 0.01 + self.rng.normal(0, 0.5))
                else:
                    # Uniform distribution
                    point[f"feature_{j}"] = float(self.rng.uniform(-2, 2))
            data.append(point)
        return data
