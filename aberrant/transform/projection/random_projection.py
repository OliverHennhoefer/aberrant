"""Random projection transformer for dimensionality reduction."""

import math

import numpy as np

from aberrant.base.transformer import BaseTransformer


class RandomProjection(BaseTransformer):
    def __init__(
        self, n_components: int, keys: list[str] | None = None, seed: int | None = None
    ) -> None:
        """
        Initialize the RandomProjection transformer.

        Implements the binary approach for random projections using sparse random matrices.
        This provides a computationally efficient way to reduce dimensionality while
        approximately preserving distances (Johnson-Lindenstrauss lemma).

        Reference:
            Achlioptas D. (2003) "Database-friendly random projections:
            Johnson-Lindenstrauss with binary coins"
            https://doi.org/10.1016/S0022-0000(03)00025-4

        Args:
            n_components: Target number of dimensions after transformation.
            keys: Feature names. If None, inferred from first sample.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If n_components is greater than the number of features.
        """
        super().__init__()

        if n_components < 1:
            raise ValueError("n_components must be greater than 0")
        self.n_components = n_components
        self.feature_names = list(keys) if keys is not None else None
        self.seed = seed

        self.n_dimensions = 0
        self.random_matrix: np.ndarray = np.array([])

        if self.feature_names is not None:
            if len(self.feature_names) != len(set(self.feature_names)):
                raise ValueError("Feature names cannot contain duplicates")
            self._initialize_random_matrix()

    def _initialize_random_matrix(self) -> None:
        """
        Initialize the random projection matrix.

        Raises:
            ValueError: If feature names are not set.
        """
        if self.feature_names is None:
            raise ValueError(
                "Feature names must be set before initializing random matrix"
            )
        self.n_dimensions = len(self.feature_names)
        if self.n_components > self.n_dimensions:
            raise ValueError(
                f"The number of n_components ({self.n_components}) has to be less or equal to the number of features ({self.n_dimensions})"
            )
        else:
            rng = np.random.default_rng(self.seed)
            # Achlioptas' sparse projection uses entries in
            # {-sqrt(3 / k), 0, sqrt(3 / k)} so that E[||Rx||^2] = ||x||^2.
            self.random_matrix = np.sqrt(3.0 / self.n_components) * rng.choice(
                [-1, 0, 1],
                size=(self.n_dimensions, self.n_components),
                p=[1 / 6, 2 / 3, 1 / 6],
            )

    @staticmethod
    def _validate_input(x: dict[str, float]) -> None:
        """Validate projection inputs before establishing persistent schema."""
        for key, value in x.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Feature '{key}' must be numeric") from e
            if not math.isfinite(numeric):
                raise ValueError(f"Feature '{key}' must be finite")

    def _vectorize(self, x: dict[str, float]) -> np.ndarray:
        """Build a stable-order vector and reject extra schema fields."""
        if self.feature_names is None:
            raise RuntimeError("Feature schema is not initialized")
        data_vector = np.array([x[key] for key in self.feature_names], dtype=float)
        if len(x) != len(self.feature_names):
            unexpected = sorted(set(x).difference(self.feature_names))
            raise ValueError(f"Input contains unexpected feature(s): {unexpected}")
        return data_vector

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Learn the number of dimensions from the first data point.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Raises:
            ValueError: If n_components is greater than the number of features in x.
        """
        self._validate_input(x)
        if self.feature_names is None and len(x) >= 1:
            self.feature_names = list(x.keys())
            self._initialize_random_matrix()

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point using random projection.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Returns:
            Transformed data point as dictionary with component names as keys.

        Raises:
            RuntimeError: If called before learning feature names.
        """

        self._validate_input(x)
        if self.feature_names is None:
            raise RuntimeError(
                "Cannot transform before learning. Call learn_one() first or provide keys."
            )

        data_vector = self._vectorize(x)
        transformed_x = self.random_matrix.T @ data_vector
        return {f"component_{i}": float(val) for i, val in enumerate(transformed_x)}

    def __repr__(self) -> str:
        """Return string representation of the transformer."""
        return f"RandomProjection(n_components={self.n_components}, seed={self.seed})"
