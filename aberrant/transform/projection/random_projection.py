"""Random projection transformer for dimensionality reduction."""

import numpy as np

from aberrant.base.transformer import BaseTransformer
from aberrant.transform.projection._schema import ProjectionSchema
from aberrant.utils.validation import coerce_feature_values


class RandomProjection(BaseTransformer):
    """Sparse Achlioptas random projection for streaming feature mappings.

    A fixed matrix maps the input vector to ``n_components`` output values named
    ``component_0``, ``component_1``, and so on. Matrix entries are sampled from
    ``{-sqrt(3/k), 0, sqrt(3/k)}``, where ``k`` is ``n_components``. Learning
    establishes feature order but does not adapt the matrix afterward.

    Args:
        n_components: Number of projected dimensions. It cannot exceed the
            number of input features.
        keys: Explicit feature order. If omitted, the first learned mapping's
            insertion order is used.
        seed: Seed for the transformer's local NumPy generator.

    References:
        Achlioptas, D. (2003). Database-friendly random projections:
        Johnson-Lindenstrauss with binary coins.
        https://doi.org/10.1016/S0022-0000(03)00025-4
    """

    def __init__(
        self, n_components: int, keys: list[str] | None = None, seed: int | None = None
    ) -> None:
        """Initialize the projection and, when possible, its random matrix."""
        super().__init__()

        if n_components < 1:
            raise ValueError("n_components must be greater than 0")
        self.n_components = n_components
        self._schema = ProjectionSchema(n_components, keys)
        self.seed = seed

        self.n_dimensions = 0
        self.random_matrix: np.ndarray = np.array([])

        if keys is not None:
            self.random_matrix = self._create_random_matrix(len(keys))
            self.n_dimensions = len(keys)

    @property
    def feature_names(self) -> list[str] | None:
        """Established feature names in projection order."""
        return self._schema.feature_names

    def _create_random_matrix(self, n_dimensions: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return np.asarray(
            np.sqrt(3.0 / self.n_components)
            * rng.choice(
                [-1, 0, 1],
                size=(n_dimensions, self.n_components),
                p=[1 / 6, 2 / 3, 1 / 6],
            ),
            dtype=np.float64,
        )

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Learn the number of dimensions from the first data point.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Raises:
            ValueError: If the input is invalid or ``n_components`` exceeds its
                feature count.
        """
        if not x:
            return
        prepared = self._schema.prepare(x)
        if not self._schema.is_established:
            matrix = self._create_random_matrix(len(prepared.names))
            self.random_matrix = matrix
            self.n_dimensions = len(prepared.names)
        self._schema.commit(prepared)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point using random projection.

        Args:
            x: A dictionary with feature names as keys and values as data point dimensions.

        Returns:
            Transformed data point as dictionary with component names as keys.

        Raises:
            RuntimeError: If called before learning feature names.
            ValueError: If values are non-numeric or non-finite, or the feature
                schema differs from the learned schema.
        """

        if self.feature_names is None:
            coerce_feature_values(x)
            raise RuntimeError(
                "Cannot transform before learning. Call learn_one() first or provide keys."
            )

        data_vector = self._schema.prepare(x).values
        transformed_x = self.random_matrix.T @ data_vector
        return {f"component_{i}": float(val) for i, val in enumerate(transformed_x)}

    def __repr__(self) -> str:
        """Return string representation of the transformer."""
        return f"RandomProjection(n_components={self.n_components}, seed={self.seed})"
