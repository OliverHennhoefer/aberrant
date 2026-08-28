"""Incremental Principal Component Analysis transformer."""

import math

import numpy as np

from aberrant.base.transformer import BaseTransformer


class IncrementalPCA(BaseTransformer):
    """
    Incremental Principal Component Analysis for online dimensionality reduction.

    Implements an uncentered online PCA algorithm that can process data points one
    at a time, maintaining principal components without storing all historical data.
    Inputs are projected around the origin; center or standardize them upstream when
    conventional mean-centered PCA semantics are required.

    Args:
        n_components: Number of principal components to keep.
        n0: Initial number of samples for warm-up phase.
        keys: Feature names. If None, inferred from first sample.
        tol: Tolerance for considering residual significance.
        forgetting_factor: Weight for new values (0 < f < 1). If None, uses 1/t.

    Examples:
        ```python
        from aberrant.transform.projection import IncrementalPCA

        pca = IncrementalPCA(n_components=2)
        pca.learn_one({"x": 1.0, "y": 2.0, "z": 3.0})
        transformed = pca.transform_one({"x": 1.5, "y": 2.5, "z": 3.5})
        ```
    """

    def __init__(
        self,
        n_components: int,
        n0: int = 50,
        keys: list[str] | None = None,
        tol: float = 1e-7,
        forgetting_factor: float | None = None,
    ) -> None:
        """
        Initialize the IncrementalPCA transformer.

        Args:
            n_components (int): Number of principal components to keep.
            n0 (int): Initial number of samples for warm-up phase before switching to online mode. Default is 50.
            keys (Optional[list[str]]): List of feature names. If None, they will be inferred from the first sample. Default is None.
            tol (float): Tolerance for considering whether a new data point contributes significantly to the subspace. Default is 1e-7.
            forgetting_factor (Optional[float]): If None (default) it is a stationary process. Larger f will give new values more weight. Must be in the interval ]0, 1[.

        Implements an online PCA algorithm based on the incremental SVD approach from:
        - Brand, M. (2002). "Incremental Singular Value Decomposition of Uncertain Data with Missing Values"
        - Arora, R., Cotter, A., Livescu, K., & Srebro, N. (2012). "Stochastic optimization for PCA and PLS"

        The implementation follows the 'incRpca' function from the R package 'onlinePCA: Online Principal Component Analysis'
        and uses the mathematical framework described in:
        - Cardot, H. & Degras, D. (2015). "Online Principal Component Analysis in High Dimension: Which Algorithm to Choose?"

        Algorithm Steps:
        1. Initialization phase: Collect n0 samples and perform uncentered PCA
        2. Online phase: For each new sample x_t at time t:
           - Apply forgetting factor: λ ← (1-f)λ where f = 1/t
           - Scale new sample: x ← √f * x_t
           - Project onto current subspace: x̂ = U^T x
           - Compute residual: r = x - U x̂
           - If ||r|| > tol, expand subspace with normalized residual
           - Update eigendecomposition of diag(λ) + x̂x̂^T
           - Keep top n_components eigenvalues/vectors
        """
        super().__init__()

        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n0 <= 0:
            raise ValueError("n0 must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if forgetting_factor is not None and not (0 < forgetting_factor < 1):
            raise ValueError("forgetting_factor must be 0 < forgetting_factor < 1")

        self.n_components: int = n_components
        self.n0: int = n0
        self.feature_names: list[str] | None = list(keys) if keys is not None else None
        if self.feature_names is not None and len(self.feature_names) != len(
            set(self.feature_names)
        ):
            raise ValueError("Feature names cannot contain duplicates")
        self.tol: float = tol
        self.forgetting_factor = forgetting_factor

        # State variables
        self.window: list[np.ndarray] = []  # Store data points during initialization
        self.n0_reached: bool = False  # Flag for switching to online mode
        self.n_samples_seen: int = 0  # Total number of samples processed
        self.n_features: int = 0  # Number of features in the data

        # PCA components (eigenvalues and eigenvectors)
        self.values: np.ndarray = np.array([])  # Eigenvalues (variances)
        self.vectors: np.ndarray = np.array([])  # Eigenvectors (loadings)

        self._check_n_features()

    def _check_n_features(self) -> None:
        """Validate n_components against feature count."""
        if self.feature_names is not None:
            self.n_features = len(self.feature_names)
            if self.n_components > len(self.feature_names):
                raise ValueError(
                    f"n_components ({self.n_components}) must be <= number of features ({len(self.feature_names)})"
                )

    @staticmethod
    def _validate_input(x: dict[str, float]) -> None:
        """Reject values that would irreversibly poison PCA state."""
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

    def _update_online_pca(self, data_vector: np.ndarray) -> None:
        """
        Update PCA components using online algorithm.

        Args:
            data_vector: New data point as numpy array.
        """
        # Compute forgetting factor
        f = (
            1.0 / self.n_samples_seen
            if self.forgetting_factor is None
            else self.forgetting_factor
        )

        # Update eigenvalues with forgetting factor
        lambda_updated = (1 - f) * self.values

        # Scale new data point and projection onto current subspace
        x_scaled = np.sqrt(f) * data_vector
        xhat = self.vectors.T @ x_scaled

        # Compute residual and check if subspace expansion is needed
        residual = x_scaled - self.vectors @ xhat
        norm_residual = float(np.linalg.norm(residual))

        if norm_residual >= self.tol:
            lambda_updated, xhat = self._expand_subspace(
                lambda_updated, xhat, residual, norm_residual
            )

        # Update eigendecomposition and store results
        self._update_eigendecomposition(lambda_updated, xhat)

    def _expand_subspace(
        self,
        lambda_updated: np.ndarray,
        xhat: np.ndarray,
        residual: np.ndarray,
        norm_residual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Expand subspace when residual is significant.

        Args:
            lambda_updated: Current eigenvalues.
            xhat: Projected data point.
            residual: Orthogonal residual.
            norm_residual: Norm of the residual.

        Returns:
            Extended eigenvalues and projected data point.
        """
        k = len(lambda_updated) + 1

        # Extend eigenvalues
        lambda_extended = np.zeros(k)
        lambda_extended[: len(lambda_updated)] = lambda_updated

        # Extend projection
        xhat_extended = np.zeros(k)
        xhat_extended[: len(xhat)] = xhat
        xhat_extended[-1] = norm_residual

        # Extend vector matrix
        U_extended = np.zeros((self.n_features, k))
        U_extended[:, : self.vectors.shape[1]] = self.vectors
        U_extended[:, -1] = residual / norm_residual

        self.vectors = U_extended
        return lambda_extended, xhat_extended

    def _update_eigendecomposition(
        self, lambda_updated: np.ndarray, xhat: np.ndarray
    ) -> None:
        """
        Compute and apply eigendecomposition update.

        Args:
            lambda_updated: Updated eigenvalues.
            xhat: Projected data point.
        """
        # Compute eigendecomposition of updated covariance matrix
        matrix_to_decompose = np.diag(lambda_updated) + np.outer(xhat, xhat)
        eigenvalues, eigenvectors = np.linalg.eig(matrix_to_decompose)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only top n_components
        if len(eigenvalues) > self.n_components:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]

        # Update stored values
        self.values = eigenvalues
        self.vectors = self.vectors @ eigenvectors

    def _initialize_pca(self, data_vector: np.ndarray) -> None:
        """
        Handle initialization phase by collecting data until n0 samples.

        Args:
            data_vector: New data point to add to initialization window.
        """
        self.window.append(data_vector)
        if len(self.window) >= self.n0:
            # Perform initial full PCA and switch to online mode
            initial_data = np.array(self.window)
            u, s, vt = np.linalg.svd(initial_data, full_matrices=False)
            s = s / np.sqrt(max(1, initial_data.shape[0] - 1))

            # Store vectors as (n_features, n_components) to match online phase
            rotation = vt.T  # Transpose to get (n_features, n_components)
            self.values = s[: self.n_components] ** 2
            self.vectors = rotation[:, : self.n_components]
            self.n0_reached = True
            self.window = []

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update PCA components incrementally using a single sample.

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Raises:
            ValueError: If `n_components` is greater than the number of features in `x`.
        """
        self._validate_input(x)
        if self.feature_names is None:
            self.feature_names = list(x.keys())
            self._check_n_features()

        data_vector = self._vectorize(x)

        if self.n0_reached:
            self._update_online_pca(data_vector)
        else:
            self._initialize_pca(data_vector)

        self.n_samples_seen += 1

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point using the learned PCA components.

        Args:
            x (Dict[str, float]): A dictionary with feature names as keys and values as the data point dimensions.

        Returns:
            Transformed data point as dictionary with component names as keys.
        """

        self._validate_input(x)
        data_vector = self._vectorize(x) if self.feature_names is not None else None
        if self.n0_reached:
            if data_vector is None:
                raise RuntimeError(
                    "Cannot transform before learning. Call learn_one() first or provide keys."
                )

            transformed_x = self.vectors.T @ data_vector
            return {f"component_{i}": float(val) for i, val in enumerate(transformed_x)}
        else:
            # Return zeros during initialization phase
            return {f"component_{i}": 0.0 for i in range(self.n_components)}

    def __repr__(self) -> str:
        """Return string representation of the transformer."""
        return (
            f"IncrementalPCA(n_components={self.n_components}, n0={self.n0}, "
            f"tol={self.tol}, forgetting_factor={self.forgetting_factor})"
        )
