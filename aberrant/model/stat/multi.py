"""Statistical models for multivariate moving window analysis."""

from collections import deque
from collections.abc import Sequence

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


def _covariance(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    ddof: int = 1,
) -> float:
    """
    Calculate the covariance between two arrays using numpy for efficiency.

    Args:
        x: First dataset.
        y: Second dataset.
        ddof: Delta degrees of freedom for Bessel correction.

    Returns:
        Covariance value.

    Raises:
        ValueError: If datasets have different lengths.
    """
    x_array = np.asarray(x)
    y_array = np.asarray(y)

    if len(x_array) != len(y_array):
        raise ValueError("Both datasets must have the same length.")

    if len(x_array) <= ddof:
        return 0.0

    return float(np.cov(x_array, y_array, ddof=ddof)[0, 1])


def _initialize_bivariate_windows(
    window: dict[str, deque[float]],
    names: tuple[str, ...],
    window_size: int,
) -> None:
    """Initialize both feature windows from a validated two-feature schema."""
    for name in names:
        window.setdefault(name, deque(maxlen=window_size))


class MovingCovariance(BaseModel):
    """
    Moving covariance anomaly detection model.

    Calculates the difference between the covariance of a window with a new value
    and the covariance of the current window. Designed for bivariate data streams.

    Args:
        window_size: Number of recent values to consider.
        bias: If False, applies Bessel correction (ddof=1).
        keys: Feature names for the two variables. If None, uses first learned keys.
        abs_diff: If True, returns absolute difference.

    Raises:
        ValueError: If window_size is not positive.

    Examples:
        ```python
        from aberrant.model.stat import MovingCovariance

        model = MovingCovariance(window_size=10)
        model.learn_one({"x": 1.0, "y": 2.0})
        score = model.score_one({"x": 1.5, "y": 2.5})
        ```
    """

    def __init__(
        self,
        window_size: int,
        bias: bool = True,
        keys: list[str] | None = None,
        abs_diff: bool = True,
    ) -> None:
        """Initialize the moving covariance model."""
        super().__init__()

        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.window_size = window_size
        self.window: dict[str, deque[float]] = {}
        self._schema = FeatureSchema(names=keys, expected_size=2)
        self.bias = bias
        self.abs_diff = abs_diff
        if self._schema.names is not None:
            _initialize_bivariate_windows(
                self.window,
                self._schema.names,
                self.window_size,
            )

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: Dictionary with exactly two key-value pairs.

        Raises:
            ValueError: If input doesn't contain exactly two features.
        """
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            _initialize_bivariate_windows(
                self.window,
                prepared.names,
                self.window_size,
            )

        # Validate the complete point before mutating either feature window.
        for name, value in zip(prepared.names, prepared.values, strict=True):
            self.window[name].append(float(value))
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score based on covariance change.

        Calculates covariance(window + x) - covariance(window).

        Args:
            x: Data point to score.

        Returns:
            Covariance difference. Returns 0.0 if insufficient data.
        """
        prepared = self._schema.preview(x)
        names = self._schema.names
        if names is None or len(self.window[names[0]]) < 2:
            return 0.0

        window_0 = self.window[names[0]]
        window_1 = self.window[names[1]]

        if len(window_0) != len(window_1):
            raise ValueError("Window lengths must match.")

        # Create score windows efficiently using numpy
        score_0 = np.append(window_0, prepared.values[0])
        score_1 = np.append(window_1, prepared.values[1])

        ddof = 0 if self.bias else 1

        # Calculate covariances
        window_cov = _covariance(window_0, window_1, ddof=ddof)
        score_cov = _covariance(score_0, score_1, ddof=ddof)

        difference = score_cov - window_cov
        return abs(difference) if self.abs_diff else difference

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"MovingCovariance(window_size={self.window_size}, "
            f"bias={self.bias}, abs_diff={self.abs_diff})"
        )


class MovingCorrelationCoefficient(BaseModel):
    """Score the candidate-induced change in bivariate Pearson correlation.

    The detector compares the correlation of the retained two-feature window
    with the correlation after temporarily appending the candidate. It returns
    the absolute change by default; set ``abs_diff=False`` to preserve its sign.
    Windows with fewer than two paired values have correlation ``0.0``.

    Args:
        window_size: Maximum number of recent paired observations to retain.
        bias: Use population normalization when true and sample normalization
            when false. The normalization cancels in Pearson correlation but is
            also applied consistently to covariance and standard deviations.
        keys: Explicit names for the two features. If omitted, the first
            successfully learned mapping establishes a sorted schema.
        abs_diff: Return the magnitude of the change when true.
    """

    def __init__(
        self,
        window_size: int,
        bias: bool = True,
        keys: list[str] | None = None,
        abs_diff: bool = True,
    ) -> None:
        """Initialize the bounded bivariate window."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: dict[str, deque[float]] = {}
        self._schema = FeatureSchema(names=keys, expected_size=2)
        self.bias = bias
        self.abs_diff = abs_diff
        if self._schema.names is not None:
            _initialize_bivariate_windows(
                self.window,
                self._schema.names,
                self.window_size,
            )

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one validated, finite bivariate observation.

        Args:
            x: Mapping containing exactly the established two feature names.

        Raises:
            ValueError: If values are invalid or the feature schema differs.
        """
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            _initialize_bivariate_windows(
                self.window,
                prepared.names,
                self.window_size,
            )
        for name, value in zip(prepared.names, prepared.values, strict=True):
            self.window[name].append(float(value))
        self._schema.commit(prepared)

    def _correlation_coefficient(
        self,
        window_0: Sequence[float],
        window_1: Sequence[float],
    ) -> float:
        len_0 = len(window_0)
        len_1 = len(window_1)
        if len_0 != len_1:
            raise ValueError("Both windows must have the same length.")
        if len_0 < 2:
            return 0.0
        n = len_0 if self.bias else len_0 - 1
        mean_0 = sum(window_0) / len_0
        mean_1 = sum(window_1) / len_1
        cov = _covariance(window_0, window_1, ddof=0 if self.bias else 1)
        std_0 = (sum((_ - mean_0) ** 2 for _ in window_0) / n) ** 0.5
        std_1 = (sum((_ - mean_1) ** 2 for _ in window_1) / n) ** 0.5
        if std_0 == 0 or std_1 == 0:
            return 0.0
        else:
            return float(cov / (std_0 * std_1))

    def score_one(self, x: dict[str, float]) -> float:
        """Return the correlation change induced by a candidate observation.

        Args:
            x: Candidate with exactly the established two feature names.

        Returns:
            Absolute or signed correlation difference according to ``abs_diff``.
        """
        prepared = self._schema.preview(x)
        names = self._schema.names
        if names is None:
            return 0.0
        score_window_0 = list(self.window[names[0]])
        score_window_1 = list(self.window[names[1]])
        score_window_0.append(float(prepared.values[0]))
        score_window_1.append(float(prepared.values[1]))
        corr_coeff_diff = self._correlation_coefficient(
            score_window_0, score_window_1
        ) - self._correlation_coefficient(
            self.window[names[0]], self.window[names[1]]
        )
        return abs(corr_coeff_diff) if self.abs_diff else corr_coeff_diff


class MovingMahalanobisDistance(BaseModel):
    """Score squared Mahalanobis distance from a recent reference window.

    The score uses the retained observations' feature mean and covariance
    matrix. It is ``0.0`` until three observations have been learned. A small,
    scale-aware diagonal term is added only when the covariance matrix is
    singular.

    Args:
        window_size: Maximum number of recent observations to retain.
        bias: Pass population normalization to NumPy covariance when true;
            use sample normalization when false.
        keys: Explicit feature order. If omitted, the first successfully
            learned mapping establishes a sorted schema.
    """

    def __init__(
        self, window_size: int, bias: bool = True, keys: list[str] | None = None
    ) -> None:
        """Initialize the bounded multivariate window."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: deque[list[float]] = deque([], maxlen=window_size)
        self._schema = FeatureSchema(names=keys)
        self.bias = bias

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one validated, finite observation.

        Args:
            x: Feature mapping matching the established schema.
        """
        prepared = self._schema.preview(x)
        self.window.append(prepared.values.tolist())
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Calculate squared Mahalanobis distance to the window mean.

        Args:
            x: Candidate feature mapping; it is not appended by this method.

        Returns:
            Squared Mahalanobis distance, or ``0.0`` before three reference
            observations exist.
        """
        prepared = self._schema.preview(x)
        if not self._schema.is_established or len(self.window) < 3:
            return 0.0
        previous_points = np.array(list(self.window))
        cov_matrix = np.atleast_2d(
            np.cov(previous_points, rowvar=False, bias=self.bias)
        )
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Add scale-aware regularization to handle singular matrices.
            regularization = 1e-6 * np.eye(cov_matrix.shape[0])
            if np.trace(cov_matrix) > 0:
                regularization *= np.trace(cov_matrix) / cov_matrix.shape[0]
            inv_cov_matrix = np.linalg.inv(cov_matrix + regularization)

        feature_mean = np.mean(previous_points, axis=0)
        diff = prepared.values - feature_mean
        return float(diff.T @ inv_cov_matrix @ diff)
