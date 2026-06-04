"""Statistical models for multivariate moving window analysis."""

from collections import deque

import numpy as np

from aberrant.base.model import BaseModel


def _covariance(x: np.ndarray | list, y: np.ndarray | list, ddof: int = 1) -> float:
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

    Example:
        >>> model = MovingCovariance(window_size=10)
        >>> model.learn_one({"x": 1.0, "y": 2.0})
        >>> score = model.score_one({"x": 1.5, "y": 2.5})
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
        self.feature_names: list[str] | None = list(keys) if keys is not None else None
        self.bias = bias
        self.abs_diff = abs_diff
        self._ensure_window_initialized()

    def _ensure_window_initialized(self) -> None:
        """Ensure feature windows exist when feature names are known."""
        if self.feature_names is None:
            return
        if len(self.feature_names) != 2:
            raise ValueError("Exactly two feature keys are required.")
        for name in self.feature_names:
            if name not in self.window:
                self.window[name] = deque(maxlen=self.window_size)

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: Dictionary with exactly two key-value pairs.

        Raises:
            ValueError: If input doesn't contain exactly two features.
        """
        if len(x) != 2:
            raise ValueError("Input must contain exactly two key-value pairs.")

        if self.feature_names is None:
            self.feature_names = sorted(x.keys())  # Sort for consistency
        self._ensure_window_initialized()

        missing = [name for name in self.feature_names if name not in x]
        if missing:
            raise ValueError(f"Input is missing required feature(s): {missing}")

        # Add values to windows
        for name in self.feature_names:
            if name in x and isinstance(x[name], int | float):
                self.window[name].append(float(x[name]))

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score based on covariance change.

        Calculates covariance(window + x) - covariance(window).

        Args:
            x: Data point to score.

        Returns:
            Covariance difference. Returns 0.0 if insufficient data.
        """
        if self.feature_names is None or len(self.window[self.feature_names[0]]) < 2:
            return 0.0

        window_0 = self.window[self.feature_names[0]]
        window_1 = self.window[self.feature_names[1]]

        if len(window_0) != len(window_1):
            raise ValueError("Window lengths must match.")

        # Create score windows efficiently using numpy
        score_0 = np.append(window_0, x[self.feature_names[0]])
        score_1 = np.append(window_1, x[self.feature_names[1]])

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
    """
    A simple moving model that calculates the difference between the correlation coefficient from the window with a new value
    to the correlation coefficient from the window of the most recent values.
    """

    def __init__(
        self,
        window_size: int,
        bias: bool = True,
        keys: list[str] | None = None,
        abs_diff: bool = True,
    ) -> None:
        """Initialize a new instance of MovingCorrelationCoefficient.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving correlation coefficient.
            bias (bool): False if bessel correction should not be used.
            keys (list[str]): Keys for the moving window. If None, the first keys learned are used.
            abs_diff (bool): If True absolute is given back, else covariance(window + score) - covariance(window)
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: dict[str, deque[float]] = {}
        self.feature_names: list[str] | None = list(keys) if keys is not None else None
        self.bias = bias
        self.abs_diff = abs_diff
        self._ensure_window_initialized()

    def _ensure_window_initialized(self) -> None:
        """Ensure feature windows exist when feature names are known."""
        if self.feature_names is None:
            return
        if len(self.feature_names) != 2:
            raise ValueError("Exactly two feature keys are required.")
        for name in self.feature_names:
            if name not in self.window:
                self.window[name] = deque([], maxlen=self.window_size)

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains other than two key-value pairs.
        """
        if len(x) != 2:
            raise ValueError("Input must contain exactly two key-value pairs.")
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
        self._ensure_window_initialized()

        missing = [name for name in self.feature_names if name not in x]
        if missing:
            raise ValueError(f"Input is missing required feature(s): {missing}")
        if isinstance(x[self.feature_names[0]], int | float) and isinstance(
            x[self.feature_names[1]], int | float
        ):
            self.window[self.feature_names[0]].append(float(x[self.feature_names[0]]))
            self.window[self.feature_names[1]].append(float(x[self.feature_names[1]]))

    def _correlation_coefficient(self, window_0, window_1) -> float:
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
            return cov / (std_0 * std_1)

    def score_one(self, x: dict[str, float]) -> float:
        """Calculate and return the correlation coefficient difference of the values in the windows.
        Args:
            x (Dict): Single datapoint to be added temporarily to calculate the correlation coefficient.
        Returns:
            float: The correlation coefficient difference of the values in the window. 0 if the window is empty or has less than 2 data points.
        """
        if self.feature_names is None:
            return 0.0
        score_window_0 = list(self.window[self.feature_names[0]])
        score_window_1 = list(self.window[self.feature_names[1]])
        score_window_0.append(x[self.feature_names[0]])
        score_window_1.append(x[self.feature_names[1]])
        corr_coeff_diff = self._correlation_coefficient(
            score_window_0, score_window_1
        ) - self._correlation_coefficient(
            self.window[self.feature_names[0]], self.window[self.feature_names[1]]
        )
        return abs(corr_coeff_diff) if self.abs_diff else corr_coeff_diff


class MovingMahalanobisDistance(BaseModel):
    """
    A simple moving model that calculates the mahalanobis distance of the last values
    and the correlation matrix of most recent values.
    """

    def __init__(
        self, window_size: int, bias: bool = True, keys: list[str] | None = None
    ) -> None:
        """Initialize a new instance of MovingMahalanobisDistance.
        Args:
            window_size (int): The number of recent values to consider for calculating the mahalanobis distance.
            bias (bool): False if bessel correction should not be used.
            keys (list[str]): Keys for the moving window. If None, the first keys learned are used.
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window_size = window_size
        self.window: deque[list[float]] = deque([], maxlen=window_size)
        self.feature_names: list[str] | None = keys
        self.bias = bias

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        """
        if self.feature_names is None:
            self.feature_names = list(x.keys())
        datapoint = [x[key] for key in self.feature_names]
        if all(isinstance(val, int | float) for val in datapoint):
            self.window.append(datapoint)

    def score_one(self, x: dict[str, float]) -> float:
        """Calculate and return the mahalanobis distance from one given point to the window's feature mean.
        Args:
            x (Dict): Single datapoint.
        Returns:
            float: The mahalanobis distance. 0 if the window is empty or has less than 3 data points.
        """
        if self.feature_names is None or len(self.window) < 3:
            return 0
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
        x_vector = np.array([x[key] for key in self.feature_names])
        diff = x_vector - feature_mean
        return float(diff.T @ inv_cov_matrix @ diff)
