"""Statistical models for univariate moving window analysis."""

from collections import deque

from aberrant.base.model import BaseModel


def _extract_univariate_value(
    x: dict[str, float], feature_name: str | None
) -> tuple[str, float]:
    """Validate a univariate sample and return its stable feature name and value."""
    if len(x) != 1:
        raise ValueError("Input must contain exactly one key-value pair.")

    sample_feature = next(iter(x))
    if feature_name is not None and sample_feature != feature_name:
        raise ValueError(
            f"Input feature must be {feature_name!r}, got {sample_feature!r}."
        )

    return feature_name or sample_feature, float(x[sample_feature])


class _BaseMovingUnivariate(BaseModel):
    """Base class for univariate moving window models to reduce code duplication."""

    def __init__(
        self, window_size: int, key: str | None = None, abs_diff: bool = True
    ) -> None:
        """Initialize base moving univariate model."""
        super().__init__()

        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: Dictionary with exactly one key-value pair.

        Raises:
            ValueError: If input doesn't contain exactly one feature.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def _extract_value(self, x: dict[str, float]) -> float:
        """Extract single value from input dictionary efficiently."""
        _, value = _extract_univariate_value(x, self.feature_name)
        return value


class MovingAverage(_BaseMovingUnivariate):
    """
    Moving average anomaly detection model.

    Calculates the difference between the arithmetic mean of a window with a new value
    and the mean of the current window.

    Args:
        window_size: Number of recent values to consider.
        key: Feature name. If None, uses first learned key.
        abs_diff: If True, returns absolute difference.

    Example:
        >>> model = MovingAverage(window_size=10)
        >>> model.learn_one({"value": 1.0})
        >>> score = model.score_one({"value": 1.5})
    """

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score based on mean change.

        Args:
            x: Data point to score.

        Returns:
            Mean difference. Returns 0.0 if window is empty.
        """
        if len(self.window) == 0:
            return 0.0

        window_sum = sum(self.window)
        current_mean = window_sum / len(self.window)
        new_value = self._extract_value(x)
        new_mean = (window_sum + new_value) / (len(self.window) + 1)

        difference = new_mean - current_mean
        return abs(difference) if self.abs_diff else difference

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"MovingAverage(window_size={self.window_size}, abs_diff={self.abs_diff})"
        )


class MovingHarmonicAverage(_BaseMovingUnivariate):
    """
    Moving harmonic average anomaly detection model.

    Calculates the difference between the harmonic mean of a window with a new value
    and the harmonic mean of the current window. Zero values are ignored.

    Args:
        window_size: Number of recent values to consider.
        key: Feature name. If None, uses first learned key.
        abs_diff: If True, returns absolute difference.

    Example:
        >>> model = MovingHarmonicAverage(window_size=10)
        >>> model.learn_one({"value": 2.0})
        >>> score = model.score_one({"value": 3.0})
    """

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Zero values are ignored as they would make harmonic mean undefined.

        Args:
            x: Dictionary with exactly one key-value pair.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        if value != 0:
            self.window.append(value)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score based on harmonic mean change.

        Args:
            x: Data point to score.

        Returns:
            Harmonic mean difference. Returns 0.0 if insufficient data.
        """
        if len(self.window) == 0:
            return 0.0

        new_value = self._extract_value(x)
        if new_value == 0:  # Skip zero values
            return 0.0

        # Calculate harmonic means
        current_reciprocal_sum = sum(1 / val for val in self.window)
        if current_reciprocal_sum == 0:
            raise ValueError("Harmonic mean is undefined for this window.")
        current_harmonic = len(self.window) / current_reciprocal_sum

        # Include new value
        all_values = list(self.window) + [new_value]
        new_reciprocal_sum = sum(1 / val for val in all_values)
        if new_reciprocal_sum == 0:
            raise ValueError("Harmonic mean is undefined after adding the sample.")
        new_harmonic = len(all_values) / new_reciprocal_sum

        difference = new_harmonic - current_harmonic
        return abs(difference) if self.abs_diff else difference

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"MovingHarmonicAverage(window_size={self.window_size}, "
            f"abs_diff={self.abs_diff})"
        )


class MovingGeometricAverage(BaseModel):
    """A simple moving model that calculates the difference between geometric average of a window + new value and the window.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
            abs_diff (bool): When true (default) returns abs() from the difference

    """

    def __init__(
        self,
        window_size: int,
        key: str | None = None,
        absoluteValues=False,
        abs_diff=True,
    ) -> None:
        """Initialize a new instance of MovingGeometricAverage.
        Args:
            window_size (int): The number of recent absolute values to consider for calculating the moving geometric average.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            absoluteValues (bool): Default is False. If True, the class is calculating the growth between the data point and then calculating
                the geometric avarage from window_size - 1 values.
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.absoluteValues = absoluteValues
        self.abs_diff: bool = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values. The value has to be greater then 0.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        if value > 0:
            self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the gemetric average of the current window (including the new data point)
        and the gemotric average of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new geometric average (including `x`) and the current geometric average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        _, value = _extract_univariate_value(x, self.feature_name)
        if value <= 0:
            raise ValueError("MovingGeometricAverage requires positive values.")

        actual_window_length = len(self.window)
        if actual_window_length <= 1 or (
            actual_window_length <= 2 and self.absoluteValues
        ):
            return 0.0
        else:
            if self.absoluteValues:
                window_growth = [
                    self.window[i + 1] / self.window[i]
                    for i in range(actual_window_length - 1)
                ]
                score_factor = value / self.window[-1]
            else:
                window_growth = self.window
                score_factor = value
            window_product = 1
            for value in window_growth:
                window_product *= value
            window_geo = window_product ** (1 / (len(window_growth)))
            score_geo = (window_product * score_factor) ** (
                1 / (len(window_growth) + 1)
            )
            return (
                abs(score_geo - window_geo) if self.abs_diff else score_geo - window_geo
            )


class MovingMedian(BaseModel):
    """A simple moving model that calculates the difference between median of a window + new value and the window.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingMedian.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving median.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the median of the current window (including the new data point)
        and the median of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new median (including `x`) and the current median of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0
        sorted_data = sorted(self.window)
        window_score = sorted_data.copy()
        _, value = _extract_univariate_value(x, self.feature_name)
        window_score.append(value)
        window_score.sort()

        mid_index_old = actual_window_length // 2
        mid_index_new = len(window_score) // 2

        if actual_window_length % 2 == 1:
            # If odd, return the middle element
            median_old = sorted_data[mid_index_old]
            median_new = (
                window_score[mid_index_new - 1] + window_score[mid_index_new]
            ) / 2
        else:
            # If even, return the average of the two middle elements
            median_old = (
                sorted_data[mid_index_old - 1] + sorted_data[mid_index_old]
            ) / 2
            median_new = window_score[mid_index_new]
        score = median_new - median_old
        return abs(score) if self.abs_diff else score

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"MovingMedian(window_size={len(self.window)}, abs_diff={self.abs_diff})"


class MovingQuantile(BaseModel):
    """A simple moving model that calculates the difference between quantile of a window + new value and the window.
    Attributes:
        window (collections.deque): A fixed-size deque storing the most recent values.
    """

    def __init__(
        self, window_size: int, key: str | None = None, quantile=0.5, abs_diff=True
    ) -> None:
        """Initialize a new instance of MovingQuantile.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving quantle.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            quantile (float): The quantile set for this instance. Default is 0.5
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be between 0 and 1.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.quantile = quantile
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def _quantile(self, sorted_list: list) -> float:
        window_length = len(sorted_list)
        rank = (
            window_length - 1
        ) * self.quantile  # Calculate the index of the quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, window_length - 1)
        fraction = rank - lower_index  # Calculate the fractional part
        if lower_index == upper_index:
            return sorted_list[lower_index]
        else:
            # Interpolation is necessary
            return (1 - fraction) * sorted_list[lower_index] + fraction * sorted_list[
                upper_index
            ]

    def score_one(self, x) -> float:
        """Calculate and return the difference between the quantile of the current window (including the new data point)
        and the quantile of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new quantile (including `x`) and the current quantile of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length <= 0:
            return 0

        sorted_window: list = sorted(self.window)
        score_data: list = sorted_window.copy()
        _, value = _extract_univariate_value(x, self.feature_name)
        score_data.append(value)
        score_data.sort()

        score = self._quantile(score_data) - self._quantile(sorted_window)
        return abs(score) if self.abs_diff else score


class MovingVariance(BaseModel):
    """A simple moving model that calculates the difference between variance of a window + new value and the window."""

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingVariance.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving variance.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the moving variance of the current window (including the new data point)
        and the moving variance of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving variance (including `x`) and the current moving variance of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length < 1:
            return 0
        else:
            score_window = list(self.window)
            _, value = _extract_univariate_value(x, self.feature_name)
            score_window.append(value)

            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            squared_diffs_window = [(value - mean_window) ** 2 for value in self.window]
            squared_diffs_score = [(value - mean_score) ** 2 for value in score_window]

            variance_window = sum(squared_diffs_window) / len(self.window)
            variance_score = sum(squared_diffs_score) / len(score_window)
            score = variance_score - variance_window
            return abs(score) if self.abs_diff else score

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"MovingVariance(window_size={len(self.window)}, abs_diff={self.abs_diff})"
        )


class MovingInterquartileRange(BaseModel):
    """A simple moving model that calculates the difference between interquartile range of a window + new value and the window."""

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingInterquartileRange.
        Args:
            window_size (int): The number of recent values to consider for calculating the
                                moving median.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point. The keys are feature names,
                                  and the values are the corresponding feature values.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def __score_one_quantile(self, sorted_list, quantile) -> float:
        """Calculate and return the value of a quantile of the values in the window.
        Returns:
            float: The quantile of the values in the window. 0 if the window is empty"""
        list_len = len(sorted_list)
        rank = (list_len - 1) * quantile  # Calculate the index of the quantile
        lower_index = int(rank)
        upper_index = min(lower_index + 1, list_len - 1)
        fraction = rank - lower_index
        if lower_index == upper_index:
            return sorted_list[lower_index]
        else:
            # Interpolation is necessary
            return (1 - fraction) * sorted_list[lower_index] + fraction * sorted_list[
                upper_index
            ]

    def score_one(self, x) -> float:
        """Calculate and return the difference between the interquartile range of the current window (including the new data point)
        and the moving interquartile range of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving interquartile range (including `x`) and the current moving average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        self.actual_window_length = len(self.window)
        if self.actual_window_length <= 0:
            return 0

        sorted_data = sorted(self.window)
        score_window = sorted_data.copy()
        _, value = _extract_univariate_value(x, self.feature_name)
        score_window.append(value)
        score_window.sort()
        iqr_window = self.__score_one_quantile(
            sorted_data, 0.75
        ) - self.__score_one_quantile(sorted_data, 0.25)
        iqr_score = self.__score_one_quantile(
            score_window, 0.75
        ) - self.__score_one_quantile(score_window, 0.25)
        score = iqr_score - iqr_window
        return abs(score) if self.abs_diff else score


class MovingAverageAbsoluteDeviation(BaseModel):
    """A simple moving model that calculates the difference between absolute deviation of a window + new value and the window."""

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingAverageAbsoluteDeviation.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving average absolute deviation.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the moving average absolute deviation of the current window (including the new data point)
        and the moving average absolute deviation of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new moving average (including `x`) and the current moving average of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            _, value = _extract_univariate_value(x, self.feature_name)
            score_window.append(value)
            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)
            dev_window = (
                sum(abs(value - mean_window) for value in self.window)
                / actual_window_length
            )
            dev_score = sum(abs(value - mean_score) for value in score_window) / len(
                score_window
            )

            score = dev_score - dev_window
            return abs(score) if self.abs_diff else score


class MovingKurtosis(BaseModel):
    """A simple moving model that calculates the difference between kurtosis of a window + new value and the window."""

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingKurtosis.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving kurtosis.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
            abs_diff (bool): When true (default) returns abs() from the difference.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the kurtosis of the current window (including the new data point)
        and the kurtosis of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new kurtosis (including `x`) and the current kurtosis of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            _, value = _extract_univariate_value(x, self.feature_name)
            score_window.append(value)
            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            central_moment_4_window = (
                sum((value - mean_window) ** 4 for value in self.window)
                / actual_window_length
            )
            central_moment_4_score = sum(
                (value - mean_score) ** 4 for value in score_window
            ) / len(score_window)

            std_4_window = (
                sum((value - mean_window) ** 2 for value in self.window)
                / actual_window_length
            ) ** 2
            std_4_score = (
                sum((value - mean_score) ** 2 for value in score_window)
                / len(score_window)
            ) ** 2

            if std_4_window == 0 or std_4_score == 0:
                return 0
            else:
                kurtosis_window = central_moment_4_window / std_4_window
                kurtosis_score = central_moment_4_score / std_4_score
                score = kurtosis_score - kurtosis_window
                return abs(score) if self.abs_diff else score


class MovingSkewness(BaseModel):
    """A simple moving model that calculates the difference between skewness of a window + new value and the window."""

    def __init__(self, window_size: int, key: str | None = None, abs_diff=True) -> None:
        """Initialize a new instance of MovingSkewness.
        Args:
            window_size (int): The number of recent values to consider for calculating the moving skewness.
            key (str): optional, if None is given, the first learned key-value pair will set the key
            abs_diff (bool): When true (default) returns abs() from the difference
        Raises:
            ValueError: If window_size is not a positive integer."""
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        self.window: deque[float] = deque([], maxlen=window_size)
        self.feature_name: str | None = key
        self.abs_diff = abs_diff

    def learn_one(self, x: dict[str, float]) -> None:
        """Update the model with a single resource point.
        Args:
            x (Dict[str, float]): A dictionary representing a single resource point.
        Raises:
            AssertionError: If the input dictionary contains more than one key-value pair or is empty.
        """
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        self.window.append(value)

    def score_one(self, x) -> float:
        """Calculate and return the difference between the skewness of the current window (including the new data point)
        and the skewness of the original window.

        Args:
            x (Dict[str, float]): A dictionary representing a single data point. It is expected to contain one key-value pair,
            where the value will be used in the calculation.

        Returns:
            float: The difference between the new skewness (including `x`) and the current skewness of the window.
            If the original window has zero or fewer elements, returns 0.
        """
        actual_window_length = len(self.window)
        if actual_window_length == 0:
            return 0
        else:
            score_window = list(self.window)
            _, value = _extract_univariate_value(x, self.feature_name)
            score_window.append(value)

            mean_window = sum(self.window) / actual_window_length
            mean_score = sum(score_window) / len(score_window)

            central_moment_3_window = (
                sum((value - mean_window) ** 3 for value in self.window)
                / actual_window_length
            )
            central_moment_3_score = sum(
                (value - mean_score) ** 3 for value in score_window
            ) / len(score_window)

            std_3_window = (
                sum((value - mean_window) ** 2 for value in self.window)
                / actual_window_length
            ) ** (3 / 2)
            std_3_score = (
                sum((value - mean_score) ** 2 for value in score_window)
                / len(score_window)
            ) ** (3 / 2)
            if std_3_score == 0 or std_3_window == 0:
                return 0
            else:
                skewness_window = central_moment_3_window / std_3_window
                skewness_score = central_moment_3_score / std_3_score
                score = skewness_score - skewness_window
                return abs(score) if self.abs_diff else score
