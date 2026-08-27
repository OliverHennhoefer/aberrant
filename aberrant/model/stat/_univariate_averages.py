"""Arithmetic, harmonic, and geometric moving-window detectors."""

from __future__ import annotations

from aberrant.model.stat._univariate_base import (
    _BaseMovingUnivariate,
    _extract_univariate_value,
)


class MovingAverage(_BaseMovingUnivariate):
    """Score the change in arithmetic mean after adding a candidate value."""

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        window_sum = sum(self.window)
        current_mean = window_sum / len(self.window)
        new_mean = (window_sum + self._extract_value(x)) / (len(self.window) + 1)
        return self._difference(new_mean, current_mean)

    def __repr__(self) -> str:
        return (
            f"MovingAverage(window_size={self.window_size}, abs_diff={self.abs_diff})"
        )


class MovingHarmonicAverage(_BaseMovingUnivariate):
    """Score harmonic-mean change, ignoring zero values during learning."""

    def learn_one(self, x: dict[str, float]) -> None:
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        if value != 0:
            self.window.append(value)

    def score_one(self, x: dict[str, float]) -> float:
        if not self.window:
            return 0.0

        new_value = self._extract_value(x)
        if new_value == 0:
            return 0.0

        current_reciprocal_sum = sum(1 / value for value in self.window)
        if current_reciprocal_sum == 0:
            raise ValueError("Harmonic mean is undefined for this window.")
        current_harmonic = len(self.window) / current_reciprocal_sum

        new_reciprocal_sum = sum(1 / value for value in [*self.window, new_value])
        if new_reciprocal_sum == 0:
            raise ValueError("Harmonic mean is undefined after adding the sample.")
        new_harmonic = (len(self.window) + 1) / new_reciprocal_sum
        return self._difference(new_harmonic, current_harmonic)

    def __repr__(self) -> str:
        return (
            f"MovingHarmonicAverage(window_size={self.window_size}, "
            f"abs_diff={self.abs_diff})"
        )


class MovingGeometricAverage(_BaseMovingUnivariate):
    """Score geometric-mean changes for values or successive growth factors."""

    def __init__(
        self,
        window_size: int,
        key: str | None = None,
        absoluteValues: bool = False,
        abs_diff: bool = True,
    ) -> None:
        super().__init__(window_size, key, abs_diff)
        self.absoluteValues = absoluteValues

    def learn_one(self, x: dict[str, float]) -> None:
        self.feature_name, value = _extract_univariate_value(x, self.feature_name)
        if value > 0:
            self.window.append(value)

    def score_one(self, x: dict[str, float]) -> float:
        value = self._extract_value(x)
        if value <= 0:
            raise ValueError("MovingGeometricAverage requires positive values.")

        window_length = len(self.window)
        if window_length <= 1 or (window_length <= 2 and self.absoluteValues):
            return 0.0

        if self.absoluteValues:
            factors = [
                self.window[index + 1] / self.window[index]
                for index in range(window_length - 1)
            ]
            score_factor = value / self.window[-1]
        else:
            factors = list(self.window)
            score_factor = value

        product = 1.0
        for factor in factors:
            product *= factor
        current_geometric = product ** (1 / len(factors))
        new_geometric = (product * score_factor) ** (1 / (len(factors) + 1))
        return self._difference(new_geometric, current_geometric)
