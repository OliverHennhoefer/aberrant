"""Statistical models for streaming anomaly detection."""

from aberrant.model.stat._univariate_averages import (
    MovingAverage,
    MovingGeometricAverage,
    MovingHarmonicAverage,
)
from aberrant.model.stat._univariate_moments import (
    MovingAverageAbsoluteDeviation,
    MovingInterquartileRange,
    MovingKurtosis,
    MovingSkewness,
    MovingVariance,
)
from aberrant.model.stat._univariate_order import (
    MovingMedian,
    MovingQuantile,
)
from aberrant.model.stat.multi import (
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)

__all__ = [
    "MovingAverage",
    "MovingAverageAbsoluteDeviation",
    "MovingCorrelationCoefficient",
    "MovingCovariance",
    "MovingGeometricAverage",
    "MovingHarmonicAverage",
    "MovingInterquartileRange",
    "MovingKurtosis",
    "MovingMahalanobisDistance",
    "MovingMedian",
    "MovingQuantile",
    "MovingSkewness",
    "MovingVariance",
]
