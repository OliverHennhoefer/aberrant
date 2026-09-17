"""Time-series anomaly detection models."""

from aberrant.model.timeseries.damp import XLagDAMP
from aberrant.model.timeseries.rolling_matrix_profile import RollingMatrixProfile

__all__ = ["RollingMatrixProfile", "XLagDAMP"]
