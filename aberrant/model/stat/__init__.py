"""Statistical models for streaming anomaly detection."""

from aberrant.model.stat.ecdf import OnlineCOPOD, OnlineECOD
from aberrant.model.stat.moving import MovingMAD, MovingZScore
from aberrant.model.stat.multi import (
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingMahalanobisDistance,
)

__all__ = [
    "MovingCorrelationCoefficient",
    "MovingCovariance",
    "MovingMAD",
    "MovingMahalanobisDistance",
    "MovingZScore",
    "OnlineCOPOD",
    "OnlineECOD",
]
