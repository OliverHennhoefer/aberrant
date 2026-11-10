"""Online calibration of anomaly and forecast scores.

A calibrator turns a stream of scores into a threshold that ``1 - alpha`` of
future scores should stay below, learned online with delayed feedback. The
methods are ports of the online conformal and Bayesian methods compared in
the Timestral Uncertainty Arena.
"""

from aberrant.calibration._calibrator import ScoreCalibrator
from aberrant.calibration._registry import (
    CALIBRATORS,
    CalibratorParameter,
    CalibratorSpec,
    Coverage,
    OnlineCalibrator,
)
from aberrant.calibration._window import ScoreHistory, conformal_quantile, pinball

__all__ = [
    "CALIBRATORS",
    "CalibratorParameter",
    "CalibratorSpec",
    "Coverage",
    "OnlineCalibrator",
    "ScoreCalibrator",
    "ScoreHistory",
    "conformal_quantile",
    "pinball",
]
