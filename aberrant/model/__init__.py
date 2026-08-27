"""Anomaly detection models for streaming data.

Available submodules:
    - distance: Distance-based models (KNN, LocalOutlierFactor)
    - graph: Graph-stream models (ISCONNA)
    - iforest: Isolation forest variants
    - sketch: Sketch-based streaming models
    - stat: Statistical models
    - svm: SVM-based models
    - timeseries: Time-series discord models
    - deep: Deep learning models (requires torch)

Also available directly:
    - NullModel, RandomModel, ThresholdModel, QuantileThreshold
"""

from aberrant.model.null import NullModel
from aberrant.model.quantile_threshold import QuantileThreshold
from aberrant.model.random import RandomModel
from aberrant.model.threshold import ThresholdModel

__all__ = [
    "NullModel",
    "RandomModel",
    "ThresholdModel",
    "QuantileThreshold",
]
