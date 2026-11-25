"""Anomaly detection models for streaming data.

Available submodules:
    - distance: Distance-based models (KNN, LocalOutlierFactor)
    - forest: Tree ensembles (isolation, half-space, random-cut and histogram forests)
    - graph: Graph-stream models (ISCONNA)
    - sketch: Sketch-based streaming models
    - stat: Statistical models
    - svm: SVM-based models
    - timeseries: Time-series discord and seasonal forecast-residual models
    - deep: Deep learning models (requires torch)

Also available directly:
    - NullModel, RandomModel, ThresholdModel
"""

from aberrant.model.null import NullModel
from aberrant.model.random import RandomModel
from aberrant.model.threshold import ThresholdModel

__all__ = [
    "NullModel",
    "RandomModel",
    "ThresholdModel",
]
