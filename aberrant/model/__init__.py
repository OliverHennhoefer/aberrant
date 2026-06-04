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

import importlib
from typing import Any

# Lazy imports - only import when accessed to avoid torch dependency
__all__ = [
    "NullModel",
    "RandomModel",
    "ThresholdModel",
    "QuantileThreshold",
]


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    if name == "NullModel":
        module = importlib.import_module("aberrant.model.null")
        return module.NullModel
    if name == "RandomModel":
        module = importlib.import_module("aberrant.model.random")
        return module.RandomModel
    if name == "ThresholdModel":
        module = importlib.import_module("aberrant.model.threshold")
        return module.ThresholdModel
    if name == "QuantileThreshold":
        module = importlib.import_module("aberrant.model.quantile_threshold")
        return module.QuantileThreshold
    raise AttributeError(f"module 'aberrant.model' has no attribute '{name}'")
