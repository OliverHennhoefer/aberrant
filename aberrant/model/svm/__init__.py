"""SVM-based models for streaming anomaly detection."""

from aberrant.model.svm.adaptive import IncrementalOneClassSVMAdaptiveKernel

__all__ = [
    "IncrementalOneClassSVMAdaptiveKernel",
]
