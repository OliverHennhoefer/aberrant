"""SVM-based models for streaming anomaly detection."""

from aberrant.model.svm.adaptive import IncrementalOneClassSVMAdaptiveKernel
from aberrant.model.svm.gadget import GraphGatedOneClassSVM

__all__ = [
    "GraphGatedOneClassSVM",
    "IncrementalOneClassSVMAdaptiveKernel",
]
