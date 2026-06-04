"""Distance-based anomaly detection models."""

import importlib
from typing import Any

__all__ = [
    "KNN",
    "LocalOutlierFactor",
    "CellNeighborhoodDetector",
    "NETS",
    "SDOStream",
    "StationaryRegionNeighborDetector",
    "STARE",
]


def __getattr__(name: str) -> Any:
    """Lazy import of model classes."""
    if name == "KNN":
        module = importlib.import_module("aberrant.model.distance.knn")
        return module.KNN
    if name == "LocalOutlierFactor":
        module = importlib.import_module("aberrant.model.distance.lof")
        return module.LocalOutlierFactor
    if name in {"CellNeighborhoodDetector", "NETS"}:
        module = importlib.import_module("aberrant.model.distance.nets")
        return getattr(module, name)
    if name == "SDOStream":
        module = importlib.import_module("aberrant.model.distance.sdostream")
        return module.SDOStream
    if name in {"StationaryRegionNeighborDetector", "STARE"}:
        module = importlib.import_module("aberrant.model.distance.stare")
        return getattr(module, name)
    raise AttributeError(f"module 'aberrant.model.distance' has no attribute '{name}'")
