"""Graph-stream anomaly detection models."""

import importlib
from typing import Any

__all__ = [
    "AnoEdgeL",
    "ISCONNA",
    "MIDAS",
    "SignedGraphSketchDetector",
    "StreamSpot",
]


def __getattr__(name: str) -> Any:
    """Lazy import of graph model classes."""
    if name == "AnoEdgeL":
        module = importlib.import_module("aberrant.model.graph.anoedge")
        return module.AnoEdgeL
    if name == "ISCONNA":
        module = importlib.import_module("aberrant.model.graph.isconna")
        return module.ISCONNA
    if name == "MIDAS":
        module = importlib.import_module("aberrant.model.graph.midas")
        return module.MIDAS
    if name in {"SignedGraphSketchDetector", "StreamSpot"}:
        module = importlib.import_module("aberrant.model.graph.streamspot")
        return getattr(module, name)
    raise AttributeError(f"module 'aberrant.model.graph' has no attribute '{name}'")
