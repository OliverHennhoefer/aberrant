"""Deep learning models for anomaly detection (optional torch dependency)."""

import importlib
from typing import Any

__all__ = [
    "Autoencoder",
    "KitNET",
    "OnlineAutoencoderEnsemble",
]


def __getattr__(name: str) -> Any:
    """Lazy import of deep model classes."""
    if name == "Autoencoder":
        module = importlib.import_module("aberrant.model.deep.autoencoder")
        return module.Autoencoder
    if name in {"KitNET", "OnlineAutoencoderEnsemble"}:
        module = importlib.import_module("aberrant.model.deep.kitnet")
        return getattr(module, name)
    raise AttributeError(f"module 'aberrant.model.deep' has no attribute '{name}'")
