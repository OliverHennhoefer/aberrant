"""Deep learning models for anomaly detection (optional torch dependency)."""

import importlib
from typing import TYPE_CHECKING

from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble

if TYPE_CHECKING:
    from aberrant.model.deep.autoencoder import Autoencoder as Autoencoder

__all__ = [
    "OnlineAutoencoderEnsemble",
]


def __getattr__(name: str) -> object:
    """Load the optional torch autoencoder on explicit access."""
    if name == "Autoencoder":
        try:
            module = importlib.import_module("aberrant.model.deep.autoencoder")
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "Autoencoder requires the optional 'dl' dependencies"
                ) from exc
            raise
        autoencoder = module.Autoencoder
        globals()[name] = autoencoder
        return autoencoder
    raise AttributeError(f"module 'aberrant.model.deep' has no attribute '{name}'")
