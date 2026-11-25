"""Deep learning models for anomaly detection.

``OnlineAutoencoderEnsemble`` needs only NumPy. ``Autoencoder`` and the
PyTorch architectures it trains need the optional ``dl`` extra and load on
first access.
"""

import importlib
from typing import TYPE_CHECKING

from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble

if TYPE_CHECKING:
    from aberrant.model.deep._architecture import Architecture as Architecture
    from aberrant.model.deep._autoencoders import (
        VanillaAutoencoder as VanillaAutoencoder,
    )
    from aberrant.model.deep._autoencoders import (
        VanillaLSTMAutoencoder as VanillaLSTMAutoencoder,
    )
    from aberrant.model.deep.autoencoder import Autoencoder as Autoencoder

_TORCH_EXPORTS = {
    "Architecture": "aberrant.model.deep._architecture",
    "Autoencoder": "aberrant.model.deep.autoencoder",
    "VanillaAutoencoder": "aberrant.model.deep._autoencoders",
    "VanillaLSTMAutoencoder": "aberrant.model.deep._autoencoders",
}

__all__ = ["OnlineAutoencoderEnsemble"]


def __getattr__(name: str) -> object:
    """Load the PyTorch components on explicit access."""
    if name not in _TORCH_EXPORTS:
        raise AttributeError(f"module 'aberrant.model.deep' has no attribute '{name}'")
    try:
        module = importlib.import_module(_TORCH_EXPORTS[name])
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                f"{name} requires the optional 'dl' dependencies"
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value
