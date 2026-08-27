"""Neural network architecture base class for deep learning models."""

import abc
import secrets

import torch
from torch import nn


class Architecture(abc.ABC, nn.Module):
    """
    Abstract base class for defining neural network architectures.

    This class ensures that any neural network architecture can be plugged into
    online anomaly detection models. It provides a consistent interface and
    device handling capabilities.

    Subclasses must implement the `forward` and `input_size` methods.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """
        Initialize the architecture.

        Args:
            device: The device to run the model on. If None, uses CPU.
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.to(self.device)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """
        The expected input size for the network.

        Returns:
            Number of input features.
        """
        raise NotImplementedError

    @staticmethod
    def make_torch_generator(
        seed: int | None,
        device: torch.device | str = "cpu",
    ) -> torch.Generator:
        """Create an independently seeded, model-owned generator."""
        generator = torch.Generator(device=device)
        generator.manual_seed(seed if seed is not None else secrets.randbits(63))
        return generator

    def __repr__(self) -> str:
        """Return a string representation of the architecture."""
        return f"{self.__class__.__name__}(input_size={self.input_size}, device={self.device})"
