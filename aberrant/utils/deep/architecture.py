"""Neural network architectures for autoencoder models."""

import math
from typing import cast

import torch
from torch import nn

from aberrant.base.architecture import Architecture


def _linear(
    in_features: int,
    out_features: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> nn.Linear:
    """Create a Linear layer without touching PyTorch's default RNG."""
    layer = nn.Linear(in_features, out_features, device="meta")
    layer.to_empty(device=device)
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
    if layer.bias is not None:
        fan_in = layer.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
    return layer


def _lstm(
    input_size: int,
    hidden_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    batch_first: bool,
) -> nn.LSTM:
    """Create an LSTM without touching PyTorch's default RNG."""
    layer = nn.LSTM(
        input_size,
        hidden_size,
        batch_first=batch_first,
        device="meta",
    )
    layer.to_empty(device=device)
    bound = 1.0 / math.sqrt(hidden_size)
    for parameter in layer.parameters():
        nn.init.uniform_(parameter, -bound, bound, generator=generator)
    return layer


class VanillaAutoencoder(Architecture):
    """
    Simple feedforward autoencoder with ReLU activations.

    Architecture: input -> 64 -> 32 -> 16 -> 32 -> 64 -> output

    Args:
        input_size: Number of input features.
        seed: Random seed for reproducibility (optional).
    """

    def __init__(
        self,
        input_size: int,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._input_size = input_size
        self._initialization_generator = self.make_torch_generator(seed, self.device)

        self.encoder = nn.Sequential(
            _linear(
                input_size,
                64,
                generator=self._initialization_generator,
                device=self.device,
            ),
            nn.ReLU(),
            _linear(
                64,
                32,
                generator=self._initialization_generator,
                device=self.device,
            ),
            nn.ReLU(),
            _linear(
                32,
                16,
                generator=self._initialization_generator,
                device=self.device,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            _linear(
                16,
                32,
                generator=self._initialization_generator,
                device=self.device,
            ),
            nn.ReLU(),
            _linear(
                32,
                64,
                generator=self._initialization_generator,
                device=self.device,
            ),
            nn.ReLU(),
            _linear(
                64,
                input_size,
                generator=self._initialization_generator,
                device=self.device,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return cast(torch.Tensor, decoded)

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self._input_size


class VanillaLSTMAutoencoder(Architecture):
    """
    LSTM-based autoencoder for sequence data.

    Args:
        input_size: Number of input features.
        seed: Random seed for reproducibility (optional).
    """

    def __init__(
        self,
        input_size: int,
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._input_size = input_size
        self._initialization_generator = self.make_torch_generator(seed, self.device)

        self.encoder = _lstm(
            input_size,
            64,
            generator=self._initialization_generator,
            device=self.device,
            batch_first=True,
        )
        self.hidden_to_latent = _linear(
            64,
            16,
            generator=self._initialization_generator,
            device=self.device,
        )

        self.latent_to_hidden = _linear(
            16,
            64,
            generator=self._initialization_generator,
            device=self.device,
        )
        self.decoder = _lstm(
            64,
            input_size,
            generator=self._initialization_generator,
            device=self.device,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM encoder and decoder."""
        _, (h_n, _) = self.encoder(x)
        latent = self.hidden_to_latent(h_n.squeeze(0))
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        decoded, _ = self.decoder(hidden)
        return cast(torch.Tensor, decoded)

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self._input_size
