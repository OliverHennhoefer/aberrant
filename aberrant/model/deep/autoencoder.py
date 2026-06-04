"""Online autoencoder for anomaly detection."""

import torch
from torch import optim

from aberrant.base.architecture import Architecture
from aberrant.base.model import BaseModel
from aberrant.utils.deep.loss_func import AutoencoderLoss


class Autoencoder(BaseModel):
    """
    Online autoencoder for anomaly detection.

    This model trains an autoencoder architecture incrementally on data points
    and uses reconstruction error as an anomaly score.

    Args:
        model: The neural network architecture (encoder-decoder).
        optimizer: PyTorch optimizer for training.
        criterion: Loss function for reconstruction error.

    Example:
        >>> from torch import nn, optim
        >>> from aberrant.utils.deep.architecture import VanillaAutoencoder
        >>>
        >>> architecture = VanillaAutoencoder(input_size=10)
        >>> autoencoder = Autoencoder(
        ...     model=architecture,
        ...     optimizer=optim.Adam(architecture.parameters()),
        ...     criterion=nn.MSELoss()
        ... )
    """

    def __init__(
        self,
        model: Architecture,
        optimizer: optim.Optimizer,
        criterion: AutoencoderLoss,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self._feature_order: list[str] | None = None

        # Pre-allocate tensors on the correct device to avoid repeated creation
        device = model.device
        self.x_tensor = torch.empty(
            1, self.model.input_size, dtype=torch.float32, device=device
        )

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the autoencoder with a single data point.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        # Set model to training mode
        self.model.train()

        # Efficiently load data into pre-allocated tensor without creating new tensors
        self._dict_to_tensor(x, self.x_tensor)

        # Forward pass and backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(self.x_tensor)
        loss = self.criterion(output, self.x_tensor)
        loss.backward()
        self.optimizer.step()

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Reconstruction error as anomaly score.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Efficiently load data into pre-allocated tensor
        self._dict_to_tensor(x, self.x_tensor)

        with torch.no_grad():
            output = self.model(self.x_tensor)
            loss = self.criterion(output, self.x_tensor)
        return loss.item()

    def _dict_to_tensor(self, x: dict[str, float], tensor: torch.Tensor) -> None:
        """
        Efficiently convert dictionary to tensor without creating intermediate tensors.

        Args:
            x: Input feature dictionary.
            tensor: Pre-allocated tensor to fill (modified in-place).

        Raises:
            ValueError: If the input schema does not match the architecture or the
                schema established by the first sample.
        """
        if self._feature_order is None:
            feature_order = sorted(x.keys())
            if len(feature_order) != self.model.input_size:
                raise ValueError(
                    f"Expected {self.model.input_size} features, got "
                    f"{len(feature_order)}."
                )
            self._feature_order = feature_order
        elif len(x) != len(self._feature_order) or any(
            key not in x for key in self._feature_order
        ):
            feature_order = sorted(x.keys())
            raise ValueError(
                f"Input features must be {self._feature_order}, got {feature_order}."
            )

        # Fill tensor directly from dictionary values
        for i, key in enumerate(self._feature_order):
            tensor[0, i] = x[key]

    def __repr__(self) -> str:
        """Return a string representation of the autoencoder."""
        return (
            f"Autoencoder(model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"criterion={self.criterion.__class__.__name__})"
        )
