"""Online autoencoder for anomaly detection."""

import torch
from torch import optim

from aberrant.base.architecture import Architecture
from aberrant.base.model import BaseModel
from aberrant.utils.deep.loss_func import AutoencoderLoss
from aberrant.utils.validation import FeatureSchema, PreparedFeatures


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
        self._schema = FeatureSchema(expected_size=model.input_size)

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
        prepared = self._schema.preview(x)

        # Set model to training mode
        self.model.train()

        # Efficiently load data into pre-allocated tensor without creating new tensors
        self._fill_tensor(prepared, self.x_tensor)

        # Forward pass and backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(self.x_tensor)
        loss = self.criterion(output, self.x_tensor)
        loss.backward()
        self.optimizer.step()
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a single data point.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Reconstruction error as anomaly score.
        """
        prepared = self._schema.preview(x)

        # Set model to evaluation mode
        self.model.eval()

        # Efficiently load data into pre-allocated tensor
        self._fill_tensor(prepared, self.x_tensor)

        with torch.no_grad():
            output = self.model(self.x_tensor)
            loss = self.criterion(output, self.x_tensor)
        return float(loss.item())

    @staticmethod
    def _fill_tensor(prepared: PreparedFeatures, tensor: torch.Tensor) -> None:
        """
        Efficiently convert dictionary to tensor without creating intermediate tensors.

        Args:
            prepared: Validated feature values in stable schema order.
            tensor: Pre-allocated tensor to fill (modified in-place).
        """
        tensor[0].copy_(torch.as_tensor(prepared.values, device=tensor.device))

    def __repr__(self) -> str:
        """Return a string representation of the autoencoder."""
        return (
            f"Autoencoder(model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"criterion={self.criterion.__class__.__name__})"
        )
