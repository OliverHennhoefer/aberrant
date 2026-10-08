"""Online autoencoder for anomaly detection."""

from itertools import chain

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

    Built-in architectures retain no event history or computation graphs.
    Input, loss, and gradient checks reject detected numerical overflow before
    stepping the optimizer. Supplied architectures and optimizers control their
    own retained state and arithmetic; their counters, internal moments, and
    parameter updates can still overflow or lose precision on long streams.

    Args:
        model: The neural network architecture (encoder-decoder).
        optimizer: PyTorch optimizer for training.
        criterion: Loss function for reconstruction error.

    Examples:
        ```python
        from torch import nn, optim

        from aberrant.model.deep import Autoencoder
        from aberrant.utils.deep.architecture import VanillaAutoencoder

        architecture = VanillaAutoencoder(input_size=10)
        autoencoder = Autoencoder(
            model=architecture,
            optimizer=optim.Adam(architecture.parameters()),
            criterion=nn.MSELoss(),
        )
        ```
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

        # The supplied module owns tensor placement and dtype, including changes
        # made through standard PyTorch .to(), .double(), and .float() calls.
        device, dtype = self._input_options()
        self.x_tensor = torch.empty(
            1, self.model.input_size, device=device, dtype=dtype
        )

    def _input_options(self) -> tuple[torch.device, torch.dtype]:
        reference = next(chain(self.model.parameters(), self.model.buffers()), None)
        if reference is None:
            return self.model.device, torch.float32
        return reference.device, reference.dtype

    def _input_tensor(self) -> torch.Tensor:
        """Reuse input storage until the module's placement or dtype changes."""
        device, dtype = self._input_options()
        if self.x_tensor.device != device or self.x_tensor.dtype != dtype:
            self.x_tensor = torch.empty(
                1, self.model.input_size, device=device, dtype=dtype
            )
        return self.x_tensor

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the autoencoder with a single data point.

        Args:
            x: Feature dictionary with string keys and float values.
        """
        prepared = self._schema.preview(x)
        x_tensor = self._input_tensor()

        # Set model to training mode
        self.model.train()

        # Efficiently load data into pre-allocated tensor without creating new tensors
        self._fill_tensor(prepared, x_tensor)

        # Forward pass and backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(x_tensor)
        loss = self.criterion(output, x_tensor)
        if not torch.isfinite(loss).all():
            raise OverflowError("Autoencoder loss exceeds the model's numeric range")
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                self.optimizer.zero_grad(set_to_none=True)
                raise OverflowError("Autoencoder gradient exceeds the numeric range")
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
        x_tensor = self._input_tensor()

        # Set model to evaluation mode
        self.model.eval()

        # Efficiently load data into pre-allocated tensor
        self._fill_tensor(prepared, x_tensor)

        with torch.no_grad():
            output = self.model(x_tensor)
            loss = self.criterion(output, x_tensor)
        if not torch.isfinite(loss).all():
            raise OverflowError("Autoencoder loss exceeds the model's numeric range")
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
        if not torch.isfinite(tensor).all():
            raise OverflowError("Input exceeds the autoencoder tensor's numeric range")

    def __repr__(self) -> str:
        """Return a string representation of the autoencoder."""
        return (
            f"Autoencoder(model={self.model.__class__.__name__}, "
            f"optimizer={self.optimizer.__class__.__name__}, "
            f"criterion={self.criterion.__class__.__name__})"
        )
