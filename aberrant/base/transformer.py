"""Base transformer interface for online data transformation."""

import abc
from typing import overload

from aberrant.base.pipeline import Pipeline
from aberrant.base.protocols import ModelProtocol, TransformerProtocol


class BaseTransformer(abc.ABC):
    """
    Abstract base class for online transformers.

    This class defines the interface for transformers that can learn from and transform
    data points incrementally. Transformers modify the input data while maintaining
    the streaming nature of the processing.

    Subclasses must implement the `learn_one` and `transform_one` methods.
    """

    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the transformer with a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
               and the values are the corresponding feature values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """
        Transform a single data point.

        Args:
            x: A dictionary representing a single data point to transform.

        Returns:
            A dictionary with transformed feature values.
        """
        raise NotImplementedError

    @overload
    def __or__(self, other: TransformerProtocol) -> "Pipeline[TransformerProtocol]": ...

    @overload
    def __or__(self, other: ModelProtocol) -> "Pipeline[ModelProtocol]": ...

    def __or__(
        self, other: TransformerProtocol | ModelProtocol
    ) -> "Pipeline[TransformerProtocol] | Pipeline[ModelProtocol]":
        if isinstance(other, TransformerProtocol) and not isinstance(
            other, ModelProtocol
        ):
            return Pipeline(self, other)
        return Pipeline(self, other)

    def __repr__(self) -> str:
        """Return a string representation of the transformer."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the transformer."""
        return self.__repr__()
