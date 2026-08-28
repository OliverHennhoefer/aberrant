"""Base transformer interface for online data transformation."""

import abc
from typing import overload

from aberrant.base.pipeline import Pipeline
from aberrant.base.protocols import ModelProtocol, TransformerProtocol


class BaseTransformer(abc.ABC):
    """
    Abstract base class for online transformers.

    Transformers learn and transform one feature mapping at a time. A standalone
    transformer does not prescribe whether learning happens before or after
    transformation; :class:`~aberrant.base.pipeline.Pipeline` deliberately uses
    post-update transformations during ``learn_one``.

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
