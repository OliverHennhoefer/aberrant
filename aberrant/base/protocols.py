"""Structural contracts for online pipeline components."""

from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias, runtime_checkable

FeatureMap: TypeAlias = dict[str, float]


class LearnerProtocol(Protocol):
    """A component that updates itself from one feature mapping."""

    def learn_one(self, x: FeatureMap) -> None:
        """Update the component from one sample."""
        ...


@runtime_checkable
class TransformerProtocol(LearnerProtocol, Protocol):
    """Structural interface accepted for transformer pipeline stages."""

    def transform_one(self, x: FeatureMap) -> FeatureMap:
        """Transform one sample without updating learned state."""
        ...


@runtime_checkable
class TransactionalTransformerProtocol(TransformerProtocol, Protocol):
    """A transformer that restores learned state when a downstream update fails."""

    def learning_transaction(self, x: FeatureMap) -> AbstractContextManager[None]:
        """Keep updates on success and restore prior state when the context raises."""
        ...


@runtime_checkable
class ModelProtocol(LearnerProtocol, Protocol):
    """Structural interface accepted for a terminal anomaly model."""

    def score_one(self, x: FeatureMap) -> float:
        """Score one sample without updating learned state."""
        ...
