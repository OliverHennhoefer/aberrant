"""Structural contracts for online pipeline components."""

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
class ModelProtocol(LearnerProtocol, Protocol):
    """Structural interface accepted for a terminal anomaly model."""

    def score_one(self, x: FeatureMap) -> float:
        """Score one sample without updating learned state."""
        ...
