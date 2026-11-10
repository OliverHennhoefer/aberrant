"""Structural contracts for online pipeline components."""

from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias, runtime_checkable

from aberrant.base.forecast import Forecast

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
class TransactionalLearnerProtocol(LearnerProtocol, Protocol):
    """A learner that can restore its state when a learning pass fails.

    The context remains open through downstream learning. Implementations must
    restore their own state when the context raises; the pipeline cannot make
    nontransactional components atomic.
    """

    def learning_transaction(self, x: FeatureMap) -> AbstractContextManager[None]:
        """Keep updates on success and restore prior state when the context raises."""
        ...


@runtime_checkable
class TransactionalTransformerProtocol(
    TransformerProtocol, TransactionalLearnerProtocol, Protocol
):
    """A transformer with the transactional learner capability."""


@runtime_checkable
class ModelProtocol(LearnerProtocol, Protocol):
    """Structural interface accepted for a terminal anomaly model."""

    def score_one(self, x: FeatureMap) -> float:
        """Score one sample without updating learned state."""
        ...


@runtime_checkable
class ForecasterProtocol(Protocol):
    """Structural interface accepted for an online forecaster.

    A forecaster learns one target series. ``forecast_one()`` forecasts the
    next ``horizon`` values from the series' own past without learning, and
    ``learn_one(y)`` then learns the value that arrived.
    """

    @property
    def horizon(self) -> int:
        """The largest lead a forecast holds."""
        ...

    def learn_one(self, y: float) -> None:
        """Learn the next value of the series."""
        ...

    def forecast_one(self) -> Forecast | None:
        """Forecast leads ``1..horizon``, or ``None`` before readiness."""
        ...


@runtime_checkable
class RegressorProtocol(Protocol):
    """Structural interface accepted for an online regressor.

    A regressor predicts a target from the covariates of the same event.
    ``predict_one(x)`` predicts without learning, and ``learn_one(x, y)``
    learns the target once it is known, which may be later.
    """

    def learn_one(self, x: FeatureMap, y: float) -> None:
        """Learn the target ``y`` of the event with covariates ``x``."""
        ...

    def predict_one(self, x: FeatureMap) -> Forecast | None:
        """Predict the target at lead 0, or ``None`` before readiness."""
        ...
