"""Base model interface for online anomaly detection."""

import abc
from collections.abc import Iterable

from aberrant.base._parameters import Parameterized
from aberrant.base.readiness import Readiness


class BaseModel(Parameterized, abc.ABC):
    """
    Abstract base class for online anomaly detection models.

    Online models process one observation at a time. ``score_one`` evaluates an
    observation against the model's current reference state; ``learn_one``
    incorporates an observation into that state.

    Every model follows one score convention: a higher score is more
    anomalous, the score is 0.0 until the model has learned enough to judge,
    and it is the model's raw statistic. Map scores to a common scale with
    ``aberrant.calibration``. ``readiness`` says whether the model can judge;
    the default suits a model that judges from the first event.

    Subclasses implement ``learn_one`` and ``score_one`` and store every
    constructor parameter under its own name. ``get_params``, ``clone``,
    ``reset`` and ``repr`` follow from that, and ``learn_many`` and
    ``score_many`` loop over the single-event methods unless a model has a
    faster batch path.
    """

    @abc.abstractmethod
    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
                and the values are the corresponding feature values.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute the anomaly score for a single data point.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
                and the values are the corresponding feature values.

        Returns:
            The anomaly score: higher is more anomalous, and ``0.0`` while
            the model is not ready.
        """
        raise NotImplementedError

    @property
    def readiness(self) -> Readiness:
        """Whether the model has learned enough to score; ``0.0`` until then."""
        return Readiness(True)

    def learn_many(self, xs: Iterable[dict[str, float]]) -> None:
        """Learn the events of ``xs`` in order."""
        for x in xs:
            self.learn_one(x)

    def score_many(self, xs: Iterable[dict[str, float]]) -> list[float]:
        """Score the events of ``xs`` against the current state, without learning."""
        return [self.score_one(x) for x in xs]
