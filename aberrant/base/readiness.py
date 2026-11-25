"""How much a streaming component has learned, and whether that is enough."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReadinessUnit(StrEnum):
    """What a component counts while it learns its reference."""

    EVENTS = "events"
    BUCKETS = "buckets"
    OBSERVERS = "observers"
    ACTIVE_GRAPHS = "active_graphs"


@dataclass(frozen=True, slots=True)
class Readiness:
    """Whether a component has learned enough to do its job.

    A model that is not ready scores ``0.0``, a transformer that is not ready
    does not transform, and a forecaster that is not ready returns ``None``.

    Args:
        ready: Whether the component does its job now.
        observed: Learned history, counted in ``unit``.
        required: History the component needs, or ``None`` when that depends
            on the data, for example on how many covariates events carry.
        unit: What ``observed`` and ``required`` count.
        reason: Why the component is not ready, when that is not a count.
    """

    ready: bool
    observed: int = 0
    required: int | None = 0
    unit: ReadinessUnit = ReadinessUnit.EVENTS
    reason: str | None = None

    @classmethod
    def after(
        cls,
        observed: int,
        required: int,
        unit: ReadinessUnit = ReadinessUnit.EVENTS,
    ) -> Readiness:
        """Return the readiness of a component that needs ``required`` units."""
        return cls(observed >= required, observed, required, unit)
