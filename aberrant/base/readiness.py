"""Optional, detached readiness descriptions for streaming components."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Readiness:
    """Whether a component has a usable learned reference, not score validity.

    ``observed`` and ``required`` describe the same history unit. Readiness does
    not guarantee that every possible candidate satisfies the input domain.
    """

    ready: bool
    observed: int = 0
    required: int = 0
    reason: str | None = None


@runtime_checkable
class ReadinessProtocol(Protocol):
    """Optional capability; existing transformer/model protocols are unchanged."""

    @property
    def readiness(self) -> Readiness:
        """Return a detached description without changing learned state."""
        ...
