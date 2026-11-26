"""Programmatic numeric graph nodes and detached execution observations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from aberrant.base.protocols import TransformerProtocol
from aberrant.base.readiness import Readiness
from aberrant.exceptions import PipelineError

INPUT = "$input"
"""The node ID of the event a graph receives."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One transformer applied to one predecessor's complete feature mapping.

    The node passes features on once the transformer's ``readiness`` is ready.
    """

    id: str
    transformer: TransformerProtocol
    input: str = INPUT


@dataclass(frozen=True, slots=True)
class MergeNode:
    """Concatenate predecessor mappings in ``inputs`` order without collisions."""

    id: str
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.inputs, str):
            raise ValueError("Merge inputs must be a sequence of node IDs")
        object.__setattr__(self, "inputs", tuple(self.inputs))


type GraphComponent = GraphNode | MergeNode
"""A node of a `GraphPipeline`."""
type FeatureValues = tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class NodeState:
    node_id: str
    updates: int
    readiness: Readiness
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LearningOutcome:
    model_updated: bool
    model_updates: int
    nodes: tuple[NodeState, ...]
    readiness: Readiness


@dataclass(frozen=True, slots=True)
class NodeObservation:
    """Detached values from a call already executed; observation never replays it."""

    node_id: str
    phase: Literal["learn", "transform", "score", "merge"]
    inputs: FeatureValues
    output: FeatureValues | float | None
    duration_ns: int


type Observer = Callable[[NodeObservation], None]
"""A callback that receives each node's observation after it ran."""


class GraphExecutionError(PipelineError):
    """A graph operation failed with transformer rollback completed."""

    def __init__(self, node_id: str, phase: str, message: str) -> None:
        self.node_id = node_id
        self.phase = phase
        self.detail = message
        super().__init__(f"Graph node {node_id!r} failed during {phase}: {message}")

    def __reduce__(self) -> tuple[object, tuple[str, str, str]]:
        return type(self), (self.node_id, self.phase, self.detail)


class GraphStateUncertainError(GraphExecutionError):
    """A nontransactional learner may have mutated before an unsuccessful pass.

    This graph permanently refuses further execution. Restore a known snapshot
    or construct a fresh graph; retrying the same object cannot repair its state.
    """
