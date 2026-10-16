"""Programmatic numeric graph nodes and detached execution observations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from aberrant.base.exceptions import PipelineError
from aberrant.base.protocols import TransformerProtocol
from aberrant.base.readiness import Readiness

INPUT = "$input"


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One transformer applied to one predecessor's complete feature mapping.

    Custom transformers without public readiness must specify
    ``minimum_updates`` explicitly, including zero for stateless components.
    """

    id: str
    transformer: TransformerProtocol
    input: str = INPUT
    minimum_updates: int | None = None


@dataclass(frozen=True, slots=True)
class MergeNode:
    """Concatenate predecessor mappings in ``inputs`` order without collisions."""

    id: str
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.inputs, str):
            raise ValueError("Merge inputs must be a sequence of node IDs")
        object.__setattr__(self, "inputs", tuple(self.inputs))


GraphComponent: TypeAlias = GraphNode | MergeNode
FeatureValues: TypeAlias = tuple[tuple[str, float], ...]


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


Observer: TypeAlias = Callable[[NodeObservation], None]


class GraphNotReadyError(PipelineError):
    """Scoring was requested before a complete learned path was ready."""


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
