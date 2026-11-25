"""Standalone programmatic composition of branched online numeric pipelines.

No persistence format, deployment machinery, or visualization dependency is
required. Existing linear :class:`aberrant.base.Pipeline` semantics are unchanged.
"""

from aberrant.compose._graph import GraphPipeline
from aberrant.compose._types import (
    INPUT,
    GraphComponent,
    GraphExecutionError,
    GraphNode,
    GraphNotReadyError,
    GraphStateUncertainError,
    LearningOutcome,
    MergeNode,
    NodeObservation,
    NodeState,
    Observer,
)

__all__ = [
    "INPUT",
    "GraphComponent",
    "GraphExecutionError",
    "GraphNode",
    "GraphNotReadyError",
    "GraphPipeline",
    "GraphStateUncertainError",
    "LearningOutcome",
    "MergeNode",
    "NodeObservation",
    "NodeState",
    "Observer",
]
