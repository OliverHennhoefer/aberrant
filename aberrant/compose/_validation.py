"""Construction-time graph checks and finite, detached feature boundaries."""

from collections.abc import Sequence
from heapq import heappop, heappush

from aberrant.base.exceptions import ConfigurationError
from aberrant.base.protocols import ModelProtocol, TransformerProtocol
from aberrant.base.readiness import ReadinessProtocol
from aberrant.compose._types import INPUT, GraphComponent, GraphNode, MergeNode
from aberrant.utils.validation import coerce_finite_number, coerce_integer_count


def predecessors(node: GraphComponent) -> tuple[str, ...]:
    return (node.input,) if isinstance(node, GraphNode) else tuple(node.inputs)


def ordered_nodes(
    nodes: Sequence[GraphComponent], model: ModelProtocol, output: str
) -> tuple[GraphComponent, ...]:
    """Reject ambiguous ownership and return a deterministic topological order."""
    if not isinstance(model, ModelProtocol) or isinstance(model, TransformerProtocol):
        raise ConfigurationError("The graph terminal must be an unambiguous model")
    by_id = _index_nodes(nodes, model)
    if output != INPUT and output not in by_id:
        raise ConfigurationError(f"Unknown graph output {output!r}")
    children, degrees = _dependencies(nodes, by_id)
    ordered = _topological_order(by_id, children, degrees)
    _require_connected(by_id, output)
    return ordered


def _index_nodes(
    nodes: Sequence[GraphComponent], model: ModelProtocol
) -> dict[str, GraphComponent]:
    by_id: dict[str, GraphComponent] = {}
    owners = {id(model)}
    for node in nodes:
        if not isinstance(node, GraphNode | MergeNode):
            raise ConfigurationError(
                "nodes must contain GraphNode or MergeNode objects"
            )
        if not isinstance(node.id, str) or not node.id or node.id in (INPUT, "$model"):
            raise ConfigurationError(
                "Graph node IDs must be non-empty and not $input or $model"
            )
        if node.id in by_id:
            raise ConfigurationError(f"Duplicate graph node ID: {node.id!r}")
        by_id[node.id] = node
        if isinstance(node, GraphNode):
            _validate_transformer(node, owners)
        sources = predecessors(node)
        if not sources or any(not isinstance(source, str) for source in sources):
            raise ConfigurationError("Every graph node requires named inputs")
        if len(sources) != len(set(sources)):
            raise ConfigurationError("Merge inputs cannot repeat a predecessor")
    return by_id


def _dependencies(
    nodes: Sequence[GraphComponent], by_id: dict[str, GraphComponent]
) -> tuple[dict[str, list[str]], dict[str, int]]:
    children: dict[str, list[str]] = {key: [] for key in by_id}
    degrees: dict[str, int] = {}
    for node in nodes:
        degrees[node.id] = 0
        for source in predecessors(node):
            if source == INPUT:
                continue
            if source not in by_id:
                raise ConfigurationError(f"Unknown predecessor {source!r}")
            children[source].append(node.id)
            degrees[node.id] += 1
    return children, degrees


def _topological_order(
    by_id: dict[str, GraphComponent],
    children: dict[str, list[str]],
    degrees: dict[str, int],
) -> tuple[GraphComponent, ...]:
    queue: list[str] = []
    for key, degree in degrees.items():
        if degree == 0:
            heappush(queue, key)
    ordered = []
    while queue:
        key = heappop(queue)
        ordered.append(by_id[key])
        for child in children[key]:
            degrees[child] -= 1
            if degrees[child] == 0:
                heappush(queue, child)
    if len(ordered) != len(by_id):
        raise ConfigurationError("Graph contains a cycle")
    return tuple(ordered)


def _require_connected(by_id: dict[str, GraphComponent], output: str) -> None:
    reachable: set[str] = set()
    pending = [output]
    while pending:
        key = pending.pop()
        if key != INPUT and key not in reachable:
            reachable.add(key)
            pending.extend(predecessors(by_id[key]))
    if reachable != set(by_id):
        raise ConfigurationError("Every node must contribute to the graph output")


def _validate_transformer(node: GraphNode, owners: set[int]) -> None:
    if not isinstance(node.transformer, TransformerProtocol) or isinstance(
        node.transformer, ModelProtocol
    ):
        raise ConfigurationError("GraphNode requires an unambiguous transformer")
    if id(node.transformer) in owners:
        raise ConfigurationError("Each component instance must belong to one node")
    owners.add(id(node.transformer))
    if node.minimum_updates is not None:
        coerce_integer_count(node.minimum_updates, label="minimum_updates", minimum=0)
    elif not isinstance(node.transformer, ReadinessProtocol):
        raise ConfigurationError(
            f"Custom transformer {node.id!r} needs explicit minimum_updates or readiness"
        )


def checked_features(
    value: object, expected: tuple[str, ...] | None = None
) -> dict[str, float]:
    """Own each branch mapping, preserving an already established feature order."""
    if not isinstance(value, dict) or not value:
        raise ValueError("Graph features must be a non-empty dict[str, float]")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError("Graph feature names must be non-empty strings")
    if expected is not None and set(value) != set(expected):
        raise ValueError(f"Feature schema changed; expected {expected!r}")
    return {
        key: coerce_finite_number(value[key], label=f"Feature {key!r}")
        for key in (expected or tuple(value))
    }
