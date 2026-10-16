# Branched numeric pipelines

`aberrant.compose.GraphPipeline` applies a directed acyclic graph of online
transformers to each observation and learns exactly one terminal anomaly model.
It is a standalone Python API using the base package dependencies. Applications
own saved designs, bindings, user interfaces, calibration and alarm policies.

```python
from aberrant.compose import GraphNode, GraphPipeline, MergeNode, Select
from aberrant.model.stat import MovingCovariance
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import IncrementalPCA

graph = GraphPipeline(
    [
        GraphNode("selected", Select(["voltage", "current"])),
        GraphNode("scaled", StandardScaler(), "selected"),
        GraphNode("pca", IncrementalPCA(1, n0=5), "scaled"),
        GraphNode("raw", Select(["temperature"])),
        MergeNode("features", ("pca", "raw")),
    ],
    MovingCovariance(window_size=32),
    output="features",
    model_minimum_updates=2,
)

for index in range(20):
    x = {
        "voltage": 230.0 + index % 3,
        "current": 4.0 + index * 0.1,
        "temperature": 20.0 + index * 0.05,
    }
    if graph.readiness.ready:
        score = graph.score_one(x)  # Score against the previously learned state.
    graph.learn_one(x)

assert graph.model_updates == 16
assert graph.readiness.ready
```

Temperature bypasses both scaling and projection. Moving the merge before PCA
would project temperature as well. Each node runs once per pass, even when its
output feeds several branches. Merge inputs concatenate features in the declared
order; duplicate feature names raise an error rather than silently overwriting
values. Use `Rename({"component_0": "left_component"})` on one branch to resolve
collisions. `Select`, `Rename` and `Passthrough` also work with ordinary `|`
pipelines and declarative transformer catalog entries.

## Initialization and learning order

`GraphNode` takes one predecessor ID, defaulting to `INPUT` (`"$input"`).
`MergeNode` takes an ordered tuple of predecessor IDs. Construction rejects
cycles, missing references, disconnected nodes and shared component instances
under different node IDs. Node IDs must not use `"$input"` or `"$model"`.
Topological ties are resolved by node ID, independently of construction order.

Learning uses post-update transformations, as in a linear pipeline. In a graph,
however, a node learns only when all its inputs are ready. A PCA branch therefore
initializes without sending provisional zeros to the detector. The observation
which makes a node ready may reach its successors in that same pass. A raw branch
can remain ready while another branch initializes; the merge waits for both
outputs from the current observation. It never reuses an older branch output.

`model_updates` counts successful terminal updates. Transformer initialization
does not advance that count. `model_minimum_updates` and a model's optional
public readiness must both be satisfied before scoring. Models without public
readiness use the supplied update count; choose it from the model's contract or
catalog metadata. Geometric/harmonic means report retained positive/nonzero
history, rather than counting ignored observations. These descriptions do not
guarantee that an arbitrary candidate satisfies a model's input domain.

`last_learning_outcome` contains the terminal-update flag, cumulative model
updates, per-node update counts, readiness and blocked predecessors. Snapshots
are frozen dataclasses. `score_one` is read-only and raises `GraphNotReadyError`
until the complete path and terminal are ready. Use one serial owner for graph
execution; concurrent calls are not supported.

The graph API does not change existing `Pipeline` or `DetectorConfig` version 1.
Linear pipelines retain their eager learning and provisional transformer
outputs. A `GraphPipeline` already ends in a model and cannot accept another
stage through `|`.

## Statistical interpretation

Online `IncrementalPCA` is uncentered and adapts its subspace. Put a centering
or standardizing transformer upstream when appropriate. Updating a scaler or
projection does not re-express samples, boundaries or model parameters retained
downstream. Applications must choose suitable model/transform combinations and
decide whether to rebuild or recalibrate when representations change. This
graph is an online execution mechanism, not a batch fit/transform estimator.

`MinMaxScaler(clip=True)` explicitly clamps out-of-reference candidates to the
configured range. Its default remains `False`, preserving extrapolation.

## Custom components, failures and observations

Custom nodes satisfy `TransformerProtocol`. Supply `minimum_updates` explicitly
when a transformer has no `readiness: Readiness` property, including zero for a
stateless transform. Inputs and outputs are detached finite mappings; one
component cannot mutate a sibling branch's input. Feature schemas become fixed
after a successful learning pass. Reordered mappings retain their established
feature order.

Implement `learning_transaction(x)` to restore transformer state if any later
component fails. All participating contexts remain open through terminal
learning. Contexts must propagate exceptions and restore state; they must not
suppress failures. Ordinary transactional failure raises `GraphExecutionError`.
An error after touching a nontransactional learner, or a rollback failure,
raises `GraphStateUncertainError` and permanently disables execution of that
graph. Restore a known checkpoint or construct a fresh graph. No atomic rollback
is promised for arbitrary model implementations.

Pass an optional `observer` callable to receive immutable `NodeObservation`
records for calls already executed. Each record includes `node_id`, `phase`,
`inputs`, `output` and `duration_ns`; feature mappings are tuples of key/value
pairs. Terminal events use `"$model"`. Events describe actual calls, including
calls later rolled back by a failing pass. Observer exceptions are isolated and
counted by `observer_error_count`. Observation does not run a component again.
Observers are process-local and excluded from pickled graph state; reattach
one after restoration. The library retains no observation history.

See the executable [graph example](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/graph_pipeline.py)
and the [composition API](../api/compose.md).
