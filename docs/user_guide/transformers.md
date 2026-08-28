# Transformers

Transformers incrementally learn preprocessing state and map one feature
dictionary to another. They implement `learn_one(x)` and `transform_one(x)` and
can precede another transformer or one terminal model in a pipeline.

## Available transforms

| Transformer | Learned state | Output | Important semantics |
| --- | --- | --- | --- |
| `MinMaxScaler` | Per-feature running minimum and maximum | Same keys, values mapped to `feature_range` | A value outside the learned extrema can transform outside the requested range; a constant learned feature maps to the lower bound |
| `StandardScaler` | Per-feature count, mean, and population variance accumulator | Same keys, centered values and optionally population-standardized values | A feature with zero learned variance maps to `0.0`; `with_std=False` centers without scaling |
| `IncrementalPCA` | Warm-up SVD followed by an incremental uncentered subspace | `component_0` through `component_{n_components - 1}` | Returns zero components until `n0` events; inputs are projected around the origin, not around an internally learned mean |
| `RandomProjection` | One seeded sparse Achlioptas projection matrix | `component_0` through `component_{n_components - 1}` | `learn_one` establishes the schema and matrix once; later calls do not fit distributional parameters |

All four reject non-numeric or non-finite values before updating their
persistent state.

## Scaling one stream

The scalers maintain each feature independently. They do not impose a global
fixed key set, but `transform_one` rejects any feature that has not previously
been learned. A downstream schema-owning model normally makes a fixed key set
necessary for the complete pipeline.

```python
from aberrant.transform.preprocessing import StandardScaler

scaler = StandardScaler()
for event in [
    {"latency": 10.0, "payload": 100.0},
    {"latency": 12.0, "payload": 120.0},
    {"latency": 8.0, "payload": 80.0},
]:
    scaler.learn_one(event)

transformed = scaler.transform_one({"latency": 13.0, "payload": 90.0})
print(transformed)
```

`transform_one` does not update the learned moments. If the candidate should
affect scaling, call `learn_one` first; that is exactly what
`Pipeline.learn_one` does.

## Projection schemas

`IncrementalPCA` and `RandomProjection` use the exact order supplied through
`keys=`. Without `keys=`, they preserve the first learned dictionary's insertion
order and reject missing or additional keys afterward. Set `keys=` when schema
order must be explicit before the first event.

```python
from aberrant.transform.projection import IncrementalPCA

pca = IncrementalPCA(
    n_components=2,
    n0=3,
    keys=["x", "y", "z"],
)

for event in [
    {"x": 1.0, "y": 0.0, "z": 1.0},
    {"x": 0.0, "y": 1.0, "z": 1.0},
    {"x": 1.0, "y": 1.0, "z": 0.0},
]:
    pca.learn_one(event)

projected = pca.transform_one({"x": 0.5, "y": 0.25, "z": 1.0})
print(projected)
```

`IncrementalPCA` is uncentered. For conventional mean-centered PCA behavior,
put a `StandardScaler` or another centering transform before it. The scaler's
online state then determines the coordinate system seen by PCA.

## Compose a complete detector

This example is standalone and uses no optional dependency:

```python
from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

detector = StandardScaler() | OnlineIsolationForest(
    num_trees=10,
    window_size=32,
    seed=5,
)

events = [
    {"x": 0.0, "y": 0.1},
    {"x": 0.2, "y": -0.1},
    {"x": -0.1, "y": 0.0},
    {"x": 4.0, "y": 4.0},
]

for index, event in enumerate(events):
    score = detector.score_one(event) if index > 0 else 0.0
    detector.learn_one(event)
    print(f"event={event}, score={score:.3f}")
```

The explicit first-event guard is required because `StandardScaler` cannot
transform a feature it has never learned. In a longer application, use a clear
warm-up phase rather than treating the first event as evaluated data.

See [Pipelines](pipelines.md) for update order, valid endpoints, and custom
structural components, or [Transform API](../api/transform.md) for exact
signatures.
