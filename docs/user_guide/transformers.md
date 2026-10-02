# Transformers

Transformers incrementally learn preprocessing state and map one feature
dictionary to another. They implement `learn_one(x)` and `transform_one(x)` and
can precede another transformer or one terminal model in a pipeline.

## Available transforms

| Transformer | Learned state | Output | Important semantics |
| --- | --- | --- | --- |
| `FeatureSchemaGuard` | Configured or first-learned feature schema | Same keys and values in schema order | Rejects invalid values or a changed feature set |
| `MinMaxScaler` | Per-feature running minimum and maximum | Same keys, values mapped to `feature_range` | A value outside the learned extrema can transform outside the requested range; a constant learned feature maps to the lower bound |
| `StandardScaler` | Per-feature count, mean, and population variance accumulator | Same keys, centered values and optionally population-standardized values | A feature with zero learned variance maps to `0.0`; `with_std=False` centers without scaling |
| `RollingRobustScaler` | Per-feature recent window, median, and interquartile range | Same keys, median-centered and IQR-scaled values | Exact linearly interpolated quartiles; zero IQR uses `fallback_scale` to preserve deviations; calibration can be frozen |
| `IncrementalPCA` | Warm-up SVD followed by an incremental uncentered subspace | `component_0` through `component_{n_components - 1}` | Returns zero components until `n0` events; inputs are projected around the origin, not around an internally learned mean |
| `RandomProjection` | One seeded sparse Achlioptas projection matrix | `component_0` through `component_{n_components - 1}` | `learn_one` establishes the schema and matrix once; later calls do not fit distributional parameters |

All built-in transformers reject non-numeric or non-finite values before updating their
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

## Robust recent-window calibration

`RollingRobustScaler(window_size=256, min_samples=1, fallback_scale=1.0)`
retains at most `window_size` observations per feature and computes
`(value - median) / (Q3 - Q1)`. Quartiles interpolate linearly at
`(n - 1) * q`. A zero IQR uses `fallback_scale` instead: a constant reference
of `10.0` with the default fallback maps a candidate of `13.0` to `3.0`.
There is no clipping or normalization to `[0, 1]`.

Missing features do not advance their own windows. `sample_counts` returns a
detached snapshot of retained counts, and `is_ready` becomes true when every
learned feature has `min_samples` observations. An empty scaler is unready;
introducing a new feature can make it unready again. Use `FeatureSchemaGuard`
when the set of features must remain fixed. Transformations use provisional
statistics during warm-up, so a pipeline can accumulate history normally;
exclude that warm-up from evaluation explicitly.

Adaptive calibration follows recent data. Freeze a ready scaler before fitting
a downstream reference model when a fixed coordinate system is required:

```python
from aberrant.transform import RollingRobustScaler

adaptive = RollingRobustScaler(window_size=5, min_samples=5)
frozen = RollingRobustScaler(window_size=5, min_samples=5)
for value in [0, 2, 4, 6, 8]:
    adaptive.learn_one({"x": float(value)})
    frozen.learn_one({"x": float(value)})
frozen.freeze()

for value in [10, 12, 14, 16, 18]:
    adaptive.learn_one({"x": float(value)})
    frozen.learn_one({"x": float(value)})

assert adaptive.transform_one({"x": 18.0}) == {"x": 1.0}
assert frozen.transform_one({"x": 18.0}) == {"x": 3.5}
```

Frozen learning still validates finite input and rejects unseen features.
`unfreeze()` resumes rolling updates from the retained window; `reset()` clears
all calibration and unfreezes while retaining constructor parameters.

Changing scaler coordinates does not re-express the points, tree boundaries,
or parameters already retained by a downstream model. Rebuild or recalibrate
that model under application control when needed. The example compares
representations, not detection accuracy. Pipeline scoring uses prior scaler
state; pipeline learning passes the same event through its post-update
representation, as for the other scalers.

For `F` supplied features, window size `W`, and `D` distinct learned feature
names, adaptive learning costs `O(F W log W)`, transformation costs `O(F)`,
and persistent storage is `O(D W)`. Statistics are cached during learning.
Frozen learning costs `O(F)`. Updates validate and stage the entire event;
non-finite statistics or output are rejected. Pipeline rollback restores
evicted observations and removes newly introduced feature state after a later
stage rejects an event. See the standalone
[frozen/adaptive example](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/rolling_robust_scaler.py).

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
