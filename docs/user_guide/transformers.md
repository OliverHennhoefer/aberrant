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
| `IncrementalPCA` | Warm-up SVD followed by an incremental uncentered subspace | `component_0` through `component_{n_components - 1}` | Returns zero components until `warm_up` events; inputs are projected around the origin, not around an internally learned mean |
| `RandomProjection` | One seeded sparse Achlioptas projection matrix | `component_0` through `component_{n_components - 1}` | `learn_one` establishes the schema and matrix once; later calls do not fit distributional parameters |
| `Lags` | Per-feature last values up to the largest lag | Every input feature plus `{feature}_lag{k}` | Before `k` earlier values, a lag repeats the oldest one |
| `Difference` | Per-feature last `lag` values | Same keys, change over `lag` events (or percent) | 0 until `lag` earlier values; a percent change from 0 is rejected |
| `ExponentialSmoother` | Per-feature smoothed level | Same keys, exponentially weighted mean | Starts from the first value; ready after `span` values |
| `Despike` | Per-feature recent window | Same keys, a spike replaced by the window median | Hampel rule on `threshold` scaled MADs; a flat window replaces nothing |
| `RollingStatistic` | Per-feature recent window | Same keys, a window statistic | Location, spread, quantiles, slope, z-scores, rank, RMS or shape |
| `Deseasonalize` | Per-feature level and spread per position of the cycle | Same keys, the value less its position's level | Levels average over `memory` cycles; one odd value moves them at most four spreads |

All built-in transformers reject non-numeric or non-finite values before updating their
persistent state.

## Scaling one stream

The scalers maintain each feature independently. They do not impose a global
fixed key set, but `transform_one` rejects any feature that has not previously
been learned. A downstream schema-owning model normally makes a fixed key set
necessary for the complete pipeline.

```python
from aberrant.transform import StandardScaler

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

`RollingRobustScaler(window_size=256, warm_up=1, fallback_scale=1.0)`
retains at most `window_size` observations per feature and computes
`(value - median) / (Q3 - Q1)`. Quartiles interpolate linearly at
`(n - 1) * q`. A zero IQR uses `fallback_scale` instead: a constant reference
of `10.0` with the default fallback maps a candidate of `13.0` to `3.0`.
There is no clipping or normalization to `[0, 1]`.

Missing features do not advance their own windows. `sample_counts` returns a
detached snapshot of retained counts, and `readiness.ready` becomes true when every
learned feature has `warm_up` observations. An empty scaler is unready;
introducing a new feature can make it unready again. Use `FeatureSchemaGuard`
when the set of features must remain fixed. Transformations use provisional
statistics during warm-up, so a pipeline can accumulate history normally;
exclude that warm-up from evaluation explicitly.

Adaptive calibration follows recent data. Freeze a ready scaler before fitting
a downstream reference model when a fixed coordinate system is required:

```python
from aberrant.transform import RollingRobustScaler

adaptive = RollingRobustScaler(window_size=5, warm_up=5)
frozen = RollingRobustScaler(window_size=5, warm_up=5)
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

## Sequences

Point detectors score each event as if it had no past. On a series that
follows its past, they score the level unless a step before them carries
time. The sequence transformers do that, one feature at a time: `Lags` adds
earlier values, `Difference` the change, `ExponentialSmoother` and
`RollingStatistic` a recent summary, `Despike` removes isolated spikes and
`Deseasonalize` a repeating cycle. `features=` limits them to some features
and passes the others through.

```python
from aberrant.model.forest import OnlineIsolationForest
from aberrant.transform import Difference, Lags

detector = Difference(features=["load"]) | Lags(lags=[1, 2], features=["load"])
detector = detector | OnlineIsolationForest(n_trees=10, window_size=64, seed=5)

for t in range(100):
    event = {"load": float(t % 10), "temperature": 20.0 + t % 3}
    score = detector.score_one(event)
    detector.learn_one(event)
```

Each transformer computes an event's output from the history before it. In a
pipeline, `learn_one` hands the next stage the output of the event it just
learned, and scoring sees the same output for the next event, so a model
learns and scores the same features. A transformer is ready once each feature
has seen enough values for a final output, such as the largest lag plus one.

An application that keeps its own state, one per feature, uses the same
computation without the transformer holding any: `initial_state()` gives a
feature's empty state, `step(state, value)` returns the next state and the
value's outputs, and `valid_state(state)` checks a restored one. A state is a
tuple of numbers and arrays of doubles that `step` never changes, so a caller
can stage the states of an event and keep or drop them as one. `lead` is how many earlier values a feature
needs before its output is final.

```python
from aberrant.transform import RollingStatistic

slope = RollingStatistic(window_size=3, statistic="slope")
state = slope.initial_state()
for value in [1.0, 2.0, 4.0]:
    state, (output,) = slope.step(state, value)
assert output == 1.5 and slope.valid_state(state)
```

## Projection schemas

`IncrementalPCA` and `RandomProjection` use the exact order supplied through
`features=`. Without `features=`, they preserve the first learned dictionary's insertion
order and reject missing or additional fields afterward. Set `features=` when schema
order must be explicit before the first event.

```python
from aberrant.transform import IncrementalPCA

pca = IncrementalPCA(
    n_components=2,
    warm_up=3,
    features=["x", "y", "z"],
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
from aberrant.model.forest import OnlineIsolationForest
from aberrant.transform import StandardScaler

detector = StandardScaler() | OnlineIsolationForest(
    n_trees=10,
    window_size=64,
    seed=5,
)

events = [
    {"x": 0.0, "y": 0.1},
    {"x": 0.2, "y": -0.1},
    {"x": -0.1, "y": 0.0},
    {"x": 4.0, "y": 4.0},
]

for event in events:
    score = detector.score_one(event)
    detector.learn_one(event)
    print(f"event={event}, score={score:.3f}")
```

Before its first learned event, a scaler maps every feature to a neutral value
(`0.0` for `StandardScaler` and `RollingRobustScaler`, the lower bound of
`feature_range` for `MinMaxScaler`), so the loop can score from the first
event. Once learning has started, a feature the scaler has not seen still
raises `ValueError`. Early scores reflect almost no learned state, so exclude a
warm-up phase from evaluation and alerting rather than treating the first
events as evaluated data.

See [Pipelines](pipelines.md) for update order, valid endpoints, and custom
structural components, or [Transform API](../api/transform.md) for exact
signatures.
