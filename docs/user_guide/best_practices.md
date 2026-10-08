# Best Practices

These recommendations follow documented library behavior. They are not a
substitute for detector-specific calibration or an application threat model.

## Make the event contract explicit

- Validate required feature names and units at the ingestion boundary.
- For schema-owning numeric models, keep the feature-key **set** fixed after the
  first successful `learn_one`. Dictionary insertion order can vary because the
  shared schema canonicalizes names.
- For `IncrementalPCA` and `RandomProjection`, pass `keys=` when order must be
  declared before learning; otherwise the first learned dictionary's order is
  retained.
- Do not encode missingness as an undocumented numeric sentinel. Impute or
  reject it under a policy the model was calibrated with.
- Reject non-finite values before they reach static threshold wrappers too.
  Stateful numeric boundaries reject them, but `ThresholdModel` is a rule
  evaluator rather than a general numeric validator.
- Keep feature units stable. A distance, reconstruction error, radius, or
  covariance threshold calibrated in one unit system is not portable to
  another.

## Define score/learn order once

Use one event-processing function for the application and test its call order.
For prequential evaluation and ordinary prospective alerting:

1. validate and construct the event;
2. call `score_one`;
3. record the score and make any threshold decision;
4. update drift monitoring from the chosen scalar;
5. call `learn_one` only if the application's learning policy accepts the
   event.

`score_one` means that the event is not learned. It does not promise that every
incidental implementation detail is immutable: query caches can be refreshed,
and `RandomModel.score_one` deliberately advances its model-local generator.

Pipeline learning uses post-update transforms. If application code performs
manual preprocessing instead, match that ordering deliberately rather than
assuming transform-then-learn is equivalent.

## Treat warm-up as an experiment parameter

- Distinguish application warm-up from a detector's minimum internal readiness.
- Suppress or separately label warm-up scores instead of interpreting a forced
  `0.0` as normal evidence.
- Record whether warm-up data is curated-normal, unlabeled, or contaminated.
- Do not use future labels to decide retrospectively which warm-up events the
  model should have learned.
- Recalibration after reset or model replacement needs its own warm-up policy.

## Calibrate score policy per detector

- Never reuse a numeric threshold merely because two detectors both return
  values in `[0, 1]`. Boundedness is not probability calibration.
- Preserve the raw detector score even when an external threshold produces a
  binary alert; it is needed for ranking metrics and incident analysis.
- For `QuantileThreshold`, decide which scores enter its window. Learning every
  score adapts to persistent anomalies and can raise the threshold.
- Set thresholds from prior calibration data or an online policy available at
  the decision time, not from labels in the evaluated interval.
- Monitor score distributions and alert rates separately. A stable alert rate
  can conceal a moving adaptive threshold.

## Handle time deliberately

- With `time_key=None`, time-aware models use arrival order. Retries, buffering,
  and partition merges can therefore change semantics.
- With an explicit `time_key`, ABERRANT accepts non-decreasing time. Equal
  values are valid; a later smaller value raises `ValueError`.
- Models based on buckets require integer-like timestamps. Do not silently
  round wall-clock values in application code.
- `score_one` previews but does not commit shared model time. A subsequent
  successful `learn_one` advances it.
- Define how late and duplicate events are handled before the model boundary;
  the library does not reorder a stream.

## Separate drift signal from drift response

A drift flag identifies statistical change in the monitored scalar, not its
cause. Before deployment, choose which signal is monitored and specify a
response matrix for data-quality failures, expected seasonal change, model
degradation, and incident bursts.

Do not assume reset is universally correct. Depending on the model and failure
mode, a safer response can be to investigate, recalibrate only the threshold,
warm a replacement model in parallel, or reject an upstream release. See
[Drift Detection](drift.md).

## Reproducibility and concurrency

- Set every exposed seed and record it with the constructor parameters and
  ABERRANT version.
- Compare stochastic models across multiple seeds; one repeat is deterministic
  evidence, not a stability estimate.
- ABERRANT's seeded NumPy and PyTorch initialization paths use model-owned
  generators where implemented. External libraries, user-supplied
  architectures, optimizers, and hardware kernels can introduce additional
  nondeterminism.
- Treat each model instance as single-owner. The package does not promise that
  concurrent `learn_one` and `score_one` calls on one instance are atomic or
  thread-safe.
- `OnlineIsolationForest(n_jobs=-1)` parallelizes work across all reported
  logical CPUs; it does not make concurrent caller access safe.
- When processing partitioned streams, give each partition its own model or
  serialize access under an application-owned ordering policy. Incremental
  model state is generally not mergeable.

## Persistence and upgrades

ABERRANT does not currently define a stable, cross-version model serialization
format. Do not document or rely on generic pickle/joblib persistence as a
package guarantee.

If an application checkpoints model objects anyway:

- own the serializer and storage security policy;
- never load an untrusted pickle-like artifact;
- record the exact Python, ABERRANT, NumPy, SciPy, and optional dependency
  versions;
- test round-trip scores and subsequent learning for every model type;
- retain the source event offset needed to resume without gaps or duplicates;
- keep a path to rebuild state from events when upgrading; the
  [public API compatibility policy](../api/index.md#compatibility-policy) does not
  guarantee cross-version checkpoint compatibility.

## Long running streams

The streaming interface alone does not guarantee constant memory or unlimited
numeric lifetime. The package-wide longevity audit exercises all built-in model
families, transformers, drift detectors, evaluation, batching, and native FAISS.
The bounds below assume fixed feature dimensions and constructor parameters;
they describe retained observations and arrays, excluding exact lifetime counters.

| Implementations | Retained state and conditions |
| --- | --- |
| `ASDIsolationForest`, `HalfSpaceTrees`, `OnlineIsolationForest`, `RandomCutForest`, `StreamRandomHistogramForest`, `XStream` | Configured windows, trees, samples, or sketches. Large `OnlineIsolationForest.learn` batches are processed in bounded chunks, and retained rows own their storage. `XStream(max_feature_cache_size=None)` permits an unlimited feature-name cache; keep a finite cache budget for schema churn. |
| `MondrianIsolationForest` | Default lifetime trees can grow indefinitely. Set `window_size=W` for periodic replacement from the latest W events. Between replacements each tree represents at most `2W - 1` events; replacement has a bounded latency spike. This is periodic rebuilding, not exact per-event sliding-window deletion. |
| `KNN`, `LocalOutlierFactor`, `SDOStream`, `CellNeighborhoodDetector`, `StationaryRegionNeighborDetector` | Configured point windows, observer budgets, cells, and bounded indexes. A custom KNN engine owns its own retention policy. |
| `MStream`, `StreamingLODA`, `StreamingRSHash` | Fixed sketches, projections, histograms, and feature statistics. Novel values do not create a persistent key per value. |
| `MIDAS`, `ISCONNA`, `AnoEdgeL`, `SignedGraphSketchDetector` | Fixed sketches or configured active-graph budgets. Novel node/graph identities do not accumulate without eviction. |
| `GraphGatedOneClassSVM`, `IncrementalOneClassSVMAdaptiveKernel` | Fixed support-vector, feature-map, and neighborhood budgets. |
| `RollingMatrixProfile`, `MultivariateRollingMatrixProfile`, `XLagDAMP`, `SeasonalResidualDetector` | Configured rolling series, lag, season, and residual arrays. DAMP rejects constant subsequences. |
| All ten univariate moving statistics, `MovingCovariance`, `MovingCorrelationCoefficient`, `MovingMahalanobisDistance` | Configured observation windows and feature matrices. Exact rolling order statistics and moments operate on bounded histories. |
| `QuantileThreshold`, `ThresholdModel`, `NullModel`, `RandomModel` | Configured score window, fixed rule configuration, or generator state. |
| `OnlineAutoencoderEnsemble` | Fixed feature-map statistics and network weights; frozen and adaptive phases retain no observation history. |
| `Autoencoder`, bundled feedforward/LSTM architectures | Fixed tensor buffers, network parameters, gradients, and ordinary optimizer state. Supplied architectures, losses, and optimizers can have their own unbounded state or numeric failures. |
| `MinMaxScaler`, `StandardScaler`, `RollingRobustScaler` | State grows with **distinct feature names**. Place `FeatureSchemaGuard` first to enforce a finite schema. Robust windows are bounded per feature, not across an unlimited feature vocabulary. |
| `FeatureSchemaGuard`, `RandomProjection`, `IncrementalPCA` | Fixed schema/projection arrays. PCA retains only its first `n0` events during initialization, then releases them. |
| `ADWIN`, `KSWIN`, `PageHinkley` | Default ADWIN uses O(log N) buckets on stationary streams; `max_window_size` bounds its represented history by discarding oldest whole buckets. KSWIN has a fixed window; Page-Hinkley uses scalar statistics. |
| `Pipeline`, `PrequentialEvaluator`, `BatchStreamer` | Pipeline retains components; evaluation retains at most `metric_window_size` labeled scores; batching retains at most `batch_size` events. Evaluation with `metric_window_size=None` explicitly retains O(N) scores when metrics are enabled. |
| NPZ loading, dataset cache, download and validation | NPZ iteration materializes the complete feature and label arrays. It is a finite benchmark source, not an out-of-core or infinite source. Downloads and hashing use bounded byte chunks; cached artifacts occupy disk. |

For bounded service operation, consume records incrementally, enforce a fixed
schema, use positive integer window/batch sizes, and keep both the input queue
and output sink bounded. `iter_evaluate` and `BatchStreamer` can consume infinite
generators lazily; `evaluate` returns only after its input ends. Close generators
when stopping early. Storing all outputs with `list`, or accepting arrivals faster
than processing, can exhaust application memory independently of model bounds.

Numeric limits remain observable even with bounded data retention:

- Exact Python counters occupy O(log N) bits; cumulative floating-point counts
  eventually lose unit precision. ISCONNA's fixed-width pattern counters saturate
  at the signed 64-bit maximum rather than wrapping negative.
- `StreamingRSHash` normalizes faded occupancy by a lifetime sample count.
  Its absolute scores can approach 1 on a long stationary stream with positive
  decay; calibrate against recent scores rather than assume a stationary scale.
- Shared integer clocks and `MIDAS`/`ISCONNA`/`AnoEdgeL` edge identifiers remain
  exact past `2**53` when supplied as Python/NumPy integers. The numeric
  identifiers in `SignedGraphSketchDetector` and categorical vectors in
  `MStream` still use float64. Passing integers through float-valued preprocessing or
  `PrequentialEvaluator`'s float event conversion loses that exactness. Supply
  integers directly to models when this matters. Floating timestamps cannot
  recover precision already lost by the caller. SDOStream's internal float64
  observer timestamps still lose single-step age precision beyond `2**53` and
  reject values outside float64's finite range before changing model state.
- A reproduced PyTorch Adam limitation is that a float32 step counter stops
  advancing at `2**24` updates. Finite loss and gradients also do not guarantee
  finite internal optimizer moments. The generic wrapper does not change the
  semantics of a caller-supplied optimizer.
- FAISS float32 squared distances and some Euclidean/moment calculations can
  overflow for large finite values. Raw distances, variances, gradients and
  scores still require a representable numeric range. Standardize inputs at
  ordinary magnitudes; cumulative sums in `StandardScaler` also have finite
  floating-point range.

The `tests/test_stream_longevity_*.py` suites cover long window turnover,
retained-object/NumPy storage bounds, released-history weak references, live
`tracemalloc` plateaus, score-only calls, lazy infinite inputs, and accelerated
large counters/timestamps. Strict expected-failure tests preserve reproduced
unresolved numeric limits. These are finite tests plus source-level state bounds,
not a proof of execution for infinitely many events, arbitrary parameter sizes,
custom components, devices, or every optional dependency build.

The audit includes CPU PyTorch and the real native FAISS index. On the tested
Windows installation FAISS required `FAISS_OPT_LEVEL=generic` and importing
`faiss` before the test runner to avoid intermittent native DLL initialization
failures. Native index sizes and FIFO scores were checked; native allocator leak
instrumentation and CUDA testing were unavailable.

Verified on 2026-10-08 with Windows, CPython 3.12.10, NumPy 2.3.5, SciPy 1.16.3,
PyTorch 2.13.0 CPU, and FAISS 1.13.0: the complete unit and integration run
finished with **1,963 passed, 109 passing subtests, 5 strict expected failures,
and 1 CUDA skip**. The four longevity suites contain 240 cases. Additional
focused suites cover rejected ensemble events, timestamp representation, and
integer-capacity contracts. The five expected
failures reproduce FAISS/SDOStream/LOF extreme-value distance errors and the two
external Adam limitations above; they are unresolved constraints. Ruff lint and
formatting, mypy across 95 source files, wheel/sdist builds, and an isolated
built-wheel import/bounded-mode smoke check also passed. Integration used a
validated local SHUTTLE artifact; no dataset downloads were needed.

## Deployment checklist

1. Pin the feature schema, units, event-time policy, model constructor, and
   package version.
2. Test first-event, warm-up-boundary, window-eviction, timestamp-tie,
   non-finite-input, and reset behavior.
3. Run a chronological canary stream and compare the complete score sequence,
   not only an aggregate metric.
4. Measure scoring/update latency and state growth at production dimensions.
5. Log model readiness, raw score, threshold value, alert decision, event time,
   and model version without logging sensitive features unnecessarily.
6. Define alert suppression, incident grouping, and operator feedback paths.
7. Keep a rebuild path from trusted events because in-memory state is not a
   durable interchange format.
