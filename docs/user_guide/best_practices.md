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
- expect pre-1.0 upgrades to require rebuilding state from events.

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
