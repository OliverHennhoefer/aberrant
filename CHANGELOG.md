# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project aims to follow
Semantic Versioning.

## [Unreleased]

### Added

- `aberrant.calibration`: online calibration of score streams with delayed
  feedback. `ScoreCalibrator` issues a threshold with `predict(key)` and learns
  each score with `update(key, score)` once it is known, so forecast leads and
  late labels work. Fifteen methods are registered in `CALIBRATORS` with their
  parameters, family and coverage (exchangeable, long run or none): rolling
  split conformal, ACI,
  DtACI, AgACI, quantile tracking, conformal PID, NexCP, ECI, SF-OGD, SAOCP,
  Bayesian belief CP, BOCPD with and without ACI, changepoint-weighted CP and
  discounted Bayes. They are ports of the Timestral Uncertainty Arena's
  methods and reproduce its thresholds on fixtures generated from it.
- `OnlineECOD` and `OnlineCOPOD` in `aberrant.model.stat`: truly online
  ECOD and COPOD that score each sample against per-feature reference
  distributions before learning it. A bounded t-digest per feature (with an
  optional half-life), an exact sliding window, or an ESOD-style hybrid of
  the two holds the reference; an optional SPOT-style generalized Pareto
  tail keeps scoring values past the observed range; `explain_one` returns
  each feature's contribution. Window scores equal the batch methods, and
  `aggregation="pyod"` reproduces PyOD's ECOD and COPOD exactly.
- The README footer shows the BMWK funding acknowledgement, as in nonconform.
- A weekly OpenSSF Scorecard analysis that publishes its results to
  scorecard.dev and code scanning, and a Scorecard badge in the README.
- A README figure of three sensor streams scored event by event, rendered in
  light and dark variants by the reproducible `docs/scripts/stream_figure.py`.
- `docs/img/social-preview.png` for the repository's social preview card.
- `CITATION.cff`, so GitHub and Zenodo can offer citation metadata.
- Tagged releases now create a GitHub release with the matching changelog
  section and the built distributions attached.
- CI uploads branch coverage to Codecov through OIDC, without a stored token,
  and the README shows a coverage badge. Coverage status checks are
  informational and never block a pull request.
- Three executed example notebooks with Open in Colab badges: a prequential
  quickstart on Shuttle, MIDAS on a graph stream, and quantile-threshold
  alerts with Page-Hinkley drift detection.
- A "Coming from river or PySAD" guide that maps their loops and components to
  aberrant.
- A README figure in which `XLagDAMP` finds the one anomalous heartbeat in a
  noisy 30,000-sample ECG from the UCR Time Series Anomaly Archive, rendered
  by the reproducible `docs/scripts/discord_figure.py`.
- `benchmarks/compare_peers.py` measures score-then-learn throughput and
  ROC-AUC of the hot detectors against river and PySAD with matched parameters.
  It is a manual benchmark and not part of CI, and it needs PySAD 0.7 or later,
  whose `fit_score_partial` scores an event before learning it.

### Changed

- Workflow tokens follow least privilege: write permissions are declared only
  on the jobs that need them, and the GitHub release is created by the PyPI
  publishing job after a successful upload.
- `SECURITY.md` links the private advisory form and states a 90-day
  coordinated disclosure timeline.
- Shortened the README: one-line feature bullets, a production link to
  Aberration and the integration guide at the top, and the integration,
  dataset, related-project and contributing sections left to the docs.
- Rewrote the README opening around real-time stream detection, with a
  dataset-backed quickstart, a comparison with river and PySAD, and links to
  related projects.
- Updated package metadata: a new summary, keywords, development-status,
  audience, topic and typing classifiers, and Python 3.13 and 3.14 support.
- Extended the CI test matrix to CPython 3.13 and 3.14.
- The docs site and issue templates use the lowercase `aberrant` name and the
  "real-time anomaly detection for data streams" tagline.
- `StandardScaler`, `MinMaxScaler` and `RollingRobustScaler` return neutral
  values (`0.0`, or the lower bound of `feature_range` for `MinMaxScaler`)
  before their first learned event instead of raising `ValueError`, so a
  score-before-learn loop can start on the first event. A feature that was
  not seen after learning has started still raises `ValueError`.
- The README quickstart scores each event before learning it; the migration
  guide's aberrant loop uses the same score-then-learn order.
- The README quickstart runs `OnlineIsolationForest` on raw Shuttle features
  instead of two scalers in front of `HalfSpaceTrees`, since its splits do not
  depend on feature scale. The pipeline example notes that its
  `StandardScaler` is there for `IncrementalPCA`.
- Feature validation checks a plain `dict` sample with the committed feature
  names and finite `float` or `numpy.float64` values in one pass. Any other
  sample takes the complete validation path and is accepted or rejected as
  before. Numeric coercion no longer calls NumPy per value. Half-Space Trees
  scores and learns about 45% more events per second.
- `StreamingLODA` finds the bins of all projections with one vectorized
  comparison instead of one `np.searchsorted` call per projection, fixes the
  bin widths when warm-up ends and fits the warm-up histograms in one pass.
  Score sequences are unchanged, and it scores and learns about 45 times more
  events per second. The warm-up histograms are fitted in row batches of at
  most about a million edge comparisons, so large warm-ups with many bins need
  no more memory than before.
- `HalfSpaceTrees` walks its complete trees with flat loops instead of
  recursion, pivots the window masses over one list of all nodes and scores
  from the validated values without building an array. `HSTNode` and `HSTLeaf`
  are slotted dataclasses. Score sequences are unchanged, and it scores and
  learns about 2.5 times more events per second.
- `OnlineIsolationForest.learn_one` and `score_one` route a single row through
  each tree with plain comparisons instead of partitioning a one-row batch at
  every node, and they no longer start worker threads, so `n_jobs` only
  applies to `learn_batch` and `score_batch`. Branches split on a feature
  index instead of a one-hot projection vector, derive their bounding box from
  their subtree when they collapse instead of updating it on every learn and
  unlearn, and the tree depth limit is computed only when a leaf may split.
  Score sequences are unchanged, and the default forest scores and learns
  about 40 times more events per second.
- `LocalOutlierFactor` keeps, for every window point, a sorted list of at
  most `2 * k` nearest window points while learning, so k-distances and the
  neighborhoods of the query's neighbors are read from those lists instead of
  measuring the window again for each of them. Ties keep the order of a full
  sort of the window, and distances are summed in NumPy's order over a
  feature-major window. Score sequences are unchanged, and with the defaults it
  scores and learns about 200 times more events per second. Learning now
  measures the new point against the window.
- `XStream` descends all chains at once with array operations instead of
  walking every chain level by level, reuses the chain paths when it learns
  the point it has just scored, and projects a sample with one weighted
  `np.bincount`. Window swaps exchange the current and reference sketches
  instead of copying them, and counters use the smallest integer type that
  holds a full window, `int16` for the default window of 256. Score sequences
  are unchanged, and with the defaults it scores and learns about 80 times
  more events per second. A sample whose bins overflow still raises
  `OverflowError`, now before any counter changes.
- `StreamRandomHistogramForest` plans an insertion into all trees at once:
  the moments and selected attributes of every branch on the point's path are
  updated together with array operations, and only the node that must be
  rebuilt is folded again, value by value. Leaves keep no moments, since
  inserting into a leaf rebuilds it from its points. Learning the point that
  was just scored applies the insertions that scoring planned instead of
  planning them again; pickles leave that preview out. Score sequences are
  unchanged, and with the defaults it scores and learns about 100 times more
  events per second.
- `RandomCutForest` walks a point down all trees at once: each level samples
  the cuts of every tree with array operations, while every tree still draws
  from its own generator. Scoring walks the query as its insertion would
  instead of inserting it, removing it and copying the generator state, and
  learning the point that was just scored reuses those walks, walking again
  only below the branches that removing the oldest point changed. Boxes are
  kept as the lower corner over the negated upper corner, so one array
  operation merges two, and removals stop updating boxes at the first one
  that stays the same. The trees share each stored point instead of copying
  it, and pickles leave the planned walks out. Score sequences, tree
  structure and generator states are unchanged, and with the defaults it
  scores and learns 4 to 5 times more events per second. Scoring a point
  whose distance to a stored point overflows float64 no longer raises from
  inside the trial insertion, since the walk never splits a leaf.
- `MondrianIsolationForest` inserts a point into all trees at once. Boxes
  nest along a walk, so only the blocks near the leaf that the point lies
  outside sample a split time; one array operation computes their extension
  rates for every tree, and the blocks above a change only count the point and
  grow the bounds it lies beyond. Scoring keeps the walks of every tree, so
  learning the point that was just scored reuses them; pickles leave them out.
  Window rebuilds insert each window point into all trees at once, and a tree
  that fails during a rebuild leaves the trees after it as a tree-by-tree
  rebuild would. Score sequences, tree structure and generator states are
  unchanged, and with the defaults it scores and learns about 5 times more
  events per second. `MondrianLeaf.from_point` now stores float64 bounds for
  any numeric point.
- `IncrementalOneClassSVMAdaptiveKernel` computes the kernels between a point
  and all support vectors, and the distances behind each gamma estimate, with
  array operations instead of one pair at a time. It standardizes with the
  running statistics once per call instead of once per vector, and restacks
  the support vectors only when they change. Score sequences, support vectors,
  gamma and generator states are unchanged, and with the defaults it scores
  and learns about 18 to 20 times more events per second.
- `ASDIsolationForest` scores a point by walking all trees at once over flat
  node arrays, which each retraining fills while it builds the trees. Trees
  are built from one row per feature, so every node takes the minima and
  maxima of all features with two reductions, and draws its split feature
  with `Generator.integers` instead of the equivalent but slower
  `Generator.choice`. Score sequences, trees and generator states are
  unchanged. With the defaults it scores and learns about 2 times more events
  per second, and about 2.5 to 3 times more with PySAD's window of 2,048
  events.
- `StreamingRSHash` hashes a point into all of its tables with integer array
  operations instead of a Python loop over every table and subspace feature,
  and reads and increments its counters with one array operation each.
  Learning a point right after scoring it reuses the buckets that scoring
  computed; pickles leave them out. Score sequences, counts and generator
  states are unchanged, and with the defaults it scores and learns about 9
  times more events per second.
- `ISCONNA` updates the one sketch cell per row that an edge touches as plain
  Python numbers instead of copying the cells into small arrays and masking
  them, and takes the minima and maxima of its scores without NumPy. Score
  sequences and sketches are unchanged, and with the defaults it scores and
  learns about 3.5 to 4 times more events per second.
- `MovingCovariance`, `MovingCorrelationCoefficient` and
  `MovingMahalanobisDistance` read their windows of Python floats into arrays
  with `np.fromiter` instead of `np.asarray` and `np.array`, which first walk
  the whole window to find its shape and type. Scores are unchanged. With a
  window of 1,000 events, `MovingCovariance` scores and learns about 2 times
  more events per second, `MovingMahalanobisDistance` about 1.6 times and
  `MovingCorrelationCoefficient` about 1.1 times.
- `MStream` reads and increments the counts of all numeric and categorical
  features with one array operation each instead of a Python loop, and skips
  the integer check when no feature is categorical. The integer check compares
  the distance to the nearest integer directly instead of calling
  `np.allclose`, with the same threshold. Scores and counts are unchanged, and
  with the defaults it scores and learns about 1.7 times more events per
  second than 1.4.0.
- `RollingMatrixProfile` and `MultivariateRollingMatrixProfile` compute the
  screening convolution with SciPy's real FFTs directly, which gives the bits
  of `scipy.signal.fftconvolve` without its general shape handling, and read
  the stored window statistics into an array with `np.fromiter`. Scores,
  matches and channel distances are unchanged. With a subsequence length of
  32 and the default window, both score and learn about 1.4 times more events
  per second.
- `XLagDAMP` learning the value it just scored reuses the backward search that
  scoring ran from the same history, instead of searching again; pickles
  leave that search out. Scores, the best-so-far score and the history are
  unchanged. With a subsequence length of 32 and the default lag, scoring
  then learning each value runs about 1.9 times more events per second.
- Removed state that was written but never read: an accumulator and a
  counter in `ADWIN`, an attribute `MovingInterquartileRange.score_one` set
  on the model while scoring, and the unused `NumericScalar` alias in
  `aberrant.utils.validation`. Scores and drift decisions are unchanged.

## [1.4.0] - 2026-10-30

### Added

- Reusable `DetectorConfig.resolve()` and `resolve_detector_config()` plans with
  canonical configuration, capabilities, fingerprints and fresh detector builds.
- Immutable `ConfigurationIssue` records on configuration errors, including
  complete nested parameter locations and independent field diagnostics.

### Changed

- Learning transactions are a learner capability shared by transformers and
  terminal models in both linear and graph pipelines.
- Constructor parameter descriptors now share coercion, defaults, schema output
  and error locations. Detector construction and evaluator initialization resolve
  parameters once per component occurrence, without a process-wide cache.

- Require Python 3.11 or newer to use patched filelock releases; continuously
  tested Python versions are now 3.11 and 3.12 on Windows and Linux.

### Fixed

- Require filelock 4.2.4 or newer for dataset cache locking and update the locked
  certifi, msgpack, setuptools, and virtualenv dependencies to patched releases.
- Align Ruff's pre-commit hooks with version 0.17.2 and preserve the existing
  positional constructor APIs when enabling its new lint rules.

## [1.3.0] - 2026-10-16

### Added

- Standalone programmatic `GraphPipeline` for feature branching/merging with
  staged post-update readiness, one update per node, detached branch values,
  transformer rollback through terminal learning, explicit uncertain-state
  failures, and optional detached execution observations.
- Public `Select`, `Rename`, and `Passthrough` transformers and catalog entries;
  catalog output-name resolution; optional component readiness descriptions.
- `MinMaxScaler(clip=False)` preserves extrapolation by default and offers
  explicit clipping for bounded-input applications.

Existing linear `Pipeline` behaviour and `DetectorConfig` version 1 are retained.

### Fixed

- Use the symmetric eigensolver for online PCA covariance updates, avoiding
  complex round-off bases and implicit imaginary-value truncation.

## [1.2.0] - 2026-10-08

### Added

- Package-wide streaming longevity regressions covering bounded retention,
  eviction, lazy infinite inputs, deep optimizer/graph storage, and accelerated
  counter/timestamp aging; documented remaining numeric and retention limits.
- Optional bounded-history modes for `MondrianIsolationForest(window_size=...)`
  and `ADWIN(max_window_size=...)`, preserving default lifetime semantics.

- Core-only `SeasonalResidualDetector` with additive Holt-Winters forecasts,
  causal raw/scaled residual scores and explanations, two-cycle initialization,
  bounded seasonal state, reset/readiness, catalog construction, and a seeded
  fixed-cadence example.
- `RollingRobustScaler` with bounded per-feature median/IQR calibration,
  readiness diagnostics, deviation-preserving zero-IQR fallback, explicit
  freeze/unfreeze controls, transactional pipeline rollback, catalog support,
  and a deterministic comparison of frozen and adaptive calibration.
- `PrequentialEvaluator` with immutable per-event records and cumulative results,
  score-before-learn ordering, explicit warm-up/readiness and label-free learning
  policies, catalog construction/fingerprints, separate model-call timings, and
  optional bounded-window or exact full-stream ranking metrics.

### Changed

- Updated locked fsspec to 2026.6.0 to resolve CVE-2026-104851 reported by
  dependency-audit CI.
- Hardened long-running processing against oversized-batch storage retention,
  non-integer capacities that disable batching, warm-up completion or eviction,
  deep Mondrian/Random Cut tree recursion,
  fixed-width sketch counter wrapping, lossy integer clocks/IDs, and several
  large-time or finite-input numeric failures. Page-Hinkley now updates its mean
  without accumulating an overflowing lifetime sum.

- Unified the bounded radius-neighbor detectors and ISCONNA's preview/learning
  transitions to remove duplicated scoring and state-update logic.
- Built-in pipeline transformers now restore their learned state when a later
  stage rejects an event. Custom transformers can opt into this behavior through
  `TransactionalTransformerProtocol`.
- Simplified FAISS retention around validated vectors and staged index updates;
  its `window` and `keys` diagnostics return owned snapshots.
- Updated all pinned CodeQL steps to 4.38.2 and grouped their Dependabot
  updates to keep action versions aligned.
- Updated the locked urllib3 dependency to 2.8.0 to fix three security
  advisories reported by the dependency audit.
- Added a CI policy check requiring all CodeQL steps to use the same commit.

### Fixed

- Bounded components share strict integer-capacity validation; NumPy integer
  capacities normalize before storage and ADWIN's cross-parameter arithmetic.
- SDOStream rejects timestamps outside its float64 storage range before state
  changes; RSHash preserves mixed integer/float elapsed times without overflow
  or loss of small gaps at large clock values.
- KitNET rejects overflowing training events without partially updating its
  ensemble, feature-map transition, schema, counters, or random generator.
- Shared linear-quantile interpolation across rolling robust scaling and moving
  statistics, preserving narrow-window IQRs and constant subnormal references.
- Geometric and harmonic catalog warm-up now counts retained values, with the
  geometric minimum resolved from `absoluteValues`, so evaluation requires an
  explicit readiness policy instead of admitting warm-up sentinel scores.
- Prequential evaluation rejects complex labels before scoring or learning,
  including NumPy complex scalars that previously lost their imaginary part.
- Prevented RSHash decay underflow from poisoning subsequent learning.
- Kept radius-neighbor cell candidates consistent with distance tolerances.
- Matched autoencoder input tensors to the supplied module's device and dtype.
- Restored cached artifacts when metadata publication fails and closed
  interleaved NPZ streams independently.
- Kept PCA components unchanged if online eigendecomposition fails.

### Development

- Aligned the pre-commit Ruff version with the lockfile and applied its formatting.
- Updated uv to 0.12.18 in GitHub Actions and relaxed the minimum version
  requirement so Dependabot can update the lockfile.
- Updated virtualenv to a patched release for CVE-2026-102925 and included
  development dependencies in the security audit.

## [1.1.0] - 2026-09-25

### Added

- `MultivariateRollingMatrixProfile` exact, causal joint subsequence scores over
  bounded history, with common-match indices, per-channel distance explanations,
  normalized/raw distances, catalog and pipeline integration, and a seeded
  phase-change example. Scalar matrix-profile numerical helpers are shared.
- `RollingMatrixProfile` exact, causal subsequence scores and nearest-match
  indices over bounded history, with normalized/raw Euclidean distances,
  constant-window support, catalog integration, and a streaming example.
- Immutable, machine-readable catalogs for every built-in model, transformer,
  and similarity engine, including parameter schemas, optional dependency
  availability, and configuration-resolved operational capabilities.
- Versioned `DetectorConfig` construction with allowlisted component IDs,
  nested similarity-engine configuration, feature-width validation, and stable
  configuration fingerprints.
- Public `FeatureSchemaGuard` input transformer and typed catalog configuration
  errors.
- Stable `aberrant.similarity.FaissSimilaritySearchEngine` import path.

### Changed

- Updated the pinned build backend to Hatchling 1.32.4 and refreshed GitHub
  Actions dependencies, including aligned CodeQL 4.38.1 steps.

## [1.0.0] - 2026-09-07

### Added

- Documented compatibility policy for the stable public API.
- Release tag/version validation before distribution builds and publication.
- API reference entries for `ModelPipeline` and `TransformerPipeline`, with
  documentation rebuilds triggered by package source changes.
- Public API exports for `aberrant.model.svm`, `aberrant.model.stat`, `aberrant.stream`,
  `aberrant.transform`, and deep lazy exports.
- Regression tests for feature-order stability in `OnlineIsolationForest`.
- Regression tests for `keys=` initialization in multivariate statistical models.
- Optional dependency extras for benchmark tooling.
- Repository standards and CI/security/release workflows.
- `OnlineAutoencoderEnsemble` deep detector with phased warm-up, tests, docs,
  and an example script.
- `MStream` sketch-based streaming detector with `aberrant.model.sketch` public API,
  unit/integration tests, docs, and example script.
- `StreamingLODA` sketch-based detector with `aberrant.model.sketch` public API,
  unit/integration tests, docs, and an example script.
- `SDOStream` bounded-memory observer-based detector in `aberrant.model.distance`
  with unit/integration tests, docs, and example script.
- `MIDAS` graph-stream detector in `aberrant.model.graph` with public export,
  unit/integration tests, docs, and example script.
- `AnoEdgeL` graph-stream detector in `aberrant.model.graph` with public export,
  unit/integration tests, docs, and example script.
- `XLagDAMP` pure-online time-series discord detector based on the original
  authors' X-Lag Amnesic DAMP implementation, with bounded history, reference
  formula tests, synthetic integration coverage, docs, and example script.
- Production hardening CI jobs for base-install smoke and optional-extras smoke.
- Trusted SHA256 checksums for built-in dataset artifacts.
- `py.typed` marker for downstream type-checker support.
- Stateful `FeatureSchema`, `MonotonicClock`, `NumericEventBoundary`, and
  `EdgeEventBoundary` components with explicit preview/commit semantics.
- Typed dataset cache, metadata, download, validation, and stream components.

### Changed

- Mondrian and random-histogram trees now represent leaves and branches separately, with complete bounds and children instead of optional node fields.

- Paper-derived custom variants now have accurate canonical public names:
  `CellNeighborhoodDetector`, `StationaryRegionNeighborDetector`,
  `SignedGraphSketchDetector`, `OnlineAutoencoderEnsemble`,
  `MondrianIsolationForest`, `StreamingRSHash`, and `StreamingLODA`.
- `StreamRandomHistogramForest` now uses `max_depth` exclusively; the historical
  `max_bins` parameter and attribute were removed. Per-node random values are
  generated lazily so memory scales with visited nodes instead of complete-tree
  capacity. Exact seeded score sequences may change under the new derivation.
- `OnlineIsolationForest` now enforces deterministic feature ordering and key-set checks.
- Dataset module version now uses `aberrant.__version__` as single source of truth.
- Documentation was rewritten to match the current public API.
- `aberrant[dev]` now includes `torch` and `scikit-learn` so the full test suite can
  run from the dev environment.
- Integration tests now run by default (no environment-variable gate).
- CI and release workflows now run `mypy aberrant` as a full-package type gate.
- `faiss-cpu` moved from base dependency to optional `aberrant[faiss]`.
- Public import tests are split into base-install and extras-install smoke tests.
- Dataset streaming now defaults to non-interactive mode (no progress bars unless enabled).
- PyTorch architectures now initialize with model-owned generators without
  replacing process-wide RNG state.
- Public package routers now expose concrete static types; optional exports are
  explicit and excluded from wildcard imports.
- MStream and XStream now store initialized algorithm state in single typed
  state objects, and isolation trees use explicit leaf/branch unions.

### Fixed

- Reject PCA configurations with fewer warm-up samples than retained components
  and prevent zero-residual normalization at `tol=0` from corrupting projections.
- Honor negative and zero quantile thresholds when classifying scores. Equality
  now consistently returns `1.0`, including at zero; below nonpositive thresholds
  the output is `0.0`.
- Compute geometric means in the log domain to avoid window-product overflow
  and underflow.
- Stabilize Mahalanobis scoring for singular and ill-conditioned covariance
  matrices so ordinary correlated observations cannot produce negative distances.
- Apply the configured formatter across the package, examples, and tests.

- Isolate the release wheel smoke test from the checkout and verify that imports resolve inside the installation environment.

- Include all non-integration tests in pull-request CI and coverage, including root-level pipeline, validation, dataset, and workflow checks.

- Give transformer and model pipelines distinct runtime capabilities and one canonical stage sequence. `Pipeline(...)` and `|` retain valid execution behavior; use `TransformerPipeline` and `ModelPipeline` in place of generic pipeline annotations.

- Validate entire transformer samples before updating scaler statistics or
  committing projection schemas; failed initialization can be retried.

- Cache only candidate cell neighborhoods in the stationary-region detector;
  recompute exact distances so query order and adjacent-cell updates cannot
  reuse stale neighbor counts.

- Preserve per-tree sampling membership when online isolation forests evict
  observations, preventing negative or drifting tree populations.

- Simplify cell-neighborhood scoring to one bounded index; preserve fractional
  scarcity scores and prevent score-only queries from accumulating cache entries.

- Restored Achlioptas' target-dimension normalization in `RandomProjection`.
- Kept ADWIN variance exact when old buckets are removed.
- Made the FAISS engine return mean L2 distance instead of mean squared L2
  distance, matching its public contract.
- Hardened dataset cache metadata/download publication and limited cache clearing
  to files owned by the dataset manager.
- Rejected non-finite observations before they can poison persistent scaler,
  drift-detector, threshold, projection, or autoencoder state.
- `keys=` initialization path in multivariate statistical detectors.
- Deep tests now skip gracefully when `torch` is unavailable.
- Packaging metadata now correctly declares MIT classifier.
- Dataset cache validation now enforces trusted checksum verification.

### Removed

- Legacy `aberrant.stream.streamer` parquet module.
- `aberrant.model.svm/todo.txt` from distributable package contents.
- Historical model aliases and the `DatasetStreamer` forwarding facade.
- The `aberrant.model.stat.uni` compatibility module; univariate statistics are
  imported from `aberrant.model.stat`.
