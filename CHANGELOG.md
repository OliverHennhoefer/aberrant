# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project aims to follow
Semantic Versioning.

## [Unreleased]

### Added

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
