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
- Optional dependency extras for `parquet` and benchmark tooling.
- Repository standards and CI/security/release workflows.
- `KitNET` online deep detector with phased warm-up, tests, docs, and example script.
- `MStream` sketch-based streaming detector with `aberrant.model.sketch` public API,
  unit/integration tests, docs, and example script.
- `LODA` sketch-based streaming detector with `aberrant.model.sketch` public API,
  unit/integration tests, docs, and example script.
- `SDOStream` bounded-memory observer-based detector in `aberrant.model.distance`
  with unit/integration tests, docs, and example script.
- `MIDAS` graph-stream detector in `aberrant.model.graph` with public export,
  unit/integration tests, docs, and example script.
- `AnoEdgeL` graph-stream detector in `aberrant.model.graph` with public export,
  unit/integration tests, docs, and example script.
- Production hardening CI jobs for base-install smoke and optional-extras smoke.
- Trusted SHA256 checksums for built-in dataset artifacts.
- `py.typed` marker for downstream type-checker support.

### Changed

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

### Fixed

- `keys=` initialization path in multivariate statistical detectors.
- Deep tests now skip gracefully when `torch` is unavailable.
- Packaging metadata now correctly declares MIT classifier.
- Dataset cache validation now enforces trusted checksum verification.

### Removed

- Legacy `aberrant.stream.streamer` parquet module.
- `aberrant.model.svm/todo.txt` from distributable package contents.
