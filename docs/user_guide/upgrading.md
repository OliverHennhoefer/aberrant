# Upgrading to 2.0

aberrant 2.0 is a clean break. Nothing from 1.x is kept as an alias or shim,
so code that ran on 1.5 fails loudly on 2.0 where it needs a change: with an
`ImportError`, a `TypeError` for an unknown or positional argument, or a
`ConfigurationError` for an old configuration. This page lists every change
that needs one. The [changelog](../changelog.md) has the details and the
reasons.

Pickles written by 1.x do not load in 2.0. They raise
`IncompatibleStateError`; rebuild each component and let it learn again from
your events.

## Every constructor takes keyword arguments

```python
HalfSpaceTrees(25, 8)  # 1.x
HalfSpaceTrees(n_trees=25, height=8)  # 2.0
```

Only the wrapped component of `ResidualDetector(model, target=...)` and the
method of `ScoreCalibrator(method, alpha=...)` stay positional. Integer
parameters reject floats and booleans: a value such as `25.0` read from JSON
needs `int(...)`.

## Renamed classes

| 1.x | 2.0 | Catalog ID |
| --- | --- | --- |
| `StreamingLODA` | `LODA` | `loda` |
| `StreamingRSHash` | `RSHash` | `rshash` |
| `XLagDAMP` | `DAMP` | `damp` |
| `IncrementalOneClassSVMAdaptiveKernel` | `AdaptiveOneClassSVM` | `adaptive_one_class_svm` |
| `StreamRandomHistogramForest` | `RHF` | `rhf` |
| `SignedGraphSketchDetector` | `StreamSpot` | `streamspot` |
| `AnoEdgeL`, `XStream`, `MStream`, `SDOStream` | unchanged | `anoedge_l`, `xstream`, `mstream`, `sdostream` |

## Renamed parameters

| Concept | 2.0 name | 1.x names |
| --- | --- | --- |
| Tree count | `n_trees` | `n_estimators`, `num_trees` |
| Warm-up | `warm_up` | `warm_up_samples`, `warmup_samples`, `warm_up_buckets`, `warm_up_slides`, `warm_up_graphs`, `warm_up_observers`, `init_sample_size`, `min_samples`, `n0` |
| Hash rows and width | `n_hashes`, `sketch_width` | `count_min_rows`/`count_min_cols`, `rows`/`buckets`, `hash_num`/`bins`, `cms_num_hashes`/`cms_width` |
| Fixed feature order | `features` | `keys` |

Single renames: `RSHash(n_members=...)` was `components_num`,
`MondrianIsolationForest(lifetime=...)` was `lambda_`, and `SDOStream`'s
`max_observers`, `time_scale`, `inactive_quantile` and `n_neighbors` were `k`,
`T`, `qv` and `x_neighbors`. `DAMP(min_std=...)` replaces its constant
subsequence threshold. `LODA`, `CellNeighborhoodDetector` and `StreamSpot`
no longer take `eps`.

## Moved imports

| 1.x | 2.0 |
| --- | --- |
| `aberrant.model.iforest` | `aberrant.model.forest` |
| exceptions from `aberrant.base` | `aberrant.exceptions` |
| `aberrant.stream` | `aberrant.datasets` |
| `Architecture`, `VanillaAutoencoder` from `aberrant.base` or `aberrant.utils.deep` | `aberrant.model.deep` |
| `Readiness` from `aberrant.compose` | `aberrant.base` |
| `Select`, `Rename`, `Passthrough` from `aberrant.compose` | `aberrant.transform` |
| `Forecast` from anywhere else | `aberrant.forecast` |

Every public name has exactly one import path.

## Removed, and what to use instead

| Removed | Use |
| --- | --- |
| `QuantileThreshold`, `predict_one`, `predict_threshold` | Compare `score_one` with a threshold from `ScoreCalibrator`, such as `ScoreCalibrator("split", alpha=0.01, window=1000)` |
| `normalize_score`, `score_scale` | Raw scores, mapped to a common scale by `aberrant.calibration` |
| `SeasonalResidualDetector` | `ResidualDetector(HoltWinters(season_length=m), target="x")` |
| `StationaryRegionNeighborDetector` | `CellNeighborhoodDetector` |
| `MovingAverage`, `MovingVariance`, `MovingMedian`, `MovingQuantile` and six other univariate moving statistics | `MovingZScore` or `MovingMAD` |
| `aberrant.similarity`, `similarity_engine` | `KNN(backend="numpy")` (default) or `KNN(backend="faiss")` |
| `OnlineIsolationForest(n_jobs=...)`, `learn_batch`, `score_batch` | `learn_many`, `score_many` on every model |
| `is_ready`, `ReadinessProtocol`, `WarmupRequirement`, `GraphNotReadyError`, `minimum_updates` | `readiness` on every component |
| `CalibratorParameter` | `CalibratorSpec.parameters` and `CalibratorSpec.schema()` |
| `DatasetManager`, `NpzStreamer`, `BatchStreamer`, cache functions | `load`, `DatasetStream` and `data_dir()` in `aberrant.datasets` |
| `GraphGatedOneClassSVM`, `AdaptiveOneClassSVM.get_model_info()`, `conformal_quantile`, `pinball` | No replacement |

## Changed behaviour

- **Readiness.** Every model, transformer, forecaster and pipeline reports
  `readiness`: whether it has learned enough, and how much it has seen and
  needs. A model scores `0.0` until it is ready, and so does a
  `GraphPipeline` instead of raising. `PrequentialEvaluator` excludes events
  while the model is not ready; `warmup` excludes further events on top.
- **Scores.** Higher is more anomalous, `0.0` until ready, and the raw
  statistic otherwise. `OnlineIsolationForest` scores `0.0`, not `1.0`, before
  it can normalize.
- **Drift detectors.** `update(x)` returns whether this value completed a
  drift, so a loop reads `if detector.update(value):`.
- **Configurations.** Detector configurations and the catalog manifest use
  format version 2. Rename IDs and parameters as above and set
  `"version": 2`; a version 1 configuration is rejected.
- **Contract.** `get_params()`, `clone()`, `reset()` and a `repr` of the
  non-default arguments work the same way on every component.
- **Datasets.** `load(Dataset.SHUTTLE)` returns a replayable `DatasetStream`:
  iterate it directly instead of calling `.stream()`. Files are cached in
  `ABERRANT_DATA_DIR` or `~/.cache/aberrant`, so the first load downloads
  again.
- **Dependencies.** The core installation depends on NumPy alone.
