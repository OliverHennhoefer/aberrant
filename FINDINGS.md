# Model Implementation Review

Reviewed on 2026-06-04.

## Scope and Method

This review covers every concrete `BaseModel` implementation under
`aberrant/model`: 40 public model classes across 29 implementation modules.
Internal tree, sketch, and SVM helper classes were reviewed with their public
model.

For each model, the source, dedicated unit tests, and available example were
read. Paper-derived models were compared with the primary paper and, where one
could be identified, an author or reference implementation. A model is called
"paper-faithful" below only when its core state, update rule, and score match the
published algorithm. A useful paper-inspired detector can still be correct as a
custom model, but it should not imply score parity with the named algorithm.

Unit tests establish implementation behavior, not scientific equivalence.
Reference-formula or benchmark-parity tests remain necessary before claiming
reproduction of a paper's results.

## Executive Summary

- The simple baselines, threshold models, moving statistics, generic
  autoencoder wrapper, KNN adapter, and most bounded streaming implementations
  are internally coherent after the fixes in this review.
- Correctness defects were fixed in univariate schema handling, geometric and
  harmonic scoring edge cases, Mahalanobis distance, LOF duplicate handling,
  autoencoder schema handling, ASD Isolation Forest, quantile threshold
  normalization, and random-histogram RNG isolation.
- Several named paper models are substantial approximations rather than
  reproductions: `NETS`, `STARE`, `MStream`, `AnoEdgeL`, `ISCONNA`,
  `StreamSpot`, `HalfSpaceTrees`, `ASDIsolationForest`,
  `StreamRandomHistogramForest`, `KitNET`, `RandomCutForest`, and
  `MondrianForest`.
- `GADGETSVM` does not implement the published GADGET algorithm.
- `IncrementalOneClassSVMAdaptiveKernel` has a fundamental coordinate-system
  consistency problem and should be treated as experimental.
- Every implementation module has dedicated unit tests. Excluding the optional
  FAISS-backed KNN test file, the model suite is at 93% statement coverage.

## Correctness Defects Fixed

| Area | Defect | Resolution and regression coverage |
| --- | --- | --- |
| Univariate moving statistics | Eight models documented "first learned key" but never saved it, allowing the learned feature to change between samples. | Added one shared schema/value extractor and a cross-model feature-lock test. |
| `MovingGeometricAverage` | Returned anomaly score `1` during cold start, despite its documentation saying `0`; nonpositive query values could produce a complex result. | Cold score is now `0.0`; nonpositive scoring values are rejected; tests cover both. |
| `MovingHarmonicAverage` | Reciprocal sums can cancel to zero and raise an unhelpful `ZeroDivisionError`. | Raises an explicit `ValueError`; covered by a focused test. |
| `MovingQuantile` | Accepted quantiles outside `[0, 1]`. | Added constructor validation and boundary tests. |
| `MovingMahalanobisDistance` | Ignored `bias`; one-feature input failed because `np.cov` returns a scalar. | Uses the requested covariance bias and promotes covariance to 2-D; tests cover population/sample covariance in one dimension. |
| `LocalOutlierFactor` | Removed all zero-distance neighbors before considering duplicates, rather than excluding only the point itself. | Neighbor selection now accepts an explicit self index and retains duplicate points; covered directly. |
| `Autoencoder` | Silently ignored extra features and could fail late on missing features or architecture-width mismatch. | First sample must match `input_size`; subsequent samples must match the established schema; covered by tests. |
| `ASDIsolationForest` | Used an incorrect `c(2)`, allowed `max_samples=1` and later divided by zero, duplicated every chunk-boundary sample, and could stop at a constant randomly selected feature while other features varied. | Corrected `c(2)`, validation, disjoint chunk buffering, and variable-feature selection; all have focused tests. |
| `QuantileThreshold` | Negative scores produced values below the documented normalized range. | Normalized output is clipped at `0.0`; covered by a test. |
| `StreamRandomHistogramForest` | Seeding an instance mutated Python's process-global random state. | All randomness now uses the instance-local NumPy generator; covered by a global-state regression test. |
| Evaluation loops | Seven examples, eleven integration-test files, and several source/documentation snippets learned each evaluated point before scoring it, causing prequential data leakage and, for KNN, a self-neighbor. | Corrected every affected evaluation loop to score before learning. The corrected non-FAISS integration suite passes; ASD's deterministic PR-AUC snapshot increased from the old range to `0.9884`. |

## Remaining Findings

### High Priority: Algorithm Identity and Correctness

1. **`GADGETSVM` is not GADGET.**
   The published GADGET algorithm is distributed online prediction and
   stochastic optimization using approximate distributed averaging. This class
   is a graph-gated ensemble of custom linear one-class SVM updates. The name
   should be deprecated or changed before presenting it as a paper
   implementation.

2. **`IncrementalOneClassSVMAdaptiveKernel` uses inconsistent feature
   coordinates.**
   Running mean/std values change over time, but existing support vectors remain
   in the coordinate system used when they were inserted. Kernel comparisons
   therefore mix incompatible representations. The first sample is also omitted
   from running-stat updates, and the stored `(mean, std)` update is not a valid
   Welford accumulator. Fixing this requires a model-level design decision:
   store raw support vectors and standardize on demand, or freeze/reproject the
   normalization state.

3. **`AnoEdgeL` does not implement AnoEdge-L's dense-submatrix search.**
   The paper/reference implementation maps edges into higher-order sketches and
   evaluates whether the mapped cell belongs to a locally dense submatrix. The
   current model computes a custom local-neighborhood rarity/density score.

4. **`ISCONNA` omits the published pattern component.**
   ISCONNA combines burst/frequency information with consecutive
   presence/absence pattern scores and aggregates three intermediate scores.
   The current class is a frequency-only conditional-surprise detector.

5. **`STARE` is not the published KDE/top-n algorithm.**
   Published STARE uses fixed grid-cell kernel centers, kernel density
   estimation, cumulative net-change skipping, and top-n outlier retrieval.
   This class uses radius-neighbor counts and a per-query bounded score.

6. **`NETS` is a point-scoring approximation.**
   Published NETS performs exact distance-based window outlier detection through
   set-level net-effect updates and two-level dimensional filtering. This class
   uses related cell bookkeeping and pruning, but returns a custom continuous
   score for one query point and randomly chooses a subspace.

7. **`HalfSpaceTrees` does not construct published Half-Space Trees.**
   HST recursively propagates and bisects feature-space intervals. This class
   samples every node threshold independently from `[0.15, 0.85]`, so child
   partitions do not represent recursive half spaces. The mass-window mechanics
   are useful, but the tree distribution is materially different.

8. **`ASDIsolationForest` is not published iForestASD.**
   Published iForestASD periodically retrains an Isolation Forest on a recent
   sliding window. This class builds one tree from each disjoint chunk and keeps
   a queue of those trees. The fixed implementation is coherent as a lightweight
   streaming iForest variant, but the name implies a different algorithm.

9. **`StreamRandomHistogramForest` is neither a forest nor STREamRHF.**
   STREamRHF incrementally builds trees and chooses splits using feature
   kurtosis. This class is an ensemble of independent fixed random histograms
   with decay. Its name and `examples/models/streamRHF.py` overstate fidelity.

### Material Paper Variants

10. **`MStream` differs from the author implementation.**
    Original MStream hashes each individual attribute and the complete record.
    This class hashes singleton and optional pairwise views, so it cannot capture
    the same full-record interactions or reproduce author scores.

11. **`StreamSpot` uses different sketches and clustering.**
    The original uses Hamming-distance graph sketches and a different
    clustering process. This class uses signed count sketches, Euclidean
    distances, and online centroid updates.

12. **`KitNET` is a simplified KitNET-style detector.**
    The phase structure and ensemble architecture are recognizable, but feature
    mapping, normalization, denoising, and autoencoder behavior differ from the
    authors' implementation. Benchmark and score parity should not be expected.

13. **`RandomCutForest` uses a custom score.**
    Tree insertion and deletion follow Random Cut Tree mechanics, but
    `codisp * (1 + distance / scale)` plus optional exponential normalization is
    not the paper/reference displacement or codisp score.

14. **`MondrianForest` is a custom anomaly hybrid.**
    Online block extension follows Mondrian mechanics, while anomaly scoring is
    Isolation Forest path-length normalization. The original Mondrian Forest is
    supervised and defines no such anomaly score.

15. **`RSHash` and `LODA` are bounded streaming adaptations.**
    Their central ideas are present, but online normalization/fading in
    `RSHash`, and fixed projection/bin selection in `LODA`, differ from the
    published procedures.

16. **`MIDAS(use_relational=True)` is a custom extension.**
    The base edge chi-square score is close to MIDAS. The endpoint-independence
    score fused by `max` is not the published MIDAS-R relational sketch.

17. **`LocalOutlierFactor` uses exactly `k` neighbors.**
    The original LOF definition includes every point tied at the k-distance.
    This implementation deliberately keeps exactly `k`, so scores can differ on
    tied neighborhoods. It is also a sliding-window recomputation, not the 2007
    incremental LOF update algorithm.

18. **Time-aware `score_one` calls can mutate model state.**
    `MStream`, `MIDAS`, `AnoEdgeL`, `ISCONNA`, and `StreamSpot` advance or decay
    buckets while scoring a future timestamp. This behavior is tested for
    `AnoEdgeL`, but it is surprising relative to otherwise non-mutating
    `score_one` implementations and should be documented as a shared contract.

19. **`OnlineIsolationForest` has no instance seed.**
    It imports NumPy's process-global random functions. This makes independent
    reproducibility and isolation from caller RNG state difficult even though
    the implementation otherwise closely follows the authors' code structure.

20. **Several tree models silently impute missing features with zero.**
    `ASDIsolationForest`, `HalfSpaceTrees`, and
    `StreamRandomHistogramForest` use `x.get(feature, 0.0)`. This is internally
    consistent but can hide schema errors. Most newer models reject schema
    changes; the library should choose and document one policy.

## Per-Model Review Matrix

| Model | Review outcome | Tests and example review |
| --- | --- | --- |
| `NullModel` | Correct baseline. | Dedicated tests sufficient; no example needed. |
| `RandomModel` | Correct instance-local random baseline. | Dedicated deterministic/range tests sufficient; no example needed. |
| `ThresholdModel` | Correct static boundary detector. | Dedicated scalar/dict/corridor tests sufficient; no example needed. |
| `QuantileThreshold` | Correct after normalized lower-bound fix. | Added negative-score coverage; dedicated tests sufficient. |
| `MovingAverage` | Correct after shared feature-schema fix. | Existing formula/window tests plus new schema test sufficient. |
| `MovingHarmonicAverage` | Correct for defined nonzero harmonic means after explicit undefined-case handling. | Added schema and reciprocal-cancellation coverage. |
| `MovingGeometricAverage` | Correct for positive-valued inputs after cold-score/query validation fixes. | Added cold, nonpositive, and schema coverage. |
| `MovingMedian` | Correct after feature-schema fix. | Formula/window tests plus schema test sufficient. |
| `MovingQuantile` | Correct after feature-schema and quantile validation fixes. | NumPy parity and validation tests sufficient. |
| `MovingVariance` | Correct population-variance change score after feature-schema fix. | NumPy parity and schema tests sufficient. |
| `MovingInterquartileRange` | Correct interpolated IQR-change score after feature-schema fix. | Dedicated tests plus schema test sufficient. |
| `MovingAverageAbsoluteDeviation` | Correct after feature-schema fix. | Dedicated tests plus schema test sufficient. |
| `MovingKurtosis` | Correct population-kurtosis change score after feature-schema fix. | SciPy comparison coverage plus schema test sufficient. |
| `MovingSkewness` | Correct population-skewness change score after feature-schema fix. | SciPy comparison coverage plus schema test sufficient. |
| `MovingCovariance` | Correct bivariate covariance-change model. | Bias/formula/schema tests sufficient. |
| `MovingCorrelationCoefficient` | Correct bivariate correlation-change model. | Bias/formula/schema tests sufficient. |
| `MovingMahalanobisDistance` | Corrected bias and one-dimensional behavior; singular matrices are regularized. | Added population/sample and 1-D parity coverage. |
| `Autoencoder` | Correct generic online reconstruction wrapper after fail-fast schema validation. | Added width/schema regression; example exists. |
| `KitNET` | Internally coherent simplified variant; not author-score faithful. | Strong phase/state/unit coverage; example exists; no author-parity test. |
| `KNN` | Correct adapter to `BaseSimilaritySearchEngine`; behavior depends on engine semantics. | Dedicated tests exist and example exists; one FAISS-backed test aborts the Python 3.13 process in this environment. |
| `LocalOutlierFactor` | Core LOF formula is coherent after duplicate/self fix; exact-k tie behavior remains a variant. | Added duplicate-neighbor regression; no example file. |
| `NETS` | Coherent bounded NETS-style point detector; not exact NETS. | Strong API/state tests and example; no reference-output test. |
| `SDOStream` | Core observer/activity/median-distance design is substantially aligned as a streaming adaptation. | Strong API/state tests and example; no reference-output test. |
| `STARE` | Coherent radius-neighbor variant; not published STARE. | Strong API/state tests and example; no KDE/top-n conformance test. |
| `AnoEdgeL` | Coherent custom sketch-local-density detector; not AnoEdge-L dense-submatrix search. | Strong behavior tests and example; no author-parity test. |
| `ISCONNA` | Coherent frequency-only sketch detector; incomplete ISCONNA. | Strong behavior tests and example; missing pattern-component tests because component is absent. |
| `MIDAS` | Base edge score is aligned; relational mode is custom. | Strong formula/state tests and example; relational reference parity absent. |
| `StreamSpot` | Coherent graph-sketch clustering variant; not original StreamSpot. | Strong behavior tests and example; no author-parity test. |
| `ASDIsolationForest` | Corrected coherent chunked streaming iForest; not published iForestASD. | Added path/chunk/split/validation tests; example exists. |
| `HalfSpaceTrees` | Mass-window logic coherent; tree construction is not paper-faithful. | Broad unit coverage; no example and no paper-conformance test. |
| `MondrianForest` | Coherent custom Mondrian/isolation hybrid. | Broad unit coverage and example; no canonical anomaly reference exists for this score. |
| `OnlineIsolationForest` | Closely follows the paper/reference implementation structure; reproducibility API remains weak. | Broad unit coverage and example; no author-output parity fixture. |
| `StreamRandomHistogramForest` | Coherent custom decayed histogram ensemble; not a forest/STREamRHF. | Added RNG-isolation test; example exists under misleading `streamRHF.py`. |
| `RandomCutForest` | Tree mechanics coherent; score is custom. | Broad unit coverage and example; no reference codisp/displacement test. |
| `XStream` | Central StreamHash/half-space-chain design is substantially aligned. | Strong unit coverage and example; no author-output parity fixture. |
| `LODA` | Coherent fixed-memory streaming LODA adaptation. | Strong unit coverage and example; no paper model-selection parity test. |
| `MStream` | Coherent sketch detector with materially different interaction views. | Strong unit coverage and example; no author-output parity test. |
| `RSHash` | Coherent bounded streaming RS-Hash adaptation. | Strong unit coverage and example; no reference-output parity test. |
| `IncrementalOneClassSVMAdaptiveKernel` | Experimental heuristic with unresolved coordinate-system defect. | Only basic/adaptation tests and example; redesign should precede more behavioral tests. |
| `GADGETSVM` | Coherent custom graph-gated SVM ensemble, but algorithm identity is incorrect. | Broad custom-behavior tests and example; no GADGET conformance is possible. |

## Test and Example Verification

Commands and outcomes after the review changes:

- `uv run ruff check .`: passed.
- `uv run mypy aberrant/model`: passed with no issues in 37 source files.
- `uv run pytest --ignore=tests/models/test_distance_knn.py --ignore=tests/integration/test_distance_knn.py -q`:
  626 passed, 55 subtests passed.
- `uv run pytest tests/models --ignore=tests/models/test_distance_knn.py -q`:
  487 passed, 55 subtests passed.
- Same suite with model coverage: 93% total model statement coverage.
- `uv run pytest tests/integration --ignore=tests/integration/test_distance_knn.py -q`:
  27 passed.
- `uv run pytest tests/models/test_distance_knn.py -q -k 'not k_parameter_effect'`:
  10 passed, 1 deselected, 4 subtests passed.
- Full `tests/models/test_distance_knn.py` aborts inside FAISS during
  `test_k_parameter_effect` on Python 3.13. This is a native optional-backend
  failure, not a Python assertion failure in `KNN`.
- Corrected score-before-learn examples completed after the review:
  `autoencoder.py` ROC-AUC `0.830`, `asd_iforest.py` ROC-AUC `0.992`,
  `gadget_svm.py` ROC-AUC `0.891`, `online_iforest.py` ROC-AUC `0.993`, and
  `streamRHF.py` ROC-AUC `0.777`.
- `examples/models/knn.py` and `examples/pipeline.py` were not executed because
  they use the same FAISS backend that aborts the Python 3.13 process in the
  dedicated KNN test.

Examples exist for all paper-derived/custom complex models except
`LocalOutlierFactor` and `HalfSpaceTrees`. Root threshold/baseline and
statistical models also have no examples; their focused unit tests are more
useful than standalone scripts. Example scripts primarily provide smoke and
metric demonstrations; they do not validate parity with paper benchmarks.

## Primary References and Author Implementations

- LOF paper: https://doi.org/10.1145/342009.335388
- NETS paper: https://doi.org/10.14778/3342263.3342269
- SDOstream paper: https://www.esann.org/sites/default/files/proceedings/2020/ES2020-143.pdf
- STARE paper: https://doi.org/10.1145/3394486.3403171
- LODA paper: https://doi.org/10.1007/s10994-015-5521-0
- RS-Hash paper: https://www.charuaggarwal.net/linearout.pdf
- MStream paper/repository: https://doi.org/10.1145/3442381.3450023 and https://github.com/Stream-AD/MStream
- MIDAS paper/repository: https://ojs.aaai.org/index.php/AAAI/article/view/5724 and https://github.com/Stream-AD/MIDAS
- AnoGraph/AnoEdge author repository: https://github.com/Stream-AD/AnoGraph
- ISCONNA paper/repository: https://arxiv.org/abs/2104.01632 and https://github.com/liurui39660/Isconna
- StreamSpot paper/repository: https://arxiv.org/abs/1607.04930 and https://github.com/sbustreamspot/sbustreamspot
- Isolation Forest paper: https://doi.org/10.1109/ICDM.2008.17
- iForestASD paper: https://www.sciencedirect.com/science/article/pii/S1474667016314999
- Half-Space Trees paper: https://www.ijcai.org/Proceedings/11/Papers/254.pdf
- Mondrian Forest paper: https://papers.nips.cc/paper/5234-mondrian-forests-efficient-online-random-forests
- Online Isolation Forest paper/repository: https://proceedings.mlr.press/v235/leveni24a.html and https://github.com/ineveLoppiliF/Online-Isolation-Forest
- Random Cut Forest paper/reference repository: https://proceedings.mlr.press/v48/guha16.html and https://github.com/aws/random-cut-forest-by-aws
- STREamRHF paper: https://doi.org/10.1109/AICCSA56895.2022.10017876
- xStream paper/reference repository: https://doi.org/10.1145/3219819.3220107 and https://github.com/cmuxstream/cmuxstream-core
- KitNET author implementation: https://github.com/ymirsky/KitNET-py
- Published GADGET algorithm: https://jmlr.org/papers/v17/15-494.html
