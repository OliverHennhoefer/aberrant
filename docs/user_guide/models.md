# Models

ABERRANT groups detectors by the structure they model, not by a single shared
score scale. Choose a family from the event shape and anomaly mechanism first;
then compare warm-up, state growth, latency, and calibration within that family.

## Choose by problem shape

| Problem | Useful starting point | Why |
| --- | --- | --- |
| General multivariate events | `OnlineIsolationForest` | Incremental trees with explicit sliding-window unlearning and a bounded `[0, 1]` score |
| Periodically rebuilt reference model | `ASDIsolationForest` or `StreamRandomHistogramForest` | Replaces a forest from recent-window state rather than adapting every old split forever |
| Local density or neighborhood isolation | `LocalOutlierFactor`, `SDOStream`, or a cell-neighborhood detector | Scores relative to nearby points, observers, or radius cells |
| Fixed-size projection/frequency state | `StreamingLODA`, `MStream`, or `StreamingRSHash` | Memory is controlled by projection, histogram, or sketch dimensions |
| Dynamic edges | `MIDAS`, `ISCONNA`, `AnoEdgeL`, or `SignedGraphSketchDetector` | Models repeated edges, endpoint patterns, dense submatrices, or graph-level structure |
| Scalar time-series discords | `XLagDAMP` | Compares the current subsequence with preceding subsequences under bounded X-Lag history |
| Interpretable local statistic change | `aberrant.model.stat` | Measures candidate-induced changes in means, spread, moments, covariance, or correlation |
| Learned reconstruction error | `OnlineAutoencoderEnsemble` or optional `Autoencoder` | Uses NumPy or user-supplied PyTorch autoencoders |
| Static or adaptive score policy | `ThresholdModel` or `QuantileThreshold` | Converts feature or detector-score boundaries into a decision-oriented output |

Feature scaling is part of the model specification. Distance-based methods and
reconstruction models are scale-sensitive. `HalfSpaceTrees` specifically
expects features scaled to `[0, 1]`. Isolation methods based on randomized cuts
often need less scaling, but a transform can still change their geometry.

## Score contracts

### Isolation forests

| Class | Default score | Built-in warm-up | State policy |
| --- | --- | --- | --- |
| `ASDIsolationForest` | `[0, 1]`; `0.0` before a forest exists | First complete `window_size` | Bounded recent window; complete forest replacement every `retrain_interval` events |
| `HalfSpaceTrees` | `[0, 1]`; `0.0` before reference mass exists | First complete `window_size` | Fixed trees and two window-mass counters; no retained sample window |
| `MondrianIsolationForest` | `[0, 1]`; `0.0` through the first learned event | At least two learned events | Growing online Mondrian trees; no automatic forgetting or state bound |
| `OnlineIsolationForest` | `[0, 1]`; early scores are possible but not calibrated | No explicit zero-score warm-up | At most `window_size` retained events plus trees; evicted events are unlearned |
| `RandomCutForest` | Raw non-negative CoDisp by default; optional `[0, 1)` transform | `warmup_samples` inserted shingles, defaulting to `sample_size` | At most `sample_size` shingled points plus shingle history |
| `StreamRandomHistogramForest` | Non-negative summed log leaf-mass score | First complete `window_size` | Bounded alternating reference/current-window forest state |
| `XStream` | `[0, 1]`; `0.0` until ready | Projection initialization plus one complete reference window | Fixed count-min sketches, chains, and bounded feature-hash cache when configured |

`MondrianIsolationForest` is an ABERRANT hybrid: it uses online Mondrian block
extension but Isolation Forest path-length scoring. The original
[Mondrian Forest](https://proceedings.neurips.cc/paper_files/paper/2014/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html)
is supervised and does not define this anomaly score.

`OnlineIsolationForest.n_jobs=1` executes tree work sequentially;
`n_jobs=-1` uses all logical CPUs reported by the operating system. Positive
values request that many worker threads.

The raw and normalized Random Cut Forest forms preserve ranking but not numeric
scale:

```python
from aberrant.model.iforest import RandomCutForest

history = [
    {"x": 0.0, "y": 0.1},
    {"x": 0.1, "y": -0.1},
    {"x": -0.1, "y": 0.0},
    {"x": 0.2, "y": 0.1},
]
query = {"x": 4.0, "y": 4.0}

raw_detector = RandomCutForest(
    n_trees=10,
    sample_size=4,
    warmup_samples=4,
    normalize_score=False,
    seed=3,
)
bounded_detector = RandomCutForest(
    n_trees=10,
    sample_size=4,
    warmup_samples=4,
    normalize_score=True,
    score_scale=8.0,
    seed=3,
)

for event in history:
    raw_detector.learn_one(event)
    bounded_detector.learn_one(event)

print(f"raw CoDisp: {raw_detector.score_one(query):.3f}")
print(f"bounded transform: {bounded_detector.score_one(query):.3f}")
```

Primary method sources include
[Isolation Forest](https://doi.org/10.1109/ICDM.2008.17),
[Half-Space Trees](https://www.ijcai.org/Proceedings/11/Papers/254.pdf),
[Online Isolation Forest](https://proceedings.mlr.press/v235/leveni24a.html),
[Random Cut Forest](https://proceedings.mlr.press/v48/guha16.html), and
[xStream](https://doi.org/10.1145/3219819.3220107). Class docstrings and the
[isolation API](../api/models/iforest.md) identify the exact source used by each
implementation.

### Distance and neighborhood detectors

| Class | Default score | Built-in warm-up | State policy |
| --- | --- | --- | --- |
| `KNN` | Engine-defined; the FAISS engine returns mean Euclidean distance to the `k` nearest retained points | Engine-defined; FAISS returns `0.0` before `warm_up` | Engine-defined; FAISS uses a `window_size` sliding window |
| `LocalOutlierFactor` | Non-negative LOF; values near `1` indicate density comparable to neighbors, larger values indicate lower local density | Returns `0.0` until more than `k` points are learned | At most `window_size` retained points |
| `CellNeighborhoodDetector` | `[0, 1]` from radius-neighbor scarcity | At least `warm_up_slides * slide_size` events and `k + 1` retained points | At most `window_size` points plus cell indexes |
| `SDOStream` | Non-negative median distance to nearest active observers | `warm_up_observers`, defaulting to at least `x_neighbors` | Exactly up to `k` observers with exponentially faded activity |
| `StationaryRegionNeighborDetector` | `[0, 1]` from radius-neighbor scarcity | At least `warm_up_slides * slide_size` events and `k + 1` retained points | At most `window_size` points plus cells and query caches |

For the FAISS-backed KNN, configure `warm_up >= k`; otherwise the engine can
be warm while still holding too few points for the requested neighbor count.
Install it with `aberrant[faiss]`.

`CellNeighborhoodDetector` is a point-scoring, NETS-inspired adaptation, not
the paper's exact set-level outlier procedure. Likewise,
`StationaryRegionNeighborDetector` uses radius counts and per-cell cache
invalidation rather than STARE's exact KDE/top-*n* procedure. The distinction
is intentional and documented in their
[API reference](../api/models/distance.md). `SDOStream` follows the fixed
observer and exponential-fading structure described in the
[SDOstream paper](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-143.pdf).

### Sketch detectors

| Class | Default score | Built-in warm-up | State policy |
| --- | --- | --- | --- |
| `MStream` | Non-negative `log1p` transform of candidate-inclusive multi-aspect sketch statistics | A bucket-index distance of `warm_up_buckets` from the first learned bucket; default `0` | Fixed numeric, categorical, and record sketches |
| `StreamingLODA` | Non-negative mean negative log histogram density | Exactly `warm_up_samples` learned events | Fixed projections and histograms; bin edges freeze after warm-up, counts optionally decay |
| `StreamingRSHash` | `[0, 1]` from low faded hash occupancy | Exactly `warm_up_samples` learned events | Fixed subspaces and count tables with fading |

These are bounded streaming adaptations. `StreamingLODA` fixes its projection
count and histogram edges rather than reproducing the original paper's model
selection procedure; `StreamingRSHash` adds online standardization and fading
instead of reproducing the paper's window procedure. See the primary
[LODA](https://doi.org/10.1007/s10994-015-5521-0),
[MStream](https://doi.org/10.1145/3442381.3450023), and
[RS-Hash](https://www.charuaggarwal.net/linearout.pdf) sources for the methods
from which they derive.

For all three classes, `time_key=None` uses one-based arrival order. With an
explicit `time_key`, timestamps must be finite and non-decreasing; MStream
additionally requires integer-like time buckets. The time field is removed
before feature vectorization. MStream warm-up uses the integer bucket-index
distance, so gaps between explicit bucket numbers count toward
`warm_up_buckets` even when no event was observed in those buckets.

### Graph-stream detectors

| Class | What it scores | Default score | State policy |
| --- | --- | --- | --- |
| `AnoEdgeL` | Candidate edge membership in maintained local dense submatrices | Non-negative raw score; optional `[0, 1)` | Fixed higher-order sketches and configured submatrices |
| `ISCONNA` | Edge/endpoint frequency, consecutive-width, and gap patterns | Non-negative raw combined G-test score; optional `[0, 1)` | Fixed count-min sketches |
| `MIDAS` | Candidate-inclusive edge microcluster statistic; MIDAS-R also scores endpoints | Non-negative raw chi-square-style score; optional `[0, 1)` | Fixed count-min sketches |
| `SignedGraphSketchDetector` | Change in a graph's signed shingle sketch relative to online centroids | Non-negative distance-plus-novelty score; optional `[0, 1)` | At most `max_graphs` sketches plus fixed centroids |

`AnoEdgeL`, `ISCONNA`, and `MIDAS` require integer-like source, destination,
and explicit timestamp values. Their timestamps are non-decreasing integer
buckets; equal timestamps represent edges in the same bucket. If
`time_key=None`, each arrival receives the next implicit bucket.

`SignedGraphSketchDetector` additionally requires a graph identifier. Its
identifiers are finite numeric values, and its explicit timestamps are also
integer-like. It is a signed-count/Euclidean-centroid structural adaptation;
it does not reproduce StreamSpot's Hamming-sketch clustering workflow.

This complete edge-stream example uses no dataset download:

```python
from aberrant.model.graph import MIDAS

detector = MIDAS(
    source_key="src",
    destination_key="dst",
    time_key="t",
    warm_up_samples=2,
    normalize_score=True,
    seed=11,
)

edges = [
    {"src": 1.0, "dst": 2.0, "t": 1.0},
    {"src": 1.0, "dst": 2.0, "t": 1.0},
    {"src": 1.0, "dst": 3.0, "t": 2.0},
    {"src": 9.0, "dst": 10.0, "t": 3.0},
]

for edge in edges:
    score = detector.score_one(edge)
    detector.learn_one(edge)
    print(f"edge={int(edge['src'])}->{int(edge['dst'])}, score={score:.3f}")
```

Primary references are linked from each class, including
[MIDAS](https://ojs.aaai.org/index.php/AAAI/article/view/5724),
[ISCONNA](https://arxiv.org/abs/2104.01632),
[AnoGraph/AnoEdge](https://doi.org/10.1145/3580305.3599273), and
[StreamSpot](https://doi.org/10.1145/2939672.2939783).

### Time-series discord detection

`XLagDAMP` accepts exactly one consistently named scalar feature. It scores the
subsequence ending at the candidate event by z-normalized Euclidean distance to
its nearest preceding subsequence. Scores are `0.0` before `start_index`; after
that, high peaks are left-discord candidates. Early-abandoned non-peak scores
can be approximate. Memory is bounded by
`x_lag + subsequence_length - 1` learned values.

The implementation follows the original X-Lag Amnesic, backward-only
`lookahead=0` procedure from
[Matrix Profile XXIV](https://doi.org/10.1145/3534678.3539271). Constant
subsequences are rejected because z-normalized distance is undefined for zero
variance under the reference restriction.

### Statistical detectors

The univariate classes accept exactly one feature and retain at most
`window_size` values:

- `MovingAverage`, `MovingHarmonicAverage`, and `MovingGeometricAverage`;
- `MovingMedian` and `MovingQuantile`;
- `MovingVariance`, `MovingInterquartileRange`, and
  `MovingAverageAbsoluteDeviation`;
- `MovingKurtosis` and `MovingSkewness`.

Their score is the change in the configured statistic after provisionally
adding the candidate to the current window. It is `0.0` for an empty window.
`abs_diff=True` returns magnitude; `abs_diff=False` preserves direction.
`MovingGeometricAverage.score_one` requires a positive candidate, while
`learn_one` ignores non-positive values. Its `absoluteValues` mode operates on
successive growth factors.

```python
from aberrant.model.stat import MovingAverage

detector = MovingAverage(window_size=3, key="latency_ms")
for value in [10.0, 11.0, 9.0]:
    detector.learn_one({"latency_ms": value})

candidate = {"latency_ms": 25.0}
print(f"mean-change score: {detector.score_one(candidate):.3f}")
detector.learn_one(candidate)
```

`MovingCovariance` and `MovingCorrelationCoefficient` require exactly two
features and score candidate-induced statistic change. `MovingMahalanobisDistance`
accepts a fixed multivariate schema and returns **squared** Mahalanobis distance
from the current window mean; it returns `0.0` with fewer than three learned
points.

### SVM detectors

`IncrementalOneClassSVMAdaptiveKernel` is a custom, budgeted RBF-kernel
heuristic with running standardization and adaptive gamma. It returns the
negative decision function, so higher values are more anomalous, and bounds
support-vector state with `sv_budget`.

`GraphGatedOneClassSVM` routes updates and scoring through a user-supplied graph
of incremental linear one-class SVM heuristics. It is unrelated to the
published GADGET distributed optimization algorithm. Both classes are marked
experimental; they do not claim parity with a published one-class SVM solver.

### Reconstruction detectors

`OnlineAutoencoderEnsemble` is NumPy-backed and available in the base install.
It has three observable phases:

1. `feature_map_warmup`: collect correlation statistics for
   `feature_map_grace` events;
2. `detector_warmup`: train sub-autoencoders and the output autoencoder for
   `ad_grace` events;
3. `ready`: return non-negative reconstruction-error scores, optionally
   continuing to adapt when `adaptive_after_warmup=True`.

The parameter names (`feature_map_grace`, `ad_grace`) and phase names are
deliberately different. This is a lightweight KitNET-inspired adaptation using
raw inputs and simple NumPy autoencoders, not score- or procedure-equivalent to
the authors' normalized denoising implementation. The source method is
[Kitsune/KitNET](https://www.ndss-symposium.org/ndss-paper/kitsune-an-ensemble-of-autoencoders-for-online-network-intrusion-detection/).

`Autoencoder` requires `aberrant[dl]`. The caller supplies an ABERRANT
`Architecture`, a PyTorch optimizer, and a reconstruction loss. It has no
built-in warm-up policy; `score_one` returns the supplied criterion's loss and
`learn_one` performs one optimizer step.

### Core utility models

| Class | Contract |
| --- | --- |
| `ThresholdModel` | Stateless binary output: `1.0` when any configured ceiling or floor is violated, otherwise `0.0`. Scalar bounds apply to every supplied feature; mapping bounds ignore unconfigured feature names. |
| `QuantileThreshold` | Sliding score-window policy. It becomes ready after `min(window_size, max(10, floor(0.1 * window_size)))` learned scores, returns `1.0` at or above the quantile, and otherwise returns a ratio below `1.0`. |
| `NullModel` | Stateless baseline that always returns `0.0`. |
| `RandomModel` | Uniform `[0, 1)` baseline. Each `score_one` consumes one draw from its model-local generator; `learn_one` is a no-op. |

`QuantileThreshold` is adaptive but not contamination-resistant: learning
anomalous scores moves the score distribution it uses. Decide explicitly
whether every score, only accepted-normal scores, or a delayed/calibration
stream should update it.
