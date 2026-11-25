# Models

aberrant groups detectors by the structure they model, not by a single shared
score scale. Choose a family from the event shape and anomaly mechanism first;
then compare warm-up, state growth, latency, and calibration within that family.

## Choose by problem shape

| Problem | Useful starting point | Why |
| --- | --- | --- |
| General multivariate events | `OnlineIsolationForest` | Incremental trees with explicit sliding-window unlearning and a bounded `[0, 1]` score |
| Periodically rebuilt reference model | `ASDIsolationForest` or `RHF` | Replaces a forest from recent-window state rather than adapting every old split forever |
| Local density or neighborhood isolation | `LocalOutlierFactor`, `SDOStream`, or a cell-neighborhood detector | Scores relative to nearby points, observers, or radius cells |
| Fixed-size projection/frequency state | `LODA`, `MStream`, or `RSHash` | Memory is controlled by projection, histogram, or sketch dimensions |
| Dynamic edges | `MIDAS`, `ISCONNA`, `AnoEdgeL`, or `StreamSpot` | Models repeated edges, endpoint patterns, dense submatrices, or graph-level structure |
| Scalar time-series discords | `RollingMatrixProfile` or `DAMP` | Exact rolling nearest-subsequence scores or approximate DAMP discord scores over bounded history |
| Departures from a known seasonal forecast | `ResidualDetector(HoltWinters(...))` | Scores forecast errors from additive level, trend, and seasonal components; see [forecast-based detection](forecasting.md) |
| Changing relationships between time-series channels | `MultivariateRollingMatrixProfile` | Exact joint subsequence novelty requiring all channels to match the same historical interval |
| Interpretable local statistic change | `aberrant.model.stat` | Measures candidate-induced changes in means, spread, moments, covariance, or correlation |
| Learned reconstruction error | `OnlineAutoencoderEnsemble` or optional `Autoencoder` | Uses NumPy or user-supplied PyTorch autoencoders |
| A fixed bound on raw features | `ThresholdModel` | Returns `1.0` when a feature leaves its configured ceiling or floor; learned alarm thresholds on scores live in [calibration](calibration.md) |

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
| `OnlineIsolationForest` | `[0, 1]`; `0.0` until path lengths can be normalized | Enough learned events for a positive path-length normalization | At most `window_size` retained events plus trees; evicted events are unlearned |
| `RandomCutForest` | Raw non-negative CoDisp | `warm_up` inserted shingles, defaulting to `sample_size` | At most `sample_size` shingled points plus shingle history |
| `RHF` | Non-negative summed log leaf-mass score | First complete `window_size` | Bounded alternating reference/current-window forest state |
| `XStream` | `[0, 1]`; `0.0` until ready | Projection initialization plus one complete reference window | Fixed count-min sketches, chains, and bounded feature-hash cache when configured |

`MondrianIsolationForest` is an aberrant hybrid: it uses online Mondrian block
extension but Isolation Forest path-length scoring. The original
[Mondrian Forest](https://proceedings.neurips.cc/paper_files/paper/2014/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html)
is supervised and does not define this anomaly score.

Its default lifetime mode can keep growing on expanding streams. For bounded
retention, set `window_size`: trees periodically rebuild from the latest complete
window, representing at most twice that window minus one event between rebuilds.
This changes the reference history and introduces periodic rebuild work.

Forest scores are raw: Random Cut Forest returns its collusive displacement
(CoDisp) unchanged, and a larger value is more anomalous. To compare or
threshold scores on a common scale, calibrate them with
[`aberrant.calibration`](calibration.md).

Primary method sources include
[Isolation Forest](https://doi.org/10.1109/ICDM.2008.17),
[Half-Space Trees](https://www.ijcai.org/Proceedings/11/Papers/254.pdf),
[Online Isolation Forest](https://proceedings.mlr.press/v235/leveni24a.html),
[Random Cut Forest](https://proceedings.mlr.press/v48/guha16.html), and
[xStream](https://doi.org/10.1145/3219819.3220107). Class docstrings and the
[isolation API](../api/models/forest.md) identify the exact source used by each
implementation.

### Distance and neighborhood detectors

| Class | Default score | Built-in warm-up | State policy |
| --- | --- | --- | --- |
| `KNN` | Non-negative mean Euclidean distance to the `k` nearest retained events | `k` learned events | The last `window_size` learned events |
| `LocalOutlierFactor` | Non-negative LOF; values near `1` indicate density comparable to neighbors, larger values indicate lower local density | Returns `0.0` until more than `k` points are learned | At most `window_size` retained points, each with a list of at most `2 * k` nearest neighbors |
| `CellNeighborhoodDetector` | `[0, 1]` from radius-neighbor scarcity | At least `warm_up * slide_size` events and `k + 1` retained points | At most `window_size` points plus cell indexes |
| `SDOStream` | Non-negative median distance to nearest active observers | `warm_up`, defaulting to at least `n_neighbors` | Exactly up to `max_observers` observers with exponentially faded activity |

`KNN` computes exact float64 distances with NumPy. `backend="faiss"` searches
an exact float32 FAISS index instead, which is faster for wide or long windows
and needs `aberrant[faiss]`.

`CellNeighborhoodDetector` scores each query with the distance-based outlier
rule that NETS computes (fewer than `k` window points within `radius`), not the
paper's window-level outlier set; see the
[API reference](../api/models/distance.md). `SDOStream` follows the fixed
observer and exponential-fading structure described in the
[SDOstream paper](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-143.pdf).

### Sketch detectors

| Class | Default score | Built-in warm-up | State policy |
| --- | --- | --- | --- |
| `MStream` | Non-negative `log1p` transform of candidate-inclusive multi-aspect sketch statistics | A bucket-index distance of `warm_up` from the first learned bucket; default `0` | Fixed numeric, categorical, and record sketches |
| `LODA` | Non-negative mean negative log histogram bin probability, which ranks like the negative log density | Exactly `warm_up` learned events | Fixed projections and histograms; bin edges freeze after warm-up, counts optionally decay |
| `RSHash` | `[0, 1]` from low faded hash occupancy | Exactly `warm_up` learned events | Fixed subspaces and count tables with fading |

These are bounded streaming adaptations. `LODA` fixes its projection
count and histogram edges rather than reproducing the original paper's model
selection procedure; `RSHash` adds online standardization and fading
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
`warm_up` even when no event was observed in those buckets.

### Graph-stream detectors

| Class | What it scores | Default score | State policy |
| --- | --- | --- | --- |
| `AnoEdgeL` | Candidate edge membership in maintained local dense submatrices | Non-negative raw score | Fixed higher-order sketches and configured submatrices |
| `ISCONNA` | Edge/endpoint frequency, consecutive-width, and gap patterns | Non-negative raw combined G-test score | Fixed count-min sketches |
| `MIDAS` | Candidate-inclusive edge microcluster statistic; MIDAS-R also scores endpoints | Non-negative raw chi-square-style score | Fixed count-min sketches |
| `StreamSpot` | Change in a graph's signed shingle sketch relative to online centroids | Non-negative distance-plus-novelty score | At most `max_graphs` sketches plus fixed centroids |

`AnoEdgeL`, `ISCONNA`, and `MIDAS` require integer-like source, destination,
and explicit timestamp values. Their timestamps are non-decreasing integer
buckets; equal timestamps represent edges in the same bucket. If
`time_key=None`, each arrival receives the next implicit bucket.

`StreamSpot` additionally requires a graph identifier. Its
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
    warm_up=2,
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

`RollingMatrixProfile` gives an exact nearest-subsequence distance on every
event after warm-up, using only the current event and previously learned
values. It accepts exactly one consistently named finite numeric feature.
Higher scores indicate greater novelty; `match_one` also returns the nearest
reference's absolute, zero-based start index since the last reset.

```python
from aberrant.model.timeseries import RollingMatrixProfile

detector = RollingMatrixProfile(subsequence_length=4, window_size=32)
for value in [0.0, 1.0, 0.0, -1.0] * 20:
    event = {"value": value}
    score, reference_start = detector.match_one(event)
    # score_one(event) returns the same score without the reference index.
    if detector.readiness.ready:
        print(score, reference_start)
    detector.learn_one(event)
```

`window_size` counts the entire comparison window **including the candidate**;
the oldest sample is excluded from scoring when it would fall outside that
window. Learning retains at most `window_size` samples, defaulting to
`16 * subsequence_length`. Expired references cannot remain nearest matches.
Scoring never changes history or locks the feature name; learning works even
without preceding scoring calls.

With subsequence length `m`, the default exclusion radius is `ceil(m / 4)`.
Reference starts must be more than that radius behind the query start, so
overlap is allowed. Set `exclusion_zone=m-1` for nonoverlapping references.
The window must hold at least `m + exclusion_zone + 1` samples. Scoring starts
after `m + exclusion_zone` learned samples, without waiting for the window to
fill; earlier calls return `0.0` and no match. Use `readiness.ready` to distinguish
warm-up zeros from genuine zero distances. Scores belong to the subsequence
ending at the current event, and equal distances choose the earliest retained
reference.

By default, each subsequence is independently z-normalized using population
standard deviation, so scores describe shape rather than absolute level or
amplitude. Normalized distances range from zero to `2 * sqrt(m)`, apart from
floating-point roundoff. Two constant subsequences have distance zero even
at different levels; a constant and a nonconstant have distance `sqrt(m)`.
Use `normalize=False` for ordinary Euclidean distances when level and amplitude
changes matter. Raw distances are not normalized to `[0, 1]`; distances beyond
the floating-point range raise `OverflowError`.

The implementation recomputes an FFT distance profile and directly checks
potential minima. Memory is `O(window_size)`; typical scoring work is
`O(window_size * log(window_size))`, with up to
`O(window_size * subsequence_length)` for numerically ambiguous candidates.
Learning computes only the newest window's statistics. No restart or batch
retraining is needed as history rolls. This API supplies the latest
left-profile score and match; it does not maintain a historical profile or
return global motif/discord rankings. The exclusion and constant-window
conventions follow [STUMPY](https://stumpy.readthedocs.io/en/latest/api.html).

`MultivariateRollingMatrixProfile` extends this scoring contract to aligned
observations with one or more fixed, named channels. At each eligible historical
start it takes the **largest channel distance**, then selects the interval with
the smallest such distance: `min(reference, max(channel, distance))`. Every
channel shares the same reference start. Separate scalar detectors can each
find a familiar shape at different times and miss a new relationship between
channels; requiring a common match can expose that change.

```python
from aberrant.model.timeseries import MultivariateRollingMatrixProfile

detector = MultivariateRollingMatrixProfile(
    subsequence_length=4, window_size=32, exclusion_zone=3
)
for value in [0.0, 1.0, 0.0, -1.0] * 20:
    event = {"sensor_a": value, "sensor_b": 2 * value}
    score, reference_start, channel_distances = detector.explain_one(event)
    if detector.readiness.ready:
        print(score, reference_start, channel_distances)
    detector.learn_one(event)
```

`explain_one` returns a fresh mapping of distances at the selected common match,
in sorted feature-name order. Its maximum equals the score. These distances
describe the mismatch with that interval, not a causal diagnosis or each
channel's independently nearest match. `match_one` returns just the score and
reference index; `score_one` returns just the score. All three are read-only.

The first successful `learn_one` fixes the feature-key set; every subsequent
event must include all channels. Input order is canonicalized, but observations
must already be aligned. Warm-up lasts `m + exclusion_zone` learned events;
`explain_one` returns `(0.0, None, {})` before readiness. The same overlap rules,
earliest-match tie breaking, bounded eviction, and reset behavior as the scalar
model apply. Use `exclusion_zone=m-1` to prohibit overlapping matches.

With `normalize=True`, each channel's query and reference windows are normalized
independently. This preserves shape and relative timing differences while
removing per-channel level and amplitude differences. Constant-window distances
use the same zero/`sqrt(m)` convention as the scalar model. In raw mode, channels
with larger units can dominate the maximum; use comparable units or fixed
scaling when that is inappropriate. Raw scores beyond the floating-point range
raise `OverflowError`. Adding an identical channel does not dilute the score.

For `d` fixed channels, typical scoring costs `O(d * W * log(W))`, with direct
numerical refinement up to `O(d * W * m)` and learning cost `O(d * m)`. State and
working memory are `O(d * W)`. Scoring continues through eviction without
retraining. Both rolling models provide exact nearest distances; DAMP's
early-abandoned scores described below can be approximate.

This is a causal, bounded-window adaptation of the pre-max aggregation in
[Matrix Profile for Anomaly Detection on Multidimensional Time Series](https://arxiv.org/html/2409.09298v1#S4.SS1).
The catalog ID is `multivariate_rolling_matrix_profile`; it is available in the
core installation and in declarative model pipelines. The
[seeded streaming example](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/models/multivariate_rolling_matrix_profile.py)
demonstrates a changed phase relationship and its common-match explanation.

`DAMP` accepts exactly one consistently named scalar feature. It scores the
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

`MovingZScore` and `MovingMAD` accept exactly one feature and compare a value
with the last `window_size` learned values. `MovingZScore` returns
`|x - mean| / std`. `MovingMAD` returns the robust z-score
`|x - median| / (1.4826 * MAD)`, which a few extreme values in the window
barely move. Both return `0.0` until `warm_up` values have been learned, and
`warm_up` defaults to `window_size`.

```python
from aberrant.model.stat import MovingMAD

detector = MovingMAD(window_size=5)
for value in [10.0, 11.0, 9.0, 10.5, 9.5]:
    detector.learn_one({"latency_ms": value})

candidate = {"latency_ms": 25.0}
print(f"robust z-score: {detector.score_one(candidate):.1f}")
detector.learn_one(candidate)
```

`MovingCovariance` and `MovingCorrelationCoefficient` require exactly two
features and score the absolute change of their statistic when the candidate
is added. `MovingMahalanobisDistance`
accepts a fixed multivariate schema and returns **squared** Mahalanobis distance
from the current window mean; it returns `0.0` with fewer than three learned
points.

#### Online ECOD and COPOD

`OnlineECOD` and `OnlineCOPOD` score a sample by how far into the tails of
each feature's reference distribution its values fall, and learn the sample
afterwards. They need no scaler, since only ranks matter, and treat features
as independent, as the batch methods do. Scores are non-negative and `0.0`
until `warm_up` samples have been learned. A tail probability counts the
candidate itself, `(#{r <= x} + 1) / (n + 1)`, so it is never zero and each
feature contributes at most `log(n + 1)`.

| `memory` | Reference | Forgetting | State |
| --- | --- | --- | --- |
| `"sketch"` (default) | t-digest per feature plus a 128-value exact buffer | none, or `half_life` samples per halving | bounded by `compression` |
| `"window"` | exact last `window_size` values | hard, after `window_size` samples | `window_size` values per feature |
| `"hybrid"` | exact window plus a sketch of evicted values (ESOD) | window, plus the sketch's `half_life` | both |

With `memory="window"` the score equals batch ECOD or COPOD fitted on the
window plus the candidate. The hybrid's score is the smaller of its window and
history scores, so a sample must be unusual against both. The history starts
contributing after `warm_up` evictions, even when decay keeps its effective
weight below that count. `tail="pareto"`
fits a generalized Pareto tail beyond the reference's `tail_quantile`
quantile, as SPOT does, so scores keep growing past the observed range instead
of saturating. `aggregation="paper"` (default) follows both papers, which
share one final formula, so the two classes then score identically;
`aggregation="pyod"` reproduces PyOD's differing ECOD and COPOD.
The paper aggregation remains the default for reproducibility. The PyOD
aggregation is available for comparison; a gain on one dataset alone does
not establish a better general default.

```python
import random

from aberrant.model.stat import OnlineECOD

detector = OnlineECOD(half_life=2_000)  # a sample's weight halves every 2,000
for _ in range(500):
    event = {"cpu": random.gauss(0.3, 0.05), "latency_ms": random.gauss(12.0, 1.5)}
    score = detector.score_one(event)  # score first
    detector.learn_one(event)  # then learn

# Per-feature contributions sum to the score; here "cpu" dominates.
print(detector.explain_one({"cpu": 0.95, "latency_ms": 11.5}))
```

### SVM detectors

`AdaptiveOneClassSVM` is a custom, budgeted RBF-kernel
heuristic with running standardization and adaptive gamma. It returns the
negative decision function, so higher values are more anomalous, and bounds
support-vector state with `sv_budget`. It is experimental and does not claim
parity with a published one-class SVM solver.

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

`Autoencoder` requires `aberrant[dl]`. The caller supplies an aberrant
`Architecture`, a PyTorch optimizer, and a reconstruction loss. It has no
built-in warm-up policy; `score_one` returns the supplied criterion's loss and
`learn_one` performs one optimizer step.

### Core utility models

| Class | Contract |
| --- | --- |
| `ThresholdModel` | Stateless binary output: `1.0` when any configured ceiling or floor is violated, otherwise `0.0`. Scalar bounds apply to every supplied feature; mapping bounds ignore unconfigured feature names. |
| `NullModel` | Stateless baseline that always returns `0.0`. |
| `RandomModel` | Uniform `[0, 1)` baseline. Each `score_one` consumes one draw from its model-local generator; `learn_one` is a no-op. |

Models return scores only. To turn a score into an alarm, use a
`ScoreCalibrator` from [`aberrant.calibration`](calibration.md).
