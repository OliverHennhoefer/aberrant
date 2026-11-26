# Drift Detection

aberrant's drift detectors monitor one finite scalar at a time. They do not
implement the anomaly-model protocol: `update(value)` returns whether that
observation signalled drift, and `drift_detected` keeps the answer until the
next update.

## Anomaly detection and drift detection answer different questions

- An anomaly detector asks whether one event is unusual relative to its current
  reference state.
- A drift detector asks whether a sequence of monitored observations provides
  evidence of change.

The monitored scalar can be an anomaly score, prediction error, residual,
feature value, alert rate, or another operational statistic. The choice defines
what a drift signal means. Monitoring a model score can reveal a changed score
distribution, but cannot by itself distinguish data drift, model degradation,
upstream schema changes, or a genuine rise in anomalies.

## Detector contracts

| Detector | Change signal | State and test cadence | Direction |
| --- | --- | --- | --- |
| `ADWIN` | Difference in means across candidate splits of an adaptive window | Exponential-histogram buckets; checks every `clock` observations after `warm_up` | Either direction |
| `CUSUM` | Standardized cumulative sums against a reference mean and deviation | Constant-size sums after a `warm_up` that estimates the reference, unless `target` and `scale` fix it | Configurable `"up"`, `"down"`, or `"both"` |
| `KSWIN` | Two-sample Kolmogorov-Smirnov difference between the latest `stat_size` values and a seeded random sample from the earlier fixed window | Fixed `window_size`; tests whenever the window is full | Any distributional difference exposed by the KS statistic |
| `PageHinkley` | Cumulative deviation from a running mean | Constant-size cumulative statistics after `warm_up` | Configurable `"up"`, `"down"`, or `"both"` |

All four reject non-numeric and non-finite observations. `update` returns
whether the observation signalled drift, and `reset()` clears the learned state
and detection count.

Implementation-specific details matter:

- ADWIN checks compressed bucket boundaries rather than retaining every raw
  observation. A smaller `delta` is more conservative. Its default stationary
  history uses O(log N) buckets. Set `max_window_size` for a hard history bound;
  oldest whole buckets are removed, so the retained width can fall below the
  configured maximum. The bound must be at least `2 * min_window_length` and
  changes the history available for detecting slow drift.
- KSWIN requires `stat_size < window_size / 2`. aberrant uses the exact
  two-sample KS p-value, as `scipy.stats.ks_2samp(method="exact")` computes
  it, and requires both `p_value <= alpha` and a KS statistic greater than
  `0.1`. `seed` controls its historical sampling.
- CUSUM standardizes each observation, so `k` and `h` are in standard
  deviations and mean the same on every signal. `k` is half the shift it
  detects fastest and `h` sets the false-alarm rate. On normal observations
  the defaults `k = 0.5`, `h = 10` and `warm_up = 100` pass about 20,000
  observations between false alarms and find a one-deviation shift after
  about 20. A shorter `warm_up` estimates the deviation less reliably and
  alarms more often.
  After a signal it estimates the reference again, so the new level becomes
  normal. `direction` and `onset` tell which way the mean moved and how many
  observations ago the change most likely began.
- Page-Hinkley `delta` is the tolerated change magnitude, `threshold` controls
  signaling, and `alpha` fades cumulative sums. Values closer to `1` retain
  more history.

## Complete change-point example

The stream below shifts from zero to one. The parameters are intentionally
sensitive so the short demonstration detects the increase deterministically.

```python
from aberrant.drift import PageHinkley

detector = PageHinkley(
    warm_up=10,
    delta=0.0,
    threshold=4.0,
    alpha=1.0,
    mode="up",
)
stream = [0.0] * 30 + [1.0] * 30

detected_at: int | None = None
for index, value in enumerate(stream):
    if detector.update(value):
        detected_at = index
        print(f"upward change detected at index {index}")
        break

assert detected_at is not None
```

`drift_detected` describes only the **last** update. Read and handle it in the
same iteration. ADWIN, CUSUM and Page-Hinkley soft-reset on the update after a signal;
KSWIN keeps only its latest `stat_size` values when it signals, so its next
test comes `window_size - stat_size` observations later. Their `n_detections`
counters remain available until an explicit `reset()`.

## Monitor a model score

Keep score production, drift monitoring, and model learning in a deliberate
order:

```python
from aberrant.drift import PageHinkley
from aberrant.model.stat import MovingZScore

model = MovingZScore(window_size=5)
monitor = PageHinkley(
    warm_up=5,
    delta=0.0,
    threshold=0.25,
    mode="up",
)
events = [{"value": value} for value in [0.0] * 8 + [1.0] * 8]

for index, event in enumerate(events):
    score = model.score_one(event)
    if monitor.update(score):
        print(f"score distribution changed at event {index}")

    model.learn_one(event)
```

This example monitors the z-score, not the raw feature. A
different signal can produce a different detection time and operational
meaning.

## Own the response policy

A drift flag does not update, reset, or replace an anomaly model. Define the
response separately and test it as application logic. Possible responses
include:

- open an investigation without modifying the detector;
- recalibrate an external threshold on a controlled recent reference period;
- create a fresh model and warm it alongside the current model;
- call the model's `reset()`, or warm `model.clone()` alongside it;
- roll back or quarantine an upstream data change.

Do not automatically relearn from an interval merely because it triggered
drift; that interval can contain the incident being detected.

## Primary sources

- Bifet and Gavaldà,
  [*Learning from Time-Changing Data with Adaptive Windowing*](https://doi.org/10.1137/1.9781611972771.42)
  (ADWIN).
- Raab, Heusinger, and Schleif,
  [*Reactive Soft Prototype Computing for Concept Drift Streams*](https://doi.org/10.1016/j.neucom.2019.11.111)
  (KSWIN).
- Page,
  [*Continuous Inspection Schemes*](https://doi.org/10.1093/biomet/41.1-2.100)
  (the cumulative sum behind CUSUM and Page-Hinkley).
- Montgomery, *Introduction to Statistical Quality Control*, chapter 9 (the
  tabular CUSUM and its run lengths).

See [Drift API](../api/drift.md) for exact parameters and observable
properties.
