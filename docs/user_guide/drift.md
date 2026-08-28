# Drift Detection

ABERRANT's drift detectors monitor one finite scalar at a time. They do not
implement the anomaly-model protocol: call `update(value)`, then inspect
`drift_detected` for the observation just processed.

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
| `ADWIN` | Difference in means across candidate splits of an adaptive window | Exponential-histogram buckets; checks every `clock` observations after `grace_period` | Either direction |
| `KSWIN` | Two-sample Kolmogorov-Smirnov difference between the latest `stat_size` values and a seeded random sample from the earlier fixed window | Fixed `window_size`; tests whenever the window is full | Any distributional difference exposed by the KS statistic |
| `PageHinkley` | Cumulative deviation from a running mean | Constant-size cumulative statistics after `min_instances` | Configurable `"up"`, `"down"`, or `"both"` |

All three reject non-numeric and non-finite observations. `update` returns the
detector itself, while `reset()` clears its learned state and detection count.

Implementation-specific details matter:

- ADWIN checks compressed bucket boundaries rather than retaining every raw
  observation. A smaller `delta` is more conservative.
- KSWIN requires `stat_size < window_size / 2`. ABERRANT uses SciPy's
  asymptotic two-sample KS test and requires both `p_value <= alpha` and a KS
  statistic greater than `0.1`. `seed` controls its historical sampling.
- Page-Hinkley `delta` is the tolerated change magnitude, `threshold` controls
  signaling, and `alpha` fades cumulative sums. Values closer to `1` retain
  more history.

## Complete change-point example

The stream below shifts from zero to one. The parameters are intentionally
sensitive so the short demonstration detects the increase deterministically.

```python
from aberrant.drift import PageHinkley

detector = PageHinkley(
    min_instances=10,
    delta=0.0,
    threshold=4.0,
    alpha=1.0,
    mode="up",
)
stream = [0.0] * 30 + [1.0] * 30

detected_at: int | None = None
for index, value in enumerate(stream):
    detector.update(value)
    if detector.drift_detected:
        detected_at = index
        print(f"upward change detected at index {index}")
        break

assert detected_at is not None
```

`drift_detected` describes only the **last** update. Read and handle it in the
same iteration. ADWIN and Page-Hinkley soft-reset on the update after a signal;
KSWIN clears its window immediately when it signals. Their `n_detections`
counters remain available until an explicit `reset()`.

## Monitor a model score

Keep score production, drift monitoring, and model learning in a deliberate
order:

```python
from aberrant.drift import PageHinkley
from aberrant.model.stat import MovingAverage

model = MovingAverage(window_size=5, key="value")
monitor = PageHinkley(
    min_instances=5,
    delta=0.0,
    threshold=0.25,
    mode="up",
)
events = [{"value": value} for value in [0.0] * 8 + [1.0] * 8]

for index, event in enumerate(events):
    score = model.score_one(event)
    monitor.update(score)

    if monitor.drift_detected:
        print(f"score distribution changed at event {index}")

    model.learn_one(event)
```

This example monitors the moving-mean change score, not the raw feature. A
different signal can produce a different detection time and operational
meaning.

## Own the response policy

A drift flag does not update, reset, or replace an anomaly model. Define the
response separately and test it as application logic. Possible responses
include:

- open an investigation without modifying the detector;
- recalibrate an external threshold on a controlled recent reference period;
- create a fresh model and warm it alongside the current model;
- call a documented model-specific `reset()` where one exists;
- roll back or quarantine an upstream data change.

Do not assume every model exposes `reset()`, and do not automatically relearn
from an interval merely because it triggered drift; that interval can contain
the incident being detected.

## Primary sources

- Bifet and Gavaldà,
  [*Learning from Time-Changing Data with Adaptive Windowing*](https://doi.org/10.1137/1.9781611972771.42)
  (ADWIN).
- Raab, Heusinger, and Schleif,
  [*Reactive Soft Prototype Computing for Concept Drift Streams*](https://doi.org/10.1016/j.neucom.2019.11.111)
  (KSWIN).
- Page,
  [*Continuous Inspection Schemes*](https://doi.org/10.1093/biomet/41.1-2.100)
  (the sequential cumulative-sum foundation of Page-Hinkley).

See [Drift API](../api/drift.md) for exact parameters and observable
properties.
