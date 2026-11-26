# Quickstart

This page starts with an offline, base-install example. It makes the event
lifecycle, warm-up policy, and score ordering explicit rather than hiding them
inside a helper.

## Score, then learn

At event *t*, prequential (test-then-train) processing uses the state learned
through event *t - 1*:

1. Construct the event mapping.
2. Call `score_one(event)` without learning the candidate.
3. Record or act on that score.
4. Call `learn_one(event)` to update state for the next event.

The first 64 observations below are an application-defined warm-up and are not
included in the ranked results.

```python
import numpy as np

from aberrant.model.forest import OnlineIsolationForest
from aberrant.transform import StandardScaler

rng = np.random.default_rng(7)
normal = rng.normal(loc=0.0, scale=1.0, size=(400, 2))
anomalies = rng.normal(loc=5.0, scale=0.5, size=(16, 2))
stream = np.vstack([normal, anomalies])

detector = StandardScaler() | OnlineIsolationForest(
    n_trees=25,
    max_leaf_samples=32,
    window_size=256,
    seed=7,
)

warm_up = 64
scores: list[tuple[int, float]] = []

for index, values in enumerate(stream):
    event = {"x": float(values[0]), "y": float(values[1])}

    if index >= warm_up:
        scores.append((index, detector.score_one(event)))

    detector.learn_one(event)

for index, score in sorted(scores, key=lambda item: item[1], reverse=True)[:8]:
    print(f"event={index}, anomaly_score={score:.3f}")
```

The `StandardScaler` learns the event before producing the value used by
`OnlineIsolationForest.learn_one`. Scoring does not call either component's
learning method and uses scaler state from prior learned events. See
[Pipelines](user_guide/pipelines.md) for the complete ordering contract.

## Convert scores into a decision signal

An anomaly score is evidence, not a universal probability. A threshold is an
application policy that must be calibrated on the selected detector's score
distribution.

A `ScoreCalibrator` learns that threshold online. Ask for the threshold before
reporting the score, so each score is compared with earlier scores only:

```python
from aberrant.calibration import ScoreCalibrator

score_stream = [0.10, 0.12, 0.09, 0.15, 0.11, 0.13, 0.92, 0.14]
threshold = ScoreCalibrator("split", alpha=0.2, window_size=5)

for t, score in enumerate(score_stream):
    limit = threshold.predict(t)
    threshold.update(t, score)
    alarm = t >= 5 and score > limit  # let the window fill first
    print(f"score={score:.2f}, threshold={limit:.2f}, alarm={alarm}")
```

`split` uses the conformal quantile of the last `window` scores. Until the
window holds enough scores for that quantile, the threshold is only the
largest absolute score seen so far, so the example ignores alarms while the
window fills. The
[calibration guide](user_guide/calibration.md) compares the fifteen methods.

## Continue from here

<div class="grid cards" markdown>

-   **Select a model**

    Compare score scales, state bounds, warm-up rules, and input contracts.

    [Models](user_guide/models.md)

-   **Evaluate honestly**

    Compute average precision and ROC AUC without training on the event being
    evaluated.

    [Evaluation](user_guide/evaluation.md)

-   **Use benchmark streams**

    Replay labelled benchmarks, verified before they are read.

    [Benchmark datasets](user_guide/datasets.md)

-   **Monitor change**

    Feed a scalar signal into ADWIN, KSWIN, or Page-Hinkley and own the response
    policy explicitly.

    [Drift detection](user_guide/drift.md)

</div>
