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

from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

rng = np.random.default_rng(7)
normal = rng.normal(loc=0.0, scale=1.0, size=(400, 2))
anomalies = rng.normal(loc=5.0, scale=0.5, size=(16, 2))
stream = np.vstack([normal, anomalies])

detector = StandardScaler() | OnlineIsolationForest(
    num_trees=25,
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

`QuantileThreshold` maintains a sliding score window. Call its `score_one`
before `learn_one` so the candidate score is compared with prior scores only:

```python
from aberrant.model import QuantileThreshold

score_stream = [0.10, 0.12, 0.09, 0.15, 0.11, 0.13, 0.92, 0.14]
threshold = QuantileThreshold(quantile=0.8, window_size=5)

for score in score_stream:
    threshold_input = {"score": score}
    thresholded_score = threshold.score_one(threshold_input)
    is_anomaly = thresholded_score == 1.0
    threshold.learn_one(threshold_input)
    print(
        f"score={score:.2f}, "
        f"thresholded_score={thresholded_score:.2f}, "
        f"anomaly={is_anomaly}"
    )
```

During its initial window, `QuantileThreshold.score_one` returns `0.0`. Once
ready, it returns `1.0` at or above the learned quantile and a normalized value
below `1.0` otherwise. It is not a calibrated anomaly probability.

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

    Download, validate, cache, and batch registered NPZ datasets.

    [Streaming datasets](user_guide/streaming.md)

-   **Monitor change**

    Feed a scalar signal into ADWIN, KSWIN, or Page-Hinkley and own the response
    policy explicitly.

    [Drift detection](user_guide/drift.md)

</div>
