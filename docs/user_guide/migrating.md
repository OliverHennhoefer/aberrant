# Coming from river or PySAD

aberrant uses river's event-by-event contract, so most river code ports by
changing imports. PySAD code needs one extra step: events become
`dict[str, float]` instead of NumPy rows.

## The loop

=== "aberrant"

    ```python
    from aberrant.model.iforest import HalfSpaceTrees
    from aberrant.transform import MinMaxScaler

    detector = MinMaxScaler() | HalfSpaceTrees(seed=42)

    for event in [{"x": 0.1, "y": 0.4}, {"x": 0.2, "y": 0.5}, {"x": 0.9, "y": 0.1}]:
        detector.learn_one(event)
        score = detector.score_one(event)
    ```

=== "river"

    ```python
    from river import anomaly, preprocessing

    detector = preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(seed=42)

    for event in [{"x": 0.1, "y": 0.4}, {"x": 0.2, "y": 0.5}, {"x": 0.9, "y": 0.1}]:
        detector.learn_one(event)
        score = detector.score_one(event)
    ```

=== "PySAD"

    ```python
    import numpy as np
    from pysad.models import HalfSpaceTrees

    X = np.array([[0.1, 0.4], [0.2, 0.5], [0.9, 0.1]])
    detector = HalfSpaceTrees(feature_mins=X.min(axis=0), feature_maxes=X.max(axis=0))

    for row in X:
        score = detector.fit_score_partial(row)
    ```

Higher scores mean more anomalous in all three libraries. aberrant keeps
`score_one` and `learn_one` separate: for prequential (test-then-train)
evaluation, call `score_one` first, which is what PySAD's
`fit_score_partial` does internally. Note that a scaler cannot transform a
feature before it has learned one event; see the
[transformer guide](transformers.md).

## Rows to events

PySAD and most NumPy pipelines produce rows. Name the columns once and turn
each row into a mapping:

```python
import numpy as np

from aberrant.model.sketch import StreamingLODA

X = np.random.default_rng(0).normal(size=(500, 3))
names = ["temperature", "pressure", "vibration"]

detector = StreamingLODA(seed=0)
scores = []
for row in X:
    event = dict(zip(names, row.tolist(), strict=True))
    scores.append(detector.score_one(event))
    detector.learn_one(event)
```

For a labelled benchmark, `PrequentialEvaluator` replaces PySAD's streamers
and metric wrappers; see [Evaluation](evaluation.md).

## Equivalent components

The table maps components that implement the same published method. Defaults
and score scales can still differ, so compare results rather than assuming
identical numbers.

| river | PySAD | aberrant |
| --- | --- | --- |
| `anomaly.HalfSpaceTrees` | `models.HalfSpaceTrees` | `model.iforest.HalfSpaceTrees` (expects features in `[0, 1]`; add a `MinMaxScaler`) |
| `anomaly.LODA` | `models.LODA` | `model.sketch.StreamingLODA` |
| `anomaly.LocalOutlierFactor` | — | `model.distance.LocalOutlierFactor` |
| — | `models.RSHash` | `model.sketch.StreamingRSHash` |
| — | `models.RobustRandomCutForest` | `model.iforest.RandomCutForest` |
| — | `models.IForestASD` | `model.iforest.ASDIsolationForest` |
| `anomaly.QuantileFilter` | `transform.ConformalProbabilityCalibrator` (calibrated scores) | `model.QuantileThreshold` (adaptive alert threshold); for conformal p-values see [nonconform](https://github.com/OliverHennhoefer/nonconform) |
| `drift.ADWIN`, `drift.KSWIN`, `drift.PageHinkley` | — | `drift.ADWIN`, `drift.KSWIN`, `drift.PageHinkley` (same `update` / `drift_detected` interface) |
| `preprocessing.StandardScaler`, `preprocessing.MinMaxScaler` | — | `transform.StandardScaler`, `transform.MinMaxScaler` |
| `compose.Pipeline` or `\|` | — | `\|` (see [Pipelines](pipelines.md)) |
| `evaluate` with a metric | `evaluation.AUROCMetric`, `utils.ArrayStreamer` | `evaluate.PrequentialEvaluator` |

aberrant also covers families that neither library ships: graph-edge
detectors (`MIDAS`, `AnoEdgeL`, `ISCONNA`), time-series discords
(`RollingMatrixProfile`, `XLagDAMP`), online SVMs and autoencoders. The
[model guide](models.md) lists them all.
