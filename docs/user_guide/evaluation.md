# Evaluation

Streaming evaluation must preserve time order and the distinction between a
score and an alert decision. The recommended baseline is prequential
(test-then-train) evaluation: score event *t* with state learned through
*t - 1*, then learn event *t*.

Install the metric dependency with `aberrant[eval]`:

```bash
python -m pip install "aberrant[eval]"
```

## Stateful evaluator

`PrequentialEvaluator` owns the score-before-learn loop. Pass a structural model
or a `DetectorConfig`; configurations construct fresh models and record the
normalized configuration, including seeds, and its fingerprint in each result.

```python
from aberrant.catalog import ComponentConfig, DetectorConfig
from aberrant.evaluate import PrequentialEvaluator

config = DetectorConfig(
    model=ComponentConfig("online_isolation_forest", {"seed": 42}),
    transformers=(ComponentConfig("standard_scaler"),),
)
evaluator = PrequentialEvaluator(
    config,
    warmup=64,
    metrics=True,
    metric_window_size=1000,
)

# Substitute any ordered iterable of (feature dictionary, binary label).
events = [({"temperature": float(i % 7)}, int(i == 79)) for i in range(100)]
for record in evaluator.iter_evaluate(events):
    if record.score is not None:
        print(record.index, record.label, record.score)

result = evaluator.result()
print(result.average_precision, result.roc_auc, result.config_fingerprint)
```

Every event before the model's `readiness` is ready is excluded as
`"not_ready"`, whatever the model counts: events, buckets, observers or active
graphs, and for a pipeline the readiness of each stage. Readiness is read from
the prior state, before the candidate is scored. An explicit `warmup` excludes
that many further **successful learning calls** first, as `"warmup"`, for a
calibration period on top of the model's own. Excluded events have
`score=None`, and their labels do not enter metrics.

Labels must be binary real numeric values (normal `0`, anomaly `1`), booleans, or
`None` for unknown. Real NumPy scalars are accepted; complex labels are rejected
before scoring or learning. Labels are supplied only to
metrics and records; the model receives only feature dictionaries. The default
policy learns every event, including evaluated anomalies. An optional
`learn_filter(event, score)` can skip learning without using labels. Its score
is `None` during exclusion, and its descriptive `learn_policy_name` is recorded.
The learning callback must return a Python boolean.

Metrics are disabled by default, so the base install can still produce traces,
counters and timings. `metrics=True` loads scikit-learn from the `eval` extra.
The default metric window holds the latest 1000 **labeled, scored events** and
results report its size and prevalence separately from cumulative prevalence.
Set `metric_window_size=None` explicitly for exact full-stream ranking metrics;
this retains all labeled scores and uses `O(n)` memory. A finite window uses
`O(window_size)` retention. AP is `None` when no positive labels are present;
ROC AUC is `None` unless both classes are present. Raw trace scores preserve the
model's scale, and metrics rank higher scores as more anomalous, as every model
scores.

Records are yielded as events are processed and never retained by the evaluator.
Write them to a file or another sink when a full trace is needed. Results include
separate cumulative score and learn call durations in nanoseconds; divide by
`n_scored` and `n_learned` respectively for means. Timings exclude validation,
callback and metric work. Peak process memory and tail latency require external
benchmark instrumentation. Computing a result evaluates metrics on the retained
window and does not call the model.

`update`, `iter_evaluate`, and `evaluate` continue the same evaluator and model
state; `evaluate` consumes a stream and returns a cumulative result. Create a
fresh evaluator for each comparison. Existing models are never reset or cloned,
and datasets are never reordered. Record dataset identity/order and callback
implementations alongside the result; a config fingerprint identifies model
construction, not the entire experiment. Model errors propagate without
advancing evaluator statistics, but model rollback depends on the model itself.
Do not retry a failed update unless that model's state is known to be recoverable.

The [seeded example](https://github.com/OliverHennhoefer/aberrant/blob/main/examples/prequential_evaluation.py)
compares fresh models using the same ordered stream and learning policy. See
the [Evaluation API](../api/evaluate.md) for signatures and result fields.

## Complete offline evaluation

This program uses a clean synthetic warm-up, then evaluates a chronologically
fixed mixture. Labels are recorded for metrics but never passed to the
unsupervised detector.

```python
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.forest import OnlineIsolationForest
from aberrant.transform import StandardScaler

rng = np.random.default_rng(42)
warm_up_values = rng.normal(size=(128, 2))

normal_values = rng.normal(size=(300, 2))
anomaly_values = rng.normal(loc=4.0, scale=0.6, size=(30, 2))
evaluation_values = np.vstack([normal_values, anomaly_values])
evaluation_labels = np.concatenate(
    [
        np.zeros(len(normal_values), dtype=np.int64),
        np.ones(len(anomaly_values), dtype=np.int64),
    ]
)

order = rng.permutation(len(evaluation_values))
evaluation_values = evaluation_values[order]
evaluation_labels = evaluation_labels[order]

detector = StandardScaler() | OnlineIsolationForest(
    n_trees=25,
    window_size=256,
    seed=42,
)

for values in warm_up_values:
    detector.learn_one({"x": float(values[0]), "y": float(values[1])})

labels: list[int] = []
scores: list[float] = []
for values, label in zip(evaluation_values, evaluation_labels, strict=True):
    event = {"x": float(values[0]), "y": float(values[1])}

    scores.append(detector.score_one(event))
    labels.append(int(label))
    detector.learn_one(event)

prevalence = float(np.mean(labels))
average_precision = average_precision_score(labels, scores)
roc_auc = roc_auc_score(labels, scores)

print(f"Anomaly prevalence: {prevalence:.3f}")
print(f"Average precision: {average_precision:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")
```

The permutation is created once before streaming and is seeded, so the program
defines one reproducible arrival order. Do not reshuffle between models being
compared.

## Use metric names precisely

**Average precision (AP)** summarizes the precision-recall curve as a weighted
mean of precision values, with each increase in recall supplying the weight.
That is what scikit-learn's
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
computes. It is not the trapezoidal area under an interpolated precision-recall
curve, so label it *average precision*, not “PR-AUC.”

**ROC AUC** summarizes the true-positive-rate/false-positive-rate ranking curve.
Use scikit-learn's
[`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
on continuous anomaly scores, with anomaly label `1` and higher scores meaning
more anomalous.

For rare anomalies, report AP or a precision-recall curve because precision
exposes the false-positive burden among flagged events. ROC AUC remains useful
as a ranking measure but can look favorable in operationally poor regions of a
highly imbalanced problem. Saito and Rehmsmeier analyze this distinction in
[*The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*](https://doi.org/10.1371/journal.pone.0118432).

Record anomaly prevalence alongside AP: precision, and therefore AP, depends on
class prevalence. Neither AP nor ROC AUC evaluates alert volume, delay, or the
cost of a false alert.

## Anomaly ranges

AP and ROC AUC score each event on its own. In a time series an anomaly is
usually a range: a long range found late outweighs a short one found on time,
and a score that rises one step before the range counts as a false positive.
`vus` measures the volume under the range-based ROC and PR surfaces of
Paparrizos et al. (*Volume Under the Surface*, VLDB 2022). Each labeled range
counts as one event, scores within a soft margin around it get partial
credit, and the metric averages over every margin from 0 to `window`.

```python
from aberrant.evaluate import vus

labels = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
scores = [0.1, 0.9, 0.8, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.1]
result = vus(labels, scores, window=4)
print(result.pr, result.roc)
```

It follows `RangeAUC_volume_opt` of the reference package `vus` 0.0.7,
threshold grid included, and needs NumPy only. VUS-PR is the measure
Liu and Paparrizos recommend for comparing time-series detectors (*The
Elephant in the Room*, NeurIPS 2024). Take `window` near one
period of the series, and report it.

## Warm-up and contamination

Warm-up is part of the experiment specification. Report:

- how many events were excluded from metrics;
- whether warm-up data was known to be normal, unlabeled, or potentially
  contaminated;
- whether preprocessing learned during warm-up;
- the exact point at which each model's internal score became ready;
- whether evaluated anomalies were subsequently learned.

Learning every evaluated event is a valid adaptive prequential protocol, but
anomalies can then contaminate the future reference state. Learning only
label-confirmed normal events answers a different question and assumes labels
are available at update time. Compare those policies explicitly rather than
silently filtering by future labels.

## Threshold-dependent evaluation

AP and ROC AUC evaluate ranking without selecting a threshold. An alerting
system also needs threshold-dependent measures chosen for its costs, such as:

- precision, recall, false alerts per time unit, and alert rate;
- detection delay for labeled anomaly intervals;
- fraction of incidents detected within a service-level window;
- alert grouping and suppression behavior.

Choose a fixed threshold on an earlier calibration segment, or update an online
threshold using scores available before the candidate. Do not search the
evaluated labels for the threshold that maximizes the reported result; that is
evaluation leakage.

## Reproducible comparisons

For every compared model:

1. preserve the same event order and feature preparation;
2. use the same warm-up and learn/skip policy;
3. set every exposed seed and repeat stochastic models across multiple seeds;
4. report constructor parameters and package version;
5. measure update latency, scoring latency, peak memory, and state growth in
   addition to predictive metrics;
6. retain the full score sequence so temporal failures can be inspected.

Prequential evaluation and its alternatives are discussed by Gama, Sebastião,
and Rodrigues in
[*Issues in Evaluation of Stream Learning Algorithms*](https://doi.org/10.1145/1557019.1557060).
