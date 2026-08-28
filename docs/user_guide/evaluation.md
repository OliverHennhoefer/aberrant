# Evaluation

Streaming evaluation must preserve time order and the distinction between a
score and an alert decision. The recommended baseline is prequential
(test-then-train) evaluation: score event *t* with state learned through
*t - 1*, then learn event *t*.

Install the metric dependency with `aberrant[eval]`:

```bash
python -m pip install "aberrant[eval]"
```

## Complete offline evaluation

This program uses a clean synthetic warm-up, then evaluates a chronologically
fixed mixture. Labels are recorded for metrics but never passed to the
unsupervised detector.

```python
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest import OnlineIsolationForest
from aberrant.transform.preprocessing import StandardScaler

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
    num_trees=25,
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
