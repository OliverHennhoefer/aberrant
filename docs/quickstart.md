# Quickstart

This page shows a complete streaming loop with a model and a dataset stream.

## 1. Build a model

```python
from aberrant.model.iforest import OnlineIsolationForest

model = OnlineIsolationForest(
    num_trees=50,
    max_leaf_samples=32,
    window_size=512,
)
```

## 2. Load a stream

```python
from aberrant.stream.dataset import Dataset, load

dataset = load(Dataset.SHUTTLE)
```

## 3. Online train + score

```python
from sklearn.metrics import average_precision_score

labels, scores = [], []

for i, (x, y) in enumerate(dataset.stream()):
    # Warm up on normal points only.
    if i < 2000:
        if y == 0:
            model.learn_one(x)
        continue

    score = model.score_one(x)
    model.learn_one(x)
    labels.append(y)
    scores.append(score)

print("PR-AUC:", average_precision_score(labels, scores))
```

## 4. Add thresholding

`OnlineIsolationForest` returns scores in `[0, 1]`. You can add a threshold model:

```python
from aberrant.model import ThresholdModel

threshold = ThresholdModel(ceiling={"score": 0.8})

for x in stream_of_points:
    score = model.score_one(x)
    is_anomaly = threshold.score_one({"score": score}) == 1.0
```
