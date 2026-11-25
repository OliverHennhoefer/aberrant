from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.distance.sdostream import SDOStream
from aberrant.stream import Dataset, load
from aberrant.transform import StandardScaler

model = SDOStream(
    max_observers=128,
    time_scale=256.0,
    inactive_quantile=0.3,
    n_neighbors=8,
    distance="euclidean",
    warm_up=12,
    seed=42,
)
pipeline = StandardScaler() | model

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

warmup_count = 0
for x, y in dataset.stream():
    if warmup_count < 2_000:
        if y == 0:
            pipeline.learn_one(x)
            warmup_count += 1
        continue

    score = pipeline.score_one(x)
    pipeline.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
