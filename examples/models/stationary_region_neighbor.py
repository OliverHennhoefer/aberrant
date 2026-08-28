from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.distance import StationaryRegionNeighborDetector
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing import StandardScaler

model = StationaryRegionNeighborDetector(
    k=40,
    radius=1.5,
    window_size=1024,
    slide_size=128,
    skip_threshold=0.1,
    warm_up_slides=4,
    predict_threshold=0.5,
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
