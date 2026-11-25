from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.datasets import Dataset, load
from aberrant.model.sketch import LODA
from aberrant.transform import StandardScaler

model = StandardScaler() | LODA(
    n_projections=64,
    n_bins=24,
    sparsity=0.3,
    warm_up=256,
    decay=1.0,
    time_key=None,
    pseudocount=0.5,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

warmup_count = 0
for x, y in dataset:
    if warmup_count < 2_000:
        if y == 0:
            model.learn_one(x)
            warmup_count += 1
        continue

    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
