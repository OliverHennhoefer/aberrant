from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.datasets import Dataset, load
from aberrant.model.sketch import RSHash

model = RSHash(
    n_members=24,
    n_hashes=4,
    sketch_width=512,
    subspace_size=3,
    decay=0.01,
    warm_up=128,
    time_key="t",
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

warmup_count = 0
for i, (x, y) in enumerate(dataset):
    sample = dict(x)
    sample["t"] = float(i)

    if warmup_count < 2_000:
        if y == 0:
            model.learn_one(sample)
            warmup_count += 1
        continue

    score = model.score_one(sample)
    model.learn_one(sample)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
