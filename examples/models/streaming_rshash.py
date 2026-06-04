from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.sketch import StreamingRSHash
from aberrant.stream.dataset import Dataset, load

model = StreamingRSHash(
    components_num=24,
    hash_num=4,
    bins=512,
    subspace_size=3,
    decay=0.01,
    warm_up_samples=128,
    time_key="t",
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

warmup_count = 0
for i, (x, y) in enumerate(dataset.stream()):
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

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
