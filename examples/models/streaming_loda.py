from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.sketch import StreamingLODA
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing import StandardScaler

model = StandardScaler() | StreamingLODA(
    n_projections=64,
    n_bins=24,
    sparsity=0.3,
    warm_up_samples=256,
    decay=1.0,
    time_key=None,
    pseudocount=0.5,
    predict_threshold=0.75,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

warmup_count = 0
for x, y in dataset.stream():
    if warmup_count < 2_000:
        if y == 0:
            model.learn_one(x)
            warmup_count += 1
        continue

    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
