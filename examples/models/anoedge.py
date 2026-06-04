from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.graph import AnoEdgeL
from aberrant.stream.dataset import Dataset, load

model = AnoEdgeL(
    source_key="src",
    destination_key="dst",
    time_key="t",
    # Compact sketch works well for this mapped tabular stream.
    count_min_rows=128,
    count_min_cols=128,
    num_hashes=4,
    num_dense_submatrices=1,
    time_decay_factor=1.0,
    warm_up_samples=0,
    normalize_score=False,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)
warmup_count = 0

for i, (x, y) in enumerate(dataset.stream()):
    # Shuttle is tabular, so we build synthetic "edges" from two discrete features.
    sample = {"src": x["feature_0"], "dst": x["feature_8"], "t": float(i // 32)}
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
