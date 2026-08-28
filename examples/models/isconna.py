import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.graph import ISCONNA

rng = np.random.default_rng(42)
model = ISCONNA(
    source_key="src",
    destination_key="dst",
    time_key="t",
    count_min_rows=8,
    count_min_cols=1024,
    time_decay_factor=0.5,
    warm_up_samples=128,
    normalize_score=False,
    seed=42,
)

labels, scores = [], []
warmup_count = 0

for t in range(10_000):
    if rng.random() < 0.12:
        label = 1
        if rng.random() < 0.5:
            src = int(rng.integers(1_000, 1_100))
            dst = int(rng.integers(1_100, 1_200))
        else:
            src_community = int(rng.integers(0, 4))
            dst_community = (src_community + int(rng.integers(1, 4))) % 4
            src = src_community * 50 + int(rng.integers(0, 50))
            dst = dst_community * 50 + int(rng.integers(0, 50))
    else:
        label = 0
        community = int(rng.integers(0, 4))
        start = community * 50
        src_local = int(rng.integers(0, 50))
        if rng.random() < 0.85:
            dst_local = (src_local + 1) % 50
        else:
            dst_local = (src_local + 2) % 50
        src = start + src_local
        dst = start + dst_local

    sample = {"src": float(src), "dst": float(dst), "t": float(t)}

    if warmup_count < 2_000:
        if label == 0:
            model.learn_one(sample)
            warmup_count += 1
        continue

    score = model.score_one(sample)
    model.learn_one(sample)
    labels.append(label)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
