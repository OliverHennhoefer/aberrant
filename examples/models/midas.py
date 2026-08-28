import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.graph import MIDAS

rng = np.random.default_rng(42)
model = MIDAS(
    source_key="src",
    destination_key="dst",
    time_key="t",
    count_min_rows=4,
    count_min_cols=1024,
    warm_up_samples=128,
    use_relational=True,
    normalize_score=False,
    seed=42,
)

labels, scores = [], []
warmup_count = 0

for i in range(10_000):
    bucket = i // 32

    if rng.random() < 0.12:
        label = 1
        if rng.random() < 0.5:
            src = int(rng.integers(1_000, 1_100))
            dst = int(rng.integers(1_100, 1_200))
        else:
            src = int(rng.integers(200, 260))
            dst = int(rng.integers(260, 320))
    else:
        label = 0
        community = int(rng.integers(0, 4))
        start = community * 25
        src_local = int(rng.integers(0, 25))
        dst_local = (src_local + 1) % 25 if rng.random() < 0.9 else (src_local + 2) % 25
        src = start + src_local
        dst = start + dst_local

    sample = {"src": float(src), "dst": float(dst), "t": float(bucket)}

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
