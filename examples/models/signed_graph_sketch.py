import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.graph import SignedGraphSketchDetector

rng = np.random.default_rng(42)
model = SignedGraphSketchDetector(
    graph_key="graph",
    source_key="src",
    destination_key="dst",
    edge_type_key="etype",
    time_key="t",
    sketch_dim=256,
    shingle_size=2,
    num_clusters=8,
    max_graphs=256,
    warm_up_graphs=16,
    normalize_score=False,
    seed=42,
)

labels, scores = [], []
warmup_count = 0

for i in range(10_000):
    graph_id = int(rng.integers(0, 40))
    bucket = i // 8

    if rng.random() < 0.15:
        label = 1
        src = int(rng.integers(1_000, 1_250))
        dst = int(rng.integers(2_000, 2_250))
        etype = 1
    else:
        label = 0
        src = graph_id * 10
        dst = graph_id * 10 + 1
        etype = 0

    sample = {
        "graph": float(graph_id),
        "src": float(src),
        "dst": float(dst),
        "etype": float(etype),
        "t": float(bucket),
    }

    if warmup_count < 2_000:
        if label == 0:
            model.learn_one(sample)
            warmup_count += 1
        continue

    score = model.score_one(sample)
    model.learn_one(sample)
    labels.append(label)
    scores.append(score)

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
