from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest import RandomCutForest
from aberrant.stream.dataset import Dataset, load

model = RandomCutForest(
    n_trees=20,
    sample_size=256,
    shingle_size=1,
    warmup_samples=256,
    normalize_score=True,
    score_scale=8.0,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 5000 and y == 0:
        model.learn_one(x)
        continue

    score = model.score_one(x)
    model.learn_one(x)
    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
