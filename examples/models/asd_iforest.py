from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest.asd import ASDIsolationForest
from aberrant.stream.dataset import Dataset, load

model = ASDIsolationForest(n_estimators=750, max_samples=2750, seed=1)
labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 10_000 and y == 0:
        model.learn_one(x)
        continue

    if i < 10_000:
        continue

    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
