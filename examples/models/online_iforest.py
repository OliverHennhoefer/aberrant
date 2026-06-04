from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest.online import OnlineIsolationForest
from aberrant.stream.dataset import Dataset, load

model = OnlineIsolationForest(
    num_trees=20,
    max_leaf_samples=32,
    type="adaptive",
    subsample=1.0,
    window_size=512,
    branching_factor=2,
    metric="axisparallel",
    n_jobs=1,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if y == 0 and i < 5000:
        model.learn_one(x)
        continue

    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
