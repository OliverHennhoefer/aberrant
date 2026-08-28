from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest import MondrianIsolationForest
from aberrant.stream.dataset import Dataset, load

# lambda_ is the Mondrian lifetime budget: larger values allow finer partitions.
model = MondrianIsolationForest(
    n_estimators=120, subspace_size=128, lambda_=1.0, seed=1
)
labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 10_000:
        if y == 0:
            model.learn_one(x)
        continue
    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
