from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.svm import GraphGatedOneClassSVM
from aberrant.stream.dataset import Dataset, load

model = GraphGatedOneClassSVM()
labels, scores = [], []
dataset = load(Dataset.FRAUD)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 2_000:
        if y == 0:
            model.learn_one(x)
        continue
    score = model.score_one(x)
    model.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
