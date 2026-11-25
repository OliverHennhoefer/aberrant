from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.distance import KNN
from aberrant.stream import Dataset, load
from aberrant.transform import MinMaxScaler

model = KNN(k=45, window_size=250)
pipeline = MinMaxScaler() | model
labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 2_000:
        if y == 0:
            pipeline.learn_one(x)
        continue

    score = pipeline.score_one(x)
    pipeline.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
