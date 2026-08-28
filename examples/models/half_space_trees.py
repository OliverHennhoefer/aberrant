from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest import HalfSpaceTrees
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing import MinMaxScaler

model = MinMaxScaler() | HalfSpaceTrees(
    n_trees=10,
    height=8,
    window_size=250,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)
warmup_count = 0

for x, y in dataset.stream():
    if warmup_count < 500:
        if y == 0:
            model.learn_one(x)
            warmup_count += 1
        continue

    score = model.score_one(x)
    model.learn_one(x)
    labels.append(y)
    scores.append(score)
    if len(scores) == 1_000:
        break

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
