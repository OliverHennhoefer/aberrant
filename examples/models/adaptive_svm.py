from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.datasets import Dataset, load
from aberrant.model.svm.adaptive import AdaptiveOneClassSVM
from aberrant.transform import StandardScaler

scaler = StandardScaler()
model = AdaptiveOneClassSVM(
    nu=0.1,
    sv_budget=25,
    initial_gamma=0.5,
    adaptation_rate=0.3,
    gamma_bounds=(0.1, 5.0),
    seed=1,
)
pipeline = scaler | model
labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset):
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
