from sklearn.metrics import average_precision_score

from aberrant.datasets import Dataset, load
from aberrant.model.distance import KNN
from aberrant.transform import IncrementalPCA, MinMaxScaler, StandardScaler

# Baseline pipeline: scaling + KNN.
baseline_pipeline = MinMaxScaler() | KNN(k=55, window_size=250)

# PCA pipeline: scaling + incremental PCA + KNN.
pca_pipeline = (
    StandardScaler()
    | IncrementalPCA(n_components=3, warm_up=100)
    | KNN(k=55, window_size=250)
)

labels, baseline_scores, pca_scores = [], [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset):
    if i < 2_000:
        if y == 0:
            baseline_pipeline.learn_one(x)
            pca_pipeline.learn_one(x)
        continue

    labels.append(y)
    baseline_scores.append(baseline_pipeline.score_one(x))
    pca_scores.append(pca_pipeline.score_one(x))

    baseline_pipeline.learn_one(x)
    pca_pipeline.learn_one(x)

baseline_average_precision = average_precision_score(labels, baseline_scores)
pca_average_precision = average_precision_score(labels, pca_scores)

print(
    "Baseline Pipeline (Scaler + KNN): "
    f"average precision = {round(baseline_average_precision, 3)}"
)
print(
    "PCA Pipeline (Scaler + PCA + KNN): "
    f"average precision = {round(pca_average_precision, 3)}"
)
