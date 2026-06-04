from sklearn.metrics import average_precision_score

from aberrant.model.distance.knn import KNN
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing.scaler import MinMaxScaler, StandardScaler
from aberrant.transform.projection.incremental_pca import IncrementalPCA
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

# Baseline pipeline: scaling + KNN.
baseline_pipeline = MinMaxScaler() | KNN(
    k=55,
    similarity_engine=FaissSimilaritySearchEngine(window_size=250, warm_up=50),
)

# PCA pipeline: scaling + incremental PCA + KNN.
pca_pipeline = (
    StandardScaler()
    | IncrementalPCA(n_components=3, n0=100)
    | KNN(
        k=55,
        similarity_engine=FaissSimilaritySearchEngine(window_size=250, warm_up=50),
    )
)

labels, baseline_scores, pca_scores = [], [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
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

baseline_pr_auc = average_precision_score(labels, baseline_scores)
pca_pr_auc = average_precision_score(labels, pca_scores)

print(f"Baseline Pipeline (Scaler + KNN): PR-AUC = {round(baseline_pr_auc, 3)}")
print(f"PCA Pipeline (Scaler + PCA + KNN): PR-AUC = {round(pca_pr_auc, 3)}")
