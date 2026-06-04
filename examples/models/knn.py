from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.distance.knn import KNN
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing.scaler import MinMaxScaler
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
model = KNN(k=45, similarity_engine=engine)
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

print(f"PR-AUC: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC-AUC: {round(roc_auc_score(labels, scores), 3)}")
