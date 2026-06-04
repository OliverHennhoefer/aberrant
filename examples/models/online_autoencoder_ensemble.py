from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.deep import OnlineAutoencoderEnsemble
from aberrant.stream.dataset import Dataset, load

model = OnlineAutoencoderEnsemble(
    max_ae_size=4,
    feature_map_grace=256,
    ad_grace=512,
    learning_rate=0.03,
    hidden_ratio=0.75,
    adaptive_after_warmup=False,
    seed=42,
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
