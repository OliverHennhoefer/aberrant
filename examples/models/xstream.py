from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.iforest import XStream
from aberrant.stream.dataset import Dataset, load

model = XStream(
    k=64,
    n_chains=50,
    depth=12,
    cms_width=512,
    cms_num_hashes=4,
    window_size=256,
    init_sample_size=256,
    density=0.25,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 5000 and y == 0:
        model.learn_one(x)
        continue

    score = model.score_one(x)
    model.learn_one(x)
    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
