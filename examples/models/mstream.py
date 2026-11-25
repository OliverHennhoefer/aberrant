from sklearn.metrics import average_precision_score, roc_auc_score

from aberrant.model.sketch import MStream
from aberrant.stream import Dataset, load

model = MStream(
    n_hashes=1,
    sketch_width=512,
    alpha=0.5,
    time_key="t",
    warm_up=4,
    seed=42,
)

labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    # MStream's author implementation applies log10(1 + x), so adapt this
    # signed benchmark into its required numeric domain.
    sample = {key: abs(value) for key, value in x.items()}
    sample["t"] = float(i // 128)

    if i < 5000 and y == 0:
        model.learn_one(sample)
        continue

    score = model.score_one(sample)
    model.learn_one(sample)
    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
