from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn, optim

from aberrant.model.deep.autoencoder import Autoencoder
from aberrant.stream.dataset import Dataset, load
from aberrant.transform.preprocessing.scaler import MinMaxScaler
from aberrant.utils.deep.architecture import VanillaAutoencoder

architecture = VanillaAutoencoder(input_size=9, seed=1)
model = Autoencoder(
    model=architecture,
    optimizer=optim.Adam(architecture.parameters(), lr=0.005, weight_decay=0),
    criterion=nn.MSELoss(),
)
pipeline = MinMaxScaler() | model
labels, scores = [], []
dataset = load(Dataset.SHUTTLE)

for i, (x, y) in enumerate(dataset.stream()):
    if i < 10_000:
        if y == 0:
            pipeline.learn_one(x)
        continue

    score = pipeline.score_one(x)
    pipeline.learn_one(x)

    labels.append(y)
    scores.append(score)

print(f"Average precision: {round(average_precision_score(labels, scores), 3)}")
print(f"ROC AUC: {round(roc_auc_score(labels, scores), 3)}")
