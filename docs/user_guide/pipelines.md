# Pipelines

Pipelines chain transformers and models with `|`.

## Learning order

Pipeline learning uses a **post-update transform** at every transformer stage:

1. The transformer learns from its current input.
2. The same input is transformed with the transformer's updated state.
3. The transformed value is passed to the next transformer or terminal model.

`score_one` and `transform_one` do not update transformer state. For online
evaluation, score a sample before calling `learn_one` when the score must only
depend on previously observed samples.

A model is terminal: a pipeline can contain any number of transformers followed
by at most one model, and no component can be appended after that model.

## Example: scaler + KNN

```python
from aberrant.model.distance import KNN
from aberrant.transform.preprocessing import MinMaxScaler
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
pipeline = MinMaxScaler() | KNN(k=45, similarity_engine=engine)
```

## Example: scaler + PCA + KNN

```python
from aberrant.model.distance import KNN
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import IncrementalPCA
from aberrant.utils.similar.faiss_engine import FaissSimilaritySearchEngine

engine = FaissSimilaritySearchEngine(window_size=250, warm_up=50)
pipeline = StandardScaler() | IncrementalPCA(n_components=3, n0=100) | KNN(
    k=45,
    similarity_engine=engine,
)
```

## Operational tip

Keep thresholding outside the model pipeline when you need fast runtime policy
changes (for example, changing alert sensitivity without retraining).
