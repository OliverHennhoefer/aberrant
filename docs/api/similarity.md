# Similarity Search API

`KNN` takes a similarity engine that keeps a window of observations and
answers nearest-neighbour queries. `FaissSimilaritySearchEngine` needs
`aberrant[faiss]`; subclass `BaseSimilaritySearchEngine` to plug in another
index.

::: aberrant.similarity.BaseSimilaritySearchEngine

::: aberrant.similarity.FaissSimilaritySearchEngine
