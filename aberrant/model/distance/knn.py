from aberrant.base.model import BaseModel
from aberrant.base.similarity import BaseSimilaritySearchEngine


class KNN(BaseModel):
    """Nearest-neighbor anomaly scorer backed by a search engine.

    ``learn_one`` appends the observation to the supplied engine. ``score_one``
    returns exactly the scalar produced by ``engine.search(x, n_neighbors=k)``;
    its range and orientation therefore belong to the engine's contract. With
    :class:`~aberrant.utils.similar.faiss_engine.FaissSimilaritySearchEngine`,
    the score is the mean Euclidean distance to the ``k`` nearest retained
    observations, and higher values are more anomalous.

    Args:
        k: Number of neighbors requested for each score. Must be positive.
        similarity_engine: Mutable search engine that owns the reference window.
    """

    def __init__(self, k: int, similarity_engine: BaseSimilaritySearchEngine) -> None:
        """Initialize a nearest-neighbor scorer."""
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k: int = k
        self.engine: BaseSimilaritySearchEngine = similarity_engine

    def learn_one(self, x: dict[str, float]) -> None:
        """Append one observation to the search engine.

        Args:
            x: Feature mapping to retain for subsequent queries.
        """
        self.engine.append(x)

    def score_one(self, x: dict[str, float]) -> float:
        """Query the engine for a ``k``-neighbor scalar.

        Args:
            x: Feature mapping to query without appending it.

        Returns:
            The value returned by the configured engine.
        """
        return self.engine.search(x, n_neighbors=self.k)

    def __repr__(self) -> str:
        """Return a string representation of the KNN model."""
        return f"KNN(k={self.k}, similarity_engine={self.engine.__class__.__name__})"
