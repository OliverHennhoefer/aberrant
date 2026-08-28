"""Base similarity search engine interface."""

import abc


class BaseSimilaritySearchEngine(abc.ABC):
    """
    Abstract base class for similarity search engines.

    This class defines the interface for engines that store observations and
    reduce a nearest-neighbor query to one scalar. The interface does not impose
    whether that scalar is a distance, dissimilarity, or similarity; callers must
    follow the concrete engine's contract.

    Subclasses must implement the `append` and `search` methods.
    """

    @abc.abstractmethod
    def append(self, x: dict[str, float]) -> None:
        """
        Add a data point to the search engine.

        Args:
            x: A dictionary representing a single data point. The keys are feature names,
                and the values are the corresponding feature values.
        """
        pass

    @abc.abstractmethod
    def search(self, x: dict[str, float], n_neighbors: int) -> float:
        """
        Search for the n nearest neighbors of a data point.

        Args:
            x: A dictionary representing the query data point.
            n_neighbors: The number of nearest neighbors to find.

        Returns:
            The engine-specific scalar summary of the nearest-neighbor query.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the search engine."""
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        """Return a human-readable string representation of the search engine."""
        return self.__repr__()
