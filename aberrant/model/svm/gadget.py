from collections import deque

import numpy as np

from aberrant.base.model import BaseModel


class IncrementalOneClassSVM:
    """
    Incremental One-Class SVM with corrected gradient calculation and regularization.
    """

    def __init__(
        self, learning_rate: float = 0.01, nu: float = 0.5, lambda_reg: float = 0.01
    ):
        self.w = None  # Weight vector
        self.rho = 0.0  # Bias term
        self.learning_rate = learning_rate
        self.nu = nu  # Anomaly rate parameter
        self.lambda_reg = lambda_reg  # Regularization parameter

    def learn_one(self, x_vec: np.ndarray) -> None:
        if self.w is None:
            self.w = np.zeros_like(x_vec)

        decision = np.dot(self.w, x_vec)
        loss = max(0, self.rho - decision)

        # Gradient updates with regularization
        if loss > 0:
            # w update: gradient = -x_vec + lambda_reg * w
            self.w += self.learning_rate * (x_vec - self.lambda_reg * self.w)
            # rho update: gradient = (nu - 1)
            self.rho += self.learning_rate * (self.nu - 1)
        else:
            # Only apply regularization to w
            self.w -= self.learning_rate * self.lambda_reg * self.w
            # rho update: gradient = nu
            self.rho += self.learning_rate * self.nu

    def score_one(self, x_vec: np.ndarray) -> float:
        return self.rho - np.dot(self.w, x_vec) if self.w is not None else 0.0


class GraphGatedOneClassSVM(BaseModel):
    """
    Graph-gated ensemble of incremental linear one-class SVM heuristics.

    The graph determines which node models are updated and scored. This custom
    anomaly detector is unrelated to the published GADGET distributed
    averaging and optimization algorithm.
    """

    def __init__(
        self,
        graph: dict[int, list[int]] | None = None,
        threshold: float = 0.0,
        learning_rate: float = 0.01,
        nu: float = 0.5,
        lambda_reg: float = 0.01,
    ):
        # Set default graph if None provided
        if graph is None:
            graph = {0: [1], 1: [2], 2: []}

        # Collect all unique nodes from graph
        all_nodes: set[int] = set()
        for node, neighbors in graph.items():
            all_nodes.add(node)
            all_nodes.update(neighbors)
        self.graph = {node: list(neighbors) for node, neighbors in graph.items()}
        # Ensure all nodes have entries (handle nodes only in values)
        self.graph.update({node: [] for node in all_nodes if node not in self.graph})

        self.threshold = threshold
        self.learning_rate = learning_rate
        self.nu = nu
        self.lambda_reg = lambda_reg
        self.feature_order = None  # Tuple for fast comparisons

        # Initialize SVMs for all nodes
        self.svms = {
            node: IncrementalOneClassSVM(learning_rate, nu, lambda_reg)
            for node in all_nodes
        }

        # Precompute root nodes (nodes with no incoming edges)
        child_nodes = set()
        for neighbors in self.graph.values():
            child_nodes.update(neighbors)
        self.root_nodes = [node for node in all_nodes if node not in child_nodes]
        # Handle empty graph case
        if not self.root_nodes and all_nodes:
            self.root_nodes = [min(all_nodes)]

    def _get_feature_vector(self, x: dict[str, float]) -> np.ndarray:
        """Efficient feature vector conversion with tuple-based order tracking."""
        if self.feature_order is None:
            self.feature_order = tuple(sorted(x.keys()))

        if tuple(sorted(x.keys())) != self.feature_order:
            raise ValueError("Inconsistent feature keys")

        return np.fromiter((x[k] for k in self.feature_order), dtype=np.float64)

    def learn_one(self, x: dict[str, float]) -> None:
        x_vec = self._get_feature_vector(x)
        visited = set()
        # Use deque for efficient BFS
        queue = deque(self.root_nodes)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            svm = self.svms[node]
            svm.learn_one(x_vec)
            score = svm.score_one(x_vec)

            if score > self.threshold:
                # Add neighbors to queue
                queue.extend(self.graph[node])

    def score_one(self, x: dict[str, float]) -> float:
        if self.feature_order is None:
            return 0.0

        x_vec = self._get_feature_vector(x)
        max_score = -np.inf
        visited = set()
        queue = deque(self.root_nodes)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            score = self.svms[node].score_one(x_vec)
            max_score = max(max_score, score)

            if score > self.threshold:
                queue.extend(self.graph[node])

        return max(max_score, 0.0)  # Ensure non-negative minimum


# Backward-compatible alias for the historical, misleading public name.
GADGETSVM = GraphGatedOneClassSVM
