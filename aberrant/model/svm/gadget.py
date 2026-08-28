from collections import deque

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


class IncrementalOneClassSVM:
    """
    Incremental One-Class SVM with corrected gradient calculation and regularization.
    """

    def __init__(
        self, learning_rate: float = 0.01, nu: float = 0.5, lambda_reg: float = 0.01
    ) -> None:
        self.w: np.ndarray | None = None  # Weight vector
        self.rho = 0.0  # Bias term
        self.learning_rate = learning_rate
        self.nu = nu  # Anomaly rate parameter
        self.lambda_reg = lambda_reg  # Regularization parameter

    def learn_one(self, x_vec: np.ndarray) -> None:
        weights = self.w
        if weights is None:
            weights = np.zeros_like(x_vec)
            self.w = weights

        decision = float(np.dot(weights, x_vec))
        loss = max(0, self.rho - decision)

        # Gradient updates with regularization
        if loss > 0:
            # w update: gradient = -x_vec + lambda_reg * w
            weights += self.learning_rate * (x_vec - self.lambda_reg * weights)
            # rho update: gradient = (nu - 1)
            self.rho += self.learning_rate * (self.nu - 1)
        else:
            # Only apply regularization to w
            weights -= self.learning_rate * self.lambda_reg * weights
            # rho update: gradient = nu
            self.rho += self.learning_rate * self.nu

    def score_one(self, x_vec: np.ndarray) -> float:
        if self.w is None:
            return 0.0
        return self.rho - float(np.dot(self.w, x_vec))


class GraphGatedOneClassSVM(BaseModel):
    """
    Graph-gated ensemble of incremental linear one-class SVM heuristics.

    The graph determines which node models are updated and scored. This custom
    anomaly detector is unrelated to the published GADGET distributed
    averaging and optimization algorithm.

    Traversal starts at nodes with no incoming edge. A node's outgoing edges
    are traversed only when its local score exceeds ``threshold``. Each visited
    node owns an incremental linear hinge-style heuristic.

    Args:
        graph: Directed adjacency mapping from integer node identifiers to
            child-node lists. ``None`` uses the chain ``0 -> 1 -> 2``. The
            mapping and child lists are copied.
        threshold: Local score gate for traversing outgoing edges.
        learning_rate: Step size used by every node model.
        nu: Coefficient in every node model's bias update. This custom update
            does not provide the guarantees of a solved One-Class SVM.
        lambda_reg: Weight-decay coefficient used by every node model.
    """

    def __init__(
        self,
        graph: dict[int, list[int]] | None = None,
        threshold: float = 0.0,
        learning_rate: float = 0.01,
        nu: float = 0.5,
        lambda_reg: float = 0.01,
    ) -> None:
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
        self._schema = FeatureSchema()

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

    def learn_one(self, x: dict[str, float]) -> None:
        prepared = self._schema.preview(x)
        x_vec = prepared.values
        visited: set[int] = set()
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
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            return 0.0

        x_vec = prepared.values
        max_score = -np.inf
        visited: set[int] = set()
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

        return float(max(max_score, 0.0))  # Ensure non-negative minimum
