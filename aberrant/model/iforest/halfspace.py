"""Half-Space Trees (HST) for streaming anomaly detection."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema


@dataclass(slots=True)
class HSTLeaf:
    """Leaf node in a Half-Space Tree."""

    l_mass: int = 0  # Current window mass (learning)
    r_mass: int = 0  # Reference window mass (scoring)

    def pivot_mass(self) -> None:
        """Copy learning mass to reference mass and reset learning mass."""
        self.r_mass = self.l_mass
        self.l_mass = 0

    def reset_mass(self) -> None:
        """Reset all mass counters (for full reset)."""
        self.l_mass = 0
        self.r_mass = 0


@dataclass(slots=True)
class HSTNode:
    """Internal node in a Half-Space Tree."""

    feature: int
    threshold: float
    left: "HSTNode | HSTLeaf" = field(default_factory=HSTLeaf)
    right: "HSTNode | HSTLeaf" = field(default_factory=HSTLeaf)
    l_mass: int = 0
    r_mass: int = 0

    def pivot_mass(self) -> None:
        """Copy learning mass to reference mass and reset learning mass."""
        self.r_mass = self.l_mass
        self.l_mass = 0
        self.left.pivot_mass()
        self.right.pivot_mass()

    def reset_mass(self) -> None:
        """Recursively reset all mass counters (for full reset)."""
        self.l_mass = 0
        self.r_mass = 0
        self.left.reset_mass()
        self.right.reset_mass()


class HalfSpaceTrees(BaseModel):
    """
    Half-Space Trees for streaming anomaly detection.

    Half-Space Trees (HST) is an ensemble method for detecting anomalies
    in streaming data. It builds multiple random trees that partition
    the feature space using half-space cuts (axis-aligned splits).

    The algorithm tracks the "mass" (visit count) at each node during
    training. Anomalies are identified by having low mass - they fall
    into regions of feature space that are rarely visited.

    Each tree first creates the randomly perturbed work space described in the
    paper. Every internal node then selects a random feature and bisects that
    feature's current interval at its midpoint. Reference and latest-window
    masses are recorded at every traversed node.

    The random work spaces are constructed for inputs in ``[0, 1]``. Scale
    other feature ranges before passing them to this detector.

    Args:
        n_trees: Number of trees in the ensemble. Default is 10.
        height: Maximum depth of each tree. Default is 8.
        window_size: Number of samples per reference window. After
            window_size samples, mass counters are reset. Default is 250.
        seed: Random seed for reproducibility. Default is None.

    Examples:
        ```python
        from aberrant.model.iforest import HalfSpaceTrees
        from aberrant.transform.preprocessing import MinMaxScaler

        stream = [
            {"x": 0.0, "y": 0.1},
            {"x": 0.2, "y": 0.1},
            {"x": 0.1, "y": 0.3},
        ]
        pipeline = MinMaxScaler() | HalfSpaceTrees(n_trees=5, seed=42)
        pipeline.learn_one(stream[0])
        scores = []
        for point in stream[1:]:
            score = pipeline.score_one(point)
            scores.append(score)
            pipeline.learn_one(point)
        assert len(scores) == 2
        ```

    References:
        Tan, S. C., Ting, K. M., & Liu, T. F. (2011). Fast anomaly
        detection for streaming data. In Proceedings of the Twenty-Second
        International Joint Conference on Artificial Intelligence
        (pp. 1511-1516).
        https://www.ijcai.org/Proceedings/11/Papers/254.pdf
    """

    def __init__(
        self,
        n_trees: int = 10,
        height: int = 8,
        window_size: int = 250,
        seed: int | None = None,
    ) -> None:
        if n_trees <= 0:
            raise ValueError("n_trees must be positive")
        if height <= 0:
            raise ValueError("height must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        """Initialize or reset internal state."""
        self._schema = FeatureSchema()
        self._n_features: int = 0
        self._trees: list[HSTNode | HSTLeaf] = []
        # Every node of every tree, so a window pivot is one flat loop.
        self._nodes: list[HSTNode | HSTLeaf] = []
        self._workspaces: list[tuple[np.ndarray, np.ndarray]] = []
        self._samples_in_window: int = 0
        self._reference_window_size: int = 0
        self._initialized: bool = False

    def _build_tree(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        depth: int = 0,
    ) -> HSTNode | HSTLeaf:
        """
        Recursively build a random half-space tree.

        Args:
            depth: Current depth in the tree.

        Returns:
            Root node of the (sub)tree.
        """
        if depth >= self.height:
            return HSTLeaf()

        # Algorithm 1: choose a dimension and bisect its propagated interval.
        feature = int(self.rng.integers(0, self._n_features))
        threshold = float((lower[feature] + upper[feature]) / 2.0)

        left_upper = upper.copy()
        left_upper[feature] = threshold
        right_lower = lower.copy()
        right_lower[feature] = threshold

        return HSTNode(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(lower, left_upper, depth + 1),
            right=self._build_tree(right_lower, upper, depth + 1),
        )

    def _initialize_trees(self) -> None:
        """Build paper-faithful random work spaces and midpoint HS-Trees."""
        trees: list[HSTNode | HSTLeaf] = []
        workspaces: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(self.n_trees):
            center = self.rng.uniform(0.0, 1.0, size=self._n_features)
            half_width = 2.0 * np.maximum(center, 1.0 - center)
            lower = center - half_width
            upper = center + half_width
            workspaces.append((lower.copy(), upper.copy()))
            trees.append(self._build_tree(lower, upper))
        nodes: list[HSTNode | HSTLeaf] = []
        pending = list(trees)
        while pending:
            node = pending.pop()
            nodes.append(node)
            if isinstance(node, HSTNode):
                pending += (node.left, node.right)
        self._trees = trees
        self._nodes = nodes
        self._workspaces = workspaces
        self._initialized = True

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a new observation.

        This increments mass counters along the path to the leaf
        in each tree.

        Args:
            x: Feature dictionary with string keys and float values.
                Values should be in [0, 1] range for best results.
        """
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            self._n_features = len(prepared.names)

        # Build trees on first sample
        if not self._initialized:
            self._initialize_trees()

        self._update_masses(prepared.values.tolist())

        self._samples_in_window += 1

        if self._samples_in_window >= self.window_size:
            self._reference_window_size = self._samples_in_window
            self._reset_masses()
            self._samples_in_window = 0
        self._schema.commit(prepared)

    def _update_masses(self, x: list[float]) -> None:
        """Increment the learning mass of every node on each tree's path to ``x``."""
        levels = range(self.height)
        for tree in self._trees:
            # Every tree is complete with ``height`` internal levels, so the first
            # ``height`` steps only meet ``HSTNode`` and end on an ``HSTLeaf``.
            # The untyped cursor spares a per-node isinstance check.
            node: Any = tree
            for _ in levels:
                node.l_mass += 1
                # A statement instead of a conditional expression saves one jump
                # per node, which is measurable in this loop.
                if x[node.feature] < node.threshold:  # noqa: SIM108
                    node = node.left
                else:
                    node = node.right
            node.l_mass += 1

    def _reset_masses(self) -> None:
        """Pivot masses: copy learning masses to reference, reset learning masses."""
        for node in self._nodes:
            node.r_mass = node.l_mass
            node.l_mass = 0

    def score_one(self, x: dict[str, float]) -> float:
        """
        Compute anomaly score for a point.

        The score is based on the mass (visit frequency) accumulated
        along the path to the leaf. Lower mass indicates anomaly.

        The score is normalized to [0, 1] where higher values indicate
        more anomalous points.

        Args:
            x: Feature dictionary with string keys and float values.

        Returns:
            Anomaly score in [0, 1]. Higher = more anomalous.
        """
        values = self._schema.preview_values(x)
        if not self._initialized or self._reference_window_size == 0:
            return 0.0

        total_score = self._total_mass_score(values)

        max_score: int = self.n_trees * self._reference_window_size * (2**self.height)
        if max_score <= 0:
            return 0.0

        return min(max(1.0 - total_score / max_score, 0.0), 1.0)

    def _total_mass_score(self, x: list[float]) -> float:
        """
        Sum the depth-weighted reference mass of each tree's scoring node.

        Each tree descends towards ``x`` until it reaches a leaf or a node whose
        reference mass is at most 10% of the window, and contributes that node's
        ``r_mass * 2**depth``.

        Args:
            x: Feature values in schema order.

        Returns:
            Weighted mass summed over the trees.
        """
        size_limit = 0.1 * self.window_size
        levels = range(self.height)
        total = 0.0
        for tree in self._trees:
            # As in ``_update_masses``, ``height`` steps cannot pass a leaf.
            node: Any = tree
            weight = 1  # 2**depth
            for _ in levels:
                if node.r_mass <= size_limit:
                    break
                if x[node.feature] < node.threshold:  # noqa: SIM108
                    node = node.left
                else:
                    node = node.right
                weight += weight
            total += float(node.r_mass * weight)
        return total

    def __repr__(self) -> str:
        return (
            f"HalfSpaceTrees(n_trees={self.n_trees}, height={self.height}, "
            f"window_size={self.window_size}, initialized={self._initialized})"
        )
