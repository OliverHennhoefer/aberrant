"""Half-Space Trees (HST) for streaming anomaly detection."""

from dataclasses import dataclass, field

import numpy as np

from aberrant.base.model import BaseModel


@dataclass
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


@dataclass
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

    IMPORTANT: This algorithm assumes features are scaled to [0, 1].
    Use MinMaxScaler in a pipeline for best results.

    Args:
        n_trees: Number of trees in the ensemble. Default is 10.
        height: Maximum depth of each tree. Default is 8.
        window_size: Number of samples per reference window. After
            window_size samples, mass counters are reset. Default is 250.
        seed: Random seed for reproducibility. Default is None.

    Example:
        >>> from aberrant.transform.preprocessing import MinMaxScaler
        >>> from aberrant.model.iforest import HalfSpaceTrees
        >>> pipeline = MinMaxScaler() | HalfSpaceTrees(n_trees=25)
        >>> for point in stream:
        ...     score = pipeline.score_one(point)
        ...     pipeline.learn_one(point)
        ...     if score > 0.5:  # Threshold for anomaly
        ...         print("Anomaly detected!")

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
        self.feature_names: list[str] | None = None
        self._n_features: int = 0
        self._trees: list[HSTNode | HSTLeaf] = []
        self._workspaces: list[tuple[np.ndarray, np.ndarray]] = []
        self._samples_in_window: int = 0
        self._reference_window_size: int = 0
        self._initialized: bool = False
        self._x_array: np.ndarray = np.empty(0)

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
        self._trees = trees
        self._workspaces = workspaces
        self._initialized = True

    def _validate_schema(self, x: dict[str, float]) -> None:
        """Set the first feature schema or reject any later schema change."""
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
            self._n_features = len(self.feature_names)
            self._x_array = np.zeros(self._n_features)
            return

        if set(x) != set(self.feature_names):
            expected = ", ".join(self.feature_names)
            received = ", ".join(sorted(x))
            raise ValueError(
                "Inconsistent feature keys. "
                f"Expected [{expected}], received [{received}]."
            )

    def _vectorize(self, x: dict[str, float]) -> np.ndarray:
        """Validate and convert a sample using the established feature order."""
        self._validate_schema(x)
        if self.feature_names is None:
            raise RuntimeError("Feature schema is not initialized")
        for index, feature in enumerate(self.feature_names):
            self._x_array[index] = x[feature]
        return self._x_array

    def learn_one(self, x: dict[str, float]) -> None:
        """
        Update the model with a new observation.

        This increments mass counters along the path to the leaf
        in each tree.

        Args:
            x: Feature dictionary with string keys and float values.
                Values should be in [0, 1] range for best results.
        """
        x_array = self._vectorize(x)

        # Build trees on first sample
        if not self._initialized:
            self._initialize_trees()

        # Update mass in each tree
        for tree in self._trees:
            self._update_mass(tree, x_array)

        self._samples_in_window += 1

        if self._samples_in_window >= self.window_size:
            self._reference_window_size = self._samples_in_window
            self._reset_masses()
            self._samples_in_window = 0

    def _update_mass(self, node: HSTNode | HSTLeaf, x: np.ndarray) -> None:
        """
        Recursively update learning mass counters along the path to leaf.

        Args:
            node: Current node.
            x: Feature vector.
        """
        node.l_mass += 1
        if isinstance(node, HSTLeaf):
            return

        if x[node.feature] < node.threshold:
            self._update_mass(node.left, x)
        else:
            self._update_mass(node.right, x)

    def _reset_masses(self) -> None:
        """Pivot masses: copy learning masses to reference, reset learning masses."""
        for tree in self._trees:
            tree.pivot_mass()

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
        if not self._initialized or self._reference_window_size == 0:
            return 0.0

        x_array = self._vectorize(x)

        total_score = 0.0
        for tree in self._trees:
            total_score += self._compute_tree_score(tree, x_array, depth=0)

        max_score = self.n_trees * self._reference_window_size * (2**self.height)
        if max_score <= 0:
            return 0.0

        return float(np.clip(1.0 - total_score / max_score, 0.0, 1.0))

    def _compute_tree_score(
        self, node: HSTNode | HSTLeaf, x: np.ndarray, depth: int
    ) -> float:
        """
        Compute weighted mass score for a single tree.

        Args:
            node: Current node.
            x: Feature vector.
            depth: Current depth.
        Returns:
            Weighted mass contribution.
        """
        if isinstance(node, HSTLeaf) or node.r_mass <= 0.1 * self.window_size:
            return node.r_mass * (2**depth)

        if x[node.feature] < node.threshold:
            return self._compute_tree_score(node.left, x, depth + 1)
        return self._compute_tree_score(node.right, x, depth + 1)

    def __repr__(self) -> str:
        return (
            f"HalfSpaceTrees(n_trees={self.n_trees}, height={self.height}, "
            f"window_size={self.window_size}, initialized={self._initialized})"
        )
