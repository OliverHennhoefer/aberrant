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
    l_mass_left: int = 0  # Current window left mass
    l_mass_right: int = 0  # Current window right mass
    r_mass_left: int = 0  # Reference window left mass
    r_mass_right: int = 0  # Reference window right mass

    def pivot_mass(self) -> None:
        """Copy learning masses to reference masses and reset learning masses."""
        self.r_mass_left = self.l_mass_left
        self.r_mass_right = self.l_mass_right
        self.l_mass_left = 0
        self.l_mass_right = 0
        self.left.pivot_mass()
        self.right.pivot_mass()

    def reset_mass(self) -> None:
        """Recursively reset all mass counters (for full reset)."""
        self.l_mass_left = 0
        self.l_mass_right = 0
        self.r_mass_left = 0
        self.r_mass_right = 0
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

    This implementation samples each node threshold independently in
    ``[0.15, 0.85]``. The paper's tree construction recursively bisects
    propagated feature-space intervals, so this class is an HST-inspired
    variant rather than an exact implementation.

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
        self._samples_in_window: int = 0
        self._total_samples_learned: int = 0  # Never resets, used for scoring check
        self._reference_window_size: int = 0  # Size of last complete window for scoring
        self._initialized: bool = False
        self._x_array: np.ndarray = np.empty(0)

    def _build_tree(self, depth: int = 0) -> HSTNode | HSTLeaf:
        """
        Recursively build a random half-space tree.

        Args:
            depth: Current depth in the tree.

        Returns:
            Root node of the (sub)tree.
        """
        if depth >= self.height:
            return HSTLeaf()

        # Random feature and threshold
        feature = int(self.rng.integers(0, self._n_features))
        # Use padding to avoid narrow splits near boundaries
        padding = 0.15
        threshold = float(self.rng.uniform(padding, 1.0 - padding))

        return HSTNode(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(depth + 1),
            right=self._build_tree(depth + 1),
        )

    def _initialize_trees(self) -> None:
        """Build all trees in the ensemble."""
        self._trees = [self._build_tree() for _ in range(self.n_trees)]
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
        if not x:
            raise ValueError("Input dictionary cannot be empty")

        # Initialize on first sample
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())
            self._n_features = len(self.feature_names)
            self._x_array = np.zeros(self._n_features)

        # Build trees on first sample
        if not self._initialized:
            self._initialize_trees()

        # Convert to array
        for i, f in enumerate(self.feature_names):
            self._x_array[i] = x.get(f, 0.0)

        # Update mass in each tree
        for tree in self._trees:
            self._update_mass(tree, self._x_array, depth=0)

        # Track samples for scoring availability
        self._samples_in_window += 1
        self._total_samples_learned += 1

        # Check if window is complete - reset masses for new window
        if self._samples_in_window >= self.window_size:
            # Save current window size for scoring normalization
            self._reference_window_size = self._samples_in_window
            self._reset_masses()
            self._samples_in_window = 0

    def _update_mass(self, node: HSTNode | HSTLeaf, x: np.ndarray, depth: int) -> None:
        """
        Recursively update learning mass counters along the path to leaf.

        Args:
            node: Current node.
            x: Feature vector.
            depth: Current depth.
        """
        if isinstance(node, HSTLeaf):
            node.l_mass += 1
            return

        # Traverse based on split, updating learning masses
        if x[node.feature] < node.threshold:
            node.l_mass_left += 1
            self._update_mass(node.left, x, depth + 1)
        else:
            node.l_mass_right += 1
            self._update_mass(node.right, x, depth + 1)

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
        # Need to have seen at least one sample to have valid trees
        if not self._initialized or self._total_samples_learned == 0:
            return 0.0

        if self.feature_names is None:
            return 0.0

        # Convert to array
        for i, f in enumerate(self.feature_names):
            self._x_array[i] = x.get(f, 0.0)

        # Determine whether to use reference masses (after first pivot) or
        # learning masses (during first window)
        use_reference = self._reference_window_size > 0

        # Accumulate weighted mass from all trees
        total_score = 0.0
        for tree in self._trees:
            tree_score = self._compute_tree_score(
                tree, self._x_array, depth=0, use_reference=use_reference
            )
            total_score += tree_score

        # Compute maximum possible score using appropriate window size
        if use_reference:
            effective_window = self._reference_window_size
        else:
            effective_window = self._samples_in_window
        max_score = self.n_trees * effective_window * ((2 ** (self.height + 1)) - 1)

        if max_score <= 0:
            return 0.0

        # Normalize and invert (low mass = high anomaly score)
        normalized = total_score / max_score
        return 1.0 - normalized

    def _compute_tree_score(
        self, node: HSTNode | HSTLeaf, x: np.ndarray, depth: int, use_reference: bool
    ) -> float:
        """
        Compute weighted mass score for a single tree.

        Args:
            node: Current node.
            x: Feature vector.
            depth: Current depth.
            use_reference: If True, use reference masses (r_mass). If False, use
                learning masses (l_mass).

        Returns:
            Weighted mass contribution.
        """
        if isinstance(node, HSTLeaf):
            mass = node.r_mass if use_reference else node.l_mass
            return mass * (2**depth)

        # Get mass and traverse
        if x[node.feature] < node.threshold:
            mass = node.r_mass_left if use_reference else node.l_mass_left
            return mass * (2**depth) + self._compute_tree_score(
                node.left, x, depth + 1, use_reference
            )
        else:
            mass = node.r_mass_right if use_reference else node.l_mass_right
            return mass * (2**depth) + self._compute_tree_score(
                node.right, x, depth + 1, use_reference
            )

    def __repr__(self) -> str:
        return (
            f"HalfSpaceTrees(n_trees={self.n_trees}, height={self.height}, "
            f"window_size={self.window_size}, initialized={self._initialized})"
        )
