from collections import deque

import numpy as np

from aberrant.base.model import BaseModel


class IncrementalOneClassSVMAdaptiveKernel(BaseModel):
    """
    Experimental incremental one-class kernel model with adaptive gamma.

    This is a custom budgeted support-vector heuristic, not an implementation
    of a published incremental One-Class SVM optimizer. Running standardization
    changes over time while stored support vectors remain in their historical
    coordinate systems; callers should treat scores as heuristic.
    """

    def __init__(
        self,
        nu: float = 0.1,
        initial_gamma: float = 1.0,
        gamma_bounds: tuple[float, float] = (0.001, 100.0),
        adaptation_rate: float = 0.1,
        buffer_size: int = 200,
        sv_budget: int = 100,
        tolerance: float = 1e-6,
        seed: int | None = None,
    ):
        self.nu = nu
        self.gamma = initial_gamma
        self.gamma_min, self.gamma_max = gamma_bounds
        self.adaptation_rate = adaptation_rate
        self.buffer_size = buffer_size
        self.sv_budget = sv_budget
        self.tolerance = tolerance

        # Model parameters
        self.support_vectors: list[np.ndarray] = []
        self.alpha: list[float] = []
        self.birth_sample: list[int] = []  # Track SV creation time
        self.rho: float = 0.0

        # Adaptation mechanism
        self.data_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.n_samples: int = 0

        # Precomputed kernel matrix
        self.K_sv: np.ndarray | None = None

        # Feature handling
        self.feature_order: tuple[str, ...] | None = None
        self.feature_stats: dict[str, tuple[float, float]] = {}  # For standardization

        # Random number generator
        self.rng = np.random.default_rng(seed)

    def _get_feature_vector(self, x: dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to standardized numpy array."""
        if self.feature_order is None:
            self.feature_order = tuple(sorted(x.keys()))
            # Initialize feature statistics
            for feature in self.feature_order:
                self.feature_stats[feature] = (0.0, 1.0)  # (mean, std)

        if tuple(sorted(x.keys())) != self.feature_order:
            raise ValueError("Inconsistent feature keys")

        # Standardize features using running statistics
        x_vec = np.zeros(len(self.feature_order))
        for i, feature in enumerate(self.feature_order):
            mean, std = self.feature_stats[feature]
            value = x[feature]
            x_vec[i] = (value - mean) / (std + 1e-8) if std > 0 else value

        return x_vec

    def _update_feature_stats(self, x: dict[str, float]):
        """Update running feature statistics for standardization"""
        if not self.feature_stats:
            return

        for feature, value in x.items():
            old_mean, old_std = self.feature_stats[feature]
            n = self.n_samples

            # Update mean and std using Welford's algorithm
            new_mean = old_mean + (value - old_mean) / (n + 1)
            if n > 0:
                new_std = np.sqrt(
                    (n * old_std**2 + (value - old_mean) * (value - new_mean)) / (n + 1)
                )
            else:
                new_std = 1.0

            self.feature_stats[feature] = (new_mean, new_std)

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF kernel value between two vectors."""
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.exp(-self.gamma * squared_distance)

    def _compute_kernel_row(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel values between x and all support vectors."""
        return np.array([self._rbf_kernel(x, sv) for sv in self.support_vectors])

    def _update_kernel_matrix(self, new_sv: np.ndarray | None = None):
        """Efficient kernel matrix update with corrected dimension handling"""
        n_sv = len(self.support_vectors)

        if n_sv == 0:
            self.K_sv = None
            return

        if new_sv is not None:
            # Compute kernel values between new_sv and all existing SVs (including itself)
            new_row = np.array(
                [self._rbf_kernel(new_sv, sv) for sv in self.support_vectors]
            )

            if self.K_sv is None:
                # First support vector
                self.K_sv = np.array([[new_row[0]]])
            else:
                # Existing SVs count (before adding new one)
                old_count = n_sv - 1

                # Separate existing and self-similar values
                k = new_row[:old_count]  # Kernel with existing SVs
                kappa = new_row[old_count]  # Kernel with self

                # Convert to proper shapes
                k_col = k.reshape(-1, 1)  # Column vector
                k_row = k.reshape(1, -1)  # Row vector

                # Build new kernel matrix blocks
                top = np.hstack([self.K_sv, k_col])
                bottom = np.hstack([k_row, np.array([[kappa]])])
                self.K_sv = np.vstack([top, bottom])
        else:
            # Full recompute when gamma changes or SV removed
            self.K_sv = np.zeros((n_sv, n_sv))
            for i in range(n_sv):
                for j in range(i, n_sv):
                    k_val = self._rbf_kernel(
                        self.support_vectors[i], self.support_vectors[j]
                    )
                    self.K_sv[i, j] = k_val
                    self.K_sv[j, i] = k_val

    def _update_rho(self):
        """Recalculate rho as median decision value of support vectors."""
        if not self.support_vectors or self.K_sv is None:
            self.rho = 0.0
            return

        decision_values = self.K_sv @ np.array(self.alpha)
        self.rho = np.median(decision_values)

    def _estimate_optimal_gamma(self) -> float:
        """Estimate optimal gamma based on standardized data distribution."""
        if len(self.data_buffer) < 10:
            return self.gamma

        # Standardize buffer data
        data_array = np.array(list(self.data_buffer))
        means = np.mean(data_array, axis=0)
        stds = np.std(data_array, axis=0) + 1e-8
        data_array = (data_array - means) / stds

        # Sample random pairs
        n_samples = min(50, len(data_array))
        indices = self.rng.choice(len(data_array), size=n_samples, replace=False)
        sampled_data = data_array[indices]

        # Compute pairwise distances
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, min(i + 10, n_samples)):
                dist = np.linalg.norm(sampled_data[i] - sampled_data[j])
                if dist > 1e-10:  # Skip near-identical points
                    distances.append(dist)

        if not distances:
            return self.gamma

        # Use median distance for robust estimation
        median_distance = np.median(distances)
        optimal_gamma = 1.0 / (2.0 * median_distance**2)
        return np.clip(optimal_gamma, self.gamma_min, self.gamma_max)

    def _adapt_gamma(self):
        """Adapt gamma parameter based on recent data."""
        if self.n_samples % 20 != 0:  # Adapt every 20 samples
            return

        target_gamma = self._estimate_optimal_gamma()
        gamma_diff = target_gamma - self.gamma

        if abs(gamma_diff) > 0.01 * self.gamma:
            self.gamma += self.adaptation_rate * gamma_diff
            self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
            self._update_kernel_matrix()  # Full recompute
            self._update_rho()

    def _manage_support_vectors(self, x: np.ndarray, alpha_new: float):
        """Add new support vector and manage budget using combined criteria."""
        # Add new support vector
        self.support_vectors.append(x.copy())
        self.alpha.append(alpha_new)
        self.birth_sample.append(self.n_samples)

        # Enforce budget constraint
        if len(self.support_vectors) > self.sv_budget:
            # Calculate combined removal score (low alpha + old age)
            max_alpha = max(self.alpha)
            max_age = self.n_samples - min(self.birth_sample)

            scores = []
            for alpha_val, birth in zip(self.alpha, self.birth_sample, strict=False):
                age = self.n_samples - birth
                norm_alpha = alpha_val / (max_alpha + 1e-8)
                norm_age = age / (max_age + 1e-8)
                # Higher score = better candidate for removal
                scores.append(0.4 * (1 - norm_alpha) + 0.6 * norm_age)

            idx_remove = np.argmax(scores)
            del self.support_vectors[idx_remove]
            del self.alpha[idx_remove]
            del self.birth_sample[idx_remove]

            # Full kernel recompute after removal
            self._update_kernel_matrix()
        else:
            # Incremental update for new vector
            self._update_kernel_matrix(new_sv=x)

        self._update_rho()

    def _decision_function(self, x: np.ndarray) -> float:
        """Compute decision function value with stability checks."""
        if not self.support_vectors:
            return -self.rho

        kernel_values = self._compute_kernel_row(x)
        return np.dot(self.alpha, kernel_values) - self.rho

    def learn_one(self, x: dict[str, float]):
        """Incrementally learn from one sample."""
        # Update feature statistics before standardization
        self._update_feature_stats(x)
        x_vec = self._get_feature_vector(x)
        self.n_samples += 1
        self.data_buffer.append(x_vec.copy())

        # Handle first sample
        if not self.support_vectors:
            self._manage_support_vectors(x_vec, 1.0 / (self.nu * 10))
            return

        # Compute decision value with current model
        decision_value = self._decision_function(x_vec)

        # Check margin violation with hysteresis
        if decision_value < -self.tolerance:
            alpha_new = min(-decision_value, 1.0 / self.nu)
            self._manage_support_vectors(x_vec, alpha_new)

        # Perform gamma adaptation
        self._adapt_gamma()

    def predict_one(self, x: dict[str, float]) -> int:
        """Predict if sample is normal (1) or anomaly (-1)."""
        if not self.support_vectors:
            return 1  # Default to normal

        x_vec = self._get_feature_vector(x)
        decision_value = self._decision_function(x_vec)
        return 1 if decision_value >= -self.tolerance else -1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score (higher = more anomalous)."""
        if not self.support_vectors:
            return 0.0

        x_vec = self._get_feature_vector(x)
        decision_value = self._decision_function(x_vec)
        return -decision_value

    def get_model_info(self) -> dict:
        """Get current model information."""
        return {
            "n_support_vectors": len(self.support_vectors),
            "gamma": self.gamma,
            "rho": self.rho,
            "n_samples_processed": self.n_samples,
            "buffer_size": len(self.data_buffer),
            "sv_ages": [self.n_samples - b for b in self.birth_sample],
        }
