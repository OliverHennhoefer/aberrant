from collections import deque

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema, PreparedFeatures


class IncrementalOneClassSVMAdaptiveKernel(BaseModel):
    """
    Experimental incremental one-class kernel model with adaptive gamma.

    This is a custom budgeted support-vector heuristic, not an implementation
    of a published incremental One-Class SVM optimizer. Support vectors and the
    recent-data buffer are stored in raw feature coordinates. Kernel values are
    computed after applying the current running standardization to both
    operands, so all comparisons use one consistent coordinate system.
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
    ) -> None:
        if not (0.0 < nu <= 1.0):
            raise ValueError("nu must be in (0, 1]")
        if initial_gamma <= 0.0:
            raise ValueError("initial_gamma must be positive")
        if (
            gamma_bounds[0] <= 0.0
            or gamma_bounds[0] > gamma_bounds[1]
            or not gamma_bounds[0] <= initial_gamma <= gamma_bounds[1]
        ):
            raise ValueError(
                "gamma_bounds must be positive, ordered, and contain initial_gamma"
            )
        if not (0.0 < adaptation_rate <= 1.0):
            raise ValueError("adaptation_rate must be in (0, 1]")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if sv_budget <= 0:
            raise ValueError("sv_budget must be positive")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

        self.nu = nu
        self.gamma = float(initial_gamma)
        self.gamma_min, self.gamma_max = gamma_bounds
        self.adaptation_rate = adaptation_rate
        self.buffer_size = buffer_size
        self.sv_budget = sv_budget
        self.tolerance = tolerance

        # Support vectors deliberately remain in raw coordinates.
        self.support_vectors: list[np.ndarray] = []
        self.alpha: list[float] = []
        self.birth_sample: list[int] = []
        self.rho: float = 0.0
        self.K_sv: np.ndarray | None = None

        self.data_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.n_samples: int = 0

        self._schema = FeatureSchema()
        self.feature_stats: dict[str, tuple[float, float]] = {}
        self._feature_mean: np.ndarray | None = None
        self._feature_m2: np.ndarray | None = None
        self._stats_count = 0

        self.rng = np.random.default_rng(seed)

    def _prepare_raw_features(self, x: dict[str, float]) -> PreparedFeatures:
        """Validate and vectorize a raw feature dictionary without committing it."""
        prepared = self._schema.preview(x)
        if not self._schema.is_established:
            n_features = len(prepared.names)
            self._feature_mean = np.zeros(n_features, dtype=np.float64)
            self._feature_m2 = np.zeros(n_features, dtype=np.float64)
            self.feature_stats = dict.fromkeys(prepared.names, (0.0, 1.0))
        return prepared

    def _update_feature_stats(
        self,
        x: np.ndarray,
        names: tuple[str, ...],
    ) -> None:
        """Update population mean/variance with a valid Welford accumulator."""
        if self._feature_mean is None or self._feature_m2 is None:
            raise RuntimeError("Feature statistics are not initialized")

        self._stats_count += 1
        count = self._stats_count
        delta = x - self._feature_mean
        self._feature_mean += delta / count
        delta2 = x - self._feature_mean
        self._feature_m2 += delta * delta2

        std = self._feature_std()
        self.feature_stats = {
            feature: (float(self._feature_mean[index]), float(std[index]))
            for index, feature in enumerate(names)
        }

    def _feature_std(self) -> np.ndarray:
        """Return population standard deviations with stable zero-variance scale."""
        if self._feature_m2 is None:
            raise RuntimeError("Feature statistics are not initialized")
        if self._stats_count <= 1:
            return np.ones_like(self._feature_m2)

        variance = np.maximum(self._feature_m2 / self._stats_count, 0.0)
        std = np.sqrt(variance)
        return np.where(std > 0.0, std, 1.0)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Standardize a raw vector using the current running statistics."""
        if self._feature_mean is None:
            raise RuntimeError("Feature statistics are not initialized")
        return np.asarray(
            (x - self._feature_mean) / self._feature_std(),
            dtype=np.float64,
        )

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the RBF kernel after consistently standardizing raw vectors."""
        difference = self._standardize(x1) - self._standardize(x2)
        return float(np.exp(-self.gamma * float(np.dot(difference, difference))))

    def _compute_kernel_row(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel values between a raw query and all raw support vectors."""
        return np.fromiter(
            (self._rbf_kernel(x, sv) for sv in self.support_vectors),
            dtype=np.float64,
            count=len(self.support_vectors),
        )

    def _update_kernel_matrix(self) -> None:
        """Recompute the support-vector kernel matrix in current coordinates."""
        n_sv = len(self.support_vectors)
        if n_sv == 0:
            self.K_sv = None
            return

        standardized = np.vstack(
            [self._standardize(vector) for vector in self.support_vectors]
        )
        squared_norms = np.sum(standardized * standardized, axis=1)
        distances = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * standardized @ standardized.T
        )
        np.maximum(distances, 0.0, out=distances)
        self.K_sv = np.exp(-self.gamma * distances)

    def _update_rho(self) -> None:
        """Recalculate rho as the median support-vector decision value."""
        if not self.support_vectors or self.K_sv is None:
            self.rho = 0.0
            return

        decision_values = self.K_sv @ np.asarray(self.alpha, dtype=np.float64)
        self.rho = float(np.median(decision_values))

    def _estimate_optimal_gamma(self) -> float:
        """Estimate gamma from pairwise distances in the recent raw-data buffer."""
        if len(self.data_buffer) < 10:
            return self.gamma

        data_array = np.vstack(self.data_buffer)
        standardized = np.vstack([self._standardize(row) for row in data_array])

        n_samples = min(50, len(standardized))
        indices = self.rng.choice(len(standardized), size=n_samples, replace=False)
        sampled_data = standardized[indices]

        distances: list[float] = []
        for i in range(n_samples):
            for j in range(i + 1, min(i + 10, n_samples)):
                distance = float(np.linalg.norm(sampled_data[i] - sampled_data[j]))
                if distance > 1e-10:
                    distances.append(distance)

        if not distances:
            return self.gamma

        median_distance = float(np.median(distances))
        optimal_gamma = 1.0 / (2.0 * median_distance**2)
        return float(np.clip(optimal_gamma, self.gamma_min, self.gamma_max))

    def _adapt_gamma(self) -> None:
        """Adapt gamma every 20 learned samples."""
        if self.n_samples % 20 != 0:
            return

        target_gamma = self._estimate_optimal_gamma()
        gamma_diff = target_gamma - self.gamma
        if abs(gamma_diff) <= 0.01 * self.gamma:
            return

        self.gamma += self.adaptation_rate * gamma_diff
        self.gamma = float(np.clip(self.gamma, self.gamma_min, self.gamma_max))
        self._update_kernel_matrix()
        self._update_rho()

    def _manage_support_vectors(self, x: np.ndarray, alpha_new: float) -> None:
        """Add a raw support vector and enforce the configured budget."""
        self.support_vectors.append(x.copy())
        self.alpha.append(float(alpha_new))
        self.birth_sample.append(self.n_samples)

        if len(self.support_vectors) > self.sv_budget:
            max_alpha = max(self.alpha)
            max_age = self.n_samples - min(self.birth_sample)
            scores = []
            for alpha_value, birth in zip(self.alpha, self.birth_sample, strict=True):
                age = self.n_samples - birth
                normalized_alpha = alpha_value / (max_alpha + 1e-8)
                normalized_age = age / (max_age + 1e-8)
                scores.append(0.4 * (1.0 - normalized_alpha) + 0.6 * normalized_age)

            remove_index = int(np.argmax(scores))
            del self.support_vectors[remove_index]
            del self.alpha[remove_index]
            del self.birth_sample[remove_index]

        self._update_kernel_matrix()
        self._update_rho()

    def _decision_function(self, x: np.ndarray) -> float:
        """Compute the decision value for a raw feature vector."""
        if not self.support_vectors:
            return -self.rho

        kernel_values = self._compute_kernel_row(x)
        return float(np.dot(self.alpha, kernel_values) - self.rho)

    def learn_one(self, x: dict[str, float]) -> None:
        """Incrementally learn from one sample."""
        prepared = self._prepare_raw_features(x)
        raw_vector = prepared.values
        self._update_feature_stats(raw_vector, prepared.names)
        self.n_samples += 1
        self.data_buffer.append(raw_vector.copy())

        # Running normalization changed, so existing kernel values and rho must
        # be expressed in the new common coordinate system before comparison.
        if self.support_vectors:
            self._update_kernel_matrix()
            self._update_rho()

        if not self.support_vectors:
            self._manage_support_vectors(raw_vector, 1.0 / (self.nu * 10.0))
            self._schema.commit(prepared)
            return

        decision_value = self._decision_function(raw_vector)
        if decision_value < -self.tolerance:
            alpha_new = min(-decision_value, 1.0 / self.nu)
            self._manage_support_vectors(raw_vector, alpha_new)

        self._adapt_gamma()
        self._schema.commit(prepared)

    def predict_one(self, x: dict[str, float]) -> int:
        """Predict normal (1) or anomalous (-1)."""
        prepared = self._schema.preview(x)
        if not self.support_vectors:
            return 1

        decision_value = self._decision_function(prepared.values)
        return 1 if decision_value >= -self.tolerance else -1

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score (higher means more anomalous)."""
        prepared = self._schema.preview(x)
        if not self.support_vectors:
            return 0.0

        return -self._decision_function(prepared.values)

    def get_model_info(self) -> dict[str, object]:
        """Return current model diagnostics."""
        return {
            "n_support_vectors": len(self.support_vectors),
            "gamma": self.gamma,
            "rho": self.rho,
            "n_samples_processed": self.n_samples,
            "buffer_size": len(self.data_buffer),
            "sv_ages": [self.n_samples - birth for birth in self.birth_sample],
        }
