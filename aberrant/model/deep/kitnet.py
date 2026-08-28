"""Online ensemble of lightweight autoencoders for anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema

_PHASE_FEATURE_MAP = "feature_map_warmup"
_PHASE_DETECTOR = "detector_warmup"
_PHASE_READY = "ready"


@dataclass
class _NumpyAutoencoder:
    """Single-hidden-layer online autoencoder trained with SGD."""

    input_dim: int
    hidden_dim: int
    learning_rate: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        limit = np.sqrt(6.0 / float(self.input_dim + self.hidden_dim))
        self.w1 = self.rng.uniform(
            low=-limit,
            high=limit,
            size=(self.hidden_dim, self.input_dim),
        )
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)

        self.w2 = self.rng.uniform(
            low=-limit,
            high=limit,
            size=(self.input_dim, self.hidden_dim),
        )
        self.b2 = np.zeros(self.input_dim, dtype=np.float64)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass and return hidden/output vectors."""
        hidden_linear = self.w1 @ x + self.b1
        hidden = np.tanh(hidden_linear)
        output = self.w2 @ hidden + self.b2
        return hidden, output

    def score(self, x: np.ndarray) -> float:
        """Compute RMSE reconstruction error without updating parameters."""
        _, output = self._forward(x)
        error = output - x
        return float(np.sqrt(np.mean(error * error)))

    def learn(self, x: np.ndarray) -> float:
        """Update parameters on one sample and return RMSE."""
        hidden, output = self._forward(x)
        error = output - x
        rmse = float(np.sqrt(np.mean(error * error)))

        # 0.5 * mean squared error -> gradient: (output - x) / input_dim
        grad_output = error / float(self.input_dim)

        grad_w2 = np.outer(grad_output, hidden)
        grad_b2 = grad_output

        grad_hidden = self.w2.T @ grad_output
        grad_hidden_linear = grad_hidden * (1.0 - hidden * hidden)

        grad_w1 = np.outer(grad_hidden_linear, x)
        grad_b1 = grad_hidden_linear

        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        return rmse


class OnlineAutoencoderEnsemble(BaseModel):
    """
    Online anomaly detector using an ensemble of lightweight autoencoders.

    The detector first learns feature groups from streaming correlations, then
    trains an ensemble of small autoencoders plus an output autoencoder. This
    implementation uses raw inputs, greedy correlation grouping, and simple
    NumPy autoencoders. The authors' implementation includes its own feature
    mapper and normalized denoising autoencoders, so scores are not expected to
    match it exactly.

    The model is stateful and sample-wise:
    - ``learn_one`` updates model state with a single sample.
    - ``score_one`` computes one anomaly score without mutating state.

    Warm-up phases:
    - ``feature_map_warmup``: build feature groups from correlations.
    - ``detector_warmup``: train ensemble and output autoencoders.
    - ``ready``: score samples; optionally keep adapting if enabled.

    Args:
        max_ae_size: Maximum number of input features assigned to one ensemble
            autoencoder.
        feature_map_grace: Number of learned samples used to estimate feature
            correlations before the autoencoder ensemble is created.
        ad_grace: Number of subsequent learned samples used to train the
            ensemble and output autoencoder before scoring begins. A value of
            zero trains once on the feature-map transition sample and becomes
            ready immediately.
        learning_rate: Positive stochastic-gradient step size used by every
            NumPy autoencoder.
        hidden_ratio: Hidden-layer width as a fraction of input width, in
            ``(0, 1]``. Width is rounded up and, for inputs wider than one,
            capped below the input width.
        adaptive_after_warmup: Continue training on calls to ``learn_one``
            after the model reaches the ready phase.
        seed: Seed for model-local NumPy generators. ``None`` selects
            nondeterministic generator initialization.

    References:
        Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018).
        Kitsune: An Ensemble of Autoencoders for Online Network Intrusion
        Detection. NDSS 2018.
        Original KitNET implementation: https://github.com/ymirsky/KitNET-py
    """

    def __init__(
        self,
        max_ae_size: int = 10,
        feature_map_grace: int = 5_000,
        ad_grace: int = 50_000,
        learning_rate: float = 0.1,
        hidden_ratio: float = 0.75,
        adaptive_after_warmup: bool = False,
        seed: int | None = None,
    ) -> None:
        if max_ae_size <= 0:
            raise ValueError("max_ae_size must be positive")
        if feature_map_grace <= 0:
            raise ValueError("feature_map_grace must be positive")
        if ad_grace < 0:
            raise ValueError("ad_grace must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 < hidden_ratio <= 1.0):
            raise ValueError("hidden_ratio must be in (0, 1]")

        self.max_ae_size = max_ae_size
        self.feature_map_grace = feature_map_grace
        self.ad_grace = ad_grace
        self.learning_rate = learning_rate
        self.hidden_ratio = hidden_ratio
        self.adaptive_after_warmup = adaptive_after_warmup
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset learned state while preserving hyperparameters."""
        self.rng = np.random.default_rng(self.seed)
        self._schema = FeatureSchema()
        self._phase = _PHASE_FEATURE_MAP

        self._samples_seen = 0
        self._feature_map_samples = 0
        self._detector_samples = 0

        self._sum: np.ndarray | None = None
        self._sum_sq: np.ndarray | None = None
        self._sum_cross: np.ndarray | None = None

        self._feature_groups: list[np.ndarray] = []
        self._ensemble: list[_NumpyAutoencoder] = []
        self._output_ae: _NumpyAutoencoder | None = None

    def reset(self) -> None:
        """Public state reset."""
        self._reset_state()

    @property
    def phase(self) -> str:
        """Current warm-up/training phase."""
        return self._phase

    @property
    def is_ready(self) -> bool:
        """Whether the model is ready to produce non-zero anomaly scores."""
        return self._phase == _PHASE_READY

    def _update_feature_statistics(self, x_vec: np.ndarray) -> None:
        """Accumulate first/second moments for online correlation estimates."""
        if self._sum is None or self._sum_sq is None or self._sum_cross is None:
            n_features = x_vec.size
            self._sum = np.zeros(n_features, dtype=np.float64)
            self._sum_sq = np.zeros(n_features, dtype=np.float64)
            self._sum_cross = np.zeros((n_features, n_features), dtype=np.float64)

        self._sum += x_vec
        self._sum_sq += x_vec * x_vec
        self._sum_cross += np.outer(x_vec, x_vec)
        self._feature_map_samples += 1

    def _build_feature_groups(self) -> list[np.ndarray]:
        """Build feature groups from absolute Pearson correlations."""
        if self._sum is None or self._sum_sq is None or self._sum_cross is None:
            raise RuntimeError("Feature statistics are not initialized")
        if self._feature_map_samples <= 0:
            raise RuntimeError("No feature-map samples collected")

        count = float(self._feature_map_samples)
        means = self._sum / count
        variances = (self._sum_sq / count) - (means * means)
        variances = np.clip(variances, 1e-12, None)
        std = np.sqrt(variances)

        covariance = (self._sum_cross / count) - np.outer(means, means)
        denom = np.outer(std, std)
        correlations = np.divide(
            covariance,
            denom,
            out=np.zeros_like(covariance),
            where=denom > 0.0,
        )
        abs_corr = np.abs(np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0))
        np.fill_diagonal(abs_corr, 1.0)

        n_features = abs_corr.shape[0]
        if n_features <= self.max_ae_size:
            return [np.arange(n_features, dtype=np.int32)]

        strengths = abs_corr.sum(axis=1)
        unassigned: set[int] = set(range(n_features))
        groups: list[np.ndarray] = []

        while unassigned:
            seed = max(unassigned, key=lambda idx: (float(strengths[idx]), -idx))
            group = [seed]
            unassigned.remove(seed)

            while unassigned and len(group) < self.max_ae_size:
                candidate = max(
                    unassigned,
                    key=lambda idx: (float(abs_corr[idx, group].mean()), -idx),
                )
                group.append(candidate)
                unassigned.remove(candidate)

            groups.append(np.array(sorted(group), dtype=np.int32))

        return groups

    def _hidden_dim(self, input_dim: int) -> int:
        """Compute hidden-layer width for one autoencoder."""
        if input_dim <= 1:
            return 1
        raw = int(np.ceil(input_dim * self.hidden_ratio))
        compressed = min(max(1, raw), input_dim - 1)
        return compressed

    def _spawn_rng(self) -> np.random.Generator:
        """Spawn a deterministic child RNG from model RNG state."""
        seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
        return np.random.default_rng(seed)

    def _initialize_detector(self) -> None:
        """Create the ensemble and output autoencoders after feature mapping."""
        self._feature_groups = self._build_feature_groups()
        self._ensemble = []

        for group in self._feature_groups:
            input_dim = int(group.size)
            hidden_dim = self._hidden_dim(input_dim)
            self._ensemble.append(
                _NumpyAutoencoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    learning_rate=self.learning_rate,
                    rng=self._spawn_rng(),
                )
            )

        output_dim = len(self._feature_groups)
        output_hidden = self._hidden_dim(output_dim)
        self._output_ae = _NumpyAutoencoder(
            input_dim=output_dim,
            hidden_dim=output_hidden,
            learning_rate=self.learning_rate,
            rng=self._spawn_rng(),
        )

    def _ensemble_errors(self, x_vec: np.ndarray, train: bool) -> np.ndarray:
        """Compute per-sub-autoencoder reconstruction errors."""
        if not self._feature_groups or not self._ensemble:
            raise RuntimeError("Detector is not initialized")

        errors = np.empty(len(self._ensemble), dtype=np.float64)
        for idx, (group, autoencoder) in enumerate(
            zip(self._feature_groups, self._ensemble, strict=False)
        ):
            subset = x_vec[group]
            errors[idx] = (
                autoencoder.learn(subset) if train else autoencoder.score(subset)
            )
        return errors

    def _train_detector(self, x_vec: np.ndarray) -> None:
        """Train ensemble and output autoencoders on one sample."""
        if self._output_ae is None:
            raise RuntimeError("Output autoencoder is not initialized")

        errors = self._ensemble_errors(x_vec, train=True)
        self._output_ae.learn(errors)

    def _score_detector(self, x_vec: np.ndarray) -> float:
        """Compute anomaly score from current detector state."""
        if self._output_ae is None:
            raise RuntimeError("Output autoencoder is not initialized")

        errors = self._ensemble_errors(x_vec, train=False)
        return self._output_ae.score(errors)

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with a single sample."""
        prepared = self._schema.preview(x)
        x_vec = prepared.values

        if self._phase == _PHASE_FEATURE_MAP:
            self._update_feature_statistics(x_vec)
            if self._feature_map_samples >= self.feature_map_grace:
                self._initialize_detector()
                if self.ad_grace == 0:
                    # Ensure "ready" implies detector weights saw at least one sample.
                    self._train_detector(x_vec)
                    self._detector_samples += 1
                    self._phase = _PHASE_READY
                else:
                    self._phase = _PHASE_DETECTOR
        elif self._phase == _PHASE_DETECTOR:
            self._train_detector(x_vec)
            self._detector_samples += 1
            if self._detector_samples >= self.ad_grace:
                self._phase = _PHASE_READY
        elif self._phase == _PHASE_READY and self.adaptive_after_warmup:
            self._train_detector(x_vec)

        self._samples_seen += 1
        self._schema.commit(prepared)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for a sample without mutating model state."""
        prepared = self._schema.preview(x)
        if self._phase != _PHASE_READY:
            return 0.0
        return float(max(0.0, self._score_detector(prepared.values)))

    def __repr__(self) -> str:
        return (
            f"OnlineAutoencoderEnsemble(max_ae_size={self.max_ae_size}, "
            f"feature_map_grace={self.feature_map_grace}, ad_grace={self.ad_grace}, "
            f"learning_rate={self.learning_rate}, hidden_ratio={self.hidden_ratio}, "
            f"adaptive_after_warmup={self.adaptive_after_warmup}, phase='{self._phase}', "
            f"feature_groups={len(self._feature_groups)})"
        )
