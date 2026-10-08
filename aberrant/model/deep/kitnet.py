"""Online ensemble of lightweight autoencoders for anomaly detection."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.utils.validation import FeatureSchema

_PHASE_FEATURE_MAP = "feature_map_warmup"
_PHASE_DETECTOR = "detector_warmup"
_PHASE_READY = "ready"


def _finite_array(values: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise OverflowError(f"Autoencoder {label} exceeds the float64 range")
    return values


def _rmse(error: np.ndarray) -> float:
    """Avoid squaring unscaled residuals when the RMSE is representable."""
    _finite_array(error, "residual")
    scale = float(np.max(np.abs(error)))
    if scale == 0.0:
        return 0.0
    return float(np.linalg.norm(error / scale) / math.sqrt(error.size) * scale)


@dataclass
class _AutoencoderUpdate:
    """Finite parameters and pre-update error for one proposed training step."""

    error: float
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


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
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
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
        return _rmse(error)

    def learn(self, x: np.ndarray) -> float:
        """Update parameters on one sample and return RMSE."""
        update = self.propose_update(x)
        self.apply_update(update)
        return update.error

    def propose_update(self, x: np.ndarray) -> _AutoencoderUpdate:
        """Compute a complete finite training step without changing parameters."""
        hidden, output = self._forward(x)
        error = output - x
        rmse = _rmse(error)

        # 0.5 * mean squared error -> gradient: (output - x) / input_dim
        grad_output = error / float(self.input_dim)

        grad_w2 = np.outer(grad_output, hidden)
        grad_b2 = grad_output

        grad_hidden = self.w2.T @ grad_output
        grad_hidden_linear = grad_hidden * (1.0 - hidden * hidden)

        grad_w1 = np.outer(grad_hidden_linear, x)
        grad_b1 = grad_hidden_linear

        # Stage a complete finite update before publishing weights. One extreme
        # sample must not leave NaN weights that poison every subsequent event.
        with np.errstate(over="ignore", invalid="ignore"):
            w2 = _finite_array(self.w2 - self.learning_rate * grad_w2, "weights")
            b2 = _finite_array(self.b2 - self.learning_rate * grad_b2, "bias")
            w1 = _finite_array(self.w1 - self.learning_rate * grad_w1, "weights")
            b1 = _finite_array(self.b1 - self.learning_rate * grad_b1, "bias")
        return _AutoencoderUpdate(rmse, w1, b1, w2, b2)

    def apply_update(self, update: _AutoencoderUpdate) -> None:
        """Publish an already validated training step."""
        self.w1, self.b1 = update.w1, update.b1
        self.w2, self.b2 = update.w2, update.b2


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
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
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

    def _preview_feature_statistics(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preview first/second moments without publishing a feature-map event."""
        if self._sum is None or self._sum_sq is None or self._sum_cross is None:
            n_features = x_vec.size
            previous_sum = np.zeros(n_features, dtype=np.float64)
            previous_sq = np.zeros(n_features, dtype=np.float64)
            previous_cross = np.zeros((n_features, n_features), dtype=np.float64)
        else:
            previous_sum, previous_sq = self._sum, self._sum_sq
            previous_cross = self._sum_cross

        with np.errstate(over="ignore", invalid="ignore"):
            total = _finite_array(previous_sum + x_vec, "feature sum")
            squared = _finite_array(previous_sq + x_vec * x_vec, "feature moment")
            cross = _finite_array(
                previous_cross + np.outer(x_vec, x_vec), "feature cross moment"
            )
        return total, squared, cross

    def _build_feature_groups(
        self,
        total: np.ndarray,
        squared: np.ndarray,
        cross: np.ndarray,
        samples: int,
    ) -> list[np.ndarray]:
        """Build feature groups from absolute Pearson correlations."""
        count = float(samples)
        means = total / count
        variances = (squared / count) - (means * means)
        variances = np.clip(variances, 1e-12, None)
        std = np.sqrt(variances)

        covariance = (cross / count) - np.outer(means, means)
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

    @staticmethod
    def _spawn_rng(parent: np.random.Generator) -> np.random.Generator:
        """Spawn a deterministic child RNG from model RNG state."""
        seed = int(parent.integers(0, np.iinfo(np.int32).max))
        return np.random.default_rng(seed)

    def _create_detector(
        self, groups: list[np.ndarray], rng: np.random.Generator
    ) -> tuple[list[_NumpyAutoencoder], _NumpyAutoencoder]:
        """Create the ensemble and output autoencoders after feature mapping."""
        ensemble = []

        for group in groups:
            input_dim = int(group.size)
            hidden_dim = self._hidden_dim(input_dim)
            ensemble.append(
                _NumpyAutoencoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    learning_rate=self.learning_rate,
                    rng=self._spawn_rng(rng),
                )
            )

        output_dim = len(groups)
        output_hidden = self._hidden_dim(output_dim)
        output_ae = _NumpyAutoencoder(
            input_dim=output_dim,
            hidden_dim=output_hidden,
            learning_rate=self.learning_rate,
            rng=self._spawn_rng(rng),
        )
        return ensemble, output_ae

    def _ensemble_errors(self, x_vec: np.ndarray) -> np.ndarray:
        """Compute per-sub-autoencoder reconstruction errors."""
        if not self._feature_groups or not self._ensemble:
            raise RuntimeError("Detector is not initialized")

        errors = np.empty(len(self._ensemble), dtype=np.float64)
        for idx, (group, autoencoder) in enumerate(
            zip(self._feature_groups, self._ensemble, strict=True)
        ):
            subset = x_vec[group]
            errors[idx] = autoencoder.score(subset)
        return errors

    @staticmethod
    def _train_detector(
        x_vec: np.ndarray,
        groups: list[np.ndarray],
        ensemble: list[_NumpyAutoencoder],
        output_ae: _NumpyAutoencoder | None,
    ) -> None:
        """Validate every network's update before publishing any parameters."""
        if output_ae is None:
            raise RuntimeError("Output autoencoder is not initialized")

        updates = [
            autoencoder.propose_update(x_vec[group])
            for group, autoencoder in zip(groups, ensemble, strict=True)
        ]
        errors = np.asarray([update.error for update in updates], dtype=np.float64)
        output_update = output_ae.propose_update(errors)
        for autoencoder, update in zip(ensemble, updates, strict=True):
            autoencoder.apply_update(update)
        output_ae.apply_update(output_update)

    def _score_detector(self, x_vec: np.ndarray) -> float:
        """Compute anomaly score from current detector state."""
        if self._output_ae is None:
            raise RuntimeError("Output autoencoder is not initialized")

        errors = self._ensemble_errors(x_vec)
        return self._output_ae.score(errors)

    def _learn_feature_map(self, x_vec: np.ndarray) -> None:
        """Publish moments and an optional detector transition as one event."""
        total, squared, cross = self._preview_feature_statistics(x_vec)
        samples = self._feature_map_samples + 1
        if samples >= self.feature_map_grace:
            groups = self._build_feature_groups(total, squared, cross, samples)
            # A rejected transition must not consume the next detector's seeds.
            rng = deepcopy(self.rng)
            ensemble, output_ae = self._create_detector(groups, rng)
            if self.ad_grace == 0:
                # Ensure "ready" implies detector weights saw at least one sample.
                self._train_detector(x_vec, groups, ensemble, output_ae)

            self._feature_groups, self._ensemble, self._output_ae = (
                groups,
                ensemble,
                output_ae,
            )
            self.rng = rng
            self._detector_samples = int(self.ad_grace == 0)
            self._phase = _PHASE_READY if self.ad_grace == 0 else _PHASE_DETECTOR

        self._sum, self._sum_sq, self._sum_cross = total, squared, cross
        self._feature_map_samples = samples

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model state with a single sample."""
        prepared = self._schema.preview(x)
        x_vec = prepared.values

        if self._phase == _PHASE_FEATURE_MAP:
            self._learn_feature_map(x_vec)
        elif self._phase == _PHASE_DETECTOR:
            self._train_detector(
                x_vec, self._feature_groups, self._ensemble, self._output_ae
            )
            self._detector_samples += 1
            if self._detector_samples >= self.ad_grace:
                self._phase = _PHASE_READY
        elif self._phase == _PHASE_READY and self.adaptive_after_warmup:
            self._train_detector(
                x_vec, self._feature_groups, self._ensemble, self._output_ae
            )

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
