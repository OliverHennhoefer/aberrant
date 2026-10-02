"""Exact rolling median/IQR scaling with explicit calibration control."""

import math
from collections.abc import Iterator
from contextlib import contextmanager

from aberrant.base.transformer import BaseTransformer
from aberrant.utils.statistics import linear_quantile
from aberrant.utils.validation import coerce_feature_values, coerce_finite_number


class RollingRobustScaler(BaseTransformer):
    """Center each feature by its recent median and divide by its recent IQR.

    Quartiles use linear interpolation at positions ``(n - 1) * q``. Each
    feature retains its last ``window_size`` observations independently;
    missing features neither advance nor evict that feature's window. A zero
    interquartile range uses ``fallback_scale``, preserving novel deviations
    from a constant reference instead of mapping every candidate to zero.

    ``transform_one`` is read-only and uses provisional statistics even before
    ``min_samples`` observations. Read ``is_ready`` to exclude this warm-up
    from evaluation. Freeze a ready calibration before fitting a downstream
    model when its stored references must share one coordinate system.
    Adaptive updates do not re-express downstream points or split boundaries.

    For F features in an event and window size W, adaptive learning costs
    O(F W log W), transformation O(F), and persistent storage O(D W) for D
    distinct learned features. Use a schema guard to bound D. Frozen learning
    validates inputs in O(F) and leaves calibration unchanged.

    Args:
        window_size: Maximum observations retained per feature.
        min_samples: Minimum retained observations per learned feature for
            readiness; this does not disable provisional transformation.
        fallback_scale: Positive finite divisor used only when IQR is zero.
    """

    def __init__(
        self,
        window_size: int = 256,
        min_samples: int = 1,
        fallback_scale: float = 1.0,
    ) -> None:
        for name, value in (("window_size", window_size), ("min_samples", min_samples)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min_samples > window_size:
            raise ValueError("min_samples must not exceed window_size")
        scale = coerce_finite_number(fallback_scale, label="fallback_scale")
        if scale <= 0.0:
            raise ValueError("fallback_scale must be positive")
        self.window_size = window_size
        self.min_samples = min_samples
        self.fallback_scale = scale
        self._windows: dict[str, tuple[float, ...]] = {}
        self._statistics: dict[str, tuple[float, float]] = {}
        self._frozen = False

    @property
    def sample_counts(self) -> dict[str, int]:
        """Return an owned snapshot of retained observations per feature."""
        return {feature: len(window) for feature, window in self._windows.items()}

    @property
    def is_ready(self) -> bool:
        """Whether every learned feature has at least ``min_samples`` values.

        An empty scaler is not ready. Learning a new feature can make an
        adaptive scaler unready again; use a schema guard for fixed inputs.
        """
        return bool(self._windows) and all(
            len(window) >= self.min_samples for window in self._windows.values()
        )

    @property
    def is_frozen(self) -> bool:
        """Whether learning validates events without updating calibration."""
        return self._frozen

    def freeze(self) -> None:
        """Freeze a ready calibration; unseen features are then rejected."""
        if not self.is_ready:
            raise ValueError("Cannot freeze before all learned features are ready")
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume rolling updates from the retained calibration window."""
        self._frozen = False

    def reset(self) -> None:
        """Clear all learned features and unfreeze, preserving parameters."""
        self._windows.clear()
        self._statistics.clear()
        self._frozen = False

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Restore this event's windows if a downstream pipeline update fails."""
        previous = {
            feature: (self._windows.get(feature), self._statistics.get(feature))
            for feature in x
        }
        try:
            yield
        except BaseException:
            for feature, (window, statistics) in previous.items():
                if window is None or statistics is None:
                    self._windows.pop(feature, None)
                    self._statistics.pop(feature, None)
                else:
                    self._windows[feature] = window
                    self._statistics[feature] = statistics
            raise

    def learn_one(self, x: dict[str, float]) -> None:
        """Validate and stage all feature windows before committing any update."""
        values = coerce_feature_values(x)
        if self._frozen:
            for feature in values:
                self._require_seen(feature)
            return
        windows: dict[str, tuple[float, ...]] = {}
        statistics: dict[str, tuple[float, float]] = {}
        for feature, value in values.items():
            window = (*self._windows.get(feature, ()), value)[-self.window_size :]
            ordered = sorted(window)
            median = linear_quantile(ordered, 0.5)
            iqr = linear_quantile(ordered, 0.75) - linear_quantile(ordered, 0.25)
            if not math.isfinite(median) or not math.isfinite(iqr):
                raise ValueError(f"Feature '{feature}' median and IQR must be finite")
            windows[feature] = window
            statistics[feature] = (median, iqr if iqr > 0.0 else self.fallback_scale)
        self._windows.update(windows)
        self._statistics.update(statistics)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """Scale without learning; reject unseen features or non-finite output."""
        transformed = {}
        for feature, value in coerce_feature_values(x).items():
            self._require_seen(feature)
            median, scale = self._statistics[feature]
            scaled = (value - median) / scale
            if not math.isfinite(scaled):
                # A finite result may exist even when subtraction overflows.
                scaled = value / scale - median / scale
            if not math.isfinite(scaled):
                raise ValueError(f"Feature '{feature}' scaled value must be finite")
            transformed[feature] = scaled
        return transformed

    def _require_seen(self, feature: str) -> None:
        if feature not in self._statistics:
            raise ValueError(f"Feature '{feature}' has not been seen during learning")

    def __repr__(self) -> str:
        return (
            f"RollingRobustScaler(window_size={self.window_size}, "
            f"min_samples={self.min_samples}, fallback_scale={self.fallback_scale})"
        )
