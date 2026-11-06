"""Online ECOD and COPOD: empirical tail-probability outlier detectors."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import numpy as np

from aberrant.base.model import BaseModel
from aberrant.base.readiness import Readiness
from aberrant.model.stat._tail_reference import (
    _BaseReference,
    _SketchReference,
    _WindowReference,
)
from aberrant.utils.validation import NumericEventBoundary, coerce_integer_count

_DEFAULT_WINDOW_SIZE = 1024
_DEFAULT_COMPRESSION = 100.0
_DEFAULT_TAIL_QUANTILE = 0.98
_MIN_COMPRESSION = 10.0


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


class _OnlineTailDetector(BaseModel):
    """Shared state and scoring for ``OnlineECOD`` and ``OnlineCOPOD``."""

    _pyod_variant: ClassVar[Literal["ecod", "copod"]]

    def __init__(
        self,
        memory: Literal["sketch", "window", "hybrid"] = "sketch",
        window_size: int | None = None,
        half_life: float | None = None,
        compression: float | None = None,
        tail: Literal["empirical", "pareto"] = "empirical",
        tail_quantile: float | None = None,
        min_samples: int = 64,
        aggregation: Literal["paper", "pyod"] = "paper",
        time_key: str | None = None,
    ) -> None:
        if memory not in ("sketch", "window", "hybrid"):
            raise ValueError("memory must be 'sketch', 'window' or 'hybrid'")
        if tail not in ("empirical", "pareto"):
            raise ValueError("tail must be 'empirical' or 'pareto'")
        if aggregation not in ("paper", "pyod"):
            raise ValueError("aggregation must be 'paper' or 'pyod'")
        uses_window = memory in ("window", "hybrid")
        uses_sketch = memory in ("sketch", "hybrid")

        if window_size is not None and not uses_window:
            raise ValueError("window_size applies only to memory='window' or 'hybrid'")
        resolved_window = (
            coerce_integer_count(
                _DEFAULT_WINDOW_SIZE if window_size is None else window_size,
                label="window_size",
                minimum=2,
            )
            if uses_window
            else None
        )
        if half_life is not None and not uses_sketch:
            raise ValueError("half_life applies only to memory='sketch' or 'hybrid'")
        resolved_half_life = (
            None if half_life is None else _positive_finite(half_life, "half_life")
        )
        if compression is not None and not uses_sketch:
            raise ValueError("compression applies only to memory='sketch' or 'hybrid'")
        resolved_compression = (
            _positive_finite(
                _DEFAULT_COMPRESSION if compression is None else compression,
                "compression",
            )
            if uses_sketch
            else None
        )
        if resolved_compression is not None and resolved_compression < _MIN_COMPRESSION:
            raise ValueError(f"compression must be at least {_MIN_COMPRESSION:g}")
        if tail_quantile is not None and tail != "pareto":
            raise ValueError("tail_quantile applies only to tail='pareto'")
        resolved_tail_quantile: float | None = None
        if tail == "pareto":
            resolved_tail_quantile = _positive_finite(
                _DEFAULT_TAIL_QUANTILE if tail_quantile is None else tail_quantile,
                "tail_quantile",
            )
            if resolved_tail_quantile >= 1.0:
                raise ValueError("tail_quantile must be in (0, 1)")
        min_samples = coerce_integer_count(min_samples, label="min_samples")
        if resolved_window is not None and min_samples > resolved_window:
            raise ValueError("min_samples must not exceed window_size")
        if time_key is not None and (not isinstance(time_key, str) or not time_key):
            raise ValueError("time_key must be a non-empty string or None")

        self.memory = memory
        self.window_size = resolved_window
        self.half_life = resolved_half_life
        self.compression = resolved_compression
        self.tail = tail
        self.tail_quantile = resolved_tail_quantile
        self.min_samples = min_samples
        self.aggregation = aggregation
        self.time_key = time_key
        self._reset_state()

    def _reset_state(self) -> None:
        self._boundary = NumericEventBoundary(time_key=self.time_key)
        self._primary: _BaseReference | None = None
        self._history: _SketchReference | None = None
        self._samples_seen = 0

    def reset(self) -> None:
        """Reset learned state while keeping hyperparameters."""
        self._reset_state()

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed with ``learn_one``."""
        return self._samples_seen

    @property
    def is_ready(self) -> bool:
        """Whether ``min_samples`` samples have been learned."""
        return self._samples_seen >= self.min_samples

    @property
    def readiness(self) -> Readiness:
        """Return a detached description of the warm-up state."""
        return Readiness(
            self.is_ready,
            min(self._samples_seen, self.min_samples),
            self.min_samples,
        )

    def _create_references(self, n_features: int) -> None:
        if self.memory == "sketch":
            assert self.compression is not None
            self._primary = _SketchReference(
                n_features,
                self.compression,
                half_life=self.half_life,
                tail_quantile=self.tail_quantile,
            )
            return
        assert self.window_size is not None
        self._primary = _WindowReference(
            n_features, self.window_size, tail_quantile=self.tail_quantile
        )
        if self.memory == "hybrid":
            assert self.compression is not None
            self._history = _SketchReference(
                n_features,
                self.compression,
                half_life=self.half_life,
                tail_quantile=self.tail_quantile,
            )

    def learn_one(self, x: dict[str, float]) -> None:
        """Add one sample to the reference distribution."""
        event = self._boundary.preview(x)
        vector = event.features.values
        if self._primary is None:
            self._create_references(len(vector))
        assert self._primary is not None
        evicted = self._primary.learn(vector)
        if evicted is not None and self._history is not None:
            self._history.learn(evicted)
        self._samples_seen += 1
        self._boundary.commit(event)

    def score_one(self, x: dict[str, float]) -> float:
        """Score one sample against the reference without learning it."""
        event = self._boundary.preview(x)
        if self._primary is None or not self.is_ready:
            return 0.0
        score, _ = self._evaluate(event.features.values)
        return score

    def explain_one(self, x: dict[str, float]) -> dict[str, float]:
        """Return each feature's contribution to ``score_one(x)``.

        The contributions are the per-feature negative log tail probabilities
        that the score aggregates, so they sum to the score. They are all zero
        during warm-up.
        """
        event = self._boundary.preview(x)
        names = event.features.names
        if self._primary is None or not self.is_ready:
            return dict.fromkeys(names, 0.0)
        _, terms = self._evaluate(event.features.values)
        return dict(zip(names, terms.tolist(), strict=True))

    def _evaluate(self, vector: np.ndarray) -> tuple[float, np.ndarray]:
        assert self._primary is not None
        score, terms = self._aggregate(self._primary, vector)
        history = self._history
        if history is not None and history.size >= self.min_samples:
            # ESOD flags a point only when it is an outlier against both the
            # recent window and the long-run history; as a single score that
            # is the smaller of the two.
            history_score, history_terms = self._aggregate(history, vector)
            if history_score < score:
                return history_score, history_terms
        return score, terms

    def _aggregate(
        self, reference: _BaseReference, vector: np.ndarray
    ) -> tuple[float, np.ndarray]:
        log_left, log_right = reference.tail_log_probabilities(vector)
        left = -log_left
        right = -log_right
        signs = reference.skew_signs(vector)

        if self.aggregation == "paper":
            # Both papers: a negative skewness selects the left tail, any other
            # the right; the score is the largest of the three summed scores.
            skewed = np.where(signs < 0.0, left, right)
            candidates = (left, right, skewed)
            totals = [float(np.add.reduce(terms)) for terms in candidates]
            best = int(np.argmax(totals))
            return totals[best], candidates[best]

        # PyOD: the skewness term uses np.sign, so a zero sign adds both tails,
        # and the per-feature maximum is summed.
        skewed = np.where(signs < 0.0, left, np.where(signs > 0.0, right, left + right))
        if self._pyod_variant == "ecod":
            terms = np.maximum(skewed, np.maximum(left, right))
        else:
            terms = np.maximum(skewed, 0.5 * (left + right))
        return float(np.add.reduce(terms)), terms

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"memory={self.memory!r}, window_size={self.window_size}, "
            f"half_life={self.half_life}, compression={self.compression}, "
            f"tail={self.tail!r}, tail_quantile={self.tail_quantile}, "
            f"min_samples={self.min_samples}, aggregation={self.aggregation!r}, "
            f"time_key={self.time_key!r}, samples_seen={self._samples_seen})"
        )


class OnlineECOD(_OnlineTailDetector):
    """
    Online ECOD: outlier detection with empirical cumulative distributions.

    Every feature keeps its own reference distribution, and a sample is scored
    by how far into the tails of those distributions its values fall. The
    candidate is scored as if it were appended to the reference with weight
    one, so a tail probability is ``(#{r <= x} + 1) / (n + 1)``: it is never
    zero, and with empirical tails each feature contributes at most
    ``log(n + 1)``. ``score_one``
    never changes state; call ``learn_one`` afterwards to add the sample.

    Features are treated as independent, as in both papers; dependence between
    features is not modeled. The feature schema is fixed after the first
    ``learn_one`` call, and scores are ``0.0`` until ``min_samples`` samples
    have been learned.

    Memories:

    - ``"sketch"`` (default): a weighted merging t-digest per feature with
      ``compression`` centroids at most, plus an exact buffer of recent
      values. ``half_life=None`` never forgets, which approximates the batch
      method on everything seen so far; a ``half_life`` halves a sample's
      weight every ``half_life`` learned samples. State is bounded.
    - ``"window"``: the exact last ``window_size`` values per feature. The
      score equals the batch method fitted on the window plus the candidate.
    - ``"hybrid"``: an exact window plus a sketch of the values it evicts, as
      in ESOD. The score is the smaller of the window and history scores, the
      single-score form of ESOD's rule that a point must be an outlier against
      both; the history joins once it holds ``min_samples`` weight.

    ``tail="pareto"`` replaces the empirical probability beyond each
    reference's ``tail_quantile`` quantile with a generalized Pareto tail
    fitted to the reference's own exceedances, as in SPOT, so values past the
    observed range keep scoring higher the further out they are instead of
    saturating at ``log(n + 1)``.

    For F features, scoring costs O(F W) with a window and O(F (C + B)) with
    a sketch of C centroids and a merge buffer of B = 128 values; learning
    costs O(F W) and amortized O(F (C / B + 1)) respectively. State is
    O(F W) and O(F (C + B)).

    Args:
        memory: Reference memory: ``"sketch"``, ``"window"`` or ``"hybrid"``.
        window_size: Window length for ``"window"`` and ``"hybrid"``; defaults
            to 1024. Rejected for ``"sketch"``.
        half_life: Learned samples after which a sketch weight halves, for
            ``"sketch"`` and the ``"hybrid"`` history. ``None`` never forgets.
        compression: t-digest compression for ``"sketch"`` and ``"hybrid"``;
            defaults to 100 and must be at least 10.
        tail: ``"empirical"`` tail probabilities, or ``"pareto"`` tails.
        tail_quantile: Reference quantile above which (and below whose mirror
            image) the Pareto tail applies; defaults to 0.98, as in SPOT.
            Only valid with ``tail="pareto"``.
        min_samples: Learned samples required before scoring.
        aggregation: ``"paper"`` takes the largest of the left-tail,
            right-tail and skewness-corrected sums. ``"pyod"`` reproduces
            PyOD's implementation, which sums a per-feature maximum instead.
        time_key: Event-time field to exclude from the features. Explicit
            times must be finite and non-decreasing.

    With ``aggregation="paper"`` the score is ECOD's
    ``max(O_left, O_right, O_auto)``. It is identical to ``OnlineCOPOD``'s,
    because both papers use the same final formula. ``aggregation="pyod"``
    reproduces PyOD's ``ECOD``, which sums the per-feature maximum of the three
    tail scores.

    References:
        Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., and Chen, G. H.
        (2022). ECOD: Unsupervised outlier detection using empirical
        cumulative distribution functions. IEEE Transactions on Knowledge and
        Data Engineering. https://arxiv.org/abs/2201.00382

        Allka, X., Ferrer-Cid, P., Barcelo-Ordinas, J. M., Garcia-Vidal, J.,
        and Avila-Torrado, A. (2025). ESOD: An edge streaming data outlier
        detection framework for IoT platforms.
        https://doi.org/10.1109/JIOT.2025.3579057

        Dunning, T., and Ertl, O. (2019). Computing extremely accurate
        quantiles using t-digests. https://arxiv.org/abs/1902.04023

        Siffer, A., Fouque, P.-A., Termier, A., and Largouet, C. (2017).
        Anomaly detection in streams with extreme value theory. Proceedings
        of the 23rd ACM SIGKDD Conference, 1067-1075.

        Hosking, J. R. M., and Wallis, J. R. (1987). Parameter and quantile
        estimation for the generalized Pareto distribution. Technometrics,
        29(3), 339-349.
    """

    _pyod_variant = "ecod"


class OnlineCOPOD(_OnlineTailDetector):
    """
    Online COPOD: copula-based outlier detection with empirical marginals.

    Every feature keeps its own reference distribution, and a sample is scored
    by how far into the tails of those distributions its values fall. The
    candidate is scored as if it were appended to the reference with weight
    one, so a tail probability is ``(#{r <= x} + 1) / (n + 1)``: it is never
    zero, and with empirical tails each feature contributes at most
    ``log(n + 1)``. ``score_one``
    never changes state; call ``learn_one`` afterwards to add the sample.

    Features are treated as independent, as in both papers; dependence between
    features is not modeled. The feature schema is fixed after the first
    ``learn_one`` call, and scores are ``0.0`` until ``min_samples`` samples
    have been learned.

    Memories:

    - ``"sketch"`` (default): a weighted merging t-digest per feature with
      ``compression`` centroids at most, plus an exact buffer of recent
      values. ``half_life=None`` never forgets, which approximates the batch
      method on everything seen so far; a ``half_life`` halves a sample's
      weight every ``half_life`` learned samples. State is bounded.
    - ``"window"``: the exact last ``window_size`` values per feature. The
      score equals the batch method fitted on the window plus the candidate.
    - ``"hybrid"``: an exact window plus a sketch of the values it evicts, as
      in ESOD. The score is the smaller of the window and history scores, the
      single-score form of ESOD's rule that a point must be an outlier against
      both; the history joins once it holds ``min_samples`` weight.

    ``tail="pareto"`` replaces the empirical probability beyond each
    reference's ``tail_quantile`` quantile with a generalized Pareto tail
    fitted to the reference's own exceedances, as in SPOT, so values past the
    observed range keep scoring higher the further out they are instead of
    saturating at ``log(n + 1)``.

    For F features, scoring costs O(F W) with a window and O(F (C + B)) with
    a sketch of C centroids and a merge buffer of B = 128 values; learning
    costs O(F W) and amortized O(F (C / B + 1)) respectively. State is
    O(F W) and O(F (C + B)).

    Args:
        memory: Reference memory: ``"sketch"``, ``"window"`` or ``"hybrid"``.
        window_size: Window length for ``"window"`` and ``"hybrid"``; defaults
            to 1024. Rejected for ``"sketch"``.
        half_life: Learned samples after which a sketch weight halves, for
            ``"sketch"`` and the ``"hybrid"`` history. ``None`` never forgets.
        compression: t-digest compression for ``"sketch"`` and ``"hybrid"``;
            defaults to 100 and must be at least 10.
        tail: ``"empirical"`` tail probabilities, or ``"pareto"`` tails.
        tail_quantile: Reference quantile above which (and below whose mirror
            image) the Pareto tail applies; defaults to 0.98, as in SPOT.
            Only valid with ``tail="pareto"``.
        min_samples: Learned samples required before scoring.
        aggregation: ``"paper"`` takes the largest of the left-tail,
            right-tail and skewness-corrected sums. ``"pyod"`` reproduces
            PyOD's implementation, which sums a per-feature maximum instead.
        time_key: Event-time field to exclude from the features. Explicit
            times must be finite and non-decreasing.

    With ``aggregation="paper"`` the score is COPOD's ``max(p_l, p_r, p_s)``.
    The paper approximates the empirical copula by the product of its
    marginals, so this equals ``OnlineECOD``'s paper score.
    ``aggregation="pyod"`` reproduces PyOD's ``COPOD``, which sums the
    per-feature maximum of the skewness-corrected score and the mean of the
    two tail scores.

    References:
        Li, Z., Zhao, Y., Botta, N., Ionescu, C., and Hu, X. (2020). COPOD:
        Copula-based outlier detection. IEEE International Conference on Data
        Mining. https://arxiv.org/abs/2009.09463

        Allka, X., Ferrer-Cid, P., Barcelo-Ordinas, J. M., Garcia-Vidal, J.,
        and Avila-Torrado, A. (2025). ESOD: An edge streaming data outlier
        detection framework for IoT platforms.
        https://doi.org/10.1109/JIOT.2025.3579057

        Dunning, T., and Ertl, O. (2019). Computing extremely accurate
        quantiles using t-digests. https://arxiv.org/abs/1902.04023

        Siffer, A., Fouque, P.-A., Termier, A., and Largouet, C. (2017).
        Anomaly detection in streams with extreme value theory. Proceedings
        of the 23rd ACM SIGKDD Conference, 1067-1075.

        Hosking, J. R. M., and Wallis, J. R. (1987). Parameter and quantile
        estimation for the generalized Pareto distribution. Technometrics,
        29(3), 339-349.
    """

    _pyod_variant = "copod"
