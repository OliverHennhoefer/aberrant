"""Per-feature reference distributions for empirical tail-probability detectors.

Each reference summarizes, for every feature independently, the values that
``learn`` has added, and answers two read-only questions about a candidate
vector ``x``:

- the left and right tail probabilities of every coordinate, computed as if
  ``x`` were appended to the reference with weight one, and
- the sign of every feature's skewness over the same augmented reference.

Appending the candidate is the "+1" rule ``(#{r <= x} + 1) / (n + 1)``. It is
what PyOD's ECOD and COPOD compute when they score a one-row batch, it keeps
every probability positive without an epsilon, and it bounds a feature's tail
score by ``log(n + 1)``.

Two memories are provided. ``_WindowReference`` keeps the last ``W`` values of
every feature sorted and is exact. ``_SketchReference`` keeps a weighted
merging t-digest per feature (Dunning and Ertl, 2019) whose weights optionally
decay geometrically, so its state is bounded however long the stream runs.
"""

from __future__ import annotations

import math

import numpy as np

# Smallest exceedance weight for which a generalized Pareto tail is fitted;
# below it the empirical tail probability is used.
_MIN_EXCEEDANCE_WEIGHT = 10.0

# Stored sketch weights are inflated so that decay never touches old state;
# they are renormalized before the inflation factor can overflow.
_MAX_INFLATION = 1e100

# Variance at or below this resolution of the mean counts as zero, the rule
# scipy.stats.skew uses before it reports an undefined skewness.
_FLOAT_RESOLUTION = float(np.finfo(np.float64).resolution)


def augmented_skew_signs(
    weight: float | np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Return each feature's skewness sign after adding ``x`` with weight one.

    ``weight``, ``mean``, ``m2`` and ``m3`` are the total weight, mean and
    second and third central moment sums of the reference. A feature whose
    augmented variance is numerically zero has skewness ``0``.
    """
    total = weight + 1.0
    delta = x - mean
    new_mean = mean + delta / total
    new_m2 = m2 + delta * delta * weight / total
    new_m3 = (
        m3
        + delta**3 * weight * (weight - 1.0) / (total * total)
        - 3.0 * delta * m2 / total
    )
    zero_variance = new_m2 / total <= (_FLOAT_RESOLUTION * new_mean) ** 2
    return np.where(zero_variance, 0.0, np.sign(new_m3))


def _add_moments(
    weight: float,
    mean: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    x: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Combine weighted central moment sums with one unit-weight point (Pebay)."""
    total = weight + 1.0
    delta = x - mean
    new_mean = mean + delta / total
    new_m3 = (
        m3
        + delta**3 * weight * (weight - 1.0) / (total * total)
        - 3.0 * delta * m2 / total
    )
    new_m2 = m2 + delta * delta * weight / total
    return total, new_mean, new_m2, new_m3


def _remove_moments(
    weight: float,
    mean: np.ndarray,
    m2: np.ndarray,
    m3: np.ndarray,
    x: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Invert ``_add_moments`` for a unit-weight point ``x`` in the reference."""
    remaining = weight - 1.0
    if remaining <= 0.0:
        zeros = np.zeros_like(mean)
        return 0.0, zeros, zeros.copy(), zeros.copy()
    old_mean = (weight * mean - x) / remaining
    delta = x - old_mean
    old_m2 = np.maximum(m2 - delta * delta * remaining / weight, 0.0)
    old_m3 = (
        m3
        - delta**3 * remaining * (remaining - 1.0) / (weight * weight)
        + 3.0 * delta * old_m2 / weight
    )
    return remaining, old_mean, old_m2, old_m3


def pareto_log_tail(
    values: np.ndarray,
    weights: np.ndarray,
    total: float,
    x: float,
    tail_quantile: float,
) -> float | None:
    """Return a generalized Pareto estimate of ``log P(X >= x)``, if one applies.

    ``values`` are the reference values (or centroid means) in ascending order
    with their ``weights``; ``total`` is their summed weight. Following SPOT
    (Siffer et al., 2017), the reference mass above its ``tail_quantile``
    quantile ``t`` is modeled by a generalized Pareto distribution of the
    excesses over ``t``. The excess distribution is fitted with weighted
    probability-weighted moments (Hosking and Wallis, 1987), which are closed
    form and more stable than maximum likelihood on the few dozen excesses a
    stream tail holds. A negative shape estimate, which implies a finite upper
    endpoint and therefore an infinite score past it, is replaced by the
    exponential tail of the same mean excess.

    The candidate counts as one more exceedance, matching the "+1" rule of the
    empirical probabilities, so ``P(X >= x) = (w_t + 1) / (total + 1) * S(x - t)``
    where ``w_t`` is the exceedance weight and ``S`` the fitted excess survival
    function. ``None`` means ``x`` is not beyond ``t`` or the tail holds too
    little weight to fit, and the caller keeps the empirical probability.
    """
    budget = (1.0 - tail_quantile) * total
    from_top = np.cumsum(weights[::-1])
    exceedances = int(np.searchsorted(from_top, budget, side="right"))
    if exceedances >= len(values) or exceedances == 0:
        return None
    exceedance_weight = float(from_top[exceedances - 1])
    if exceedance_weight < _MIN_EXCEEDANCE_WEIGHT:
        return None
    first = len(values) - exceedances
    threshold = float(values[first - 1])
    if x <= threshold:
        return None

    excesses = values[first:] - threshold
    excess_weights = weights[first:]
    below = np.cumsum(excess_weights) - excess_weights
    plotting = (below + 0.5 * excess_weights) / exceedance_weight
    a0 = float(np.dot(excess_weights, excesses)) / exceedance_weight
    a1 = float(np.dot(excess_weights, excesses * (1.0 - plotting))) / exceedance_weight
    denominator = a0 - 2.0 * a1
    if a0 <= 0.0 or denominator <= 0.0:
        return None
    shape = 2.0 - a0 / denominator
    scale = 2.0 * a0 * a1 / denominator
    if shape <= 0.0 or scale <= 0.0:
        shape, scale = 0.0, a0

    excess = x - threshold
    if shape == 0.0:
        log_survival = -excess / scale
    else:
        log_survival = -math.log1p(shape * excess / scale) / shape
    return math.log((exceedance_weight + 1.0) / (total + 1.0)) + log_survival


class _BaseReference:
    """Shared tail-model plumbing for the window and sketch memories."""

    def __init__(self, n_features: int, tail_quantile: float | None) -> None:
        self.n_features = n_features
        self.tail_quantile = tail_quantile

    @property
    def size(self) -> float:
        """Effective number of reference points (total weight)."""
        raise NotImplementedError

    def learn(self, x: np.ndarray) -> np.ndarray | None:
        """Add ``x``; return the vector evicted to make room for it, if any."""
        raise NotImplementedError

    def skew_signs(self, x: np.ndarray) -> np.ndarray:
        """Return each feature's skewness sign with ``x`` appended."""
        raise NotImplementedError

    def _masses(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return reference weight ``<= x`` and ``>= x`` per feature."""
        raise NotImplementedError

    def _tail_items(self, feature: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one feature's ascending values and weights for tail fitting."""
        raise NotImplementedError

    def tail_log_probabilities(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``log`` left and right tail probabilities with ``x`` appended."""
        total = self.size
        at_most, at_least = self._masses(x)
        # Summing decayed weights through different paths can put the queried
        # mass a few ulps above the running total. A probability cannot exceed
        # one, and its negative log must remain a non-negative contribution.
        log_left = np.log((np.minimum(at_most, total) + 1.0) / (total + 1.0))
        log_right = np.log((np.minimum(at_least, total) + 1.0) / (total + 1.0))
        if self.tail_quantile is not None:
            self._apply_pareto(x, total, log_left, log_right)
        return log_left, log_right

    def _apply_pareto(
        self,
        x: np.ndarray,
        total: float,
        log_left: np.ndarray,
        log_right: np.ndarray,
    ) -> None:
        assert self.tail_quantile is not None
        # Only coordinates already in an outer tail can lie beyond the fitted
        # threshold; the generous bound keeps the per-feature fit off the
        # common path without missing a candidate.
        cutoff = math.log(
            min(1.0, 2.0 * (1.0 - self.tail_quantile) + 2.0 / (total + 1.0))
        )
        for feature in np.flatnonzero(log_right <= cutoff):
            values, weights = self._tail_items(int(feature))
            estimate = pareto_log_tail(
                values, weights, total, float(x[feature]), self.tail_quantile
            )
            if estimate is not None:
                log_right[feature] = estimate
        for feature in np.flatnonzero(log_left <= cutoff):
            values, weights = self._tail_items(int(feature))
            estimate = pareto_log_tail(
                -values[::-1],
                weights[::-1],
                total,
                -float(x[feature]),
                self.tail_quantile,
            )
            if estimate is not None:
                log_left[feature] = estimate


class _WindowReference(_BaseReference):
    """Exact per-feature reference of the last ``window_size`` learned vectors."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        tail_quantile: float | None = None,
    ) -> None:
        super().__init__(n_features, tail_quantile)
        self.window_size = window_size
        self._sorted = np.empty((n_features, window_size), dtype=np.float64)
        self._arrivals = np.empty((window_size, n_features), dtype=np.float64)
        self._count = 0
        self._head = 0
        self._evictions = 0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._m2 = np.zeros(n_features, dtype=np.float64)
        self._m3 = np.zeros(n_features, dtype=np.float64)

    @property
    def size(self) -> float:
        return float(self._count)

    def learn(self, x: np.ndarray) -> np.ndarray | None:
        n = self._count
        if n < self.window_size:
            positions = np.add.reduce(self._sorted[:, :n] < x[:, np.newaxis], axis=1)
            for feature, position in enumerate(positions.tolist()):
                row = self._sorted[feature]
                row[position + 1 : n + 1] = row[position:n]
                row[position] = x[feature]
            self._arrivals[n] = x
            self._count = n + 1
            _, self._mean, self._m2, self._m3 = _add_moments(
                float(n), self._mean, self._m2, self._m3, x
            )
            return None

        evicted: np.ndarray = self._arrivals[self._head].copy()
        window = self._sorted
        removals = np.add.reduce(window < evicted[:, np.newaxis], axis=1)
        insertions = np.add.reduce(window < x[:, np.newaxis], axis=1)
        for feature, (removal, insertion) in enumerate(
            zip(removals.tolist(), insertions.tolist(), strict=True)
        ):
            row = window[feature]
            if insertion <= removal:
                row[insertion + 1 : removal + 1] = row[insertion:removal]
                row[insertion] = x[feature]
            else:
                row[removal : insertion - 1] = row[removal + 1 : insertion]
                row[insertion - 1] = x[feature]
        self._arrivals[self._head] = x
        self._head = (self._head + 1) % self.window_size

        self._evictions += 1
        if self._evictions >= self.window_size:
            # Incremental removal accumulates rounding error; refresh the
            # moments exactly once per full turnover of the window.
            self._evictions = 0
            self._mean = np.mean(self._arrivals, axis=0)
            centered = self._arrivals - self._mean
            self._m2 = np.add.reduce(centered * centered, axis=0)
            self._m3 = np.add.reduce(centered * centered * centered, axis=0)
        else:
            weight, mean, m2, m3 = _remove_moments(
                float(n), self._mean, self._m2, self._m3, evicted
            )
            _, self._mean, self._m2, self._m3 = _add_moments(weight, mean, m2, m3, x)
        return evicted

    def skew_signs(self, x: np.ndarray) -> np.ndarray:
        return augmented_skew_signs(
            float(self._count), self._mean, self._m2, self._m3, x
        )

    def _masses(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        window = self._sorted[:, : self._count]
        column = x[:, np.newaxis]
        at_most = np.add.reduce(window <= column, axis=1).astype(np.float64)
        below = np.add.reduce(window < column, axis=1)
        return at_most, (self._count - below).astype(np.float64)

    def _tail_items(self, feature: int) -> tuple[np.ndarray, np.ndarray]:
        values = self._sorted[feature, : self._count]
        return values, np.ones(self._count, dtype=np.float64)


class _SketchReference(_BaseReference):
    """Bounded per-feature reference held in weighted merging t-digests.

    Every feature has its own digest of centroids, compressed with the ``k2``
    scale function so that centroids near either tail stay small and tail
    probabilities stay accurate (Dunning and Ertl, 2019). New values wait in a
    buffer of exact points that is merged into the digests once it is full;
    queries count the buffer exactly, so a learned value is never invisible.

    Besides its mean and weight, every centroid keeps its sum of squared
    deviations, the smallest and largest value merged into it and the weight
    of the points equal to each. Its mass is read as those two atoms plus the
    rest spread uniformly with the matching mean and variance, so single
    points, pairs, repeated discrete values and the extremes of the reference
    are exact, which is where the tail scores live.

    With a ``half_life``, all weights decay by ``0.5 ** (1 / half_life)`` per
    learned vector. Scaling every weight by the same factor leaves each
    centroid's quantile position, and so the digest's size bound, unchanged;
    the decay is applied lazily by giving each new point a growing stored
    weight.
    """

    def __init__(
        self,
        n_features: int,
        compression: float,
        half_life: float | None = None,
        tail_quantile: float | None = None,
        buffer_size: int = 128,
    ) -> None:
        super().__init__(n_features, tail_quantile)
        self.compression = compression
        self.half_life = half_life
        self.buffer_size = buffer_size
        self._decay = 1.0 if half_life is None else 0.5 ** (1.0 / half_life)
        # Stored weight of the most recent point; actual = stored / inflation.
        self._inflation = 1.0
        self._stored_total = 0.0
        self._merges = 0

        empty = _Centroids.empty()
        self._centroids: list[_Centroids] = [empty] * n_features
        self._buffer = np.empty((buffer_size, n_features), dtype=np.float64)
        self._buffer_weights = np.empty(buffer_size, dtype=np.float64)
        self._buffered = 0
        self._rebuild_query()

        self._moment_weight = 0.0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._m2 = np.zeros(n_features, dtype=np.float64)
        self._m3 = np.zeros(n_features, dtype=np.float64)

    @property
    def size(self) -> float:
        return self._stored_total / self._inflation

    @property
    def n_centroids(self) -> int:
        """Total number of centroids over all features."""
        return sum(len(centroids.means) for centroids in self._centroids)

    def learn(self, x: np.ndarray) -> np.ndarray | None:
        if self._decay < 1.0:
            # Renormalize before division can overflow or the decay factor
            # underflows to zero. Zero decay simply forgets all previous mass.
            if self._inflation > _MAX_INFLATION * self._decay:
                self._renormalize(self._decay)
            else:
                self._inflation /= self._decay
            self._moment_weight *= self._decay
            self._m2 *= self._decay
            self._m3 *= self._decay
        self._buffer[self._buffered] = x
        self._buffer_weights[self._buffered] = self._inflation
        self._buffered += 1
        self._stored_total += self._inflation
        if self._buffered == self.buffer_size:
            self._merge_buffer()
        self._moment_weight, self._mean, self._m2, self._m3 = _add_moments(
            self._moment_weight, self._mean, self._m2, self._m3, x
        )
        return None

    def skew_signs(self, x: np.ndarray) -> np.ndarray:
        return augmented_skew_signs(
            self._moment_weight, self._mean, self._m2, self._m3, x
        )

    def _renormalize(self, decay: float) -> None:
        factor = decay / self._inflation
        self._centroids = [centroids.scaled(factor) for centroids in self._centroids]
        self._buffer_weights[: self._buffered] *= factor
        self._stored_total *= factor
        self._inflation = 1.0
        self._rebuild_query()

    def _merge_buffer(self) -> None:
        count = self._buffered
        buffer_weights = self._buffer_weights[:count]
        # Alternate the merge direction, as the merging digest does, so that
        # centroids are not systematically biased toward one end.
        descending = self._merges % 2 == 1
        self._merges += 1
        for feature in range(self.n_features):
            self._centroids[feature] = self._centroids[feature].merged(
                self._buffer[:count, feature],
                buffer_weights,
                self.compression,
                weight_scale=self._inflation,
                descending=descending,
            )
        self._buffered = 0
        self._rebuild_query()

    def _rebuild_query(self) -> None:
        """Lay the centroids out as zero-weight-padded arrays for queries.

        A centroid's mass is its two endpoint atoms, which are exact, plus an
        interior spread uniformly over the interval whose mean and variance
        match the interior's, clipped to the centroid's range. Matching the
        variance rather than spanning the whole range matters under decay: an
        old extreme keeps the range wide long after its weight has gone.
        """
        width = max(1, *(len(c.means) for c in self._centroids))
        shape = (self.n_features, width)
        lows = np.zeros(shape, dtype=np.float64)
        highs = np.zeros(shape, dtype=np.float64)
        low_atoms = np.zeros(shape, dtype=np.float64)
        high_atoms = np.zeros(shape, dtype=np.float64)
        interiors = np.zeros(shape, dtype=np.float64)
        starts = np.zeros(shape, dtype=np.float64)
        ends = np.zeros(shape, dtype=np.float64)
        for feature, c in enumerate(self._centroids):
            size = len(c.means)
            if size == 0:
                continue
            # A point mass is held entirely by its low atom.
            high_atom = np.where(c.lows == c.highs, 0.0, c.high_weights)
            interior = np.maximum(c.weights - c.low_weights - high_atom, 0.0)
            has_interior = interior > 1e-12 * c.weights
            safe = np.where(has_interior, interior, 1.0)
            center = np.where(
                has_interior,
                (c.weights * c.means - c.low_weights * c.lows - high_atom * c.highs)
                / safe,
                c.means,
            )
            spread = (
                c.m2s
                - c.low_weights * (c.lows - c.means) ** 2
                - high_atom * (c.highs - c.means) ** 2
                - interior * (center - c.means) ** 2
            )
            half_width = np.sqrt(3.0 * np.maximum(spread, 0.0) / safe)
            lows[feature, :size] = c.lows
            highs[feature, :size] = c.highs
            low_atoms[feature, :size] = c.low_weights
            high_atoms[feature, :size] = high_atom
            interiors[feature, :size] = np.where(has_interior, interior, 0.0)
            starts[feature, :size] = np.clip(center - half_width, c.lows, c.highs)
            ends[feature, :size] = np.clip(center + half_width, c.lows, c.highs)
        self._query_lows = lows
        self._query_highs = highs
        self._query_low_atoms = low_atoms
        self._query_high_atoms = high_atoms
        self._query_interiors = interiors
        self._query_starts = starts
        self._query_ends = ends
        self._query_spread = ends > starts
        self._query_spans = np.where(self._query_spread, ends - starts, 1.0)

    def _masses(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        column = x[:, np.newaxis]
        lows = self._query_lows
        highs = self._query_highs
        starts = self._query_starts
        ends = self._query_ends
        spans = self._query_spans
        spread = self._query_spread
        below = np.where(
            spread, np.clip((column - starts) / spans, 0.0, 1.0), starts <= column
        )
        above = np.where(
            spread, np.clip((ends - column) / spans, 0.0, 1.0), starts >= column
        )
        low_atoms = self._query_low_atoms
        high_atoms = self._query_high_atoms
        interiors = self._query_interiors
        at_most = np.add.reduce(
            low_atoms * (lows <= column)
            + high_atoms * (highs <= column)
            + interiors * below,
            axis=1,
        )
        at_least = np.add.reduce(
            low_atoms * (lows >= column)
            + high_atoms * (highs >= column)
            + interiors * above,
            axis=1,
        )
        buffered = self._buffered
        if buffered:
            values = self._buffer[:buffered]
            weights = self._buffer_weights[:buffered]
            at_most = at_most + weights @ (values <= x)
            at_least = at_least + weights @ (values >= x)
        scale = self._inflation
        return at_most / scale, at_least / scale

    def _tail_items(self, feature: int) -> tuple[np.ndarray, np.ndarray]:
        buffered = self._buffered
        centroids = self._centroids[feature]
        values = np.concatenate((centroids.means, self._buffer[:buffered, feature]))
        weights = np.concatenate((centroids.weights, self._buffer_weights[:buffered]))
        order = np.argsort(values, kind="stable")
        return values[order], weights[order] / self._inflation


class _Centroids:
    """One feature's t-digest centroids in ascending order of their means."""

    __slots__ = (
        "high_weights",
        "highs",
        "low_weights",
        "lows",
        "m2s",
        "means",
        "weights",
    )

    def __init__(
        self,
        means: np.ndarray,
        weights: np.ndarray,
        lows: np.ndarray,
        highs: np.ndarray,
        low_weights: np.ndarray,
        high_weights: np.ndarray,
        m2s: np.ndarray,
    ) -> None:
        self.means = means
        self.weights = weights
        self.lows = lows
        self.highs = highs
        self.low_weights = low_weights
        self.high_weights = high_weights
        # Weighted sum of squared deviations from the mean, per centroid.
        self.m2s = m2s

    @classmethod
    def empty(cls) -> _Centroids:
        nothing = np.empty(0, dtype=np.float64)
        return cls(nothing, nothing, nothing, nothing, nothing, nothing, nothing)

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.means,
            self.weights,
            self.lows,
            self.highs,
            self.low_weights,
            self.high_weights,
            self.m2s,
        )

    def scaled(self, factor: float) -> _Centroids:
        return _Centroids(
            self.means,
            self.weights * factor,
            self.lows,
            self.highs,
            self.low_weights * factor,
            self.high_weights * factor,
            self.m2s * factor,
        )

    def merged(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        compression: float,
        *,
        weight_scale: float,
        descending: bool,
    ) -> _Centroids:
        """Merge new points into these centroids with one compression pass.

        ``weight_scale`` converts stored weights to effective ones. Entries
        whose stored weight has decayed to zero are dropped: they no longer
        contribute mass, and merging two of them would divide by zero.
        """
        all_weights = np.concatenate((self.weights, weights))
        kept = np.flatnonzero(all_weights > 0.0)
        means = np.concatenate((self.means, values))[kept]
        order = kept[np.argsort(means, kind="stable")]
        if descending:
            order = order[::-1]
        merged = _compress(
            np.concatenate((self.means, values))[order],
            all_weights[order],
            np.concatenate((self.lows, values))[order],
            np.concatenate((self.highs, values))[order],
            np.concatenate((self.low_weights, weights))[order],
            np.concatenate((self.high_weights, weights))[order],
            np.concatenate((self.m2s, np.zeros(len(values))))[order],
            compression,
            weight_scale,
        )
        if descending:
            merged = _Centroids(*(array[::-1] for array in merged.arrays()))
        return merged


def _compress(
    means: np.ndarray,
    weights: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    low_weights: np.ndarray,
    high_weights: np.ndarray,
    m2s: np.ndarray,
    compression: float,
    weight_scale: float = 1.0,
) -> _Centroids:
    """Merge sorted centroids greedily under the ``k2`` scale function.

    This is the merge pass of the merging t-digest (Dunning and Ertl, 2019,
    Algorithm 1): a centroid absorbs its successor while the quantile span it
    covers stays within one unit of the scale function. The ``k2`` scale
    ``k(q) = delta / Z(n) * log(q / (1 - q))`` with ``Z(n) = 4 log(n / delta)
    + 24`` limits a centroid to a fixed fraction of its distance from the
    nearer end, so tail ranks carry a relative rather than an absolute error,
    which is what a negative log tail probability needs. The input may run in
    either direction; the scale function is symmetric. ``weight_scale``
    converts the stored ``weights`` to effective sample weights for ``Z(n)``.
    """
    total = float(np.sum(weights))
    if total <= 0.0 or len(means) <= 1:
        return _Centroids(means, weights, lows, highs, low_weights, high_weights, m2s)
    # Z(n) uses the effective sample size; stored weights carry the lazy decay
    # inflation, which must not make the digest coarser over time.
    effective = total / weight_scale
    normalizer = 4.0 * math.log(max(effective / compression, 1.0)) + 24.0
    step = normalizer / compression

    def q_limit(q: float) -> float:
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        # sigmoid(logit(q) + step), written to stay finite for any q.
        odds = q / (1.0 - q) * math.exp(step)
        return odds / (1.0 + odds)

    out: list[tuple[float, float, float, float, float, float, float]] = []
    items = zip(
        means.tolist(),
        weights.tolist(),
        lows.tolist(),
        highs.tolist(),
        low_weights.tolist(),
        high_weights.tolist(),
        m2s.tolist(),
        strict=True,
    )
    mean, weight, low, high, low_weight, high_weight, m2 = next(items)
    done = 0.0
    limit = 0.0
    for item in items:
        (
            next_mean,
            next_weight,
            next_low,
            next_high,
            next_low_w,
            next_high_w,
            next_m2,
        ) = item
        if done + weight + next_weight <= limit:
            delta = next_mean - mean
            combined = weight + next_weight
            m2 += next_m2 + delta * delta * weight * next_weight / combined
            weight = combined
            mean += delta * next_weight / weight
            if next_low < low:
                low, low_weight = next_low, next_low_w
            elif next_low == low:
                low_weight += next_low_w
            if next_high > high:
                high, high_weight = next_high, next_high_w
            elif next_high == high:
                high_weight += next_high_w
        else:
            out.append((mean, weight, low, high, low_weight, high_weight, m2))
            done += weight
            limit = q_limit(done / total) * total
            mean, weight, low, high, low_weight, high_weight, m2 = item
    out.append((mean, weight, low, high, low_weight, high_weight, m2))
    columns = np.array(out, dtype=np.float64).T
    return _Centroids(*(np.ascontiguousarray(column) for column in columns))
