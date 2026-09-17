"""Exact joint subsequence novelty over bounded multivariate history."""

from __future__ import annotations

import numpy as np

from aberrant.model.timeseries._matrix_profile import (
    _BaseRollingMatrixProfile,
    _multivariate_window_statistics,
    _nearest_multivariate_match,
)


class MultivariateRollingMatrixProfile(
    _BaseRollingMatrixProfile[np.ndarray, list[tuple[int, float, float]]]
):
    """Exact, causal scores for aligned multivariate subsequences.

    For each eligible historical interval, compute each channel's Euclidean
    distance to the newest subsequence and take the largest channel distance.
    The score is the minimum of these maxima, so all channels must match the
    same historical interval. Higher scores mean greater joint novelty.

    Args:
        subsequence_length: Number of consecutive events per subsequence,
            at least two.
        window_size: Retained event capacity including the candidate, defaulting
            to ``16 * subsequence_length``. Must be at least
            ``subsequence_length + exclusion_zone + 1``.
        normalize: Independently population z-normalize each channel's windows
            when true. Otherwise use raw Euclidean distances; channels with
            larger units can dominate the score.
        exclusion_zone: Exclude reference starts within this many positions of
            the query start. Defaults to ``ceil(subsequence_length / 4)``.
            Set to ``subsequence_length - 1`` for nonoverlapping matches.

    Notes:
        - Each event contains one or more consistently named finite numeric
          features representing one aligned observation. The first successful
          learning call establishes the schema; scoring never commits it.
        - Query before learning. ``score_one``, ``match_one``, and ``explain_one``
          are read-only; learning needs no preceding query.
        - Warm-up returns zero and no match until
          ``subsequence_length + exclusion_zone`` events have been learned.
          Thereafter every valid event has an eligible reference, including
          during eviction. Equal scores choose the earliest retained start.
        - Normalization removes per-channel level and amplitude differences.
          Two constant windows have distance zero; exactly one constant window
          gives distance ``sqrt(subsequence_length)`` for that channel.
        - With ``d`` channels, scoring costs typically ``O(d * W * log(W))``
          with direct numerical fallback up to ``O(d * W * m)``. Learning costs
          ``O(d * m)``; retained state and working memory are ``O(d * W)``.
        - This exposes the newest causal profile entry and its common match,
          not a historical profile or global motif ranking. Raw scores beyond
          the floating-point range raise ``OverflowError``.

    References:
        Matrix Profile for Anomaly Detection on Multidimensional Time Series.
        Uses the pre-max aggregation with a causal, bounded reference window.
        https://arxiv.org/abs/2409.09298
    """

    def learn_one(self, x: dict[str, float]) -> None:
        """Learn an aligned observation and its window statistics without scoring."""
        prepared = self._schema.preview(x)
        self._learn(prepared, prepared.values, _multivariate_window_statistics)

    def score_one(self, x: dict[str, float]) -> float:
        """Return the minimum largest-channel distance without learning."""
        return self.explain_one(x)[0]

    def match_one(self, x: dict[str, float]) -> tuple[float, int | None]:
        """Return ``(score, reference_start)`` without changing learned state.

        The absolute, zero-based index counts learned events since reset.
        Before readiness, return ``(0.0, None)``. Later learning may evict the
        returned match.
        """
        score, start, _ = self.explain_one(x)
        return score, start

    def explain_one(
        self, x: dict[str, float]
    ) -> tuple[float, int | None, dict[str, float]]:
        """Return score, common reference start, and its channel distances.

        The fresh mapping follows sorted schema order and describes distances
        at the selected common interval, not each channel's individual nearest
        match. Its maximum equals the score. Before readiness, return
        ``(0.0, None, {})``. This method does not learn or advance indices.
        """
        prepared = self._schema.preview(x)
        if not self.is_ready:
            return 0.0, None, {}
        series, query, statistics, window_start = self._query(prepared.values)
        distance, local_start, channel_distances = _nearest_multivariate_match(
            series, query, statistics, normalize=self.normalize
        )
        return (
            distance,
            window_start + local_start,
            dict(zip(prepared.names, channel_distances, strict=True)),
        )
