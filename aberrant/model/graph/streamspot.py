"""Signed-sketch structural anomaly detection for graph edge streams."""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass

import numpy as np

from aberrant._utils.validation import (
    MonotonicClock,
    PreparedTimestamp,
    coerce_event_number,
    coerce_integer_count,
)
from aberrant.base.model import BaseModel
from aberrant.exceptions import ValidationError


@dataclass(slots=True)
class _GraphState:
    """Mutable per-graph state."""

    sketch: np.ndarray
    tail: deque[bytes]


@dataclass(frozen=True, slots=True)
class _PreparedGraphEvent:
    """Validated graph event awaiting a clock commit."""

    timestamp: PreparedTimestamp
    graph_id: float
    source: float
    destination: float
    edge_type: float | None


class StreamSpot(BaseModel):
    """
    Signed-sketch detector for graph-level structural anomalies.

    The detector maintains bounded per-graph sketches over edge shingles and an
    online set of cluster centers over graph sketches. Incoming edges are scored
    by the distance between their host graph's candidate sketch and the nearest
    cluster center. This implementation uses signed count sketches and online
    Euclidean centroids; the original StreamSpot uses Hamming-distance sketches
    and a different clustering procedure.

    Notes:
    - Scores are continuous and non-negative.
    - State is bounded by ``max_graphs``, ``sketch_dim``, and ``num_clusters``.

    Args:
        graph_key: Input field identifying the graph to which an edge belongs.
        source_key: Input field containing the finite numeric source identifier.
        destination_key: Input field containing the finite numeric destination
            identifier.
        edge_type_key: Optional input field included in edge-shingle identity.
        time_key: Input field containing a non-decreasing integer-like time
            bucket. ``None`` uses one-based arrival order.
        sketch_dim: Number of signed-count coordinates per graph sketch.
        shingle_size: Number of consecutive per-graph edge tokens in a shingle.
        num_clusters: Maximum number of online Euclidean cluster centers.
        max_graphs: Maximum number of active per-graph sketches. Least-recently
            learned graph state is evicted when the limit is exceeded.
        warm_up: Number of active graph sketches required before
            scoring. It cannot exceed ``max_graphs``.
        seed: Seed for model-local shingle hashing.

    References:
        Manzoor, E., Milajerdi, S. M., & Akoglu, L. (2016). Fast Memory-efficient
        Anomaly Detection in Streaming Heterogeneous Graphs.
        https://doi.org/10.1145/2939672.2939783
        Preprint: https://arxiv.org/abs/1602.04844
        Original implementation: https://github.com/sbustreamspot/sbustreamspot-core
    """

    @staticmethod
    def _validate_required_name(value: str, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
        return value

    @staticmethod
    def _validate_optional_name(value: str | None, label: str) -> str | None:
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{label} must be a non-empty string or None")
        return value

    @staticmethod
    def _validate_key_uniqueness(
        graph_key: str,
        source_key: str,
        destination_key: str,
        edge_type_key: str | None,
        time_key: str | None,
    ) -> None:
        required_keys = [graph_key, source_key, destination_key]
        if edge_type_key is not None:
            required_keys.append(edge_type_key)
        if time_key is not None:
            required_keys.append(time_key)
        if len(set(required_keys)) != len(required_keys):
            raise ValueError(
                "graph_key, source_key, destination_key, edge_type_key, "
                "and time_key must be distinct"
            )

    @staticmethod
    def _validate_hyperparameters(
        *,
        sketch_dim: int,
        shingle_size: int,
        num_clusters: int,
        max_graphs: int,
        warm_up: int,
    ) -> None:
        if num_clusters > max_graphs:
            raise ValueError("num_clusters must be less than or equal to max_graphs")
        if warm_up > max_graphs:
            raise ValueError("warm_up must be less than or equal to max_graphs")

    def __init__(
        self,
        *,
        graph_key: str = "graph",
        source_key: str = "src",
        destination_key: str = "dst",
        edge_type_key: str | None = None,
        time_key: str | None = "t",
        sketch_dim: int = 1024,
        shingle_size: int = 2,
        num_clusters: int = 8,
        max_graphs: int = 4096,
        warm_up: int = 32,
        seed: int | None = None,
    ) -> None:
        sketch_dim = coerce_integer_count(sketch_dim, label="sketch_dim", minimum=1)
        num_clusters = coerce_integer_count(
            num_clusters, label="num_clusters", minimum=1
        )
        shingle_size = coerce_integer_count(
            shingle_size, label="shingle_size", minimum=1
        )
        warm_up = coerce_integer_count(warm_up, label="warm_up", minimum=1)
        self.graph_key = self._validate_required_name(graph_key, "graph_key")
        self.source_key = self._validate_required_name(source_key, "source_key")
        self.destination_key = self._validate_required_name(
            destination_key, "destination_key"
        )
        self.edge_type_key = self._validate_optional_name(
            edge_type_key, "edge_type_key"
        )
        self.time_key = self._validate_optional_name(time_key, "time_key")

        self._validate_key_uniqueness(
            graph_key=self.graph_key,
            source_key=self.source_key,
            destination_key=self.destination_key,
            edge_type_key=self.edge_type_key,
            time_key=self.time_key,
        )
        max_graphs = coerce_integer_count(max_graphs, label="max_graphs")
        self._validate_hyperparameters(
            sketch_dim=sketch_dim,
            shingle_size=shingle_size,
            num_clusters=num_clusters,
            max_graphs=max_graphs,
            warm_up=warm_up,
        )
        self.sketch_dim = sketch_dim
        self.shingle_size = shingle_size
        self.num_clusters = num_clusters
        self.max_graphs = max_graphs
        self.warm_up = warm_up
        self.seed = seed

        self._reset_state()

    def _reset_state(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        max_uint64 = int(np.iinfo(np.uint64).max)
        index_salt = int(
            self._rng.integers(low=0, high=max_uint64, dtype=np.uint64)
        ).to_bytes(16, byteorder="little", signed=False)
        sign_salt = int(
            self._rng.integers(low=0, high=max_uint64, dtype=np.uint64)
        ).to_bytes(16, byteorder="little", signed=False)
        self._index_key = index_salt
        self._sign_key = sign_salt

        self._graph_states: dict[float, _GraphState] = {}
        # Least recently seen first: a touch moves a graph to the end.
        self._graph_last_seen: OrderedDict[float, int] = OrderedDict()

        self._cluster_centers = np.zeros(
            (self.num_clusters, self.sketch_dim), dtype=np.float64
        )
        self._cluster_counts = np.zeros(self.num_clusters, dtype=np.float64)
        self._initialized_clusters = 0

        self._clock = MonotonicClock(integer_like=True)
        self._samples_seen = 0

    @property
    def n_samples_seen(self) -> int:
        """Number of samples processed via ``learn_one``."""
        return self._samples_seen

    def _prepare_sample(
        self,
        x: dict[str, float],
    ) -> _PreparedGraphEvent:
        if not x:
            raise ValidationError("Input dictionary cannot be empty")

        if self.graph_key not in x:
            raise ValidationError(
                f"Missing graph_key '{self.graph_key}' in input sample"
            )
        if self.source_key not in x:
            raise ValidationError(
                f"Missing source_key '{self.source_key}' in input sample"
            )
        if self.destination_key not in x:
            raise ValidationError(
                f"Missing destination_key '{self.destination_key}' in input sample"
            )
        if self.edge_type_key is not None and self.edge_type_key not in x:
            raise ValidationError(
                f"Missing edge_type_key '{self.edge_type_key}' in input sample"
            )

        graph_id = coerce_event_number(
            x[self.graph_key],
            label=f"Feature '{self.graph_key}'",
        )
        source = coerce_event_number(
            x[self.source_key],
            label=f"Feature '{self.source_key}'",
        )
        destination = coerce_event_number(
            x[self.destination_key],
            label=f"Feature '{self.destination_key}'",
        )
        edge_type: float | None = None
        if self.edge_type_key is not None:
            edge_type = coerce_event_number(
                x[self.edge_type_key],
                label=f"Feature '{self.edge_type_key}'",
            )

        if self.time_key is None:
            timestamp = self._clock.preview(implicit=True)
        else:
            if self.time_key not in x:
                raise ValidationError(
                    f"Missing time_key '{self.time_key}' in input sample"
                )
            timestamp = self._clock.preview(
                x[self.time_key],
                implicit=False,
            )
        return _PreparedGraphEvent(
            timestamp=timestamp,
            graph_id=graph_id,
            source=source,
            destination=destination,
            edge_type=edge_type,
        )

    def _edge_token(self, src: float, dst: float, edge_type: float | None) -> bytes:
        payload = bytearray()
        payload.extend(b"s")
        payload.extend(np.float64(src).tobytes())
        payload.extend(b"d")
        payload.extend(np.float64(dst).tobytes())
        if edge_type is not None:
            payload.extend(b"y")
            payload.extend(np.float64(edge_type).tobytes())
        return bytes(payload)

    def _candidate_shingle(self, tail: deque[bytes], token: bytes) -> bytes | None:
        if self.shingle_size == 1:
            return token

        expected_tail = self.shingle_size - 1
        if len(tail) < expected_tail:
            return None

        payload = bytearray()
        for part in tail:
            payload.extend(part)
            payload.append(0x1F)
        payload.extend(token)
        return bytes(payload)

    def _hash_shingle(self, shingle: bytes) -> tuple[int, float]:
        index_digest = hashlib.blake2b(
            shingle,
            digest_size=8,
            key=self._index_key,
        ).digest()
        index = int.from_bytes(index_digest, byteorder="little", signed=False) % (
            self.sketch_dim
        )

        sign_digest = hashlib.blake2b(
            shingle,
            digest_size=1,
            key=self._sign_key,
        ).digest()
        sign = 1.0 if (sign_digest[0] % 2) == 0 else -1.0
        return index, sign

    def _novelty_score(
        self,
        state: _GraphState | None,
        index: int,
        sign: float,
    ) -> float:
        if state is None:
            return 1.0
        signed_count = state.sketch[index] * sign
        return float(1.0 / (max(signed_count, 0.0) + 1.0))

    def _apply_shingle_update(self, sketch: np.ndarray, shingle: bytes | None) -> None:
        if shingle is None:
            return
        index, sign = self._hash_shingle(shingle)
        sketch[index] += sign

    def _get_graph_state(self, graph_id: float, *, create: bool) -> _GraphState | None:
        existing = self._graph_states.get(graph_id)
        if existing is not None or not create:
            return existing

        if self.shingle_size > 1:
            tail: deque[bytes] = deque(maxlen=self.shingle_size - 1)
        else:
            tail = deque()

        state = _GraphState(
            sketch=np.zeros(self.sketch_dim, dtype=np.float64),
            tail=tail,
        )
        self._graph_states[graph_id] = state
        self._touch_graph(graph_id)
        return state

    def _touch_graph(self, graph_id: float) -> None:
        self._graph_last_seen[graph_id] = self._samples_seen
        self._graph_last_seen.move_to_end(graph_id)

    def _evict_graphs_if_needed(self) -> None:
        while len(self._graph_states) > self.max_graphs:
            # The front holds the oldest time; among graphs sharing it, the
            # smallest identifier goes first.
            entries = iter(self._graph_last_seen.items())
            oldest_graph, oldest_time = next(entries)
            for graph_id, last_seen in entries:
                if last_seen != oldest_time:
                    break
                oldest_graph = min(oldest_graph, graph_id)
            self._graph_last_seen.pop(oldest_graph, None)
            self._graph_states.pop(oldest_graph, None)

    def _nearest_cluster_index_and_distance_sq(
        self,
        sketch: np.ndarray,
    ) -> tuple[int | None, float]:
        if self._initialized_clusters == 0:
            return None, 0.0

        active_centers = self._cluster_centers[: self._initialized_clusters]
        diff = active_centers - sketch
        distances_sq = np.sum(diff * diff, axis=1)
        nearest_index = int(np.argmin(distances_sq))
        nearest_dist_sq = float(distances_sq[nearest_index])
        if not np.isfinite(nearest_dist_sq):
            return nearest_index, 0.0
        return nearest_index, max(nearest_dist_sq, 0.0)

    def _update_clusters(self, sketch: np.ndarray) -> None:
        if self._initialized_clusters < self.num_clusters:
            idx = self._initialized_clusters
            self._cluster_centers[idx] = sketch
            self._cluster_counts[idx] = 1.0
            self._initialized_clusters += 1
            return

        nearest_index, _distance_sq = self._nearest_cluster_index_and_distance_sq(
            sketch
        )
        if nearest_index is None:
            return

        count = self._cluster_counts[nearest_index] + 1.0
        eta = 1.0 / count
        self._cluster_centers[nearest_index] += eta * (
            sketch - self._cluster_centers[nearest_index]
        )
        self._cluster_counts[nearest_index] = count

    def _is_warm(self) -> bool:
        return (
            len(self._graph_states) >= self.warm_up and self._initialized_clusters > 0
        )

    def learn_one(self, x: dict[str, float]) -> None:
        """Update detector state with one edge event."""
        event = self._prepare_sample(x)

        state = self._get_graph_state(event.graph_id, create=True)
        if state is None:
            raise RuntimeError("Graph state creation failed")

        token = self._edge_token(event.source, event.destination, event.edge_type)
        shingle = self._candidate_shingle(state.tail, token)
        self._apply_shingle_update(state.sketch, shingle)
        if self.shingle_size > 1:
            state.tail.append(token)

        self._update_clusters(state.sketch)
        self._touch_graph(event.graph_id)
        self._evict_graphs_if_needed()

        self._samples_seen += 1
        self._clock.commit(event.timestamp)

    def score_one(self, x: dict[str, float]) -> float:
        """Compute anomaly score for one edge event without mutating state."""
        event = self._prepare_sample(x)

        if not self._is_warm():
            return 0.0

        state = self._get_graph_state(event.graph_id, create=False)
        if state is None:
            candidate_sketch = np.zeros(self.sketch_dim, dtype=np.float64)
            tail: deque[bytes] = deque()
        else:
            candidate_sketch = state.sketch.copy()
            tail = state.tail

        token = self._edge_token(event.source, event.destination, event.edge_type)
        shingle = self._candidate_shingle(tail, token)
        novelty = 0.0
        if shingle is not None:
            index, sign = self._hash_shingle(shingle)
            novelty = self._novelty_score(state, index, sign)
            candidate_sketch[index] += sign

        _nearest_index, distance_sq = self._nearest_cluster_index_and_distance_sq(
            candidate_sketch
        )
        distance = float(np.sqrt(max(distance_sq, 0.0)))
        distance_component = distance / (1.0 + distance)
        raw_score = float(distance_component + novelty)
        if not np.isfinite(raw_score):
            raw_score = 0.0
        return raw_score
