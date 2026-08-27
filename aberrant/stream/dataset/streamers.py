"""Typed NPZ dataset streaming utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Protocol, TypeAlias

import numpy as np
from tqdm import tqdm

from aberrant.stream.dataset.registry import DatasetInfo

Sample: TypeAlias = tuple[dict[str, float], object]


class DatasetStream(Protocol):
    """Common interface for dataset sample streams."""

    def stream(self) -> Iterator[Sample]:
        """Yield individual samples."""
        ...

    def get_metadata(self) -> DatasetInfo | None:
        """Return registered dataset metadata when available."""
        ...


class NpzStreamer:
    """Row-wise iterator over a registered NPZ dataset artifact."""

    def __init__(
        self,
        file_path: str | Path,
        dataset_info: DatasetInfo | None = None,
        *,
        feature_prefix: str = "feature_",
        label_column: str = "y",
        feature_column: str = "X",
        show_progress: bool = False,
    ) -> None:
        if not feature_prefix:
            raise ValueError("feature_prefix must be non-empty")
        if not label_column:
            raise ValueError("label_column must be non-empty")
        if not feature_column:
            raise ValueError("feature_column must be non-empty")
        self.file_path = Path(file_path)
        self.dataset_info = dataset_info
        self.feature_prefix = feature_prefix
        self.label_column = label_column
        self.feature_column = feature_column
        self.show_progress = show_progress
        self._archive: np.lib.npyio.NpzFile | None = None

    def __enter__(self) -> NpzStreamer:
        if not self.file_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.file_path}")
        archive: np.lib.npyio.NpzFile | None = None
        try:
            archive = np.load(self.file_path)
            if self.feature_column not in archive:
                raise KeyError(
                    f"Feature array '{self.feature_column}' not found in NPZ file"
                )
            if self.label_column not in archive:
                raise KeyError(
                    f"Label array '{self.label_column}' not found in NPZ file"
                )
            self._archive = archive
            return self
        except Exception:
            if archive is not None:
                archive.close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def __iter__(self) -> Iterator[Sample]:
        archive = self._archive
        if archive is None:
            raise RuntimeError("NPZ file is not open; use stream() or a with block")

        features_array = archive[self.feature_column]
        labels_array = archive[self.label_column]
        if features_array.ndim != 2:
            raise ValueError("Feature array must be two-dimensional")
        if features_array.shape[0] != labels_array.shape[0]:
            raise ValueError(
                "Feature and label arrays have different lengths: "
                f"{features_array.shape[0]} vs {labels_array.shape[0]}"
            )

        n_samples, n_features = features_array.shape
        feature_names = [f"{self.feature_prefix}{index}" for index in range(n_features)]
        description = (
            f"Loading {self.dataset_info.name}"
            if self.dataset_info is not None
            else "Loading data"
        )
        progress = (
            tqdm(total=n_samples, desc=description, unit="sample")
            if self.show_progress
            else None
        )
        try:
            for feature_vector, raw_label in zip(
                features_array,
                labels_array,
                strict=True,
            ):
                features = {
                    name: float(value)
                    for name, value in zip(
                        feature_names,
                        feature_vector,
                        strict=True,
                    )
                }
                label: object = (
                    raw_label.item()
                    if isinstance(raw_label, np.generic)
                    else raw_label
                )
                yield features, label
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    def stream(self) -> Iterator[Sample]:
        """Open the artifact and yield samples row by row."""
        with self:
            yield from self

    def get_metadata(self) -> DatasetInfo | None:
        """Return registered metadata when supplied by the loader."""
        return self.dataset_info

    def __repr__(self) -> str:
        if self.dataset_info is None:
            return f"NpzStreamer(file_path={str(self.file_path)!r})"
        return (
            f"NpzStreamer(file_path={str(self.file_path)!r}, "
            f"dataset={self.dataset_info.name!r}, "
            f"n_samples={self.dataset_info.n_samples}, "
            f"n_features={self.dataset_info.n_features})"
        )


class BatchStreamer:
    """Batch samples from any typed dataset stream."""

    def __init__(self, base_streamer: DatasetStream, batch_size: int = 1000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.base_streamer = base_streamer
        self.batch_size = batch_size

    def stream(self) -> Iterator[tuple[list[dict[str, float]], list[object]]]:
        """Yield feature and label batches."""
        feature_batch: list[dict[str, float]] = []
        label_batch: list[object] = []
        for features, label in self.base_streamer.stream():
            feature_batch.append(features)
            label_batch.append(label)
            if len(feature_batch) == self.batch_size:
                yield feature_batch, label_batch
                feature_batch = []
                label_batch = []
        if feature_batch:
            yield feature_batch, label_batch

    def get_metadata(self) -> DatasetInfo | None:
        """Return metadata from the underlying stream."""
        return self.base_streamer.get_metadata()

    def __repr__(self) -> str:
        return (
            f"BatchStreamer(base_streamer={self.base_streamer!r}, "
            f"batch_size={self.batch_size})"
        )
