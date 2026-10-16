"""Stateless feature operations that also compose in ordinary pipelines."""

from collections.abc import Iterator
from contextlib import contextmanager

from aberrant.base.readiness import Readiness
from aberrant.base.transformer import BaseTransformer
from aberrant.utils.validation import coerce_feature_values


class Passthrough(BaseTransformer):
    """Copy a finite feature mapping without changing names or values."""

    @property
    def readiness(self) -> Readiness:
        return Readiness(True)

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Stateless operations need no restoration on downstream failure."""
        yield

    def learn_one(self, x: dict[str, float]) -> None:
        self.transform_one(x)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        return coerce_feature_values(x)


class Select(Passthrough):
    """Select named features in the explicitly supplied order."""

    def __init__(self, features: list[str]) -> None:
        if isinstance(features, str | bytes):
            raise ValueError("features must be a sequence of names, not a string")
        if not features or any(not isinstance(key, str) or not key for key in features):
            raise ValueError("features must contain non-empty feature names")
        if len(features) != len(set(features)):
            raise ValueError("features cannot contain duplicates")
        self.features = tuple(features)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        values = coerce_feature_values(x)
        missing = set(self.features).difference(values)
        if missing:
            raise ValueError(f"Missing selected features: {sorted(missing)}")
        return {key: values[key] for key in self.features}


class Rename(Passthrough):
    """Rename selected keys, preserving order and rejecting name collisions."""

    def __init__(self, mapping: dict[str, str]) -> None:
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in mapping.items()
        ):
            raise ValueError("mapping must contain non-empty feature names")
        self.mapping = dict(mapping)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        values = coerce_feature_values(x)
        missing = set(self.mapping).difference(values)
        if missing:
            raise ValueError(f"Missing renamed features: {sorted(missing)}")
        result: dict[str, float] = {}
        for key, value in values.items():
            renamed = self.mapping.get(key, key)
            if renamed in result:
                raise ValueError(f"Renaming produces duplicate feature {renamed!r}")
            result[renamed] = value
        return result
