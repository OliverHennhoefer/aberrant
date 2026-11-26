"""Turn categorical features into numbers."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

from aberrant._utils.validation import coerce_integer_count
from aberrant.base.readiness import Readiness
from aberrant.base.transformer import BaseTransformer
from aberrant.exceptions import ValidationError


def _category(value: object, name: str) -> str | None:
    """A value's category as text, or ``None`` when it is missing."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))
        raise ValidationError(
            f"Feature '{name}' must be a category: text or an integer, not {value}"
        )
    if isinstance(value, str | int):
        return str(value)
    raise ValidationError(f"Feature '{name}' must be a category: text or an integer")


def _feature_names(features: object, label: str) -> tuple[str, ...]:
    if isinstance(features, str | bytes) or not isinstance(features, Sequence):
        raise ValueError(f"{label} must be a sequence of names, not a string")
    names = tuple(features)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(f"{label} must be distinct, non-empty feature names")
    return names


class OneHotEncoder(BaseTransformer):
    """Replace a categorical feature by one indicator per declared category.

    Each feature in ``categories`` becomes ``{feature}_{category}`` for its
    declared categories, followed by ``{feature}_other`` for any category not
    declared, so the output keeps the same features for every event. A
    category is text or an integer, compared as text, so ``2``, ``2.0`` and
    ``"2"`` are the same category. A missing value, an
    absent feature, ``None`` or NaN, sets every indicator of the feature to
    NaN for an ``Imputer`` to fill. Other features pass through unchanged.

    Args:
        categories: Each categorical feature's categories, in output order.
    """

    def __init__(self, *, categories: Mapping[str, Sequence[str | int]]) -> None:
        if not isinstance(categories, Mapping) or not categories:
            raise ValueError("categories must map feature names to categories")
        declared: dict[str, tuple[str, ...]] = {}
        for name, values in categories.items():
            if not isinstance(name, str) or not name:
                raise ValueError("categories must map non-empty feature names")
            if isinstance(values, str | bytes) or not isinstance(values, Sequence):
                raise ValueError(f"categories of '{name}' must be a sequence")
            texts = tuple(
                str(value) for value in values if isinstance(value, str | int)
            )
            if (
                not texts
                or len(texts) != len(values)
                or len(set(texts)) != len(texts)
                or "other" in texts
            ):
                raise ValueError(
                    f"categories of '{name}' must be distinct texts or integers, "
                    "and not 'other'"
                )
            declared[name] = texts
        self.categories = {name: list(values) for name, values in categories.items()}
        self._declared = declared

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """A stateless encoder needs no restoration."""
        yield

    def learn_one(self, x: dict[str, float]) -> None:
        """Check the event; nothing is learned."""
        self.transform_one(x)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The event with each categorical feature replaced by indicators."""
        result: dict[str, float] = {}
        for name, value in x.items():
            if name in self._declared:
                result.update(self._indicators(name, value))
            else:
                result[name] = value
        for name in self._declared:
            if name not in x:
                result.update(self._indicators(name, None))
        expected = len(x) + sum(
            len(values) + (name not in x) for name, values in self._declared.items()
        )
        if len(result) != expected:
            raise ValidationError(
                "One-hot indicators clash with features of the same name"
            )
        return result

    def _indicators(self, name: str, value: object) -> dict[str, float]:
        category = _category(value, name)
        declared = self._declared[name]
        if category is None:
            return {f"{name}_{text}": math.nan for text in (*declared, "other")}
        other = category not in declared
        indicators = {f"{name}_{text}": float(text == category) for text in declared}
        indicators[f"{name}_other"] = float(other)
        return indicators


class FrequencyEncoder(BaseTransformer):
    """Replace a categorical feature by how often its category was learned.

    Each feature in ``features`` becomes the share of its learned values,
    over the last ``window_size`` events or all of them, that had the same
    category: near 1 for a common category, 0 for one never learned. A
    detector then sees a rare category as a small value. A category is text
    or an integer, compared as text as for ``OneHotEncoder``. A missing
    value, an absent feature,
    ``None`` or NaN, is not learned and comes out as NaN for an ``Imputer``
    to fill. Other features pass through unchanged.

    Args:
        features: The categorical features.
        window_size: Learned values each feature's shares count, or ``None``
            for all of them.
    """

    def __init__(
        self, *, features: Sequence[str], window_size: int | None = None
    ) -> None:
        self.features = list(_feature_names(features, "features"))
        self.window_size = (
            None
            if window_size is None
            else coerce_integer_count(window_size, label="window_size")
        )
        self._counts: dict[str, Counter[str]] = {
            name: Counter() for name in self.features
        }
        self._totals = dict.fromkeys(self.features, 0)
        self._recent: dict[str, deque[str]] = {name: deque() for name in self.features}

    @property
    def readiness(self) -> Readiness:
        """Ready once every feature has learned a category."""
        observed = min(self._totals.values())
        return Readiness.after(min(observed, 1), 1)

    def learn_one(self, x: dict[str, float]) -> None:
        """Count each observed category; missing values change nothing."""
        observed = self._observed(x)
        for name, category in observed.items():
            counts = self._counts[name]
            counts[category] += 1
            self._totals[name] += 1
            if self.window_size is None:
                continue
            recent = self._recent[name]
            recent.append(category)
            if len(recent) > self.window_size:
                leaving = recent.popleft()
                counts[leaving] -= 1
                self._totals[name] -= 1
                if not counts[leaving]:
                    del counts[leaving]

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """The event with each categorical feature replaced by its share."""
        observed = self._observed(x)
        result: dict[str, float] = {}
        for name, value in x.items():
            result[name] = (
                self._share(name, observed) if name in self._counts else value
            )
        for name in self.features:
            if name not in x:
                result[name] = math.nan
        return result

    def _observed(self, x: Mapping[str, object]) -> dict[str, str]:
        observed = {}
        for name in self.features:
            category = _category(x.get(name), name)
            if category is not None:
                observed[name] = category
        return observed

    def _share(self, name: str, observed: Mapping[str, str]) -> float:
        if name not in observed:
            return math.nan
        total = self._totals[name]
        return self._counts[name][observed[name]] / total if total else 0.0

    @contextmanager
    def learning_transaction(self, x: dict[str, float]) -> Iterator[None]:
        """Restore the counts if a pipeline update fails."""
        counts = {name: Counter(counter) for name, counter in self._counts.items()}
        totals = dict(self._totals)
        recent = {name: deque(values) for name, values in self._recent.items()}
        try:
            yield
        except BaseException:
            self._counts, self._totals, self._recent = counts, totals, recent
            raise
