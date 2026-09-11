"""Feature-schema validation for application input boundaries."""

from collections.abc import Sequence

from aberrant.base.exceptions import ValidationError
from aberrant.base.transformer import BaseTransformer
from aberrant.utils.validation import FeatureSchema, PreparedFeatures


class FeatureSchemaGuard(BaseTransformer):
    """Validate finite numeric values and a stable set of feature names.

    Supply ``features`` to enforce an application schema from the first event.
    If omitted, the first successfully learned event establishes the schema.
    The transformer preserves feature names and values while returning them in
    the established order.

    Args:
        features: Expected feature names, or ``None`` to learn them from the
            first event.
        sort_features: Sort names when learning a schema from the first event.
    """

    def __init__(
        self,
        features: Sequence[str] | None = None,
        *,
        sort_features: bool = True,
    ) -> None:
        if isinstance(features, str | bytes):
            raise ValueError("features must be a sequence of names, not a string")
        self.features = None if features is None else tuple(features)
        self.sort_features = sort_features
        self._schema = FeatureSchema(
            names=self.features,
            sort_names=sort_features,
        )

    @property
    def feature_names(self) -> tuple[str, ...] | None:
        """Return the configured or learned feature order."""
        return self._schema.names

    def learn_one(self, x: dict[str, float]) -> None:
        """Validate an event and commit its schema after successful validation."""
        prepared = self._prepare(x)
        self._schema.commit(prepared)

    def transform_one(self, x: dict[str, float]) -> dict[str, float]:
        """Validate and return an ordered, float-valued copy of an event."""
        prepared = self._prepare(x)
        return dict(zip(prepared.names, prepared.values.tolist(), strict=True))

    def _prepare(self, x: dict[str, float]) -> PreparedFeatures:
        try:
            return self._schema.preview(x)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def reset(self) -> None:
        """Forget a learned schema while preserving constructor configuration."""
        self._schema = FeatureSchema(
            names=self.features,
            sort_names=self.sort_features,
        )

    def __repr__(self) -> str:
        return (
            f"FeatureSchemaGuard(features={self.features!r}, "
            f"sort_features={self.sort_features})"
        )
