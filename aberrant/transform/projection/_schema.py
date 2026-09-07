"""Projection-specific dimensional checks over the shared feature schema."""

from collections.abc import Mapping

from aberrant.utils.validation import (
    FeatureSchema,
    PreparedFeatures,
    coerce_feature_values,
)


class ProjectionSchema(FeatureSchema):
    """Preserve projection feature order and missing-key error semantics."""

    def __init__(self, n_components: int, keys: list[str] | None) -> None:
        super().__init__(keys, sort_names=False)
        self.n_components = n_components
        if keys is not None:
            self._check_dimensions(len(keys))

    def _check_dimensions(self, count: int) -> None:
        if self.n_components > count:
            raise ValueError(
                f"The number of n_components ({self.n_components}) has to be less or equal "
                f"to the number of features ({count})"
            )

    def prepare(self, x: Mapping[str, float]) -> PreparedFeatures:
        values = coerce_feature_values(x)
        if self.names is not None:
            for name in self.names:
                if name not in values:
                    raise KeyError(name)
            unexpected = sorted(set(values).difference(self.names))
            if unexpected:
                raise ValueError(f"Input contains unexpected feature(s): {unexpected}")
        self._check_dimensions(len(values))
        return self.preview(values)

    @property
    def feature_names(self) -> list[str] | None:
        """Return an owned copy of the established feature order."""
        return None if self.names is None else list(self.names)
