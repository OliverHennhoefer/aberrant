"""Rolling calibration, causality, and pipeline failure contracts."""

import pickle

import numpy as np
import pytest

from aberrant.base import BaseModel, TransactionalTransformerProtocol
from aberrant.catalog import DetectorConfig, get_transformer_spec
from aberrant.transform import RollingRobustScaler
from aberrant.transform.preprocessing import RollingRobustScaler as PreprocessingScaler


class _RecordingModel(BaseModel):
    def __init__(self) -> None:
        self.learned: list[float] = []

    def learn_one(self, x: dict[str, float]) -> None:
        if set(x) != {"x"}:
            raise ValueError("Unexpected feature")
        self.learned.append(x["x"])

    def score_one(self, x: dict[str, float]) -> float:
        return x["x"]


@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 16])
def test_exact_linear_quartiles_match_numpy(length: int) -> None:
    values = np.random.default_rng(42).normal(size=length)
    scaler = RollingRobustScaler(window_size=16)
    for value in values:
        scaler.learn_one({"x": float(value)})
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    scale = q3 - q1 if q3 > q1 else 1.0

    assert scaler.transform_one({"x": 3.0})["x"] == pytest.approx((3 - median) / scale)


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ([1.5000000000000004, 1.5000000000000007], 1.500000000000001, 3.0),
        ([1e-323, 1.5e-323], 2e-323, 2.0),
        ([5e-324, 1e-323], 1.5e-323, 1.0),
    ],
)
def test_narrow_nonconstant_windows_keep_their_iqr(
    reference: list[float], candidate: float, expected: float
) -> None:
    scaler = RollingRobustScaler(window_size=2)
    for value in reference:
        scaler.learn_one({"x": value})

    assert scaler.transform_one({"x": candidate}) == {"x": expected}


def test_outlier_does_not_dominate_the_reference_scale() -> None:
    scaler = RollingRobustScaler(window_size=8)
    for value in [0, 1, 2, 3, 4, 5, 6, 1_000_000]:
        scaler.learn_one({"x": value})

    assert scaler.transform_one({"x": 7.0}) == {"x": 1.0}


def test_eviction_removes_old_values_from_the_quartiles() -> None:
    scaler = RollingRobustScaler(window_size=3)
    for value in [0, 10, 20]:
        scaler.learn_one({"x": value})
    assert scaler.transform_one({"x": 30.0}) == {"x": 2.0}

    scaler.learn_one({"x": 30.0})

    assert scaler.transform_one({"x": 30.0}) == {"x": 1.0}
    assert scaler.sample_counts == {"x": 3}


def test_zero_iqr_preserves_novel_positive_and_negative_deviations() -> None:
    scaler = RollingRobustScaler(fallback_scale=2.0)
    for _ in range(5):
        scaler.learn_one({"x": 10.0})

    assert scaler.transform_one({"x": 10.0}) == {"x": 0.0}
    assert scaler.transform_one({"x": 16.0}) == {"x": 3.0}
    assert scaler.transform_one({"x": 4.0}) == {"x": -3.0}


@pytest.mark.parametrize("value", [5e-324, -5e-324, 1e308, -1e308])
def test_constant_extreme_references_center_exactly(value: float) -> None:
    scaler = RollingRobustScaler()
    for _ in range(4):
        scaler.learn_one({"x": value})

    assert scaler.transform_one({"x": value}) == {"x": 0.0}


def test_sparse_windows_and_readiness_are_per_feature() -> None:
    scaler = RollingRobustScaler(window_size=3, min_samples=2)
    assert not scaler.is_ready
    scaler.learn_one({"x": 0.0})
    assert not scaler.is_ready
    # Provisional output remains available while a pipeline warms up.
    assert scaler.transform_one({"x": 2.0}) == {"x": 2.0}
    scaler.learn_one({"x": 2.0})
    assert scaler.is_ready
    scaler.learn_one({"y": 10.0})
    assert not scaler.is_ready
    assert scaler.sample_counts == {"x": 2, "y": 1}
    scaler.learn_one({"y": 20.0})
    assert scaler.is_ready
    assert scaler.transform_one({"x": 2.0, "y": 20.0}) == {"x": 1.0, "y": 1.0}
    snapshot = scaler.sample_counts
    snapshot["x"] = 100
    assert scaler.sample_counts["x"] == 2


def test_windows_remain_bounded_over_a_long_stream() -> None:
    scaler = RollingRobustScaler(window_size=7)
    for index in range(1000):
        scaler.learn_one({"x": float(index), "y": float(-index)})

    assert scaler.sample_counts == {"x": 7, "y": 7}
    assert sum(len(window) for window in scaler._windows.values()) == 14
    assert scaler.transform_one({"x": 999.0, "y": -999.0}) == {"x": 1.0, "y": -1.0}


def test_transform_is_read_only_and_repeated_candidates_do_not_change_scores() -> None:
    scaler = RollingRobustScaler(window_size=3)
    for value in [0, 2, 4]:
        scaler.learn_one({"x": value})
    before = pickle.dumps(scaler)

    for _ in range(3):
        assert scaler.transform_one({"x": 8.0}) == {"x": 3.0}

    assert pickle.dumps(scaler) == before
    with pytest.raises(ValueError, match="has not been seen"):
        scaler.transform_one({"unseen": 1.0})
    assert pickle.dumps(scaler) == before


@pytest.mark.parametrize("operation", ["learn_one", "transform_one"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "not numeric"])
def test_invalid_sample_is_atomic(operation: str, invalid: object) -> None:
    scaler = RollingRobustScaler()
    scaler.learn_one({"x": 0.0})
    before = pickle.dumps(scaler)

    with pytest.raises(ValueError):
        getattr(scaler, operation)({"x": 100.0, "new": invalid})

    assert pickle.dumps(scaler) == before


def test_non_finite_derived_statistics_reject_all_staged_features() -> None:
    scaler = RollingRobustScaler(window_size=4)
    for value in [-1e308, -1e308, 1e308]:
        scaler.learn_one({"x": 0.0, "y": value})
    before = pickle.dumps(scaler)

    with pytest.raises(ValueError, match="median and IQR must be finite"):
        scaler.learn_one({"x": 10.0, "y": 1e308})

    assert pickle.dumps(scaler) == before


def test_large_subtraction_is_rescaled_when_the_result_is_representable() -> None:
    scaler = RollingRobustScaler()
    for value in [-1e308, -1e308, 1e308]:
        scaler.learn_one({"x": value})

    assert scaler.transform_one({"x": 1e308}) == {"x": 2.0}


def test_unrepresentable_output_is_rejected_without_learning() -> None:
    scaler = RollingRobustScaler()
    scaler.learn_one({"x": -1e308})
    before = pickle.dumps(scaler)

    with pytest.raises(ValueError, match="scaled value must be finite"):
        scaler.transform_one({"x": 1e308})

    assert pickle.dumps(scaler) == before


def test_freeze_requires_readiness_and_preserves_calibration() -> None:
    scaler = RollingRobustScaler(window_size=3, min_samples=2)
    with pytest.raises(ValueError, match="ready"):
        scaler.freeze()
    scaler.learn_one({"x": 0.0})
    with pytest.raises(ValueError, match="ready"):
        scaler.freeze()
    scaler.learn_one({"x": 2.0})
    scaler.freeze()
    assert scaler.is_frozen
    before = pickle.dumps(scaler)
    scaler.learn_one({"x": 100.0})
    assert pickle.dumps(scaler) == before
    assert scaler.transform_one({"x": 4.0}) == {"x": 3.0}
    with pytest.raises(ValueError, match="has not been seen"):
        scaler.learn_one({"x": 100.0, "new": 1.0})
    with pytest.raises(ValueError, match="finite"):
        scaler.learn_one({"x": float("nan")})
    assert pickle.dumps(scaler) == before
    scaler.unfreeze()
    scaler.learn_one({"x": 4.0})
    assert not scaler.is_frozen
    assert scaler.transform_one({"x": 4.0}) == {"x": 1.0}


def test_reset_clears_readiness_and_unfreezes_without_changing_parameters() -> None:
    scaler = RollingRobustScaler(window_size=5, min_samples=2, fallback_scale=3.0)
    for value in [0, 1]:
        scaler.learn_one({"x": value})
    scaler.freeze()
    scaler.reset()

    assert scaler.sample_counts == {}
    assert not scaler.is_ready
    assert not scaler.is_frozen
    assert (scaler.window_size, scaler.min_samples, scaler.fallback_scale) == (5, 2, 3)


def test_pipeline_scores_prior_state_and_learns_post_update_state() -> None:
    scaler = RollingRobustScaler(window_size=3, min_samples=2)
    model = _RecordingModel()
    pipeline = scaler | model
    for value in [0, 10]:
        pipeline.learn_one({"x": value})
    assert scaler.is_ready
    before = pickle.dumps(pipeline)
    assert pipeline.score_one({"x": 20.0}) == 3.0
    assert pickle.dumps(pipeline) == before

    pipeline.learn_one({"x": 20.0})

    assert model.learned == [0.0, 1.0, 1.0]


def test_pipeline_rollback_restores_evicted_values_and_removes_new_features() -> None:
    scaler = RollingRobustScaler(window_size=2)
    model = _RecordingModel()
    pipeline = scaler | model
    pipeline.learn_one({"x": 0.0})
    pipeline.learn_one({"x": 2.0})
    before = pickle.dumps(pipeline)

    with pytest.raises(ValueError, match="Unexpected feature"):
        pipeline.learn_one({"x": 100.0, "new": 5.0})

    assert pickle.dumps(pipeline) == before
    assert scaler.transform_one({"x": 4.0}) == {"x": 3.0}
    assert isinstance(scaler, TransactionalTransformerProtocol)


def test_frozen_and_adaptive_calibration_diverge_after_a_level_shift() -> None:
    adaptive = RollingRobustScaler(window_size=5, min_samples=5)
    frozen = RollingRobustScaler(window_size=5, min_samples=5)
    for value in [0, 2, 4, 6, 8]:
        adaptive.learn_one({"x": value})
        frozen.learn_one({"x": value})
    frozen.freeze()
    for value in [10, 12, 14, 16, 18]:
        adaptive.learn_one({"x": value})
        frozen.learn_one({"x": value})

    assert adaptive.transform_one({"x": 18.0}) == {"x": 1.0}
    assert frozen.transform_one({"x": 18.0}) == {"x": 3.5}


def test_adaptation_does_not_reexpress_retained_downstream_references() -> None:
    adaptive = RollingRobustScaler(window_size=5, min_samples=5)
    frozen = RollingRobustScaler(window_size=5, min_samples=5)
    for value in [0, 2, 4, 6, 8]:
        adaptive.learn_one({"x": value})
        frozen.learn_one({"x": value})
    frozen.freeze()
    adaptive_model, frozen_model = _RecordingModel(), _RecordingModel()
    adaptive_pipeline = adaptive | adaptive_model
    frozen_pipeline = frozen | frozen_model
    adaptive_pipeline.learn_one({"x": 8.0})
    frozen_pipeline.learn_one({"x": 8.0})
    stored_adaptive, stored_frozen = adaptive_model.learned[0], frozen_model.learned[0]

    for value in [10, 12, 14, 16, 18]:
        adaptive_pipeline.learn_one({"x": value})
        frozen_pipeline.learn_one({"x": value})

    assert adaptive_model.learned[0] == stored_adaptive
    assert frozen_model.learned[0] == stored_frozen
    assert adaptive_pipeline.score_one({"x": 8.0}) != stored_adaptive
    assert frozen_pipeline.score_one({"x": 8.0}) == stored_frozen


def test_export_catalog_and_declarative_pipeline() -> None:
    assert PreprocessingScaler is RollingRobustScaler
    spec = get_transformer_spec("rolling_robust_scaler")
    assert spec.output_feature_count({}, 2) == 2
    assert spec.parameter_schema()["properties"]["window_size"]["default"] == 256
    config = DetectorConfig.from_mapping(
        {
            "version": 1,
            "transformers": [
                {"id": "rolling_robust_scaler", "params": {"window_size": 3}}
            ],
            "model": {"id": "null"},
        }
    )
    detector = config.build()
    detector.learn_one({"x": 1.0})
    assert detector.score_one({"x": 2.0}) == 0.0
    assert config.normalized().transformers[0].params["fallback_scale"] == 1.0


@pytest.mark.parametrize(
    "params",
    [
        {"window_size": 0},
        {"window_size": 1.5},
        {"window_size": True},
        {"min_samples": 0},
        {"min_samples": False},
        {"window_size": 2, "min_samples": 3},
        {"fallback_scale": 0.0},
        {"fallback_scale": -1.0},
        {"fallback_scale": float("nan")},
        {"fallback_scale": float("inf")},
    ],
)
def test_invalid_configuration(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RollingRobustScaler(**params)  # type: ignore[arg-type]
