"""Long-run turnover, numeric aging, and finite-state streaming regressions."""

from __future__ import annotations

import gc
import math
import tracemalloc
import weakref
from collections import deque

import numpy as np
import pytest

from aberrant.model import NullModel, QuantileThreshold, RandomModel, ThresholdModel
from aberrant.model.deep.kitnet import OnlineAutoencoderEnsemble, _NumpyAutoencoder
from aberrant.model.stat import (
    MovingAverage,
    MovingAverageAbsoluteDeviation,
    MovingCorrelationCoefficient,
    MovingCovariance,
    MovingGeometricAverage,
    MovingHarmonicAverage,
    MovingInterquartileRange,
    MovingKurtosis,
    MovingMahalanobisDistance,
    MovingMedian,
    MovingQuantile,
    MovingSkewness,
    MovingVariance,
)
from aberrant.model.timeseries import (
    MultivariateRollingMatrixProfile,
    RollingMatrixProfile,
    SeasonalResidualDetector,
    XLagDAMP,
)
from aberrant.model.timeseries.damp import _mass_distance_profile

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from aberrant.model.deep.autoencoder import Autoencoder
    from aberrant.utils.deep.architecture import (
        VanillaAutoencoder,
        VanillaLSTMAutoencoder,
    )

_UNIVARIATE = [
    MovingAverage,
    MovingAverageAbsoluteDeviation,
    MovingGeometricAverage,
    MovingHarmonicAverage,
    MovingInterquartileRange,
    MovingKurtosis,
    MovingMedian,
    MovingQuantile,
    MovingSkewness,
    MovingVariance,
]


@pytest.mark.parametrize("model_type", _UNIVARIATE)
def test_univariate_state_stays_bounded_through_hundreds_of_turnovers(
    model_type: type,
) -> None:
    model = model_type(window_size=17)
    expected: deque[float] = deque(maxlen=17)
    for index in range(4_000):
        value = 2.0 + math.sin(index * 0.173) + (index % 7) / 10
        assert math.isfinite(model.score_one({"value": value}))
        model.learn_one({"value": value})
        expected.append(value)
        assert model.window == expected
    # Inference-only calls cannot enlarge the learned sample window.
    for _ in range(200):
        assert math.isfinite(model.score_one({"value": 3.0}))
    assert model.window == expected
    with pytest.raises(ValueError):
        model.learn_one({"replacement": 3.0})
    assert model.window == expected


@pytest.mark.parametrize(
    "model_type",
    [MovingCovariance, MovingCorrelationCoefficient, MovingMahalanobisDistance],
)
def test_multivariate_statistics_bound_history_and_reject_schema_churn(
    model_type: type,
) -> None:
    model = model_type(window_size=17)
    for index in range(2_000):
        point = {"x": math.sin(index * 0.17), "y": math.cos(index * 0.21)}
        assert math.isfinite(model.score_one(point))
        model.learn_one(point)
        if isinstance(model.window, dict):
            assert set(model.window) == {"x", "y"}
            assert all(len(window) <= 17 for window in model.window.values())
        else:
            assert len(model.window) <= 17
    for index in range(100):
        with pytest.raises(ValueError):
            model.learn_one({"x": 1.0, f"new_{index}": 2.0})
    assert model._schema.names == ("x", "y")


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("multivariate", [False, True])
def test_matrix_profiles_evict_samples_statistics_and_absolute_matches(
    normalize: bool, multivariate: bool
) -> None:
    model_type = (
        MultivariateRollingMatrixProfile if multivariate else RollingMatrixProfile
    )
    model = model_type(subsequence_length=4, window_size=23, normalize=normalize)
    old_samples: list[weakref.ReferenceType] = []
    for index in range(1_500):
        point = {"x": math.sin(index * 0.17)}
        if multivariate:
            point["y"] = math.cos(index * 0.21)
        score, reference = model.match_one(point)
        assert math.isfinite(score)
        if reference is not None:
            assert max(0, index - 23 + 1) <= reference < index - 3
        model.learn_one(point)
        if multivariate and index < 50:
            old_samples.append(weakref.ref(model._history[-1]))
        assert len(model._history) <= 23
        assert len(model._statistics) <= 20
    assert model.n_samples_seen == 1_500
    assert all(reference() is None for reference in old_samples)
    # Exercise counts beyond machine-size integer and float exactness limits.
    model._samples_seen += 1 << 80
    score, reference = model.match_one(point)
    assert math.isfinite(score)
    assert reference is not None and reference >= (1 << 80)
    model.learn_one(point)
    assert model.n_samples_seen == (1 << 80) + 1_501
    assert len(model._history) == 23


def test_seasonal_residual_releases_warmup_and_preserves_large_counter_phase() -> None:
    model = SeasonalResidualDetector(season_length=7)
    for index in range(10_000):
        point = {"x": 20.0 + math.sin(2 * math.pi * (index % 7) / 7)}
        assert math.isfinite(model.score_one(point))
        model.learn_one(point)
    assert model._warmup == []
    assert len(model._seasonal) == 7
    point = {"x": 21.0}
    expected = model.explain_one(point)
    model.n_samples_seen += (1 << 80) * 7
    assert model.explain_one(point) == expected
    model.learn_one(point)
    assert len(model._seasonal) == 7


def test_damp_bounded_history_and_huge_lifetime_counters() -> None:
    model = XLagDAMP(subsequence_length=4, x_lag=23, start_index=8)
    for index in range(2_000):
        point = {"x": math.sin(index * 0.17) + math.cos(index * 0.11)}
        score = model.score_one(point)
        assert math.isfinite(score)
        model.learn_one(point)
        assert model.last_score == score
        assert len(model._history) <= 26
    expected = model.score_one({"x": 0.1})
    model._samples_seen += 1 << 80
    model._subsequences_processed += 1 << 80
    assert model.score_one({"x": 0.1}) == expected
    model.learn_one({"x": 0.1})
    assert model.n_samples_seen == (1 << 80) + 2_001


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_damp_normalization_survives_finite_extreme_magnitudes(scale: float) -> None:
    series = np.array([1.0, 2.0, 4.0, 3.0, 0.0, 1.0, 5.0, 2.0])
    query = np.array([0.0, 1.0, 5.0, 2.0])
    expected = _mass_distance_profile(series, query, eps=1e-12)
    actual = _mass_distance_profile(series * scale, query * scale, eps=1e-300)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_quantile_threshold_extremes_and_turnover_never_retain_nonfinite_boundary() -> (
    None
):
    model = QuantileThreshold(quantile=0.5, window_size=2)
    extreme = np.finfo(float).max
    for index in range(5_000):
        model.learn_one({"score": -extreme if index % 2 else extreme})
        assert model.n_scores <= 2
        if model.threshold is not None:
            assert model.threshold == 0.0
            assert model.score_one({"score": 0.0}) == 1.0
    model.reset()
    assert model.threshold is None and model.n_scores == 0


@pytest.mark.parametrize(
    "model", [NullModel(), RandomModel(), ThresholdModel(ceiling=1.0)]
)
def test_stateless_models_do_not_retain_unbounded_feature_names(model: object) -> None:
    original_keys = set(vars(model))
    for index in range(10_000):
        point = {f"new_{index}": float(index % 3)}
        assert math.isfinite(model.score_one(point))
        model.learn_one(point)
    assert set(vars(model)) == original_keys
    assert not any(
        isinstance(value, list | dict | deque) for value in vars(model).values()
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_numpy_ensemble_state_has_fixed_shapes_after_warmup(adaptive: bool) -> None:
    model = OnlineAutoencoderEnsemble(
        max_ae_size=2,
        feature_map_grace=13,
        ad_grace=17,
        learning_rate=0.02,
        adaptive_after_warmup=adaptive,
        seed=7,
    )
    shapes = None
    for index in range(4_000):
        point = {
            f"x_{channel}": math.sin(index * 0.17 + channel) for channel in range(5)
        }
        assert math.isfinite(model.score_one(point))
        model.learn_one(point)
        if model.is_ready:
            current = [
                (ae.w1.shape, ae.b1.shape, ae.w2.shape, ae.b2.shape)
                for ae in [*model._ensemble, model._output_ae]
            ]
            if shapes is None:
                shapes = current
            assert current == shapes
    assert model._feature_map_samples == 13
    assert model._detector_samples == 17
    assert len(model._feature_groups) == 3
    model._samples_seen += 1 << 80
    model.learn_one(point)
    assert math.isfinite(model.score_one(point))


def test_numpy_ensemble_rejects_overflowing_feature_moments_without_poisoning() -> None:
    model = OnlineAutoencoderEnsemble(feature_map_grace=2, ad_grace=0, seed=7)
    model.learn_one({"x": 0.2})
    before = model._sum.copy(), model._sum_sq.copy(), model._sum_cross.copy()
    with pytest.raises(OverflowError):
        model.learn_one({"x": 1e200})
    for actual, expected in zip(
        (model._sum, model._sum_sq, model._sum_cross), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert model._samples_seen == 1 and model._feature_map_samples == 1
    model.learn_one({"x": 0.3})
    assert model.is_ready and math.isfinite(model.score_one({"x": 0.4}))


def test_rejected_first_ensemble_event_does_not_lock_feature_dimensions() -> None:
    model = OnlineAutoencoderEnsemble(feature_map_grace=2, ad_grace=0, seed=7)
    with pytest.raises(OverflowError):
        model.learn_one({"x": 1e200, "y": 1e200})
    assert model._schema.names is None and model._sum is None
    model.learn_one({"other": 0.2})
    model.learn_one({"other": 0.3})
    assert model.is_ready


def test_numpy_autoencoder_stages_overflowing_updates_before_publishing_weights() -> (
    None
):
    model = _NumpyAutoencoder(
        input_dim=2,
        hidden_dim=1,
        learning_rate=1e308,
        rng=np.random.default_rng(7),
    )
    weights = [
        parameter.copy() for parameter in (model.w1, model.b1, model.w2, model.b2)
    ]
    with pytest.raises(OverflowError):
        model.learn(np.array([100.0, 50.0]))
    for actual, expected in zip(
        (model.w1, model.b1, model.w2, model.b2), weights, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert math.isfinite(model.score(np.array([1e200, 1e200])))


def test_retained_python_memory_plateaus_after_warmup_across_model_families() -> None:
    models = [
        MovingAverage(23),
        MovingCovariance(23),
        RollingMatrixProfile(4, 23),
        MultivariateRollingMatrixProfile(4, 23),
        SeasonalResidualDetector(7),
        XLagDAMP(4, 23, 8),
        QuantileThreshold(window_size=23),
        OnlineAutoencoderEnsemble(
            feature_map_grace=13,
            ad_grace=17,
            learning_rate=0.02,
            adaptive_after_warmup=True,
            seed=7,
        ),
    ]

    def run(count: int) -> None:
        for index in range(count):
            point = {"x": math.sin(index * 0.17)}
            paired = {**point, "y": math.cos(index * 0.21)}
            for model_index, model in enumerate(models):
                sample = paired if model_index in (1, 3, 7) else point
                if model_index == 6:
                    sample = {"score": point["x"]}
                if index % 4 == 0:
                    model.score_one(sample)
                model.learn_one(sample)

    tracemalloc.start()
    try:
        run(300)
        gc.collect()
        baseline, _ = tracemalloc.get_traced_memory()
        run(3_000)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
        # Allow allocator/library housekeeping, while rejecting one retained
        # point per event (and larger) across this many window turnovers.
        assert retained - baseline < 128 * 1_024
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("recurrent", [False, True])
def test_torch_autoencoder_reuses_storage_and_releases_event_graphs(
    recurrent: bool,
) -> None:
    torch = pytest.importorskip("torch")

    architecture_type = VanillaLSTMAutoencoder if recurrent else VanillaAutoencoder
    architecture = architecture_type(input_size=2, seed=7)
    model = Autoencoder(
        architecture, torch.optim.Adam(architecture.parameters()), torch.nn.MSELoss()
    )
    input_identity = id(model.x_tensor)
    outputs: deque[weakref.ReferenceType] = deque(maxlen=8)
    hook = architecture.register_forward_hook(
        lambda _model, _inputs, output: outputs.append(weakref.ref(output))
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    state_sizes = None
    try:
        for index in range(600):
            point = {"x": math.sin(index * 0.17), "y": math.cos(index * 0.21)}
            model.learn_one(point)
            assert math.isfinite(model.score_one(point))
            assert id(model.x_tensor) == input_identity
            assert all(reference() is None for reference in outputs)
            states = [
                value
                for state in model.optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            ]
            current_sizes = [value.numel() for value in states]
            if state_sizes is None:
                state_sizes = current_sizes
            assert current_sizes == state_sizes
            assert all(value.grad_fn is None for value in states)
            assert all(
                parameter.grad is None or parameter.grad.grad_fn is None
                for parameter in architecture.parameters()
            )
        gc.collect()
        assert all(reference() is None for reference in outputs)
    finally:
        hook.remove()
        torch.set_num_threads(previous_threads)


def test_torch_autoencoder_rejects_finite_input_that_overflows_tensor_or_loss() -> None:
    torch = pytest.importorskip("torch")

    architecture = VanillaAutoencoder(input_size=1, seed=7)
    model = Autoencoder(
        architecture, torch.optim.Adam(architecture.parameters()), torch.nn.MSELoss()
    )
    before = [parameter.detach().clone() for parameter in architecture.parameters()]
    for value in (1e300, 1e30):
        with pytest.raises(OverflowError):
            model.learn_one({"x": value})
        with pytest.raises(OverflowError):
            model.score_one({"x": value})
    assert model._schema.names is None and not model.optimizer.state
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(architecture.parameters(), before, strict=True)
    )
    model.learn_one({"x": 0.5})
    assert math.isfinite(model.score_one({"x": 0.5}))


@pytest.mark.xfail(
    strict=True,
    reason="PyTorch Adam uses float32 step tensors that stop incrementing at 2**24",
)
def test_external_adam_counter_keeps_advancing_after_float32_integer_limit() -> None:
    torch = pytest.importorskip("torch")
    architecture = VanillaAutoencoder(input_size=1, seed=7)
    model = Autoencoder(
        architecture, torch.optim.Adam(architecture.parameters()), torch.nn.MSELoss()
    )
    model.learn_one({"x": 0.5})
    for state in model.optimizer.state.values():
        state["step"].fill_(1 << 24)
    model.learn_one({"x": 0.5})
    assert all(
        state["step"].item() == (1 << 24) + 1
        for state in model.optimizer.state.values()
    )


@pytest.mark.xfail(
    strict=True,
    reason="Finite gradients from input 1e18 overflow float32 Adam second-moment state",
)
def test_external_adam_moments_remain_finite_for_finite_loss_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    architecture = VanillaAutoencoder(input_size=1, seed=7)
    model = Autoencoder(
        architecture, torch.optim.Adam(architecture.parameters()), torch.nn.MSELoss()
    )
    # The wrapper's loss/gradient guards pass, while Adam squares gradients in
    # its supplied parameter dtype and overflows its accumulated second moment.
    model.learn_one({"x": 1e18})
    assert all(
        torch.isfinite(value).all()
        for state in model.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
