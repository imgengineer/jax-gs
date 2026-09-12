from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.config import DataConfig, ModelConfig, RasterizationConfig, TrainConfig
from jax_gs.training._evaluation import _EvaluationRenderer


def test_evaluation_grows_intersections_before_candidates_and_reuses_plan(monkeypatch):
    configs = []
    checked = []
    config = TrainConfig(
        intersection_bucket_min_capacity=2,
        rasterizer=RasterizationConfig(max_gaussians_per_tile=2),
    )

    def factory(current):
        configs.append(current.rasterizer)

        def render():
            enough_intersections = current.rasterizer.max_intersections >= 9
            enough_candidates = current.rasterizer.max_candidates_per_tile >= 5
            return (
                jnp.asarray(float(enough_intersections and enough_candidates)),
                jnp.asarray(1.0),
                jnp.asarray([not enough_candidates]),
                jnp.asarray(not enough_intersections),
                jnp.asarray(9),
                jnp.asarray(5 if enough_intersections else 1),
            )

        return render

    monkeypatch.setattr(
        "jax_gs.training._evaluation._check_evaluation_memory_budget",
        lambda current, **kwargs: checked.append(current.rasterizer),
    )
    renderer = _EvaluationRenderer(config, 32, 32, factory)
    for _ in range(2):
        image, _, overflow, intersection_overflow = renderer(
            physical_capacity=16, intersection_capacity=4, candidate_bound=2
        )
        assert float(image) == 1.0
        assert not np.any(overflow)
        assert not bool(intersection_overflow)
    assert [(c.max_intersections, c.max_candidates_per_tile) for c in configs] == [
        (4, 2),
        (16, 2),
        (16, 8),
    ]
    assert checked == [configs[0], configs[1], configs[2], configs[2]]


def test_evaluation_respects_explicit_intersection_limit(monkeypatch):
    config = TrainConfig(rasterizer=RasterizationConfig(max_intersections=4))
    monkeypatch.setattr(
        "jax_gs.training._evaluation._check_evaluation_memory_budget",
        lambda *args, **kwargs: None,
    )
    renderer = _EvaluationRenderer(
        config,
        32,
        32,
        lambda current: (
            lambda: (
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray([False]),
                jnp.asarray(True),
                jnp.asarray(5),
                jnp.asarray(1),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="exceed configured maximum"):
        renderer(physical_capacity=16, intersection_capacity=4, candidate_bound=2)


def test_evaluation_preflight_uses_bounded_workspace(monkeypatch):
    from jax_gs import training

    config = TrainConfig(
        model=ModelConfig(capacity=1_000_000),
        rasterizer=RasterizationConfig(
            backend="intersections",
            compositor_backend="cuda_tile",
        ),
    )
    configs = []
    monkeypatch.setattr(training, "_device_memory_usage", lambda: (2**30, 32 * 2**30))

    def factory(current):
        configs.append(current)
        return lambda: (
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            jnp.asarray([False]),
            jnp.asarray(False),
            jnp.asarray(100),
            jnp.asarray(10),
        )

    renderer = _EvaluationRenderer(config, 648, 420, factory)
    renderer(
        physical_capacity=1_000_000,
        intersection_capacity=1_048_576,
        candidate_bound=2048,
    )
    assert configs[0].rasterizer.max_intersections == 1_048_576


@pytest.mark.parametrize("model_type", ["3dgs", "2dgs"])
def test_evaluation_retry_matches_complete_render(model_type):
    from jax_gs.model import GaussianModel
    from jax_gs.training import make_render_step

    config = TrainConfig(
        model_type=model_type,
        data=DataConfig(patch_size=4),
        model=ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=0),
        rasterizer=RasterizationConfig(
            backend="intersections",
            tile_size=4,
            max_gaussians_per_tile=2,
            tile_batch_size=1,
        ),
        intersection_bucket_min_capacity=2,
    )
    model = GaussianModel.from_point_cloud(
        np.asarray([[0.0, 0.0, 3.0], [0.1, 0.0, 3.1]], np.float32),
        np.asarray([[255, 0, 0], [0, 255, 0]], np.uint8),
        config.model,
    )
    args = (
        model,
        jnp.eye(4),
        jnp.asarray([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]),
        jnp.asarray(0),
    )
    renderer = _EvaluationRenderer(
        config,
        8,
        8,
        lambda current: make_render_step(current, 8, 8, _return_info=True),
    )
    actual = renderer(
        *args,
        physical_capacity=4,
        intersection_capacity=2,
        candidate_bound=1,
    )
    complete_config = replace(
        config,
        rasterizer=replace(config.rasterizer, max_intersections=16),
    )
    expected = make_render_step(complete_config, 8, 8)(*args)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)
    assert not np.any(actual[2])
    assert not bool(actual[3])
