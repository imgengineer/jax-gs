import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.config import ModelConfig
from jax_gs.model import GaussianModel


def test_fixed_capacity_point_cloud_initialization():
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], np.uint8)
    model = GaussianModel.from_point_cloud(
        points, colors, ModelConfig(capacity=16, sh_degree=2)
    )
    assert model.means.shape == (16, 3)
    assert model.sh_coeffs.shape == (16, 9, 3)
    assert int(model.active_count) == 2
    assert jnp.all(model.scales > 0.0)
    assert jnp.allclose(jnp.linalg.norm(model.normalized_quats, axis=-1), 1.0)


def test_point_cloud_scales_match_three_neighbor_rms_times_initial_scale():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        np.float32,
    )
    config = ModelConfig(
        capacity=8,
        bucket_min_capacity=8,
        initial_scale=0.25,
    )

    model = GaussianModel.from_point_cloud(points, np.zeros_like(points), config)

    expected_rms = np.sqrt(
        np.array([59.0 / 3.0, 41.0 / 3.0, 29.0 / 3.0, 101.0 / 3.0], np.float32)
    )
    expected_active_scales = np.repeat(
        (expected_rms * config.initial_scale)[:, None], 3, axis=1
    )
    np.testing.assert_allclose(
        np.asarray(model.scales[:4]), expected_active_scales, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(model.scales[4:]), config.initial_scale, rtol=1.0e-6
    )


@pytest.mark.parametrize(
    ("points", "expected_rms"),
    [
        (np.array([[0.0, 0.0, 0.0]], np.float32), np.array([1.0], np.float32)),
        (
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float32),
            np.array([2.0, 2.0], np.float32),
        ),
        (
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                np.float32,
            ),
            np.sqrt(np.array([29.0 / 2.0, 13.0 / 2.0, 34.0 / 2.0], np.float32)),
        ),
    ],
)
def test_point_cloud_scales_reduce_neighbor_count_for_small_inputs(
    points: np.ndarray, expected_rms: np.ndarray
):
    config = ModelConfig(
        capacity=4,
        bucket_min_capacity=4,
        initial_scale=0.2,
    )

    model = GaussianModel.from_point_cloud(points, np.zeros_like(points), config)

    expected = np.repeat((expected_rms * config.initial_scale)[:, None], 3, axis=1)
    np.testing.assert_allclose(
        np.asarray(model.scales[: points.shape[0]]), expected, rtol=1.0e-6
    )


def test_inactive_slots_have_zero_activated_opacity():
    model = GaussianModel.empty(ModelConfig(capacity=8, sh_degree=1))
    model.active_mask[:3] = True
    opacities = model.activated()["opacities"]
    assert jnp.all(opacities[:3] > 0.0)
    assert jnp.all(opacities[3:] == 0.0)


def test_state_dict_round_trip_can_preserve_a_larger_logical_maximum():
    config = ModelConfig(capacity=16, bucket_min_capacity=4, sh_degree=1)
    model = GaussianModel.empty(config, physical_capacity=4)
    restored = GaussianModel.from_state_dict(
        model.state_dict(), max_capacity=model.max_capacity
    )
    assert restored.capacity == 4
    assert restored.max_capacity == 16


def test_appearance_point_cloud_uses_mutually_exclusive_feature_color_state():
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], np.float32)
    colors = np.array([[255, 0, 128], [64, 255, 0]], np.uint8)
    config = ModelConfig(capacity=4, bucket_min_capacity=4, sh_degree=3)
    feature_key = jax.random.key(9)

    model = GaussianModel.from_point_cloud(
        points,
        colors,
        config,
        appearance_feature_dim=32,
        feature_key=feature_key,
    )

    assert model.has_appearance
    assert not hasattr(model, "sh0")
    assert not hasattr(model, "sh_rest")
    assert model.features.shape == (4, 32)
    assert model.colors.shape == (4, 3)
    np.testing.assert_array_equal(
        model.features[:2], jax.random.uniform(feature_key, (2, 32))
    )
    np.testing.assert_array_equal(model.features[2:], 0.0)
    np.testing.assert_array_equal(model.colors[2:], 0.0)
    np.testing.assert_allclose(
        jax.nn.sigmoid(model.colors[:2]),
        colors.astype(np.float32) / 255.0,
        atol=1.0e-6,
    )
    activated = model.activated()
    assert set(activated) == {
        "means",
        "quats",
        "scales",
        "opacities",
        "features",
        "colors",
        "active_mask",
    }
    with pytest.raises(ValueError, match="appearance"):
        _ = model.sh_coeffs

    restored = GaussianModel.from_state_dict(
        model.state_dict(), max_capacity=model.max_capacity
    )
    assert restored.has_appearance
    np.testing.assert_array_equal(restored.features[...], model.features[...])
    np.testing.assert_array_equal(restored.colors[...], model.colors[...])


def test_empty_appearance_model_builds_checkpoint_restore_target():
    model = GaussianModel.empty(
        ModelConfig(capacity=8, bucket_min_capacity=4),
        physical_capacity=4,
        appearance_feature_dim=32,
    )

    assert model.has_appearance
    assert model.features.shape == (4, 32)
    assert model.colors.shape == (4, 3)
    np.testing.assert_array_equal(model.features[...], 0.0)
    np.testing.assert_array_equal(model.colors[...], 0.0)


def test_point_cloud_worker_count_accepts_more_than_four():
    points = np.zeros((2, 3), np.float32)
    model = GaussianModel.from_point_cloud(
        points,
        points,
        ModelConfig(capacity=4),
        num_workers=5,
    )
    assert int(model.active_count) == 2
    with pytest.raises(ValueError, match="must be positive"):
        GaussianModel.from_point_cloud(
            points,
            points,
            ModelConfig(capacity=4),
            num_workers=0,
        )
