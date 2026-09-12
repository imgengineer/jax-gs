from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_gs.scene import (
    GaussianInferenceScene,
    GaussianScene,
    Scene,
    SHCompressionMode,
)
from jax_gs.scene.functional import pack_gaussian_inference_scene


def _raw_splats(count=3, *, sh_degree=None):
    quats = jnp.zeros((count, 4), dtype=jnp.float32).at[:, 0].set(1)
    values = {
        "means": nnx.Param(
            jnp.arange(count * 3, dtype=jnp.float32).reshape((count, 3))
        ),
        "scales": nnx.Param(jnp.zeros((count, 3), dtype=jnp.float32)),
        "quats": nnx.Param(quats),
        "opacities": nnx.Param(jnp.zeros((count,), dtype=jnp.float32)),
    }
    if sh_degree is None:
        values["colors"] = nnx.Param(
            jnp.linspace(0.1, 0.9, max(count * 3, 1), dtype=jnp.float32).reshape(
                (count, 3)
            )
        )
    else:
        basis_count = (sh_degree + 1) ** 2
        values["sh0"] = nnx.Param(jnp.full((count, 1, 3), 0.2, dtype=jnp.float32))
        if basis_count > 1:
            values["shN"] = nnx.Param(
                jnp.full((count, basis_count - 1, 3), 0.1, dtype=jnp.float32)
            )
    return nnx.Dict(values)


def _activated_inputs(count=4, sh_degree=-1):
    means = jnp.arange(count * 3, dtype=jnp.float32).reshape((count, 3)) / 10
    quats = jnp.zeros((count, 4), dtype=jnp.float32).at[:, 0].set(1)
    scales = jnp.full((count, 3), 0.5, dtype=jnp.float32)
    opacities = jnp.linspace(0.1, 0.9, count, dtype=jnp.float32)
    if sh_degree < 0:
        colors = (
            jnp.linspace(-1, 1, count * 3, dtype=jnp.float32).reshape((count, 3))
            if count
            else jnp.empty((0, 3), dtype=jnp.float32)
        )
    else:
        colors = (
            jnp.arange(count * (sh_degree + 1) ** 2 * 3, dtype=jnp.float32).reshape(
                (count, (sh_degree + 1) ** 2, 3)
            )
            / 100
        )
    return means, quats, scales, opacities, colors


def test_scene_public_surface_and_id_validation():
    from jax_gs import scene

    assert scene.__all__ == [
        "functional",
        "Scene",
        "GaussianScene",
        "GaussianInferenceScene",
        "SHCompressionMode",
    ]
    assert issubclass(GaussianScene, Scene)
    with pytest.raises(ValueError, match="non-empty string"):
        GaussianScene("")
    with pytest.raises(ValueError, match="non-empty string"):
        GaussianInferenceScene(3)


def test_gaussian_scene_first_put_keeps_nnx_dict_identity():
    splats = _raw_splats(3)
    scene = GaussianScene.from_splats(splats, id="world")
    assert scene.splats is splats
    assert scene.num_gaussians() == 3
    assert scene.component_names == ["world"]
    assert jnp.array_equal(scene.component_index, jnp.zeros((3,), dtype=jnp.int32))
    state = nnx.state(scene)
    assert isinstance(state["splats"]["means"], nnx.Param)


def test_gaussian_scene_appends_components_and_pads_signal():
    scene = GaussianScene.from_splats(
        _raw_splats(2),
        id="background",
        signal={"velocity": jnp.ones((2, 3), dtype=jnp.float32)},
    )
    second = _raw_splats(1)
    second["ignored_new_key"] = nnx.Param(jnp.ones((1, 2), dtype=jnp.float32))
    scene.put("car", second)
    assert scene.num_gaussians() == 3
    assert scene.component_names == ["background", "car"]
    assert jnp.array_equal(scene.component_index, jnp.asarray([0, 0, 1]))
    assert "ignored_new_key" not in scene.splats
    assert jnp.array_equal(
        scene.signal["velocity"],
        jnp.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=jnp.float32),
    )
    selected = scene.get("car")
    assert selected["name"] == "car"
    assert selected["index"] == 1
    assert selected["splats"]["means"].shape == (1, 3)
    assert selected["signal"]["velocity"].shape == (1, 3)


def test_gaussian_scene_put_and_get_validation():
    scene = GaussianScene("world")
    with pytest.raises(ValueError, match="must not be empty"):
        scene.put("", _raw_splats(1))
    with pytest.raises(ValueError, match="must not be empty"):
        scene.put("empty", nnx.Dict())
    scene.put("world", _raw_splats(1))
    with pytest.raises(ValueError, match="already exists"):
        scene.put("world", _raw_splats(1))
    with pytest.raises(KeyError, match="Unknown component"):
        scene.get("missing")
    with pytest.raises(KeyError, match="Unknown component index"):
        scene.get(2)


def test_gaussian_scene_validate_detects_row_and_membership_errors():
    scene = GaussianScene.from_splats(_raw_splats(2), id="world")
    scene.signal["bad"] = jnp.zeros((1, 2), dtype=jnp.float32)
    with pytest.raises(ValueError, match="signal array"):
        scene.validate()
    del scene.signal["bad"]
    scene.component_index = jnp.asarray([0, 1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="unknown component"):
        scene.validate()


def test_gaussian_scene_state_dict_roundtrip_and_default_component_name():
    scene = GaussianScene.from_splats(
        _raw_splats(2),
        id="world",
        signal={"label": jnp.asarray([3, 4], dtype=jnp.int32)},
    )
    state = scene.state_dict()
    restored = GaussianScene.from_state_dict(state)
    assert restored.id == scene.id
    assert restored.component_names == scene.component_names
    assert jnp.array_equal(restored.component_index, scene.component_index)
    assert jnp.array_equal(restored.signal["label"], scene.signal["label"])
    assert jnp.array_equal(restored.splats["means"][...], scene.splats["means"][...])

    state.pop("component_names")
    state.pop("component_index")
    defaulted = GaussianScene.from_state_dict(state)
    assert defaulted.component_names == ["world"]
    assert jnp.array_equal(defaulted.component_index, jnp.zeros((2,), dtype=jnp.int32))


def test_gaussian_scene_empty_state_roundtrip():
    empty = GaussianScene("empty")
    restored = GaussianScene.from_state_dict(empty.state_dict())
    assert restored.id == "empty"
    assert restored.num_gaussians() == 0
    assert restored.component_names == []


def test_gaussian_scene_topology_hooks_keep_sidecars_aligned():
    def make_scene():
        scene = GaussianScene.from_splats(
            _raw_splats(4),
            id="world",
            signal={"row": jnp.arange(4, dtype=jnp.int32)},
        )
        scene.component_names = ["a", "b"]
        scene.component_index = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        return scene

    duplicated = make_scene()
    duplicated.on_duplicate(jnp.asarray([1, 3], dtype=jnp.int32))
    assert jnp.array_equal(duplicated.component_index, jnp.asarray([0, 1, 0, 1, 1, 1]))
    assert jnp.array_equal(duplicated.signal["row"], jnp.asarray([0, 1, 2, 3, 1, 3]))

    split = make_scene()
    split.on_split(
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([0, 2, 3], dtype=jnp.int32),
    )
    assert jnp.array_equal(split.component_index, jnp.asarray([0, 0, 1, 1, 1]))
    assert jnp.array_equal(split.signal["row"], jnp.asarray([0, 2, 3, 1, 1]))

    removed = make_scene()
    removed.on_remove(jnp.asarray([False, True, False, True]))
    assert jnp.array_equal(removed.component_index, jnp.asarray([0, 0]))
    assert jnp.array_equal(removed.signal["row"], jnp.asarray([0, 2]))

    relocated = make_scene()
    relocated.on_relocate(
        jnp.asarray([0, 2], dtype=jnp.int32),
        jnp.asarray([1, 3], dtype=jnp.int32),
    )
    assert jnp.array_equal(relocated.component_index, jnp.asarray([1, 1, 1, 1]))
    assert jnp.array_equal(relocated.signal["row"], jnp.asarray([1, 1, 3, 3]))

    sampled = make_scene()
    sampled.on_sample_add(jnp.asarray([2, 0], dtype=jnp.int32))
    assert jnp.array_equal(sampled.signal["row"], jnp.asarray([0, 1, 2, 3, 2, 0]))

    permuted = make_scene()
    order = jnp.asarray([3, 1, 0, 2], dtype=jnp.int32)
    permuted.on_permute(order)
    assert jnp.array_equal(permuted.signal["row"], order)


def test_gaussian_scene_fixed_slot_transaction_uses_one_sidecar_snapshot():
    scene = GaussianScene.from_splats(
        _raw_splats(4),
        id="world",
        signal={
            "label": jnp.asarray([10, 20, 30, 40], jnp.int32),
            "vector": jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
        },
    )
    scene.component_names = ["a", "b"]
    scene.component_index = jnp.asarray([0, 1, 0, 1], jnp.int32)

    scene.apply_slot_transaction(
        source_slots=jnp.asarray([0, 1, 3], jnp.int32),
        target_slots=jnp.asarray([1, 2, 1], jnp.int32),
        valid=jnp.asarray([True, True, False]),
    )

    # Slot 2 must read old slot 1 (label 20), not slot 1 after it was
    # overwritten by the first copy. The padded third pair is a no-op even
    # though it aliases the first pair's target.
    assert jnp.array_equal(
        scene.signal["label"], jnp.asarray([10, 10, 20, 40], jnp.int32)
    )
    assert jnp.array_equal(scene.component_index, jnp.asarray([0, 0, 1, 1], jnp.int32))
    assert jnp.array_equal(
        scene.signal["vector"],
        jnp.asarray([[0, 1], [0, 1], [2, 3], [6, 7]], jnp.float32),
    )

    restored = GaussianScene.from_state_dict(scene.state_dict())
    assert jnp.array_equal(restored.component_index, scene.component_index)
    assert jnp.array_equal(restored.signal["label"], scene.signal["label"])
    assert jnp.array_equal(restored.signal["vector"], scene.signal["vector"])


@pytest.mark.parametrize(
    ("degree", "mode", "expected_shape", "expected_dtype"),
    [
        (-1, SHCompressionMode.NONE, (4, 4), jnp.float16),
        (0, SHCompressionMode.NONE, (4, 1, 3), jnp.float32),
        (1, SHCompressionMode.NONE, (4, 4, 3), jnp.float32),
        (2, SHCompressionMode.NONE, (4, 9, 3), jnp.float32),
        (3, SHCompressionMode.NONE, (4, 16, 3), jnp.float16),
        (3, SHCompressionMode.PACKED_32B, (4, 48), jnp.float16),
        (3, SHCompressionMode.PACKED_16B, (4, 48), jnp.float16),
    ],
)
def test_pack_gaussian_inference_scene_layouts(
    degree, mode, expected_shape, expected_dtype
):
    inputs = _activated_inputs(4, degree)
    means_planar, qso, colors = pack_gaussian_inference_scene(*inputs, degree, mode)
    assert means_planar.shape == (3, 4)
    assert means_planar.dtype == jnp.float32
    assert qso.shape == (4, 8)
    assert qso.dtype == jnp.float16
    assert colors.shape == expected_shape
    assert colors.dtype == expected_dtype
    assert jnp.array_equal(means_planar, inputs[0].T)
    assert jnp.array_equal(qso[:, :4], inputs[1].astype(jnp.float16))
    assert jnp.array_equal(qso[:, 7], inputs[3].astype(jnp.float16))
    if degree == -1:
        assert jnp.array_equal(colors[:, 3], jnp.zeros((4,), dtype=jnp.float16))


def test_pack_clamps_fp16_lanes_and_is_no_grad_jittable():
    means, quats, scales, opacities, colors = _activated_inputs(2, -1)
    scales = scales.at[0, 0].set(1.0e9)
    colors = colors.at[1, 2].set(-1.0e9)
    pack = jax.jit(
        partial(
            pack_gaussian_inference_scene,
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )
    )
    packed = pack(means, quats, scales, opacities, colors)
    assert float(packed[1][0, 4]) == 65504.0
    assert float(packed[2][1, 2]) == -65504.0
    gradient = jax.grad(
        lambda value: jnp.sum(
            pack_gaussian_inference_scene(
                value,
                quats,
                scales,
                opacities,
                colors,
                -1,
                SHCompressionMode.NONE,
            )[0]
        )
    )(means)
    assert jnp.array_equal(gradient, jnp.zeros_like(gradient))


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda values: values[:4] + ("-1",) + values[5:], TypeError),
        (lambda values: values[:4] + (values[4], 4, values[6]), ValueError),
        (
            lambda values: values[:5] + (-1, SHCompressionMode.PACKED_32B),
            ValueError,
        ),
        (
            lambda values: (values[0][:, :2],) + values[1:],
            ValueError,
        ),
        (
            lambda values: values[:1] + (values[1].astype(jnp.float16),) + values[2:],
            TypeError,
        ),
        (
            lambda values: values[:4] + (values[4][:, None, :],) + values[5:],
            ValueError,
        ),
    ],
)
def test_pack_validation(mutator, error):
    values = _activated_inputs(4, -1) + (-1, SHCompressionMode.NONE)
    with pytest.raises(error):
        pack_gaussian_inference_scene(*mutator(values))


def test_inference_scene_from_activated_rgb_tensors():
    inputs = _activated_inputs(5, -1)
    scene = GaussianInferenceScene.from_gaussian_tensors(
        *inputs,
        sh_degree=None,
        sh_compression="none",
        id="viewer",
    )
    assert not scene.is_empty()
    assert scene.num_gaussians == 5
    assert scene.sh_degree == -1
    assert scene.sh_compression_mode is SHCompressionMode.NONE
    assert scene.component_names == ["viewer"]
    assert jnp.array_equal(scene.qso_packed[:, :4], inputs[1].astype(jnp.float16))


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (2, lambda value: value.at[0, 0].set(0), "non-positive"),
        (3, lambda value: value.at[0].set(1.2), "outside"),
        (1, lambda value: value.at[0].set(jnp.asarray([1, 1, 1, 1])), "unit-norm"),
        (0, lambda value: value.at[0, 0].set(jnp.nan), "NaN or Inf"),
    ],
)
def test_inference_scene_activation_contract(field, replacement, match):
    inputs = list(_activated_inputs(3, -1))
    inputs[field] = replacement(inputs[field])
    with pytest.raises(ValueError, match=match):
        GaussianInferenceScene.from_gaussian_tensors(
            *inputs,
            sh_degree=None,
            sh_compression="none",
            id="bad",
        )


def test_inference_scene_warns_before_fp16_clamping():
    inputs = list(_activated_inputs(3, -1))
    inputs[2] = inputs[2].at[0, 0].set(1.0e8)
    with pytest.warns(RuntimeWarning, match="fp16 clamping"):
        scene = GaussianInferenceScene.from_gaussian_tensors(
            *inputs,
            sh_degree=None,
            sh_compression="none",
            id="large",
        )
    assert float(scene.qso_packed[0, 4]) == 65504.0


def test_inference_scene_from_training_scene_applies_activations_and_wxyz_order():
    splats = _raw_splats(2)
    splats["quats"][...] = jnp.asarray(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]], dtype=jnp.float32
    )
    splats["scales"][...] = jnp.log(
        jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=jnp.float32)
    )
    scene = GaussianScene.from_splats(splats, id="train")
    inference = GaussianInferenceScene.from_gaussian_scene(scene, id="viewer")
    expected_quats = jnp.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float16
    )
    assert jnp.array_equal(inference.qso_packed[:, :4], expected_quats)
    assert jnp.allclose(
        inference.qso_packed[:, 4:7].astype(jnp.float32),
        jnp.exp(splats["scales"][...]).astype(jnp.float16).astype(jnp.float32),
    )


def test_inference_scene_from_training_scene_concatenates_sh_and_rejects_features():
    scene = GaussianScene.from_splats(_raw_splats(2, sh_degree=3), id="sh")
    inference = GaussianInferenceScene.from_gaussian_scene(
        scene, id="viewer", sh_compression="32b"
    )
    assert inference.sh_degree == 3
    assert inference.colors_packed.shape == (2, 48)

    appearance = _raw_splats(1)
    appearance["features"] = nnx.Param(jnp.ones((1, 4), dtype=jnp.float32))
    appearance_scene = GaussianScene.from_splats(appearance, id="appearance")
    with pytest.raises(ValueError, match="appearance-optimized"):
        GaussianInferenceScene.from_gaussian_scene(appearance_scene, id="viewer")


def test_inference_scene_rejects_multi_component_training_scene():
    scene = GaussianScene.from_splats(_raw_splats(1), id="first")
    scene.put("second", _raw_splats(1))
    with pytest.raises(ValueError, match="multi-component"):
        GaussianInferenceScene.from_gaussian_scene(scene, id="viewer")


def test_inference_scene_put_get_release_and_zero_length_replacement():
    empty_inputs = _activated_inputs(0, -1)
    zero = GaussianInferenceScene.from_gaussian_tensors(
        *empty_inputs,
        sh_degree=None,
        sh_compression="none",
        id="zero",
    )
    assert zero.is_empty()

    full_inputs = _activated_inputs(2, -1)
    packed = pack_gaussian_inference_scene(*full_inputs, -1, SHCompressionMode.NONE)
    zero.put(
        "full",
        {
            "means_planar": packed[0],
            "qso_packed": packed[1],
            "colors_packed": packed[2],
            "sh_degree": -1,
            "sh_compression_mode": SHCompressionMode.NONE,
        },
    )
    assert zero.component_names == ["full"]
    selected = zero.get(0)
    assert selected["qso_packed"].shape == (2, 8)
    with pytest.raises(KeyError):
        zero.get("missing")
    zero.release()
    assert zero.is_empty()
    with pytest.raises(RuntimeError, match="released"):
        zero.get(0)


def test_inference_scene_put_validates_packed_invariants():
    scene = GaussianInferenceScene("viewer")
    with pytest.raises(TypeError, match="float16"):
        scene.put(
            "bad",
            {
                "means_planar": jnp.zeros((3, 2), dtype=jnp.float32),
                "qso_packed": jnp.zeros((2, 8), dtype=jnp.float32),
                "colors_packed": jnp.zeros((2, 4), dtype=jnp.float16),
                "sh_degree": -1,
                "sh_compression_mode": SHCompressionMode.NONE,
            },
        )
