from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_gs.experimental import (
    GaussianInferenceRenderer,
    GaussianInferenceScene,
    RenderReturn,
    rasterize_gaussian_inference_scene,
    render_scene,
)
from jax_gs.experimental.render.components import (
    GaussianInferenceRenderer as ComponentRenderer,
)
from jax_gs.experimental.render.functional import (
    RenderReturn as FunctionalRenderReturn,
)
from jax_gs.experimental.render.kernels.gaussian_inference_ops import (
    _simulate_sh_codec,
    gaussian_render_inference_only,
)
from jax_gs.scene import SHCompressionMode


def _scene(
    *,
    count: int = 2,
    sh_degree: int | None = None,
    sh_compression: str = "none",
    scene_id: str = "test",
) -> GaussianInferenceScene:
    means = jnp.stack(
        (
            jnp.linspace(-0.1, 0.1, count),
            jnp.zeros((count,)),
            jnp.full((count,), 2.0),
        ),
        axis=-1,
    ).astype(jnp.float32)
    quats = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32), (count, 1))
    scales = jnp.full((count, 3), 0.2, jnp.float32)
    opacities = jnp.linspace(0.6, 0.8, count, dtype=jnp.float32)
    if sh_degree is None:
        colors = jnp.tile(jnp.array([[0.9, 0.2, 0.5]], jnp.float32), (count, 1))
    else:
        colors = jnp.zeros((count, (sh_degree + 1) ** 2, 3), jnp.float32)
        colors = colors.at[:, 0].set(jnp.array([0.4, -0.1, 0.2], jnp.float32))
        if sh_degree > 0:
            colors = colors.at[:, 1].set(jnp.array([0.2, 0.0, -0.2], jnp.float32))
    return GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree,
        sh_compression,
        id=scene_id,
    )


def _camera(width: int = 8, height: int = 8) -> tuple[jax.Array, jax.Array]:
    viewmat = jnp.eye(4, dtype=jnp.float32)
    K = jnp.array(
        [
            [16.0, 0.0, width / 2],
            [0.0, 16.0, height / 2],
            [0.0, 0.0, 1.0],
        ],
        jnp.float32,
    )
    return viewmat, K


def _request(width: int = 8, height: int = 8) -> dict[str, object]:
    viewmat, K = _camera(width, height)
    return {"viewmat": viewmat, "K": K, "width": width, "height": height}


def test_experimental_public_imports_match_current_main_hierarchy():
    from jax_gs import experimental
    from jax_gs.experimental.render import (
        GaussianInferenceRenderer as RenderRenderer,
    )

    assert experimental.__all__ == [
        "GaussianInferenceRenderer",
        "GaussianInferenceScene",
        "RenderReturn",
        "rasterize_gaussian_inference_scene",
        "render_scene",
    ]
    assert GaussianInferenceRenderer is ComponentRenderer is RenderRenderer
    assert RenderReturn is FunctionalRenderReturn
    assert issubclass(GaussianInferenceRenderer, nnx.Module)


def test_stateless_render_and_dispatch_contract():
    scene = _scene()

    direct = rasterize_gaussian_inference_scene(scene, **_request())
    dispatched = render_scene(scene, **_request())

    assert isinstance(direct, RenderReturn)
    assert direct.frame.shape == (1, 8, 8, 3)
    assert direct.frame.dtype == jnp.float32
    assert direct.metadata["alpha"].shape == (1, 8, 8, 1)
    assert float(direct.metadata["alpha"].max()) > 0.0
    np.testing.assert_array_equal(dispatched.frame, direct.frame)
    assert dispatched.metadata["render_path"] == "inference"


@pytest.mark.parametrize("tile_size", [8, 16])
def test_stateless_render_accepts_both_tile_sizes(tile_size: int):
    result = rasterize_gaussian_inference_scene(
        _scene(), **_request(), tile_size=tile_size
    )
    assert result.frame.shape == (1, 8, 8, 3)


def test_stateless_render_normalizes_single_camera_batches():
    scene = _scene()
    viewmat, K = _camera()

    singular = rasterize_gaussian_inference_scene(
        scene, viewmat=viewmat, K=K, width=8, height=8
    )
    plural = rasterize_gaussian_inference_scene(
        scene, viewmats=viewmat[None], Ks=K[None], width=8, height=8
    )

    np.testing.assert_array_equal(plural.frame, singular.frame)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"with_ut": True}, "does not support with_ut"),
        ({"render_mode": "D"}, "render_mode='RGB' only"),
        ({"camera_model": "ortho"}, "camera_model='pinhole' only"),
        ({"tile_size": 32}, "tile_size in {8, 16}"),
        ({"sh_degree": 0}, "sh_degree/sh_compression_mode"),
        (
            {"backgrounds": jnp.zeros((1, 3))},
            "unexpected keyword argument 'backgrounds'",
        ),
        ({"unknown": True}, "unexpected keyword argument 'unknown'"),
    ],
)
def test_stateless_render_rejects_unsupported_requests(extra, message: str):
    with pytest.raises(TypeError, match=message):
        rasterize_gaussian_inference_scene(_scene(), **_request(), **extra)


def test_stateless_render_validates_camera_cardinality_and_resolution():
    scene = _scene()
    viewmat, K = _camera()
    with pytest.raises(RuntimeError, match="not both"):
        rasterize_gaussian_inference_scene(
            scene,
            viewmat=viewmat,
            viewmats=viewmat[None],
            K=K,
            width=8,
            height=8,
        )
    with pytest.raises(RuntimeError, match="leading dim"):
        rasterize_gaussian_inference_scene(
            scene,
            viewmats=jnp.tile(viewmat[None], (2, 1, 1)),
            Ks=K[None],
            width=8,
            height=8,
        )
    with pytest.raises(ValueError, match="width must be a positive integer"):
        rasterize_gaussian_inference_scene(
            scene, viewmat=viewmat, K=K, width=0, height=8
        )


def test_stateless_out_retains_container_identity_functionally():
    out = RenderReturn(
        frame=jnp.empty((1, 8, 8, 3), jnp.float32),
        metadata={"alpha": jnp.empty((1, 8, 8, 1), jnp.float32), "stale": True},
    )

    result = rasterize_gaussian_inference_scene(_scene(), out=out, **_request())

    assert result is out
    assert set(out.metadata) == {"alpha"}
    assert jnp.all(jnp.isfinite(out.frame))


def test_stateless_render_rejects_wrong_scene_and_released_scene():
    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        rasterize_gaussian_inference_scene("scene", **_request())
    scene = _scene()
    scene.release()
    with pytest.raises(ValueError, match="has been released"):
        rasterize_gaussian_inference_scene(scene, **_request())


def test_raw_packed_kernel_is_jittable_and_stops_gradients():
    scene = _scene(count=1)
    viewmat, K = _camera()

    def render(means_planar, camera):
        return gaussian_render_inference_only(
            means_planar,
            scene.qso_packed,
            scene.colors_packed,
            camera,
            K,
            8,
            8,
            scene.sh_degree,
            8,
            0.01,
            1.0e10,
            0.0,
            0.3,
            scene.sh_compression_mode,
            None,
        )

    renders, alphas = jax.jit(render)(scene.means_planar, viewmat)
    means_grad = jax.grad(lambda means: jnp.sum(render(means, viewmat)[0]))(
        scene.means_planar
    )

    assert renders.shape == (8, 8, 3)
    assert alphas.shape == (8, 8, 1)
    np.testing.assert_array_equal(means_grad, jnp.zeros_like(means_grad))


def test_sh_codecs_preserve_or_drop_higher_order_chroma():
    colors = jnp.zeros((4, 16, 3), jnp.float32)
    colors = colors.at[:, 1:, 0].set(0.25)
    colors = colors.at[:, 1:, 2].set(-0.25)
    opacities = jnp.ones((4,), jnp.float32)

    decoded_32b = _simulate_sh_codec(colors, opacities, SHCompressionMode.PACKED_32B)
    decoded_16b = _simulate_sh_codec(colors, opacities, SHCompressionMode.PACKED_16B)

    assert float(jnp.max(jnp.abs(decoded_32b[:, 1:, 0]))) > 0.2
    np.testing.assert_allclose(decoded_32b[:, 1:, 0], -decoded_32b[:, 1:, 2])
    np.testing.assert_allclose(decoded_16b[:, 1:], 0.0, atol=1.0e-6)


@pytest.mark.parametrize("compression", ["none", "32b", "16b"])
def test_degree_three_inference_modes_render(compression: str):
    result = rasterize_gaussian_inference_scene(
        _scene(sh_degree=3, sh_compression=compression), **_request()
    )
    assert result.frame.shape == (1, 8, 8, 3)
    assert jnp.all(jnp.isfinite(result.frame))


def test_stateful_renderer_matches_stateless_rgb_and_transmittance():
    scene = _scene()
    stateless = rasterize_gaussian_inference_scene(scene, **_request())

    renderer = GaussianInferenceRenderer(scene)
    stateful = renderer.render(**_request())

    assert renderer.num_gaussians == scene.num_gaussians
    assert stateful.frame.shape == (1, 8, 8, 4)
    assert stateful.frame.dtype == jnp.float16
    assert stateful.metadata == {"format": "RGBT", "channels": "RGBT"}
    np.testing.assert_array_equal(
        stateful.frame[..., :3], stateless.frame.astype(jnp.float16)
    )
    np.testing.assert_array_equal(
        stateful.frame[..., 3:],
        (1.0 - stateless.metadata["alpha"]).astype(jnp.float16),
    )


def test_stateful_renderer_out_resize_and_lifecycle():
    scene = _scene()
    out = RenderReturn(jnp.empty((1, 8, 8, 4), jnp.float16), {"stale": True})
    renderer = GaussianInferenceRenderer(scene, tile_size=16)

    result = renderer.render(out=out, **_request())
    resized = renderer.render(**_request(width=16, height=8))
    renderer.release()

    assert result is out
    assert out.metadata == {"format": "RGBT", "channels": "RGBT"}
    assert resized.frame.shape == (1, 8, 16, 4)
    assert renderer.is_released
    assert renderer.num_gaussians == 0
    with pytest.raises(RuntimeError, match="has been released"):
        renderer.render(**_request())


def test_stateful_renderer_context_manager_and_scene_mutation_guard():
    scene = _scene(count=1)
    with GaussianInferenceRenderer(scene) as renderer:
        other = _scene(count=1, scene_id="other")
        scene.put("other", other.get("other"))
        with pytest.raises(RuntimeError, match="Scene was mutated"):
            renderer.render(**_request())
    assert renderer.is_released


def test_stateful_renderer_validates_construction_and_kwargs():
    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        GaussianInferenceRenderer(None)
    with pytest.raises(ValueError, match="tile_size must be 8 or 16"):
        GaussianInferenceRenderer(_scene(), tile_size=4)
    renderer = GaussianInferenceRenderer(_scene())
    with pytest.raises(TypeError, match="does not support render_mode"):
        renderer.render(**_request(), render_mode="RGB")
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        renderer.render(**_request(), unknown=True)


def test_stateful_renderer_supports_lower_sh_degree_override():
    renderer = GaussianInferenceRenderer(_scene(sh_degree=3, sh_compression="32b"))
    result = renderer.render(**_request(), sh_degree=0)
    assert result.frame.shape == (1, 8, 8, 4)
    assert jnp.all(jnp.isfinite(result.frame))
