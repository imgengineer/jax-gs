import jax.numpy as jnp
import pytest

from jax_gs.scene import GaussianScene
from jax_gs.stage import Stage


def _scene(count=3, id="scene"):
    return GaussianScene.from_splats(
        {
            "means": jnp.zeros((count, 3), dtype=jnp.float32),
            "scales": jnp.zeros((count, 3), dtype=jnp.float32),
            "quats": jnp.zeros((count, 4), dtype=jnp.float32),
            "opacities": jnp.zeros((count,), dtype=jnp.float32),
        },
        id=id,
    )


def _capture(splats, **kwargs):
    return splats, kwargs


def test_stage_registers_and_returns_scenes_in_insertion_order():
    stage = Stage()
    ego = _scene(2, "ego")
    world = _scene(5, "world")
    stage.add_scene(ego, _capture)
    stage.add_scene(world, _capture)
    assert stage.scene_ids() == ["ego", "world"]
    assert stage.get_scene("ego") is ego
    assert stage.get_scene("world") is world


def test_stage_rejects_duplicate_scene_ids():
    stage = Stage()
    stage.add_scene(_scene(id="world"), _capture)
    with pytest.raises(ValueError, match="already registered"):
        stage.add_scene(_scene(id="world"), _capture)


@pytest.mark.parametrize("operation", ["get", "render"])
def test_stage_unknown_scene_has_available_ids(operation):
    stage = Stage()
    stage.add_scene(_scene(id="world"), _capture)
    with pytest.raises(KeyError, match="not registered.*world"):
        if operation == "get":
            stage.get_scene("missing")
        else:
            stage.render("missing")


def test_stage_render_passes_splats_as_keyword_and_forwards_kwargs():
    calls = []

    def render_fn(**kwargs):
        calls.append(kwargs)
        return "rendered"

    stage = Stage()
    scene = _scene(id="world")
    stage.add_scene(scene, render_fn)
    result = stage.render("world", camera="cam", width=64)
    assert result == "rendered"
    assert calls == [{"splats": scene.splats, "camera": "cam", "width": 64}]


def test_stage_dispatches_each_scene_to_its_own_renderer():
    stage = Stage()
    stage.add_scene(
        _scene(2, "a"),
        lambda splats, **_: ("a", splats["means"].shape[0]),
    )
    stage.add_scene(
        _scene(7, "b"),
        lambda splats, **_: ("b", splats["means"].shape[0]),
    )
    assert stage.render("a") == ("a", 2)
    assert stage.render("b") == ("b", 7)


def test_stage_preserves_arbitrary_return_arity():
    stage = Stage()
    scene = _scene()
    two = (jnp.zeros((1,)), jnp.ones((1,)))
    stage.add_scene(scene, lambda splats, **_: two)
    assert stage.render("scene") is two

    stage = Stage()
    three = (jnp.zeros((1,)), jnp.ones((1,)), {"radii": jnp.zeros((3,))})
    stage.add_scene(scene, lambda splats, **_: three)
    assert stage.render("scene") is three


def test_empty_stage_render_raises():
    with pytest.raises(KeyError, match="not registered"):
        Stage().render("scene")
