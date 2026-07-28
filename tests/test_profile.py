from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_gs.profile as profile


@pytest.fixture(autouse=True)
def _reset_capture_registry():
    profile._pending_captures.clear()
    profile._next_capture_id = 0
    profile.profiler.clear()
    yield
    profile._pending_captures.clear()


def _captures(stem: Path):
    return sorted(stem.parent.glob(f"{stem.name}_*_*.pt"))


def test_capture_inputs_without_environment_returns_original(monkeypatch):
    monkeypatch.delenv("TEST_CAPTURE", raising=False)

    def function(value):
        return value + 1

    assert profile.capture_inputs(envvar="TEST_CAPTURE")(function) is function


def test_capture_inputs_routes_ranges_and_loads_jax_payload(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv(
        "TEST_CAPTURE", f"{first}:1,{second}:1:3"
    )

    def function(value, scale=2):
        return value * scale

    wrapped = profile.capture_inputs(envvar="TEST_CAPTURE")(function)
    assert jnp.array_equal(wrapped(jnp.asarray([1.0])), jnp.asarray([2.0]))
    assert jnp.array_equal(wrapped(jnp.asarray([2.0])), jnp.asarray([4.0]))
    with pytest.raises(SystemExit, match="all captures done"):
        wrapped(jnp.asarray([3.0]))

    assert len(_captures(first)) == 1
    assert len(_captures(second)) == 2
    payload = profile.load_capture(_captures(first)[0])
    assert set(payload) == {"value", "scale"}
    assert isinstance(payload["value"], jax.Array)
    assert jnp.array_equal(payload["value"], jnp.asarray([1.0]))
    assert payload["scale"] == 2


def test_capture_exit_waits_for_all_decorators(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTURE_A", f"{tmp_path / 'a'}:1")
    monkeypatch.setenv("CAPTURE_B", f"{tmp_path / 'b'}:1")
    first = profile.capture_inputs(envvar="CAPTURE_A")(lambda value: value)
    second = profile.capture_inputs(envvar="CAPTURE_B")(lambda value: value)
    first(jnp.asarray(1.0))
    with pytest.raises(SystemExit):
        second(jnp.asarray(2.0))


@pytest.mark.parametrize(
    "spec, message",
    [
        ("a:0", "empty range"),
        ("a:-1:2", "negative values"),
        ("a:3:2", "must be >="),
        ("a:0:3,b:2:4", "claimed by multiple specs"),
        (" , ", "no specs provided"),
    ],
)
def test_capture_spec_validation(monkeypatch, spec, message):
    monkeypatch.setenv("TEST_CAPTURE", spec)
    with pytest.raises(ValueError, match=message):
        profile.capture_inputs(envvar="TEST_CAPTURE")(lambda value: value)


def test_capture_directory_and_rank_tag(monkeypatch, tmp_path):
    monkeypatch.setenv("GSPLAT_INPUT_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("TEST_CAPTURE", "nested/input:1")
    wrapped = profile.capture_inputs(envvar="TEST_CAPTURE")(lambda value: value)
    with pytest.raises(SystemExit):
        wrapped(jnp.asarray([1.0]))
    assert len(list((tmp_path / "nested").glob("input_r3_*.pt"))) == 1


def test_timeit_context_and_decorator_accumulate(monkeypatch):
    monkeypatch.setenv("TIMEIT", "1")
    with profile.timeit("context"):
        jnp.arange(8).block_until_ready()

    @profile.timeit()
    def function(value):
        return value + 1

    function(jnp.asarray(2.0)).block_until_ready()
    assert profile.profiler["context"] >= 0.0
    assert profile.profiler["function"] >= 0.0


def test_timeit_accepts_current_main_dunder_keywords(monkeypatch):
    monkeypatch.delenv("TIMEIT", raising=False)
    timer = profile.timeit()

    wrapped = timer.__call__(f=lambda value: value + 1)

    assert wrapped(2) == 3
    assert timer.__exit__(exc_type=None, exc_val=None, exc_tb=None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8", 8),
        ("8.5", 8.5),
        ("true", True),
        ("None", None),
        ("[1, 2]", [1, 2]),
        ("pinhole", "pinhole"),
    ],
)
def test_profile_override_parsing(raw, expected):
    assert profile._parse_override_value(raw) == expected
    assert profile._parse_input_override(f"value={raw}") == ("value", expected)


def test_channel_override_handles_sh_extra_depth_and_background():
    inputs = {
        "colors": jnp.zeros((1, 5, 4, 3)),
        "sh_degree": 1,
        "extra_signals": jnp.zeros((1, 5, 2)),
        "backgrounds": jnp.zeros((1, 3)),
        "render_mode": "RGB+D",
    }
    notes = profile._apply_channel_override(inputs, 6)
    assert inputs["sh_degree"] is None
    assert inputs["colors"].shape[-1] == 3
    assert inputs["extra_signals"].shape[-1] == 2
    assert inputs["backgrounds"].shape[-1] == 3
    assert notes


def test_make_replay_supports_forward_and_explicit_grad_inputs():
    def operator(means, ids):
        return means * 2.0, {"ids": ids}

    inputs = {
        "means": jnp.asarray([[1.0, 2.0]]),
        "ids": jnp.asarray([1], jnp.int32),
    }
    forward, values = profile._make_replay(operator, inputs, with_grad=False)
    output = forward(values)
    assert jnp.array_equal(output[0], inputs["means"] * 2.0)

    backward, values = profile._make_replay(operator, inputs, with_grad=True)
    loss, gradients = backward(values)
    assert float(loss) == 6.0
    assert jnp.array_equal(gradients[0], jnp.full((1, 2), 2.0))
