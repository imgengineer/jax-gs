import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import viser.transforms as tf

from jaxgs import CapacityConfig, create_gaussians, seed_gaussians
from jaxgs.render.view import RESOLUTIONS, ViewRenderer, resize_camera
from jaxgs.viewer import initial_view, view_camera


def _pool():
    return seed_gaussians(
        create_gaussians(CapacityConfig(4, sh_degree=1)),
        jnp.array([[0.0, 0.0, 2.0], [0.3, 0.1, 3.0]]),
        jnp.array([[0.8, 0.2, 0.1], [0.2, 0.8, 0.3]]),
        scale=0.06,
        opacity=0.7,
    )


@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 0.6])
def test_view_camera_preserves_pose_and_viewport_rays(aspect):
    rotation = tf.SO3.exp(np.array([0.15, -0.3, 0.2]))
    position = np.array([1.2, -0.4, 3.1])
    fov = np.deg2rad(60)
    camera = view_camera(rotation.wxyz, position, fov, aspect, 640, 360)
    center = position + rotation.as_matrix() @ np.array([0.0, 0.0, 2.0])
    np.testing.assert_allclose(
        camera.world_to_camera @ np.append(center, 1), [0, 0, 2, 1], atol=2e-7
    )
    np.testing.assert_allclose(camera.center, position, atol=2e-7)
    right_edge = position + rotation.as_matrix() @ np.array([aspect * np.tan(fov / 2), 0, 1])
    projected = camera.world_to_camera @ np.append(right_edge, 1)
    np.testing.assert_allclose(camera.fx * projected[0] / projected[2] + camera.cx, 640, atol=1e-4)
    resized = resize_camera(camera, 1920, 1080)
    np.testing.assert_array_equal(resized.world_to_camera, camera.world_to_camera)
    np.testing.assert_allclose(
        [resized.fx, resized.fy, resized.cx, resized.cy],
        np.array([camera.fx, camera.fy, camera.cx, camera.cy]) * 3,
    )


def test_automatic_initial_view_fits_active_model():
    pool = _pool().replace(xyz=_pool().xyz.at[2:].set(1e8))
    position, target, up, fov = initial_view(pool)
    assert np.linalg.norm(target) < 10
    assert np.linalg.norm(target - position) > 0
    np.testing.assert_array_equal(up, [0, -1, 0])
    assert 0 < fov < np.pi


@pytest.mark.parametrize("invalid", ["empty", "sh"])
def test_renderer_rejects_invalid_models(invalid):
    pool = _pool()
    if invalid == "empty":
        pool = pool.replace(n_active=jnp.array(0, jnp.int32))
    else:
        pool = pool.replace(sh=jnp.zeros((4, 2, 3)))
    with pytest.raises(ValueError, match="active Gaussians|SH dimension"):
        ViewRenderer(pool)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="viewer rendering requires JAX CUDA and CuTe",
)
def test_resolution_warmup_reuses_jit_for_dynamic_cameras():
    renderer = ViewRenderer(_pool(), pair_capacity=65536)
    camera = view_camera([1, 0, 0, 0], [0, 0, 0], np.pi / 3, 16 / 9, 1280, 720)
    times = renderer.warmup(camera)
    assert set(times) == set(RESOLUTIONS) and all(value > 0 for value in times.values())
    assert renderer.jitted._cache_size() == len(RESOLUTIONS)
    for width, height in RESOLUTIONS.values():
        rotation = tf.SO3.exp(np.array([0.02, -0.03, 0.01])).wxyz
        view = view_camera(rotation, [0.02, -0.01, 0], np.pi / 2, 4 / 3, width, height)
        actual, overflow = jax.device_get(renderer(view))
        assert actual.shape == (height, width, 3) and actual.dtype == np.uint8
        assert not overflow and np.any(actual)
        expected, _ = jax.device_get(renderer.eager(view))
        np.testing.assert_array_equal(actual, expected)
    assert renderer.jitted._cache_size() == len(RESOLUTIONS)


@pytest.mark.skipif(
    jax.default_backend() != "gpu" or importlib.util.find_spec("cutlass") is None,
    reason="viewer rendering requires JAX CUDA and CuTe",
)
def test_warmup_rejects_visibility_overflow():
    renderer = ViewRenderer(_pool(), pair_capacity=1)
    camera = view_camera([1, 0, 0, 0], [0, 0, 0], np.pi / 3, 16 / 9, 640, 360)
    with pytest.raises(RuntimeError, match="visibility overflow"):
        renderer.warmup(camera)


def test_colmap_initial_view_and_empty_camera_list(tmp_path):
    from PIL import Image

    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (tmp_path / "images").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 32 16 24 24 16 8\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 -1 -2 -3 1 frame.png\n\n")
    Image.fromarray(np.zeros((16, 32, 3), np.uint8)).save(tmp_path / "images/frame.png")
    position, target, up, fov = initial_view(_pool(), tmp_path)
    np.testing.assert_allclose(position, [1, 2, 3])
    np.testing.assert_allclose((target - position)[:2], 0)
    assert target[2] > position[2]
    np.testing.assert_array_equal(up, [0, -1, 0])
    np.testing.assert_allclose(fov, 2 * np.arctan(16 / 48))
    (sparse / "images.txt").write_text("")
    with pytest.raises(ValueError, match="no registered cameras"):
        initial_view(_pool(), tmp_path)


def test_websocket_viewer_camera_resolution_reset_overflow_and_disconnect(tmp_path, monkeypatch):
    import io
    import socket
    import threading
    from collections import deque
    from types import SimpleNamespace

    import msgspec
    import zstandard
    from PIL import Image
    from websockets.sync.client import connect

    from jaxgs import viewer
    from jaxgs.io_manager.checkpoint import save_gaussians

    # Exercise viser's real HTTP/websocket server and GUI callbacks with a
    # deterministic image producer; GPU image parity is checked separately.
    state, errors = {}, []
    ready = threading.Event()
    monkeypatch.setattr(viewer, "RESOLUTIONS", {"small": (32, 16), "large": (48, 24)})

    class Renderer:
        def __init__(self, pool, *, pair_capacity):
            self.pool, self.config = pool, SimpleNamespace(sh_degree=1)
            self.overflow = False
            self.cameras = []
            state["renderer"] = self
            assert pair_capacity == 256

        def warmup(self, camera):
            return {"small": 0.001, "large": 0.001}

        def __call__(self, camera):
            self.cameras.append(camera)
            pixels = np.full((camera.height, camera.width, 3), 64, np.uint8)
            return pixels, np.bool_(self.overflow)

    original = viewer.Viewer

    def capture(*args, **kwargs):
        instance = original(*args, **kwargs)
        state["viewer"] = instance
        ready.set()
        return instance

    monkeypatch.setattr(viewer, "ViewRenderer", Renderer)
    monkeypatch.setattr(viewer, "Viewer", capture)
    model = tmp_path / "model.ply"
    save_gaussians(model, _pool())
    with socket.socket() as free_port:
        free_port.bind(("127.0.0.1", 0))
        port = free_port.getsockname()[1]

    def run():
        try:
            viewer.main(
                [
                    str(model),
                    "--port",
                    str(port),
                    "--resolution",
                    "small",
                    "--max-visibility-pairs",
                    "256",
                ]
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert ready.wait(10), errors
        port = state["viewer"].server.get_port()
        with connect(
            f"ws://127.0.0.1:{port}",
            subprotocols=[f"viser-v{viewer.viser.__version__}"],
            max_size=10_000_000,
        ) as connection:
            seen = []
            pending = deque()

            def send_camera(x=0.0):
                connection.send(
                    msgspec.msgpack.encode(
                        {
                            "type": "ViewerCameraMessage",
                            "wxyz": [1.0, 0.0, 0.0, 0.0],
                            "position": [x, 0.0, 0.0],
                            "fov": float(np.pi / 3),
                            "near": 0.2,
                            "far": 1000.0,
                            "image_height": 480,
                            "image_width": 640,
                            "look_at": [x, 0.0, 2.0],
                            "up_direction": [0.0, -1.0, 0.0],
                        }
                    )
                )

            def receive(predicate):
                for _ in range(50):
                    while pending:
                        message = pending.popleft()
                        if predicate(message):
                            return message
                    packet = connection.recv(timeout=5)
                    compressed_size = int.from_bytes(packet[8:16], "little")
                    payload = msgspec.msgpack.decode(
                        zstandard.decompress(packet[16 : 16 + compressed_size])
                    )
                    messages = payload["messages"]
                    seen.extend(messages)
                    pending.extend(messages)
                raise AssertionError("expected viewer message was not received")

            send_camera()
            first = receive(
                lambda message: (
                    message["type"] == "BackgroundImageMessage" and message["rgb_data"] is not None
                )
            )
            assert Image.open(io.BytesIO(first["rgb_data"])).size == (32, 16)
            dropdown = next(message for message in seen if message["type"] == "GuiDropdownMessage")
            button = next(message for message in seen if message["type"] == "GuiButtonMessage")
            connection.send(
                msgspec.msgpack.encode(
                    {
                        "type": "GuiUpdateMessage",
                        "uuid": dropdown["uuid"],
                        "updates": {"value": "large"},
                    }
                )
            )
            changed = receive(
                lambda message: (
                    message["type"] == "BackgroundImageMessage"
                    and message["rgb_data"] is not None
                    and Image.open(io.BytesIO(message["rgb_data"])).size == (48, 24)
                )
            )
            assert changed["format"] == "jpeg"
            send_camera(0.25)
            receive(lambda message: message["type"] == "BackgroundImageMessage")
            np.testing.assert_allclose(state["renderer"].cameras[-1].world_to_camera[0, 3], -0.25)
            connection.send(
                msgspec.msgpack.encode(
                    {"type": "GuiUpdateMessage", "uuid": button["uuid"], "updates": {"value": True}}
                )
            )
            receive(lambda message: message["type"] == "SetCameraPositionMessage")
            state["renderer"].overflow = True
            send_camera(0.5)
            receive(
                lambda message: (
                    message["type"] == "BackgroundImageMessage" and message["rgb_data"] is None
                )
            )
            receive(
                lambda message: (
                    message["type"] == "GuiUpdateMessage"
                    and "exceeds pair capacity" in str(message["updates"])
                )
            )
        # Closing the connection releases any queued camera update.
    finally:
        if "viewer" in state:
            state["viewer"].stop()
        thread.join(timeout=10)
    assert not thread.is_alive() and not errors
    assert not state["viewer"].pending
