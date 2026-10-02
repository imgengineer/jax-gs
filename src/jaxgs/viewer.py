"""Browse a Gaussian PLY/NPZ with viser and a JIT-compiled JAX renderer."""

import argparse
import json
from pathlib import Path
from threading import Event, Lock
from time import perf_counter

import jax
import numpy as np
import viser
import viser.transforms as tf

from .io_manager.checkpoint import load_gaussians
from .io_manager.colmap import load_colmap_images
from .render.view import RESOLUTIONS, ViewRenderer
from .scene.camera import Camera


def view_camera(wxyz, position, fov: float, aspect: float, width: int, height: int) -> Camera:
    """Convert viser's OpenCV camera-to-world pose without dispatching GPU work."""
    rotation = tf.SO3(np.asarray(wxyz)).as_matrix().T
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ np.asarray(position)
    focal = 0.5 / np.tan(fov / 2)
    # A fixed-size texture is stretched to the browser viewport. Its ray aspect
    # follows the viewport, so resizing the browser only changes scalar inputs.
    return Camera(
        transform,
        np.float32(width * focal / aspect),
        np.float32(height * focal),
        np.float32(width / 2),
        np.float32(height / 2),
        width,
        height,
    )


def initial_view(pool, scene: Path | None = None, images: str = "images"):
    """Use a COLMAP view when supplied, otherwise fit the central model bounds."""
    xyz = np.asarray(pool.xyz)[np.asarray(pool.alive)]
    low, high = np.quantile(xyz, [0.05, 0.95], axis=0)
    center = (low + high) / 2
    distance = max(float(np.linalg.norm(high - low)), 0.01)
    if scene is None:
        return center - np.array([0, 0, distance]), center, np.array([0, -1, 0]), np.pi / 3
    frames = load_colmap_images(scene, images, resolution=-1)
    if not frames:
        raise ValueError("scene has no registered cameras")
    camera = frames[0].camera
    position = np.asarray(camera.center)
    rotation = np.asarray(camera.world_to_camera)[:3, :3]
    fov = 2 * np.arctan(camera.height / (2 * float(camera.fy)))
    return position, position + rotation[2] * distance, -rotation[1], fov


class Viewer:
    """Coalesce camera events and serialize GPU work across connected clients."""

    def __init__(self, server: viser.ViserServer, renderer: ViewRenderer, initial, resolution: str):
        self.server, self.renderer = server, renderer
        self.pending = {}
        self.lock, self.wake = Lock(), Event()
        self.stopped = Event()
        position, look_at, up, fov = initial
        server.scene.world_axes.visible = False
        server.scene.set_up_direction(up)
        server.initial_camera.position = position
        server.initial_camera.look_at = look_at
        server.initial_camera.up = up
        server.initial_camera.fov = fov
        server.initial_camera.near = 0.2
        server.initial_camera.far = 1000.0
        server.gui.add_markdown(
            f"**{int(renderer.pool.n_active):,} Gaussians** · SH{renderer.config.sh_degree}"
        )

        @server.on_client_connect
        def connect(client):
            size = client.gui.add_dropdown(
                "Resolution", options=tuple(RESOLUTIONS), initial_value=resolution
            )
            reset = client.gui.add_button("Reset view")
            status = client.gui.add_markdown("Preparing view…")

            def request(_=None):
                camera = client.camera
                width, height = RESOLUTIONS[size.value]
                snapshot = view_camera(
                    camera.wxyz.copy(),
                    camera.position.copy(),
                    camera.fov,
                    camera.aspect,
                    width,
                    height,
                )
                with self.lock:
                    self.pending[client.client_id] = (client, snapshot, status)
                    self.wake.set()

            client.camera.on_update(request)
            size.on_update(request)

            @reset.on_click
            def reset_view(_):
                with client.atomic():
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = up
                    client.camera.fov = fov
                request()

            request()

        @server.on_client_disconnect
        def disconnect(client):
            with self.lock:
                self.pending.pop(client.client_id, None)

    def stop(self):
        self.stopped.set()
        self.wake.set()

    def run(self):
        try:
            while not self.stopped.is_set():
                self.wake.wait()
                with self.lock:
                    requests = list(self.pending.values())
                    self.pending.clear()
                    self.wake.clear()
                for client, camera, status in requests:
                    if client.client_id not in self.server.get_clients():
                        continue
                    start = perf_counter()
                    image, overflow = jax.device_get(self.renderer(camera))
                    rendered = perf_counter()
                    if overflow:
                        client.scene.set_background_image(None)
                        status.content = "View exceeds pair capacity. Restart with a larger --max-visibility-pairs."
                        continue
                    client.scene.set_background_image(image, format="jpeg", jpeg_quality=90)
                    client.flush()
                    queued = perf_counter()
                    status.content = (
                        f"Render + readback: **{(rendered - start) * 1000:.1f} ms**  \n"
                        f"JPEG + queue: **{(queued - rendered) * 1000:.1f} ms**"
                    )
        finally:
            self.server.stop()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--scene", type=Path, help="use the first COLMAP camera as the initial view"
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="1280x720")
    parser.add_argument("--max-visibility-pairs", type=int, default=8_000_000)
    args = parser.parse_args(argv)
    pool = load_gaussians(args.model)
    renderer = ViewRenderer(pool, pair_capacity=args.max_visibility_pairs)
    initial = initial_view(pool, args.scene, args.images)
    position, look_at, up, fov = initial
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward), axis=1)
    camera = view_camera(tf.SO3.from_matrix(rotation).wxyz, position, fov, 16 / 9, 1280, 720)
    print("Preparing render resolutions…", flush=True)
    print(json.dumps({"warmup_seconds": renderer.warmup(camera)}), flush=True)
    server = viser.ViserServer(host=args.host, port=args.port, label="jaxgs", verbose=False)
    print(f"Viewer: http://{args.host}:{server.get_port()}", flush=True)
    try:
        Viewer(server, renderer, initial, args.resolution).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
