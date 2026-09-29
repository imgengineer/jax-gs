from dataclasses import dataclass
from pathlib import Path

import grain
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .scene.camera import Camera


@dataclass(frozen=True)
class Frame:
    image_path: Path
    camera: Camera
    resample: int = Image.Resampling.LANCZOS

    def load_rgb(self) -> np.ndarray:
        """Decode on the host; Grain workers never initialize a JAX device."""
        with Image.open(self.image_path) as image:
            image = image.convert("RGB")
            if image.size != (self.camera.width, self.camera.height):
                image = image.resize((self.camera.width, self.camera.height), self.resample)
            return np.asarray(image, dtype=np.uint8)

    def load_image(self) -> jnp.ndarray:
        return jnp.asarray(self.load_rgb().astype(np.float32) / 255.0)


def image_dataset(frames: list[Frame], *, steps: int | None = None) -> grain.IterDataset:
    """Ordered uint8 RGB images with bounded, threaded decoding and prefetch.

    Read each frame once by default. With ``steps``, repeat in frame order
    and stop before reading beyond that many training samples.
    """
    dataset = grain.MapDataset.source(frames)
    if steps is not None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        dataset = dataset.repeat().slice(slice(steps))
    return dataset.map(Frame.load_rgb).to_iter_dataset(
        grain.ReadOptions(num_threads=4, prefetch_buffer_size=8)
    )
