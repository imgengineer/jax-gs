from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jax_gs.data import (
    ColmapDataSource,
    create_grain_dataset,
    load_colmap_scene,
    mipnerf360_split_indices,
)


def _write_model(scene_dir: Path) -> None:
    model_dir = scene_dir / "sparse" / "0"
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 5, 1, 100, 80))
        file.write(struct.pack("<dddd", 50.0, 60.0, 40.0, 30.0))

    records = ((20, "b.JPG"), (10, "a.JPG"))
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(records)))
        for image_id, name in records:
            file.write(
                struct.pack(
                    "<i7di",
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    float(image_id),
                    0.0,
                    0.0,
                    5,
                )
            )
            file.write(name.encode("utf-8") + b"\x00")
            file.write(struct.pack("<Q", 0))
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", 0))

    for factor in (1, 2, 4, 8):
        image_dir = scene_dir / ("images" if factor == 1 else f"images_{factor}")
        image_dir.mkdir()
        Image.new("RGB", (50, 20), (255, 0, 0)).save(image_dir / "a.png")
        Image.new("RGB", (25, 40), (0, 128, 255)).save(image_dir / "b.png")


@pytest.fixture
def synthetic_scene(tmp_path: Path) -> Path:
    _write_model(tmp_path)
    return tmp_path


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_supports_standard_image_directories(
    synthetic_scene: Path, factor: int
) -> None:
    scene = load_colmap_scene(synthetic_scene, factor=factor, load_points=False)
    expected_name = "images" if factor == 1 else f"images_{factor}"
    assert scene.image_dir.name == expected_name
    assert scene.image_names == ("a.JPG", "b.JPG")


def test_scales_intrinsics_per_actual_image_dimensions(synthetic_scene: Path) -> None:
    scene = load_colmap_scene(synthetic_scene, factor=4, load_points=False)

    first, second = scene.images
    assert (first.width, first.height) == (50, 20)
    assert (second.width, second.height) == (25, 40)
    np.testing.assert_allclose(
        first.K, [[25.0, 0.0, 20.0], [0.0, 15.0, 7.5], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        second.K, [[12.5, 0.0, 10.0], [0.0, 30.0, 15.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        scene.worldtocams @ scene.camtoworlds,
        np.broadcast_to(np.eye(4), (len(scene.images), 4, 4)),
    )


def test_mipnerf360_filename_sorted_split() -> None:
    np.testing.assert_array_equal(
        mipnerf360_split_indices(10, "test", test_every=4), [0, 4, 8]
    )
    np.testing.assert_array_equal(
        mipnerf360_split_indices(10, "val", test_every=4), [0, 4, 8]
    )
    np.testing.assert_array_equal(
        mipnerf360_split_indices(10, "train", test_every=4),
        [1, 2, 3, 5, 6, 7, 9],
    )


def test_data_source_crop_resize_and_static_output(synthetic_scene: Path) -> None:
    scene = load_colmap_scene(synthetic_scene, factor=8, load_points=False)
    source = ColmapDataSource(
        scene,
        split="train",
        test_every=2,
        crop_size=(20, 20),
        resize=(10, 10),
    )

    assert len(source) == 1
    assert source.image_shape == (10, 10, 3)
    item = source[0]
    assert item["image"].shape == (10, 10, 3)
    assert item["image"].dtype == np.float32
    assert np.all((item["image"] >= 0.0) & (item["image"] <= 1.0))
    assert item["K"].dtype == np.float32
    np.testing.assert_allclose(
        item["K"], [[6.25, 0.0, 4.0], [0.0, 15.0, 2.5], [0.0, 0.0, 1.0]]
    )
    assert item["image_index"] == 1
    assert item["dataset_index"] == 0
    assert item["image_name"] == "b.JPG"
    assert item["camera_index"] == 0
    np.testing.assert_allclose(item["w2c"] @ item["c2w"], np.eye(4), atol=1e-6)


def test_variable_raw_sizes_require_crop_or_resize(synthetic_scene: Path) -> None:
    scene = load_colmap_scene(synthetic_scene, load_points=False)
    with pytest.raises(ValueError, match="different sizes"):
        ColmapDataSource(scene, split="all")

    source = ColmapDataSource(scene, split="all", resize=(12, 16))
    assert source[0]["image"].shape == (12, 16, 3)
    assert source[1]["image"].shape == (12, 16, 3)


def test_grain_map_dataset_wrapper(synthetic_scene: Path) -> None:
    dataset = create_grain_dataset(
        synthetic_scene,
        split="all",
        resize=(8, 12),
        factor=2,
        shuffle=False,
    )

    assert len(dataset) == 2
    assert dataset[0]["image"].shape == (8, 12, 3)
    assert dataset[0]["image_name"] == "a.JPG"


def test_grain_repeat_precedes_static_batching_and_keeps_small_scenes_trainable(
    synthetic_scene: Path,
) -> None:
    dataset = create_grain_dataset(
        synthetic_scene,
        split="all",
        resize=(8, 12),
        shuffle=False,
        repeat=True,
        batch_size=3,
        drop_remainder=True,
    )

    np.testing.assert_array_equal(dataset[0]["dataset_index"], [0, 1, 0])
    np.testing.assert_array_equal(dataset[1]["dataset_index"], [1, 0, 1])

    shuffled = create_grain_dataset(
        synthetic_scene,
        split="all",
        resize=(8, 12),
        shuffle=True,
        seed=0,
        repeat=True,
    )
    first_epoch = tuple(int(shuffled[index]["dataset_index"]) for index in range(2))
    second_epoch = tuple(int(shuffled[index]["dataset_index"]) for index in range(2, 4))
    assert set(first_epoch) == {0, 1}
    assert set(second_epoch) == {0, 1}
    assert second_epoch != first_epoch


def test_colmap_data_source_in_memory_cache(synthetic_scene: Path) -> None:
    source_uncached = ColmapDataSource(
        synthetic_scene, split="all", resize=(12, 16), cache_images=False
    )
    assert source_uncached._cache is None
    item0_uncached = source_uncached[0]

    source_cached = ColmapDataSource(
        synthetic_scene, split="all", resize=(12, 16), cache_images=True
    )
    assert source_cached._cache is not None
    item0_cached = source_cached[0]
    assert 0 in source_cached._cache
    # Second access returns cached object
    assert source_cached[0] is item0_cached
    np.testing.assert_allclose(item0_cached["image"], item0_uncached["image"])


def test_colmap_data_source_uint8(synthetic_scene: Path) -> None:
    source_f32 = ColmapDataSource(
        synthetic_scene, split="all", resize=(12, 16), uint8=False
    )
    source_u8 = ColmapDataSource(
        synthetic_scene, split="all", resize=(12, 16), uint8=True
    )
    item_f32 = source_f32[0]
    item_u8 = source_u8[0]
    assert item_u8["image"].dtype == np.uint8
    assert item_f32["image"].dtype == np.float32
    np.testing.assert_allclose(
        item_u8["image"].astype(np.float32) / 255.0,
        item_f32["image"],
        atol=1.0 / 255.0,
    )


@pytest.mark.parametrize("batch_size", [1, 6])
def test_resume_seeks_grain_batches_without_reading_prior_epochs(batch_size):
    import grain
    from jax_gs.training._data import _infinite_batches

    class CountingSource:
        reads = 0

        def __len__(self):
            return 37

        def __getitem__(self, index):
            self.reads += 1
            return {"image_index": np.int32(index)}

    source = CountingSource()
    dataset = (
        grain.MapDataset.source(source).shuffle(seed=42).repeat()
        .batch(batch_size, drop_remainder=True)
    )
    expected = [dataset[index]["image_index"] for index in range(2000, 2010)]
    source.reads = 0
    batches = _infinite_batches(dataset, num_workers=1, start_batch=2000)
    try:
        actual = [next(batches)["image_index"] for _ in range(10)]
    finally:
        batches.close()
    np.testing.assert_array_equal(actual, expected)
    assert source.reads <= 30 * batch_size


@pytest.mark.slow
def test_stump_metadata_only() -> None:
    scene_dir = Path("/home/lzc/datasets/stump")
    if not scene_dir.is_dir():
        pytest.skip("local Mip-NeRF360 stump dataset is unavailable")

    scene = load_colmap_scene(scene_dir, factor=8, load_points=False)

    assert len(scene.images) == 125
    assert scene.model.points3D == {}
    assert scene.image_dir == scene_dir / "images_8"
    assert all(image.path.is_file() for image in scene.images)
    first = scene.images[0]
    camera = scene.model.cameras[first.camera_id]
    np.testing.assert_allclose(
        first.K,
        camera.intrinsic_matrix(width=first.width, height=first.height),
    )
