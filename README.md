# jax-gs

`jax-gs` 是一个基于 JAX 与 Flax NNX 的可微 Gaussian Splatting 实现，包含渲染、训练、COLMAP 数据读取、checkpoint、导出、稠密化策略和可选 NVIDIA GPU 加速后端。

当前 API 与数值约定以 `gsplat main@2b902ff` 为兼容目标。默认路径保持 pure JAX；Pallas、cuTile 和 CUDA/XLA FFI 都必须显式启用。

## 功能

- 3D Gaussian Splatting 与 2D Gaussian Splatting。
- 3DGUT、Eval3D、相机畸变、rolling shutter 与 LiDAR 相关 API。
- 固定容量模型、packed metadata、交点溢出检测与显存预估。
- Flax NNX 训练循环、Adam、Default/MCMC 稠密化策略。
- appearance optimization、pose optimization、sparse/visible optimizer 选项。
- COLMAP 数据读取、checkpoint 恢复、PLY 与 `.splat` 导出。
- 单机多 GPU 的 JAX distributed 路径。
- 可选 Pallas compositor、CUDA FFI compositor、cuTile intersections 和 cuTile+CUB topology。

## 环境

- Python `>=3.12`
- JAX `>=0.11.0`
- CUDA 13 JAX wheel（项目依赖为 `jax[cuda13]`）
- 可选原生后端首次构建需要 `nvcc`

推荐使用 `uv`：

```bash
uv sync --all-groups
uv run python -c "import jax; print(jax.devices())"
```

CPU 与默认 JAX 路径不依赖 cuTile，也不会在普通导入时编译 CUDA 源码。

### 可选 cuTile

```bash
uv pip install --python .venv/bin/python \
  'cuda-tile[tileiras]>=1.5.0'
```

cuTile 1.5 需要较新的 NVIDIA driver，并需要 Tile IR compiler 或兼容 CUDA Toolkit。具体 GPU 支持范围取决于安装的 `tileiras` 版本。

## Python 快速开始

```python
import jax.numpy as jnp

from jax_gs import RasterizationConfig, rasterization

N = 1024
means = jnp.zeros((N, 3), dtype=jnp.float32)
quats = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32), (N, 1))
scales = jnp.full((N, 3), 0.01, dtype=jnp.float32)
opacities = jnp.full((N,), 0.5, dtype=jnp.float32)
colors = jnp.full((N, 3), 0.5, dtype=jnp.float32)
viewmats = jnp.eye(4, dtype=jnp.float32)[None]
Ks = jnp.array(
    [[[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]]],
    dtype=jnp.float32,
)

renders, alphas, info = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width=640,
    height=360,
    config=RasterizationConfig(),
)

print(renders.shape)  # [C, H, W, 3]
print(alphas.shape)   # [C, H, W, 1]
print(info["intersection_overflow"])
```

默认 `RasterizationConfig()` 使用 JAX compositor 与 JAX intersection pipeline。

## COLMAP 数据

预期目录：

```text
scene/
├── images_8/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

先检查数据：

```bash
uv run jax-gs inspect-data /path/to/scene --image-dir images_8
```

## 训练

生成默认配置：

```bash
uv run jax-gs init-config \
  --data /path/to/scene \
  --output scene.json
```

开始训练：

```bash
uv run jax-gs train \
  --config scene.json \
  --output outputs/scene
```

恢复训练：

```bash
uv run jax-gs train \
  --config scene.json \
  --resume outputs/scene/checkpoints/step_00010000
```

常用覆盖项：

```bash
uv run jax-gs train \
  --config scene.json \
  --steps 30000 \
  --strategy default \
  --capacity 1000000 \
  --target-primitives 1000000 \
  --bucket-min-capacity 65536 \
  --max-intersections 1048576 \
  --max-candidates-per-tile 2048
```

2DGS：

```bash
uv run jax-gs train \
  --config scene.json \
  --model-type 2dgs \
  --normal-loss \
  --dist-loss
```

`--target-primitives` 是可选的 DefaultStrategy 增长调度：它按剩余 refine 次数控制新增点数，同时仍受梯度候选、`--max-new-per-refine` 和逻辑/物理容量限制。它不会强制补足缺少的候选，也不会为了维持目标而抑制剪枝或强制缩小已有点集，因此目标是增长上限而非无条件精确的最终点数。未设置时保持原有阈值 densification；当前不支持 distributed 训练。

训练配置也可以直接在 Python 中构造：

```python
from jax_gs import DataConfig, RasterizationConfig, TrainConfig

config = TrainConfig(
    data=DataConfig(root="/path/to/scene", image_dir="images_8"),
    rasterizer=RasterizationConfig(max_intersections=1_048_576),
    steps=30_000,
    output_dir="outputs/scene",
)
config.save("scene.json")
```

## 渲染与导出

渲染 checkpoint：

```bash
uv run jax-gs render \
  outputs/scene/checkpoints/step_00030000 \
  --data /path/to/scene \
  --split test \
  --index 0 \
  --output render.png \
  --alpha alpha.png
```

导出：

```bash
uv run jax-gs export checkpoint_dir model.ply
uv run jax-gs export checkpoint_dir model.splat
```

## 显存与容量

JAX shape 在编译时固定。模型使用逻辑容量与物理 bucket 分离，intersection buffer 也使用固定容量。

训练前估算显存：

```bash
uv run jax-gs estimate-memory \
  --config scene.json \
  --active-target 250000 \
  --image-height 1080 \
  --image-width 1920
```

渲染与训练时应检查：

```python
info["intersection_overflow"]
info["intersection_required_count"]
info["intersection_capacity"]
info["tile_overflow"]
info["candidate_limit_exceeded"]
```

`intersection_overflow=True` 表示固定 intersection capacity 不足。训练安全状态会阻止溢出 step 提交 optimizer 更新。

## 渲染后端

后端通过 `RasterizationConfig` 独立选择：

```python
RasterizationConfig(
    backend="intersections",
    compositor_backend="jax",
    intersection_backend="auto",
    intersection_mode="auto",
    sort_backend="auto",
)
```

### Rasterizer

| `backend` | 说明 |
| --- | --- |
| `auto` | 默认 fixed-capacity sorted-intersections 路径 |
| `jax` / `intersections` | 显式选择同一 JAX 路径 |
| `reference` | 慢速对照实现，适合调试 |

### Compositor

| `compositor_backend` | 说明 |
| --- | --- |
| `jax` | 默认；覆盖面最广 |
| `pallas` | 显式 Pallas/Mosaic GPU compositor |
| `cuda_ffi` | 显式 CUDA/XLA FFI compositor |

`cuda_ffi` 当前用于 float32 3DGS intersections 路径，要求 tile size 16，支持 channel count `{1, 2, 3, 4, 8, 16, 32}`。它不支持 AbsGrad、Eval3D、2DGS 或 distributed rasterization。

### Intersections

| `intersection_backend` | 说明 |
| --- | --- |
| `auto` / `jax` | 默认 pure-JAX topology pipeline |
| `pallas` | Pallas AccuTile count/emit |
| `cuda_tile` | NVIDIA cuTile AccuTile count/emit |
| `cuda_tile_cub` | cuTile count/emit + CUDA FFI/CUB prefix、sort、offsets |

`cuda_tile_cub` 保留 JAX projection 与 AccuTile geometry state，并严格保留：

- overflow 前的 Gaussian-major 固定前缀；
- `(tile_id, depth, gaussian_id)` 排序；
- tile offsets；
- `-1` padding；
- saturated `required_count` 与 overflow metadata。

该后端要求 float32、pinhole 3DGS 和 opacity-aware AccuTile。UT、Eval3D、2DGS、AABB 与 distributed 模式会被拒绝。

显式启用完整 NVIDIA 路径：

```python
config = RasterizationConfig(
    backend="intersections",
    compositor_backend="cuda_ffi",
    intersection_backend="cuda_tile_cub",
    intersection_mode="accutile",
    tile_size=16,
    max_intersections=524_288,
    max_candidates_per_tile=2048,
)
```

原生库按源码、JAX、NVCC 和 compute capability 缓存。相关环境变量：

```text
JAX_GS_NVCC
JAX_GS_CUDA_FFI_CACHE_DIR
JAX_GS_CUDA_FFI_LIBRARY
JAX_GS_CUDA_INTERSECTIONS_FFI_LIBRARY
```

## cuTile 调优

项目提供 fresh-process profile 调优器。它计时完整 renderer，而不是在 FFI 调用中动态 benchmark：

```bash
uv run python benchmarks/autotune_cutile.py \
  --npz /path/to/scene.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --max-intersections 524288 \
  --radius-clip 3
```

根据结果设置：

```bash
export JAX_GS_CUTILE_TUNING=default
```

可选 profile：`default`、`small`、`wide`、`low_occupancy`。

手动覆盖：

```text
JAX_GS_CUTILE_COUNT_BLOCK_SIZE
JAX_GS_CUTILE_EMIT_BLOCK_SIZE
JAX_GS_CUTILE_COUNT_OCCUPANCY
JAX_GS_CUTILE_EMIT_OCCUPANCY
```

这些值是静态编译参数，改变后会触发新的 cuTile/JAX 编译。

## Benchmark

```bash
uv run python benchmarks/benchmark_rasterization.py \
  --npz /path/to/scene.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --backend intersections \
  --compositor-backend cuda_ffi \
  --intersection-backend cuda_tile_cub \
  --intersection-mode accutile \
  --max-intersections 524288 \
  --max-candidates-per-tile 2048 \
  --k 512 \
  --backward \
  --allow-unsafe
```

RTX 5090 garden workload（138,766 active Gaussians、640×360、352,091 intersections、capacity 524,288）的一次保留门禁测量约为：

| 路径 | forward | `value_and_grad` |
| --- | ---: | ---: |
| `cuda_tile_cub` + `cuda_ffi` | 0.734 ms | 1.369 ms |

性能取决于 GPU、driver、JAX/XLA、capacity、填充率与场景分布；请使用 fresh process 重测自己的 workload。

## 测试

默认测试集排除 resource-heavy 用例：

```bash
uv run pytest
```

单独运行资源密集测试：

```bash
uv run pytest -m resource_heavy
```

构建发布包：

```bash
uv build
```

项目对可选 CUDA/cuTile 路径保持 lazy import；普通导入和默认测试不应要求 `cuda.tile`、CUDA toolkit 或预编译原生库。

## 已知边界

- 默认 JAX 路径是兼容性基线；实验性 GPU 后端不会自动启用。
- CUDA FFI 与 cuTile+CUB 当前以单设备 float32 3DGS 为主要目标。
- distributed Gaussian-shard 与 cuTile/CUB 组合尚未验证。
- 高阶梯度、JVP 和部分 gsplat 实验功能并非所有后端都支持。
- 固定 capacity 改变会触发 JIT 重编译。

## License

项目使用 Apache-2.0。上游 gsplat 与 NVIDIA CUDA 适配的版权和归属见 [`NOTICE.md`](NOTICE.md)。
