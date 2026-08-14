# jax-gs

`jax-gs` 是一个基于 JAX 与 Flax NNX 的高性能可微 Gaussian Splatting 框架，支持 3DGS/2DGS 渲染、训练、COLMAP 数据读取、动态分桶容量、Checkpoint 恢复、PLY/`.splat` 导出与可选 NVIDIA 原生加速流水线。

项目严格对齐 `gsplat main@2b902ff` 的数值精度与拓扑定义。Python API 保持纯净的 JAX 默认值；CLI 在支持的 NVIDIA GPU 环境下会自动路由至最高性能的严格原生后端。

---

## 核心特性

- **完整 3DGS 与 2DGS**：支持 3DGS、2DGS、3DGUT、Eval3D、相机畸变、Rolling Shutter 与 LiDAR 扩展。
- **动态分桶与显存安全**：逻辑容量与物理存储分离，高水位自动分桶扩展，避免点数变化引发频繁 JIT 重编译，支持交点溢出检测与安全步重放。
- **Flax NNX 现代训练**：模块化训练循环、Adam 优化器、Default / MCMC 稠密化策略、`--target-primitives` 目标点数增长调度。
- **进阶优化选项**：支持 Appearance Optimization、Pose Optimization、Sparse / Visible 梯度优化。
- **多层后端加速**：
  - **Compositor**：Pure JAX、Pallas / Mosaic GPU 与 CUDA/XLA FFI（共享内存颜色加载 + Warp/Block 级遮挡尾部跳过）。
  - **Intersections**：Pure JAX、Pallas AccuTile、NVIDIA cuTile AccuTile 与 `cuda_tile_cub`（cuTile 计数/发射 + CUB 饱和前缀和、Radix Sort 与并行边界偏移量扫描）。
- **分布式支持**：基于 JAX 原生 SPMD 的单机多 GPU 分布式训练。

---

## 快速安装

- Python `>=3.12`
- JAX `>=0.11.0`（CUDA 13 支持：`jax[cuda13]`）
- 可选原生加速首次运行时需要 `nvcc`

推荐使用 `uv` 管理环境：

```bash
uv sync --all-groups
uv run python -c "import jax; print(jax.devices())"
```

### 可选 cuTile 加速

```bash
uv pip install --python .venv/bin/python 'cuda-tile[tileiras]>=1.5.0'
```

> **说明**：普通导入和纯 JAX 运行不依赖 `cuda.tile` 或预编译库。原生 FFI 库会在首次使用时由 `nvcc` 自动编译并持久化缓存。

---

## Python API 快速开始

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

# 默认 RasterizationConfig() 使用纯 JAX 后端
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

print("Render shape:", renders.shape)  # [C, H, W, 3]
print("Alpha shape:", alphas.shape)    # [C, H, W, 1]
print("Overflow:", info["intersection_overflow"])
```

显式启用最快 NVIDIA 原生流水线：

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

---

## CLI 工具与训练

### 1. 检查 COLMAP 数据

```bash
uv run jax-gs inspect-data /path/to/scene --image-dir images_8
```

### 2. 生成配置并开始训练

```bash
# 生成默认配置
uv run jax-gs init-config --data /path/to/scene --output scene.json

# 开始训练（在支持的环境下会自动启用 CUDA FFI + cuTile CUB 加速）
uv run jax-gs train --config scene.json --output outputs/scene
```

### 3. 常用训练参数

```bash
# 恢复训练
uv run jax-gs train --config scene.json --resume outputs/scene/checkpoints/step_00010000

# 指定训练步数与目标高斯点数增长调度
uv run jax-gs train --config scene.json --steps 30000 --target-primitives 1000000

# 2D Gaussian Splatting
uv run jax-gs train --config scene.json --model-type 2dgs --normal-loss --dist-loss

# 强制使用纯 JAX 渲染后端
uv run jax-gs train --config scene.json --intersection-backend jax
```

### 4. 渲染与导出

```bash
# 渲染评估集
uv run jax-gs render outputs/scene/checkpoints/step_00030000 \
  --data /path/to/scene --split test --index 0 \
  --output render.png --alpha alpha.png

# 导出标准 PLY 与 .splat
uv run jax-gs export outputs/scene/checkpoints/step_00030000 model.ply
uv run jax-gs export outputs/scene/checkpoints/step_00030000 model.splat
```

---

## 渲染后端体系

后端可通过 `RasterizationConfig` 进行组合配置：

| 模块 | 后端选项 | 说明 |
| --- | --- | --- |
| **Compositor** | `jax` | 默认纯 JAX，通用性最广 |
| | `cuda_ffi` | 原生 CUDA/XLA FFI（Shared Memory 颜色加载 + Warp/Block 遮挡跳过） |
| | `pallas` | Mosaic GPU 原生 Pallas 实验性后端 |
| **Intersections** | `auto` / `jax` | 默认纯 JAX 拓扑流水线 |
| | `cuda_tile` | NVIDIA cuTile AccuTile 计数与发射 |
| | `cuda_tile_cub` | cuTile AccuTile + CUDA FFI / CUB 饱和前缀和、Radix Sort 与并行边界扫描 |
| | `pallas` | Pallas AccuTile 计数与发射 |

### 原生加速自动路由规则

在通过 CLI 启动训练时，若满足以下条件，系统将**自动路由**至最快原生组合（`cuda_ffi` + `cuda_tile_cub` + `accutile`）：

- 单 GPU 训练（非 distributed）
- Pinhole 相机模型 3DGS（无 UT、Eval3D、AbsGrad、Appearance Optimization）
- NVIDIA GPU Compute Capability $\ge 10.0$
- 已安装 `cuda.tile` 且具备 `nvcc`（或预编译 FFI 库）

其他场景或显式指定时，严格保持用户指定或 JAX 默认路径。

---

## Benchmark 与测试

### 运行标准 Benchmark

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
  --backward \
  --allow-unsafe
```

### 运行测试集

```bash
# 运行默认测试集
uv run pytest

# 运行资源密集型独立测试
uv run pytest -m resource_heavy
```

---

## License

本项目基于 Apache-2.0 许可证分发。第三方归属与版权声明详见 [`NOTICE.md`](NOTICE.md)。
