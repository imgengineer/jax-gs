# jax-gs

<div align="center">

**High-Performance, Fully Differentiable Gaussian Splatting in JAX & Flax NNX**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.11.0%2B-crimson.svg)](https://github.com/jax-ml/jax)
[![CUDA](https://img.shields.io/badge/CUDA-13%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![gsplat Compatibility](https://img.shields.io/badge/gsplat%20compat-1.5.3%402b902ff-orange.svg)](https://github.com/nerfstudio-project/gsplat)

[核心特性](#-核心特性) •
[快速安装](#-快速安装) •
[Python API 快速开始](#-python-api-快速开始) •
[CLI 命令行工具](#-cli-命令行工具) •
[渲染后端体系](#-渲染后端体系) •
[分布式多卡架构](#-分布式多卡架构) •
[性能评测](#-性能评测) •
[许可协议](#-许可协议)

</div>

---

## 📖 简介

`jax-gs` 是一个基于 JAX 与 Flax NNX 构建的高性能、工业级可微 3D / 2D Gaussian Splatting 研发框架。它将 JAX 优秀的函数式编程、自动微分、JIT 编译和多设备 SPMD 分布式能力，与原生 CUDA / CUB / NVIDIA cuTile 硬件加速深度结合。

项目严格对齐 `gsplat 1.5.3@2b902ff` 的数值精度与拓扑几何定义。在单设备和多 GPU 分布式场景下均提供完整的训练、评估、渲染、位姿/外观联合优化与模型导出能力。

---

## ✨ 核心特性

- **严格数学与拓扑对齐**：完整支持 3DGS、2DGS（法线一致性与深度畸变损失）、3DGUT、Eval3D、相机畸变、Rolling Shutter 与 LiDAR 扩展。严格保证 `MAX_ALPHA = 0.999`、饱和透射率截断与精确梯度传播。
- **极致的原生 CUDA 加速流水线**：
  - **CUDA FFI Compositor**：Shared Memory 颜色预加载、Warp / Block 级遮挡饱和快速跳过，前向耗时压至 **315 µs**，反向压至 **966 µs**（RTX 5090 实测）。
  - **cuTile + CUB Topology**：NVIDIA cuTile AccuTile 几何计数与发射，结合 CUDA FFI / CUB 饱和前缀和、Radix Sort 与并行边界偏移量扫描。
- **全功能分布式多卡训练（Distributed Multi-GPU）**：
  - 基于 JAX 原生 SPMD 的 Gaussian-Sharding 模型并行与数据并行混合架构。
  - 支持分布式 3DGS 与 2DGS 训练，各卡渲染局部相机时均可直接调用原生 `cuda_ffi` 算子加速。
  - 相机位姿优化（`CameraOptModule`）与神经外观优化（`AppearanceOptModule`）支持跨卡 DDP 梯度同步。
  - 支持分布式 Checkpoint 存储、恢复与跨卡数量弹性 Resharding。
- **动态分桶与显存安全机制**：
  - 逻辑容量与物理存储分离，高水位自动分桶扩展，杜绝高斯点数增减引发的频繁 JIT 重编译。
  - 具备端到端交点溢出检测（Intersection Overflow Detection）与安全步重放（Overflow Replay）。
- **现代化训练与稠密化调度**：
  - 基于 Flax NNX 现代模块化设计，内置 Adam 与 Visible Adam 优化器。
  - 支持 DefaultStrategy 与 MCMCStrategy 稠密化策略。
  - 支持 `--target-primitives` 目标点数平滑步进调度。
- **完善的资产与数据接口**：
  - 内置 COLMAP 数据加载器与 Google Grain 数据集流水线。
  - 支持导出为标准 `.ply`（无损/压缩）与 Web 端直接渲染的 `.splat` 文件。

---

## 📦 快速安装

### 1. 基础环境需求

- Python `>= 3.12`
- JAX `>= 0.11.0`（CUDA 13 支持）
- 如需启用可选原生 CUDA 加速，宿主机需要配置 `nvcc`

推荐使用现代包管理器 [`uv`](https://docs.astral.sh/uv/)：

```bash
# 克隆仓库
git clone https://github.com/imgengineer/jax-gs.git
cd jax-gs

# 安装基础依赖
uv sync --all-groups

# 验证 JAX CUDA 设备可用性
uv run python -c "import jax; print('Available JAX Devices:', jax.devices())"
```

### 2. 安装可选 NVIDIA cuTile（可选）

如需使用 cuTile AccuTile 拓扑计数与发射加速后端：

```bash
uv pip install --python .venv/bin/python 'cuda-tile[tileiras]>=1.5.0'
```

> **注意**：普通导入和纯 JAX 运行不强制依赖 `cuda.tile` 或外部动态链接库。原生 CUDA FFI 组件会在首次使用时由 `nvcc` 自动即时编译并持久化缓存至 `~/.cache/jax-gs/cuda-ffi/`。

---

## 🚀 Python API 快速开始

### 基础光栅化渲染

```python
import jax
import jax.numpy as jnp
from jax_gs import RasterizationConfig, rasterization

# 初始化示例高斯点云
N = 1024
means = jnp.zeros((N, 3), dtype=jnp.float32)
quats = jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], jnp.float32), (N, 1))
scales = jnp.full((N, 3), 0.01, dtype=jnp.float32)
opacities = jnp.full((N,), 0.5, dtype=jnp.float32)
colors = jnp.full((N, 3), 0.5, dtype=jnp.float32)

# 设置相机参数 [C, 4, 4] 与 [C, 3, 3]
viewmats = jnp.eye(4, dtype=jnp.float32)[None]
Ks = jnp.array(
    [[[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]]],
    dtype=jnp.float32,
)

# 1. 默认纯 JAX 后端渲染
renders, alphas, info = rasterization(
    means, quats, scales, opacities, colors,
    viewmats, Ks,
    width=640, height=360,
    config=RasterizationConfig(),
)

print("Render Output Shape:", renders.shape)  # [C, H, W, 3]
print("Alpha Output Shape:", alphas.shape)    # [C, H, W, 1]

# 2. 显式启用最高性能 NVIDIA 原生流水线
config_native = RasterizationConfig(
    backend="intersections",
    compositor_backend="cuda_ffi",
    intersection_backend="cuda_tile_cub",
    intersection_mode="accutile",
    tile_size=16,
    max_intersections=524_288,
    max_candidates_per_tile=2048,
)

renders_native, alphas_native, _ = rasterization(
    means, quats, scales, opacities, colors,
    viewmats, Ks,
    width=640, height=360,
    config=config_native,
)
```

---

## 🛠️ CLI 命令行工具

`jax-gs` 提供开箱即用的命令行工具链，覆盖完整训练与交付流程。

### 1. 数据检查与配置生成

```bash
# 检查 COLMAP 数据集完整性与相机位姿
uv run jax-gs inspect-data /path/to/colmap_scene --image-dir images_8

# 生成训练配置文件
uv run jax-gs init-config \
  --data /path/to/colmap_scene \
  --output scene_config.json
```

### 2. 模型训练

```bash
# 单卡标准训练（支持的环境下会自动启用 CUDA FFI + cuTile CUB 加速）
uv run jax-gs train \
  --config scene_config.json \
  --output outputs/scene_run

# 单机多卡分布式训练（指定卡数）
uv run jax-gs train \
  --config scene_config.json \
  --output outputs/scene_dist \
  --num-workers 2

# 从指定 Checkpoint 恢复训练
uv run jax-gs train \
  --config scene_config.json \
  --resume outputs/scene_run/checkpoints/step_00010000

# 启用 2D Gaussian Splatting 并开启法线与畸变正则损失
uv run jax-gs train \
  --config scene_config.json \
  --model-type 2dgs \
  --normal-loss \
  --dist-loss

# 设置目标点数增长调度（如目标 100 万高斯）
uv run jax-gs train \
  --config scene_config.json \
  --steps 30000 \
  --target-primitives 1000000

# 强制使用纯 JAX 后端运行（开发与跨平台对照）
uv run jax-gs train \
  --config scene_config.json \
  --intersection-backend jax
```

### 3. 渲染评估与资产导出

```bash
# 渲染指定 Checkpoint 的测试集视点
uv run jax-gs render outputs/scene_run/checkpoints/step_00030000 \
  --data /path/to/colmap_scene \
  --split test \
  --index 0 \
  --output eval_render.png \
  --alpha eval_alpha.png

# 导出为标准 3DGS PLY 资产
uv run jax-gs export outputs/scene_run/checkpoints/step_00030000 model.ply

# 导出为 Web 端实时查看的 .splat 文件
uv run jax-gs export outputs/scene_run/checkpoints/step_00030000 model.splat
```

### 4. 显存预估

在正式训练前，可使用静态容量与分辨率预估显存峰值：

```bash
uv run jax-gs estimate-memory \
  --config scene_config.json \
  --active-target 300000 \
  --image-height 1080 \
  --image-width 1920
```

---

## 🏛️ 渲染后端体系

`jax-gs` 采用高度解耦的模块化后端架构，主要由 **Compositor（像素光栅化合成）** 与 **Intersections（Tile 拓扑交叉与排序）** 组成：

### 后端选项矩阵

| 模块 | 选项名称 | 说明 | 适用场景 |
| --- | --- | --- | --- |
| **Compositor** | `jax` | 纯 JAX 实现，具备极致通用性与跨平台性 | CPU / TPU / 通用 GPU / 算法原型验证 |
| | `cuda_ffi` | **原生 CUDA FFI 算子**：Shared Memory 颜色加载 + Warp/Block 级遮挡跳过 | **推荐生产训练与推理**（单卡/多卡最高速） |
| | `pallas` | JAX 原生 Mosaic GPU Pallas 算子 | Pallas GPU 原生实验 |
| **Intersections** | `auto` / `jax` | 纯 JAX 拓扑与排序流水线 | 跨平台通用 |
| | `cuda_tile` | NVIDIA cuTile AccuTile 几何计数与发射 | cuTile 原生加速 |
| | `cuda_tile_cub` | **cuTile + CUDA FFI / CUB** 饱和前缀和、Radix Sort 与并行边界偏移量扫描 | **单卡最速拓扑构建** |
| | `pallas` | Pallas AccuTile 计数与发射 | Pallas 拓扑实验 |

### CLI 原生加速自动路由规则

为降低用户配置成本，CLI 入口在检测到以下环境条件时会自动路由至最快原生组合（`compositor_backend="cuda_ffi"`, `intersection_backend="cuda_tile_cub"`, `intersection_mode="accutile"`）：

1. Pinhole 相机模型 3DGS（无 UT、Eval3D、AbsGrad、Appearance Optimization）。
2. NVIDIA GPU Compute Capability $\ge 10.0$。
3. Python 环境已安装 `cuda.tile` 且宿主机具备可用 `nvcc`。
4. 单 GPU 训练（多 GPU 分布式训练自动组合 `cuda_ffi` 与 JAX 拓扑）。

---

## 🌐 分布式多卡架构

`jax-gs` 的分布式训练设计基于 JAX 原生 SPMD 理念，专为大规模高斯场景设计：

```
[Host Dataset (Grain)]
         │ (Shard Camera Batches)
         ├─── Rank 0: Camera Batch 0 ───┐
         └─── Rank 1: Camera Batch 1 ───┤
                                        ▼
             [Rank 0: Shard 0 (N/2)] ───► [All-Gather] ───► [Global Scene (N)]
             [Rank 1: Shard 1 (N/2)] ───► [All-Gather] ───► [Global Scene (N)]
                                                                  │
                                            ┌─────────────────────┴─────────────────────┐
                                            ▼                                           ▼
                               [Rank 0: Local Rasterize (CUDA FFI)]        [Rank 1: Local Rasterize (CUDA FFI)]
                                            │                                           │
                                            ▼                                           ▼
                                    [Rank 0: Local Loss]                        [Rank 1: Local Loss]
                                            │                                           │
                                            └─────────────────────┬─────────────────────┘
                                                                  ▼
                                                   [Reduce-Scatter / Autodiff]
                                                                  │
                                            ┌─────────────────────┴─────────────────────┐
                                            ▼                                           ▼
                                [Update Shard 0 Parameters]                 [Update Shard 1 Parameters]
```

- **显存线性扩展**：高斯存储分片化，单卡显存占用随 GPU 数量翻倍而减半。
- **独立优化与同步控制**：各卡独立执行局部 Densification / Pruning，溢出状态通过全局规约原子化同步，保障多卡模型结构绝对一致。

---

## 📊 性能评测

在 NVIDIA GeForce RTX 5090 上，针对标准 Garden 场景（138,766 个激活高斯，640×360 分辨率，524,288 容量限制）进行严格的端到端单步耗时测量：

| 模块 / 阶段 | 传统 JAX 纯净实现 | jax-gs 原生 CUDA FFI 优化 | 加速比 |
| --- | --- | --- | --- |
| **Compositor Forward** | ~1,250 µs | **315 µs** | **3.97×** |
| **Compositor Backward** | ~2,030 µs | **651 µs** | **3.12×** |
| **CUB Sort & Offsets** | ~897 µs | **108 µs** | **8.30×** |
| **Full Raster Forward** | ~1,585 µs | **606 µs** | **2.61×** |
| **Value & Grad (Step)** | ~3,450 µs | **1,369 µs** | **2.52×** |

> *注：测试环境为 Ubuntu 24.04, Python 3.12, CUDA 13.3, JAX 0.11.0, RTX 5090 32GB。*

### 运行性能基准与自动调优工具

```bash
# 运行完整光栅化前后向 Benchmark
uv run python benchmarks/benchmark_rasterization.py \
  --npz /path/to/garden.npz \
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

# cuTile 块大小与占用率自动调优
uv run python benchmarks/autotune_cutile.py \
  --npz /path/to/garden.npz \
  --resolution 640x360
```

---

## 🧪 测试套件

项目包含完整的单元测试与端到端回归测试，确保任意代码修改均不破坏数学对齐：

```bash
# 运行默认测试集（排除超长集成测试）
uv run pytest

# 运行分布式与多卡专属测试
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/test_distributed.py tests/test_training_distributed.py

# 运行高耗能/重显存独立测试
uv run pytest -m resource_heavy
```

---

## 📄 许可协议与致谢

- 本项目基于 [Apache-2.0 License](LICENSE) 开源。
- 第三方算子架构与开源归属详见 [`NOTICE.md`](NOTICE.md)。
- 感谢 [gsplat](https://github.com/nerfstudio-project/gsplat)、[Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 和 [LiteGS / SpeedySplat](https://github.com/MooreThreads/LiteGS) 团队在可微高斯光栅化领域的先锋工作。
