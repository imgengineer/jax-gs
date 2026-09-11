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

`jax-gs` 是一个基于 JAX 与 Flax NNX 构建的高性能、工业级可微 3D / 2D Gaussian Splatting 研发框架。它将 JAX 的函数式编程、自动微分、JIT 编译和多设备 SPMD 分布式能力，与 NVIDIA cuTile Python GPU 编程模型深度结合。

项目严格对齐 `gsplat 1.5.3@2b902ff` 的数值精度与拓扑几何定义。在单设备和多 GPU 分布式场景下均提供完整的训练、评估、渲染、位姿/外观联合优化与模型导出能力。

---

## ✨ 核心特性

- **严格数学与拓扑对齐**：完整支持 3DGS、2DGS（法线一致性与深度畸变损失）、3DGUT、Eval3D、相机畸变、Rolling Shutter 与 LiDAR 扩展。严格保证 `MAX_ALPHA = 0.999`、饱和透射率截断与精确梯度传播。
- **端到端 cuTile Python 加速流水线**：
  - **cuTile Compositor**：原生 forward/backward、全像素完成后的前向提前终止与 packed Gaussian gradient buffer。
  - **cuTile Projection**：float32 Pinhole 3DGS 离散投影；权威 JAX covariance/determinant 保证 conic 与 topology 精确一致，反向使用权威 JAX 重计算 VJP。
  - **cuTile Topology**：AccuTile preparation/count/emission、饱和前缀和、有效前缀分层 mixed-width radix sorting 与 tile offsets 全部由 cuTile kernel 完成。
- **全功能分布式多卡训练（Distributed Multi-GPU）**：
  - 基于 JAX 原生 SPMD 的 Gaussian-Sharding 模型并行与数据并行混合架构。
  - 支持分布式 3DGS 与 2DGS 训练；纯 JAX 路径保持完整通用性，支持的单卡 Pinhole 3DGS 配置自动选择 cuTile 流水线。
  - 相机位姿优化（`CameraOptModule`）与神经外观优化（`AppearanceOptModule`）支持跨卡 DDP 梯度同步。
  - 支持分布式 Checkpoint 存储、恢复与跨卡数量弹性 Resharding。
- **动态分桶与显存安全机制**：
  - 逻辑容量与物理存储分离；`active_mask` 更新不改变 shape，每个新的存储/交点 bucket 最多编译一次，bucket 切换不会再清空全局 JAX 编译缓存。
  - 具备端到端交点溢出检测（Intersection Overflow Detection）与安全步重放（Overflow Replay），checkpoint 会恢复已调优的交点与 tile-candidate 高水位。
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
- NVIDIA cuTile 路径需要 CUDA 13 GPU；`cuda-tile[tileiras]>=1.6.0` 已作为项目依赖安装，无需运行时 `nvcc`

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

### 2. 验证 NVIDIA cuTile

`uv sync --all-groups` 会安装完整 cuTile 依赖。可用下列命令验证导入：

```bash
uv run python -c "import cuda.tile.jax; print('cuTile ready')"
```

> **注意**：cuTile kernel 通过 JAX bridge 编译，并使用项目的持久化 JAX compilation cache；cuTile 是唯一的原生 GPU kernel 后端，无需运行时 `nvcc`。纯 JAX 参考路径仍用于跨平台运行以及 cuTile 尚未覆盖的 2DGS、UT 和分布式配置。
>
> **运行环境约定**：项目不覆盖 `--xla_gpu_force_compilation_parallelism`，也不强制设置 OpenMP、BLAS 或构建线程数。编译与主机并行度遵循工具默认值和用户环境。

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

print("Render Output Shape:", renders.shape)  # [C, H, W, 3]
print("Alpha Output Shape:", alphas.shape)  # [C, H, W, 1]

# 2. 显式启用端到端 NVIDIA cuTile 流水线
config_native = RasterizationConfig(
    backend="intersections",
    projection_backend="cuda_tile",
    compositor_backend="cuda_tile",
    intersection_backend="cuda_tile",
    sort_backend="cuda_tile",
    intersection_mode="accutile",
    tile_size=16,
    max_intersections=524_288,
    max_candidates_per_tile=2048,
)

renders_native, alphas_native, _ = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width=640,
    height=360,
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
# 单卡标准训练（支持的环境下会自动启用端到端 cuTile 加速）
uv run jax-gs train \
  --config scene_config.json \
  --output outputs/scene_run

# 单机多卡分布式训练（选择 GPU 0、1）
CUDA_VISIBLE_DEVICES=0,1 uv run jax-gs train \
  --config scene_config.json \
  --output outputs/scene_dist \
  --distributed

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

# 显式启用完整 cuTile 流水线
uv run jax-gs train \
  --config scene_config.json \
  --projection-backend cuda_tile \
  --compositor-backend cuda_tile \
  --intersection-backend cuda_tile \
  --sort-backend cuda_tile

# 显存足够时固定全部训练 buffer shape，避免稠密化期间产生新 JIT
# capacity、intersection 和 candidates 三项都必须覆盖预跑高水位；
# candidate 超界同样会扩大静态 plan 并触发一次新 JIT。
uv run jax-gs train \
  --config scene_config.json \
  --capacity 1000000 \
  --bucket-min-capacity 1000000 \
  --max-intersections 4194304 \
  --intersection-bucket-min-capacity 4194304 \
  --max-candidates-per-tile 2048

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
| **Projection** | `jax` | 权威纯 JAX 稠密投影 | 所有相机模型、分布式与通用参考路径 |
| | `cuda_tile` | JAX 权威 covariance/determinant + cuTile 离散 Pinhole projection；JAX 重计算 VJP | 单卡 float32 Classic Pinhole 3DGS |
| **Compositor** | `jax` | 纯 JAX 实现，具备完整通用性与跨平台性 | CPU / TPU / 通用 GPU / 算法原型验证 |
| | `cuda_tile` | **cuTile forward/backward** 与 packed gradients | **推荐单卡生产训练与推理** |
| **Intersections** | `auto` / `jax` | 纯 JAX 拓扑与排序流水线 | 跨平台通用 |
| | `cuda_tile` | **cuTile AccuTile + saturated scan + stable radix + tile offsets** | **推荐单卡完整拓扑流水线** |
| **Sort** | `jax` | JAX 稳定排序参考实现 | 跨平台通用 |
| | `cuda_tile` | cuTile 稳定 LSD radix sort | cuTile 拓扑流水线 |

### CLI 原生加速自动路由规则

为降低用户配置成本，CLI 入口在检测到支持环境时会自动路由至完整 cuTile 组合（投影、合成、相交与排序后端均为 `"cuda_tile"`，相交模式为 `"accutile"`）：

1. Pinhole 相机模型 3DGS（无 UT、Eval3D、AbsGrad、Sparse Grad、Appearance Optimization）。
2. 单 GPU、非 distributed 训练。
3. NVIDIA GPU Compute Capability $\ge 8.0$ 且 `cuda.tile.jax` 可导入。
4. `tile_size=16`，使用 float32 Classic rasterization contract。

不满足条件时保留权威 JAX 路径；显式传入 backend 参数时 CLI 不覆盖用户选择。

---

## 🌐 分布式多卡架构

`jax-gs` 的分布式训练设计基于 JAX 原生 SPMD 理念，专为大规模高斯场景设计：

```
[Host Dataset (Grain)]
         │ (Shard Camera Batches)
         ├─── Rank 0: Camera Batch 0 ───┐
         └─── Rank 1: Camera Batch 1 ───┤
                                        ▼
             [Geometry Shards] ───────────► [All-Gather] ───► [Global Geometry]
             [Eligible Shared SH] ─► [Owner SH Evaluation] ─► [All-to-All Features]
                                                                  │
                                            ┌─────────────────────┴─────────────────────┐
                                            ▼                                           ▼
                               [Rank 0: Local Rasterize (JAX)]             [Rank 1: Local Rasterize (JAX)]
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

- **分片存储与优化交换**：高斯模型与优化器状态保持分片；当共享 SH 的基函数数量大于本地相机数时，所属 Rank 按目标相机求值并以固定槽位 `all_to_all` 交换直接特征，否则保留全量聚合兼容路径。Degree-3、每 Rank 单相机时，SH 通信载荷由每高斯 48 个 float 降至 3 个 float；投影几何仍会全量聚合。
- **严格投影拓扑**：均值、四元数、尺度与不透明度仍以 Rank-major 顺序聚合，避免分片形状改变 JAX conic 舍入与 AccuTile 成员关系。待投影算术具备形状无关性后，再替换为完整的可见投影 primitive 交换。
- **独立优化与同步控制**：各卡独立执行局部 Densification / Pruning，溢出状态通过全局规约原子化同步，保障多卡模型结构绝对一致。

---

## 📊 性能评测

旧原生后端的性能数据不适用于当前 cuTile 实现，因此不再展示。重新建立基线时应显式记录场景、GPU、软件版本、容量、candidate bound、warmup/hot iterations 与同步方式；不同协议的绝对时间不可直接横向比较。

### 运行性能基准与自动调优工具

```bash
# 运行完整光栅化前后向 Benchmark
uv run python benchmarks/benchmark_rasterization.py \
  --npz /path/to/garden.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --backend intersections \
  --projection-backend cuda_tile \
  --compositor-backend cuda_tile \
  --intersection-backend cuda_tile \
  --sort-backend cuda_tile \
  --intersection-mode accutile \
  --max-intersections 524288 \
  --max-candidates-per-tile 2048 \
  --backward \
  --allow-unsafe

# cuTile 块大小与占用率自动调优
uv run python benchmarks/autotune_cutile.py \
  --npz /path/to/garden.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --max-intersections 524288
```

---

## 🧪 测试套件

项目包含完整的单元测试与端到端回归测试，确保任意代码修改均不破坏数学对齐：

```bash
# 运行默认测试集（排除超长集成测试）
uv run pytest

# 运行 cuTile 投影、拓扑、合成与环境契约回归
uv run pytest tests/test_cutile_projection.py tests/test_cutile_intersections.py tests/test_cutile_compositor.py tests/test_environment.py

# 运行分布式与多卡专属测试
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/test_distributed.py tests/test_training_distributed.py

# 运行高耗能/重显存独立测试
uv run pytest -m resource_heavy
```

---

## 🗂️ 文档索引

- [`README.md`](README.md)：安装、API、后端、性能与测试入口。
- [`sandbox/perf_results.md`](sandbox/perf_results.md)：按统一字段记录的性能实验台账。
- [`NOTICE.md`](NOTICE.md)：第三方组件、许可与归属。

---

## 📄 许可协议与致谢

- 本项目基于 [Apache-2.0 License](LICENSE) 开源。
- 第三方算子架构与开源归属详见 [`NOTICE.md`](NOTICE.md)。
- 感谢 [gsplat](https://github.com/nerfstudio-project/gsplat)、[Inria 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) 和 [LiteGS / SpeedySplat](https://github.com/MooreThreads/LiteGS) 团队在可微高斯光栅化领域的先锋工作。
