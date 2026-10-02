# jaxgs

**3D Gaussian Splatting in JAX, with Flax NNX models and CuTe GPU kernels.**

jaxgs follows [LiteGS](https://github.com/MooreThreads/LiteGS)'s modular scene,
rendering and training pipeline using JAX arrays and NNX state. Production
training combines cluster culling, compact projection, tile binning, packed
rasterization and sparse parameter updates. A JAX reference path supports
small-scene correctness checks on CPU.

## Training performance

Measured on an RTX 5090 with the default 30k schedule, Optax, 8×16 tiles and a
one-million-slot pool. The bicycle experiment uses `images_4` at 1237×822,
169 training views and 25 held-out views.

| Scene | Actual updates | Final Gaussians | Training loop | Held-out PSNR |
| --- | ---: | ---: | ---: | ---: |
| bicycle | 29,913 | 993,152 | 41.28 s | 25.433 dB |

Time is the median of three runs; PSNR is the mean of the three held-out
scores. Timing includes densification, pruning, opacity decay and spatial
refinement, and excludes initialization, image preload, compilation and
checkpoint writing. These measurements cover one scene and GPU.

Contribution pruning reduced the matched JAX baseline from 43.28 to 41.28 s
(4.6% less time), with mean PSNR changing by −0.043 dB. The
[measurement and validation record](benchmarks/results/contribution_pruning_20261002.md)
contains all samples, configuration, quality differences and GPU checks.

## Features

- **Modular rendering:** cluster culling and compaction, projection, visibility
  tables and rasterization have separate Python interfaces.
- **NNX model state:** Gaussian parameters, occupancy, Adam moments and fragment
  statistics retain fixed shapes through growth and pruning. Compiled training
  donates parameter and optimizer buffers for reuse.
- **CuTe kernels:** packed half2 rasterization, contribution pruning in backward,
  fused uint8 L1+SSIM loss and compact projection pullbacks.
- **Optimizer choices:** Optax Adam by default; Muon for SH colors and a CuTe
  sparse Adam implementation are also available.
- **Reference implementation:** JAX rendering and a small-scene trainer for
  numerical comparisons and CPU debugging.

## Getting started

### Installation

Production training requires Linux, Python 3.12+, an NVIDIA GPU and a driver
compatible with CUDA 13. Dependency versions are recorded in [uv.lock](uv.lock).
The tested environment uses JAX 0.11.2, Flax 0.12.10 and CUTLASS DSL 4.8.0.

```bash
git clone https://github.com/imgengineer/jax-gs.git
cd jax-gs
uv sync --extra cute
```

The `cute` extra installs NVIDIA CUTLASS DSL. GPU kernels are written in Python
and compiled by CuTe; no separate C++ extension build is needed.

### Dataset

Use an undistorted COLMAP reconstruction with `PINHOLE` or `SIMPLE_PINHOLE`
cameras. Binary and text reconstructions are supported:

```text
scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin       # or cameras.txt
        ├── images.bin        # or images.txt
        └── points3D.bin      # or points3D.txt
```

Reconstructions directly under `sparse/` are also accepted. Use `--images`
to select an existing image folder such as `images_4`; camera intrinsics are
rescaled to its image dimensions. The default resolution caps image width at
1600 pixels.

## Train

```bash
uv run --extra cute jaxgs-train /path/to/scene --output output/model.npz
```

The trainer writes the Gaussian pool to `output/model.npz` and its resolved
configuration, epoch history and timing to `output/model.json`. Checkpoints
include parameters and occupancy in the project's NPZ format.

For bicycle `images_4` with held-out evaluation, create `experiment.toml`:

```toml
[model]
images = "images_4"
eval = true
```

```bash
uv run --extra cute jaxgs-train /path/to/bicycle \
  --config experiment.toml --output output/bicycle.npz
```

Evaluation mode uses `train_test_split.json` when present; otherwise every
eighth image is held out. The default `eval=false` trains on all registered
views. Requested iterations are rounded down to complete epochs.

### Configuration

TOML files override [the packaged defaults](src/jaxgs/config/default.toml).
CLI options override TOML values; run `jaxgs-train --help` for available flags.

| Setting | Default |
| --- | --- |
| Iterations / position LR schedule | 30,000 / 30,000 |
| Images / resolution / evaluation split | `images` / width capped at 1600 / disabled |
| Optimizer | Optax Adam |
| SH degree / cluster size / tile height × width | 3 / 128 / 8 × 16 |
| Gaussian capacity / growth target | 1,000,000 / 1,000,000 |
| Visibility pair capacity | 8,000,000 |
| Densify from / interval / opacity reset | epoch 3 / 5 / 10 |

Densification intervals use epochs. Changing `iterations` leaves the position
LR schedule length unchanged unless `position_lr_max_steps` is also overridden.
The model, optimization, pipeline and densification defaults follow LiteGS;
`runtime` contains JAX capacity, optimizer and seed settings.

Use `--optimizer muon` for SH Muon or `--optimizer cute` for CuTe sparse Adam.
The optional [8×8 bicycle configuration](benchmarks/configs/bicycle_8x8.toml)
is documented in the [tile analysis](benchmarks/results/algorithm_analysis_20261002.md),
including its pair-storage cost and measured quality difference.

### Python entry point

Training uses LiteGS's argument-group order with the project's typed JAX/NNX
configuration:

```python
from jaxgs import config, training

settings = config.load_config("experiment.toml")
report = training.start(
    settings.model,
    settings.optimization,
    settings.pipeline,
    settings.densify,
    source_path="/path/to/scene",
    model_path="output/model.npz",
    runtime=settings.runtime,
)
```

`model_path` names the checkpoint file. When `runtime` is omitted, Gaussian
capacity follows the densification group's growth target.

## Render

Load a checkpoint and call preprocessing before rendering. Use the same
capacity and SH settings as training:

```python
from flax import nnx

from jaxgs import GaussianModel, config, render
from jaxgs.io_manager.checkpoint import load_gaussians
from jaxgs.io_manager.colmap import load_colmap_images
from jaxgs.scene.cluster import world_cluster_bounds

settings = config.load_config("experiment.toml")
pp = settings.capacity
model = GaussianModel(load_gaussians("output/model.npz"))
bounds = world_cluster_bounds(model.as_arrays(), pp.cluster_size)
frames = load_colmap_images(
    "/path/to/scene", settings.model.images, resolution=settings.model.resolution
)


@nnx.jit(graph=False)
def render_view(model, camera):
    clusters, _, culled = render.render_preprocess(
        bounds, camera, model.as_arrays(), pp
    )
    return render.render(camera, culled, clusters, pp.sh_degree, pp)


output = render_view(model, frames[0].camera)
image = output.image
```

`RenderOutput` contains HWC float32 RGB in `[0, 1]`, a primitive visibility mask
and an overflow flag. Visibility overflow requires a larger pair capacity.
For CPU correctness comparisons on small pools, pass `backend="reference"`
to both rendering calls.

## Modular pipeline

1. **Cluster culling and compaction:** use world-space bounds to identify
   visible clusters and compact their IDs.
2. **Projection:** transform visible Gaussians, evaluate SH colors and project
   their covariance into screen space.
3. **Visibility table:** emit Gaussian/tile pairs and sort them in depth order.
4. **Rasterization and loss:** composite tile pixels and compute L1+SSIM with
   gradients for RGB training.
5. **Update and density control:** update visible slots, accumulate fragment
   statistics, then grow, prune and reorder at epoch boundaries.

```text
src/jaxgs/
├── config/          # Typed settings and fixed capacities
├── scene/           # NNX Gaussian model, cameras and spatial organization
├── render/          # Preprocessing, rendering and shared PyTrees
├── kernels/         # CuTe kernels and JAX bindings
├── training/        # Compiled steps, optimizers and epoch orchestration
├── reference/       # JAX correctness implementations
└── io_manager/      # COLMAP readers and NPZ checkpoints
```

`GaussianModel(nnx.Module)` owns trainable `nnx.Param` values and occupancy
Variables; `model.as_arrays()` exposes the same buffers as `GaussianArrays`.
Production binds the model, optimizer and statistics once with
`nnx.jit_partial(graph=False)`. Details and LiteGS/openpi/Tunix source references
are in the [project structure guide](docs/structure.md).

## Evaluation and benchmarks

Evaluate every eighth COLMAP view using the production renderer:

```bash
uv run --extra cute python benchmarks/evaluate_protocol.py \
  /path/to/bicycle output/bicycle.npz --backend jaxgs \
  --images images_4 --output output/quality.json
```

This evaluator uses the every-eighth split; it does not read custom
`train_test_split.json` files. Reports contain per-view PSNR and its mean.

[benchmarks/fixed_model.py](benchmarks/fixed_model.py) compares JAX and LiteGS
updates from a common Gaussian PLY without densification.
[benchmarks/packed_parity.py](benchmarks/packed_parity.py) checks native kernel
parity. LiteGS comparisons require a separate installed LiteGS checkout.

Additional records cover [loss target loading](benchmarks/results/loss_target_20261001.md),
[kernel arithmetic](benchmarks/results/kernel_arithmetic_20261001.md) and
[sequential scan](benchmarks/results/scan_training_20261002.md).
The scan experiment reduced loop time by 2.2% with `unroll=4`, but preload,
warmup and training combined took 6.2% longer; the default loop is retained.

## Development and verification

```bash
uv sync --extra cute --group dev
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute pytest -q \
  --cov=jaxgs --cov-report=term-missing
```

The verified GPU suite passes **390 tests** with **100% Python line and branch
coverage**: 2,001 statements and 292 branches. GPU DSL bodies are excluded from
Python coverage; parity, gradient, training and CUDA device checks verify their
behavior. The [contribution pruning record](benchmarks/results/contribution_pruning_20261002.md)
also documents the CUDA checker options and diagnostic limitations.

The small-scene reference trainer is available for CPU debugging:

```bash
JAX_PLATFORMS=cpu uv run jaxgs-train-reference /path/to/small-scene \
  --backend reference --steps 100 --capacity 1024 --output output/reference.npz
```

## Source attribution

The packed rasterizer, sparse Adam and AccuTile binning translate
[LiteGS](https://github.com/MooreThreads/LiteGS) code; its
[license is retained](src/jaxgs/kernels/LICENSE.LiteGS).
AccuTile includes work from speedy-splat and gaussian-splatting.
The fused L1+SSIM kernels retain the original
[MIT notice](src/jaxgs/kernels/LICENSE.fused_ssim).
