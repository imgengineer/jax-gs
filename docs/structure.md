# Project structure and training functions

Public training and rendering calls follow LiteGS. NNX state ownership and
compilation also use patterns from production JAX/Flax NNX projects.
Sources were reviewed on 2026-10-01 and are pinned to commits.

| Project and sources | Pattern used here |
| --- | --- |
| LiteGS: [training entry](https://github.com/MooreThreads/LiteGS/blob/004b95215c90c36cdaf4b354301132b700ac287b/litegs/training/trainer.py), [rendering](https://github.com/MooreThreads/LiteGS/blob/004b95215c90c36cdaf4b354301132b700ac287b/litegs/render/__init__.py) | Call training with model, optimization, pipeline and densification groups; preprocess visible clusters before projection, binning and rendering. |
| openpi: [training loop](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py), [training state](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/utils.py) | Separate model code, optimizer state and host orchestration; bind configuration with `functools.partial` before JIT. |
| Google Tunix: [Gemma3 model](https://github.com/google/tunix/blob/29ee1313e19297fc3dc62a50edbb0df9e96f0263/tunix/models/gemma3/model.py), [SFT trainer](https://github.com/google/tunix/blob/29ee1313e19297fc3dc62a50edbb0df9e96f0263/tunix/sft/peft_trainer.py) | Keep parameters in NNX modules and reuse a compiled step bound to fixed model and optimizer objects. |

openpi uses NNX with Linen bridges for its Gemma and vision backbones. Tunix's
Gemma3 implementation subclasses `nnx.Module` directly. The separate
[DeepMind Gemma transformer](https://github.com/google-deepmind/gemma/blob/c490becf42d18a3337a6ef351c24dfc77896e6d4/gemma/gm/nn/_transformer.py)
uses Linen at the reviewed commit. These implementations have different state
boundaries; the decisions below follow jaxgs's fixed shapes and custom kernels.

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `config/` | Immutable typed settings and TOML loading; capacities determine compiled shapes. |
| `data.py` | Image frames and bounded, threaded decoding with Grain. |
| `io_manager/colmap.py`, `io_manager/checkpoint.py` | COLMAP readers, standard Gaussian PLY models and fixed-capacity NPZ checkpoints. |
| `io_manager/report.py` | Final checkpoint and production training report output. |
| `scene/point.py` | `GaussianModel` owns NNX parameters and occupancy; `GaussianArrays` exposes the same buffers to kernels. |
| `render/types.py`, `scene/types.py` | Shared PyTrees and the parameter ordering used across rendering and optimization. |
| `kernels/`, `reference/` | GPU implementations and numerical correctness implementations. |
| `training/state.py` | Ownership of model buffers, optimizer moments and fragment statistics. |
| `training/step.py` | Pure single-step computation and thin JAX/NNX compilation boundaries. |
| `training/optimizer.py`, `training/muon.py` | Fixed-capacity Adam updates and the separate SH Muon transform. |
| `training/initialization.py` | Training-view selection, sparse-cloud seeding and cluster padding. |
| `training/warmup.py` | Compilation of scheduled variants using disposable copies of training buffers. |
| `training/trainer.py` | CLI and Python entry points, image preload and epoch orchestration. |
| `training/densify.py` | Fixed-capacity growth, pruning and opacity decay at epoch boundaries. |
| `training/reference_trainer.py`, `training/reference_densify.py` | Small-scene reference training and density control for correctness checks. |

Keep dependencies pointed toward shared data types and numerical components.
The single-step computation does not load images, write files or run the epoch
schedule. Reference implementations use the shared types and remain usable on
CPU. GPU bindings are imported when the production step is traced.

`trainer.start` selects views and initializes the pool through `initialization`,
binds a compiled update from `step`, then calls `warmup` before timing the epoch
loop. At completion it delegates checkpoint and report writing to `io_manager`.
These helpers do not import the trainer. The public `training.start`,
`render.render_preprocess`, `render.render` and command-line entry points retain
their existing signatures.

## Expressing training functions

The public training entry uses LiteGS's group ordering:

```python
from jaxgs import config, training

settings = config.load_config()
report = training.start(
    settings.model,  # lp
    settings.optimization,  # op
    settings.pipeline,  # pp
    settings.densify,  # dp
    source_path="/path/to/scene",
    model_path="gaussians.ply",
    runtime=settings.runtime,
)
```

Paths are explicit keyword arguments. `model_path` is the output checkpoint
file. The CLI defaults to `gaussians.ply`; PLY stores active points and loads
into an all-active pool. Explicit `.npz` paths retain capacity and occupancy.
`RuntimeConfig` supplies JAX capacities, optimizer backend and seed; when
omitted, pool capacity follows `dp.target_primitives` and other runtime values
use packaged defaults. The CLI's `train(scene, output, ...)` applies overrides
and delegates to this entry.

Rendering follows LiteGS's preprocessing/render sequence with JAX data types:

```python
from jaxgs import render

pp = settings.capacity
clusters, visible_slots, culled = render.render_preprocess(
    cluster_bounds, camera, model.as_arrays(), pp
)
output = render.render(camera, culled, clusters, degree, pp)
image = output.image
primitive_visible = output.primitive_visible
overflow = output.overflow
```

`Camera` groups view/projection information and image dimensions;
`GaussianArrays` groups parameter arrays. `clusters` contains fixed-size IDs
and a valid count; `visible_slots` is the frustum mask. Preprocessing changes
the alive mask and shares all parameter buffers. `RenderOutput` contains
clamped RGB, the binned primitive mask and an overflow flag. Use
`backend="reference"` on both calls for CPU correctness checks.

CuTe combines SH activation and projection in one kernel. Those operations
remain in `render`; preprocessing performs cluster culling and compaction.
Production training shares preprocessing, then uses its compact parameter
pullback and fused loss kernels. Evaluation uses both public rendering calls.

## Compiled steps

`compute_training_step` takes array PyTrees and returns updated arrays plus
metrics. `array_train_step` compiles that function for correctness comparisons.
The NNX wrappers call the same computation and write results into existing
Variables with `set_value`.

The production loop binds capacity, optimizer and learning-rate settings once:

```python
update = bind_train_step(
    training_state,
    capacity_config,
    max_steps=optimization.position_lr_max_steps,
    optimizer=optimizer,
    optimization=optimization,
)

loss, overflow, peak_pairs = update(
    cluster_bounds,
    camera,
    target,
    step,
    scene_radius,
    active_degree=degree,
    collect_stats=collect_stats,
    overflow=overflow,
    peak_pairs=peak_pairs,
)
```

`active_degree` and `collect_stats` remain static per call because they select
compiled SH and statistics variants. Array values such as step, target and
camera intrinsics remain dynamic. Bind again to select a different run
configuration. Warmup and timed training use the same bound callable.

This combines openpi's configuration partial with fixed NNX state binding.
jaxgs uses Flax 0.12.10's
[`nnx.jit_partial`](https://github.com/google/flax/blob/v0.12.10/flax/nnx/transforms/compilation.py)
in tree mode. The cached Variable references read current values after host
operations such as densification and opacity reset.

## State and verification

The names `TrainingState.adam`, `fragments` and `model` determine tree ordering
and donation pairing. Preserve these names and Variable identities when
changing array contents. Precompilation updates an independent working copy
and restores the original buffers in `finally`.
Warmup deduplicates the SH degree/statistics modes from the same epoch schedule
that training executes. It compiles seven modes for the 169-view bicycle 10k
schedule, instead of all eight combinations; a one-epoch SH0 run needs one.
The [performance sweep](../benchmarks/results/performance_sweep_20261002.md)
records this change and the deferred SH-gradient reconstruction prototype.

The optimizer uses per-slot moments, visibility masks and compact gradients.
Preserve this update contract when changing a training interface. GPU tests
compare array and NNX results for all three optimizer backends, check donated
buffer addresses, and verify that changing scalar values reuses compiled steps.
Mixed-resolution warmup and complete training schedules are tested separately.
The warp-vote optimization passed 394 GPU tests with 100% Python line and branch coverage;
its [validation record](../benchmarks/results/warp_vote_20261002.md)
records the gradient regressions and CUDA checks. At the initial interface
review, the CPU reference and configuration checks passed 33 tests. Interface tests cover
fixed shapes, shared parameter buffers, SH gradients and updates to NNX Variables.
At capacity 128, the interface change preserves StableHLO after normalizing the
module name for all three optimizers, each with SH0 without statistics and SH3
with statistics. Argument, output, alias and temporary memory footprints are
also identical. Per-step array/NNX comparisons use `rtol=2e-5, atol=2e-6`.

The 2026-10-02 module split passed all 56 configuration, production-training and
NNX regression cases below. The moved functions and training loop preserve
their ASTs after normalizing the three renamed helpers; compiled steps and
kernel files are unchanged. CPU imports, the installed CLI and existing Python
entry points were also checked. This validation covers behavior, with no new
performance measurement.

```bash
PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --no-sync pytest -q -p no:cacheprovider \
  tests/test_config.py tests/test_production_training.py tests/test_nnx.py
```
