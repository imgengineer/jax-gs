# jaxgs

Fixed-capacity Gaussian splatting with Flax NNX models and CuTe DSL GPU kernels. The
Gaussian model, Adam moments, tile table and densification outputs keep static
shapes when the number of live Gaussians changes.

## Install

```bash
uv sync --extra cute
```

The default dependency set includes JAX with CUDA 13 support, Flax, Optax, Chex, Grain and Pillow.
The `cute` extra adds NVIDIA CUTLASS DSL. A CUDA GPU is required for the CuTe
backend; the reference path also works on CPU.

## Layout and model state

Scene, rendering and training responsibilities follow LiteGS. GPU operations
live in the project's own `kernels` Python package:

```text
src/jaxgs/
├── config/                     # default.toml, typed training settings, static capacities
├── data.py                     # image frames and Grain input pipeline
├── io_manager/                  # COLMAP readers and shared NPZ checkpoints
├── scene/
│   ├── point.py                # GaussianModel (NNX) and GaussianArrays array view
│   ├── types.py                # parameter order and fixed-shape cluster layouts
│   ├── camera.py
│   ├── cluster.py
│   └── spatial_refine.py
├── render/
│   ├── types.py                # shared projections, tile tables and render results
│   ├── rasterizer.py           # differentiable and forward render interfaces
│   ├── projection.py
│   └── visibility_table.py
├── training/
│   ├── trainer.py              # configuration, dataset setup and production loop
│   ├── step.py                 # compiled array step and bound NNX training step
│   ├── state.py                # NNX model, Optax state and fragment statistics
│   ├── optimizer.py            # fixed-capacity Optax Adam; CuTe comparison
│   ├── densify.py              # growth and pruning
│   ├── pool_ops.py             # slot pruning shared with the reference path
│   ├── reference_trainer.py
│   └── reference_densify.py
├── kernels/                    # CuTe kernels, launchers and JAX bindings
└── reference/                  # JAX correctness implementations
```

`GaussianModel(nnx.Module)` owns `xyz`, `log_scale`, `rotation`, `opacity` and
`sh` as `nnx.Param`; occupancy masks and the active count are `nnx.Variable`.
`nnx.state(model, nnx.Param)` selects trainable parameters, and `nnx.split` /
`nnx.merge` work normally. `model.as_arrays()` provides a zero-copy array view
of type `GaussianArrays` for kernels and custom VJPs. `create_gaussians` and
`seed_gaussians` initialize these arrays; `GaussianModel(arrays)` owns them in
NNX. `model.update_from_arrays(arrays)` updates the existing Variables. The
existing NPZ checkpoint format is unchanged.

Both CLI trainers update NNX models through NNX JIT transforms. Production
binds a fixed `TrainingState` once with `nnx.jit_partial(graph=False)`, caching
the flattening of its model, Optax state and statistics. The bound step donates
these buffers. Densification changes array
contents and occupancy without changing the model structure. Optax keeps the existing
update rule and fixed-capacity moments. The pure array
`array_train_step` remains available for correctness comparisons.

`TrainingState` stores Adam in `nnx.OptState` and statistics in `nnx.Variable`;
their pytree order precedes the model so donation reuses the parameter/moment
buffers. Warmup uses one working copy, then restores the original arrays on
the same bound Variables before timed training. Tests verify buffer addresses,
parameter and optimizer parity, NNX state round-trips and compiled step reuse.
Eager Adam initialization creates independent m/v buffers for direct donation.
Production retains shared initial zeros through warmup and separates v before
training. Warmup covers each distinct camera image size and clipping range.
The [Flax binding review](benchmarks/results/flax_partial_review_20260930.md)
records the documentation, CPU dispatch probe and million-capacity resource checks.

Use `jaxgs-train` for production training and `jaxgs-train-reference` for the
small-scene correctness path. Production training defaults to Optax. Source
project names remain in attribution and external comparison scripts.

Renderer data types are defined in `render.types`; production CuTe bindings
and the reference implementation use the same PyTrees. The bounded diagnostic
render interfaces are `render.rasterizer.rasterize` and `rasterize_forward`.
Production training uses `kernels.projector.project_with_compact_pullback`
and the packed RGB kernels. Its gradient tuple follows `scene.types.PARAMETER_NAMES`:
`xyz`, `log_scale`, `rotation`, `opacity`, `sh`. Compact buffers contain a valid
visible-cluster prefix; their unused tails must not be read.

The four fragment-statistics columns are fragment count, compositing weight,
summed alpha gradient and summed squared alpha gradient. `reset_adam_slots`
accepts a boolean slot mask and clears only those moments and update counts.
The production trainer separates pool initialization, precompilation and report
writing from the epoch loop. The
[structure validation record](benchmarks/results/structure_naming_validation_20260930.json)
compares outputs, compiled operations and memory against the preceding implementation.

The JAX and NNX steps share the pure `training.step.compute_training_step` function.
Precompilation reuses one independent donated working state across SH/statistics
variants. `Camera.from_colmap` normalizes all intrinsic scalars to `float32`,
so their original Python/NumPy numeric types do not create extra JIT signatures.
The [Flax/JAX documentation review](benchmarks/results/flax_jax_review_20260930.md)
records the versioned sources, code decisions and regression checks.

### Default training configuration

[config/default.toml](src/jaxgs/config/default.toml) is the source of production
defaults. Its model, pipeline, optimization and densification sections match
LiteGS's `get_default_arg()` / `litegs/arguments.py` at commit
`004b95215c90c36cdaf4b354301132b700ac287b`. Scene and output paths are CLI
arguments. The `runtime` section contains JAX-specific capacity, optimizer
backend and seed settings.

| Setting | Default |
| --- | --- |
| Images / resolution / evaluation split | `images` / `-1` (width capped at 1600) / disabled |
| Iterations / position LR schedule length | 30,000 / 30,000 |
| Position LR, before scene-radius scaling | 0.00016 → 0.0000016 |
| SH DC / higher-order SH LR | 0.0025 / 0.00025 |
| Opacity / scale / rotation LR | 0.025 / 0.005 / 0.001 |
| Cluster size / tile height × width / SH degree | 128 / 8 × 16 / 3 |
| Densify from / interval / opacity reset interval | epoch 3 / 5 / 10 |
| Densify until | `floor(epochs * 0.8 / 10) * 10 + 1` |
| Pruning / opacity reset / growth target | weight / decay / 1,000,000 |

Iterations are rounded down to complete epochs, as in the source trainer.
Changing `iterations` does not silently change `position_lr_max_steps`.
Grain uses the native bicubic resize convention for production images.
With `model.eval=true`, an explicit `train_test_split.json` takes precedence;
otherwise every eighth image is held out. With the default `eval=false`, all
registered images train the model.

Run the defaults, or supply a TOML file containing only overrides:

```bash
uv run --extra cute jaxgs-train /path/to/scene --output model.npz
uv run --extra cute jaxgs-train /path/to/scene --config experiment.toml --output model.npz
```

Explicit CLI options override TOML values. Each training report records the
resolved configuration. Source switches outside the implemented RGB training
path (for example depth output, learned cameras, dense Adam, other pruning
modes or a different fused-loss weight) reject unsupported values before
compilation. The three deprecated threshold fields are retained for source
parity and are unused by the weight-based controller, just as upstream.

The previous 10k bicycle experiment is preserved separately in
[benchmarks/configs/bicycle_10k.toml](benchmarks/configs/bicycle_10k.toml).
It uses `images_4`, an evaluation split, densification every two epochs and
position LR ending at 0.000016 over 10,000 steps. Historical timing results
below describe that experiment, not the restored 30k defaults.

### Code checks

Dependencies are managed with `uv add` (`uv add --dev` for development tools).
Ruff checks imports and Python errors, and formats source, tests and benchmarks:

```bash
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
```

Array annotations use `chex.Array`. Model construction checks fixed pool shapes
and occupancy mask types; camera construction checks COLMAP extrinsic shapes.
The Optax transformation checks gradient/moment shapes and dtypes, and the active
mask layout. These are [Chex static assertions](https://chex.readthedocs.io/en/latest/api.html),
executed during construction or JIT tracing. Value assertions that synchronize
device arrays are confined to tests. Both dense and compact-gradient Optax updates
produce identical StableHLO before and after these checks. Tests also verify one
trace across repeated updates, no assertion callbacks, and NNX buffer donation.

### Tests and coverage

Install test dependencies with `uv sync --extra cute --group dev`, then run
the GPU suite:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute pytest -q \
  --cov=jaxgs --cov-report=term-missing --cov-report=html
```

The GPU suite passes **279 tests**, including 15 Chex contract cases. Python line
and branch coverage are both **100%** (1,694 statements and 250 branch outcomes).
The report enforces a 99% combined threshold. The exclusion policy is unchanged:
only bodies decorated with `@cute.kernel`, `@cute.jit`, or `@dsl_user_op` are
excluded, because they compile to GPU code. Python bindings and launch setup
remain covered; GPU arithmetic is checked by render parity, finite differences,
individual cotangents, full training and NNX buffer-donation tests.

New tests cover native default parity, configuration overrides and rejection,
per-property learning rates and schedule endpoints, COLMAP failure cases,
image pyramids and native resizing, evaluation splits, complete training with
densification/reset, overflow failure and checkpoint round-trips. The native
source parity test uses a sibling `LiteGS` checkout or `LITEGS_ROOT`; it skips
that comparison when neither is available. Other tests remain self-contained.

The latest tests also execute both module CLIs, train through the step-3000
opacity reset, verify failure before saving on overflow or non-finite loss,
check CPU/CuTe device boundaries and invalid backend/SH/tile/image arguments,
and compare custom-VJP inference and 12x16-tile gradients against the reference.
Open `htmlcov/index.html` after the command above for line/branch details, or see
the [Chex validation record](benchmarks/results/chex_style_profile_validation.json).
The earlier [coverage expansion record](benchmarks/results/coverage_expansion.json)
documents the increase from 96.94% line / 86.61% branch coverage to 100%.

A fresh shared-GPU profile still identifies backward rasterization and the
Optax SH update as the largest kernels. Three gradient-gather alternatives had
no stable gain in interleaved trials, so the optimizer implementation is unchanged.
The Chex validation record includes those exploratory measurements; uncontended
timing is deferred until the other GPU training process finishes.

Configuration loading and CLI help also work without importing CuTe. Both
trainers and evaluation use `io_manager.checkpoint`; `training.step` owns
compiled numerical work while `training.trainer` owns the host loop. Existing
imports of training steps from `trainer` continue to work.

Performance stayed unchanged in matched Optax measurements on RTX 5090:
three interleaved 975,104-point trials gave **3.720 / 3.721 ms per step**
(before / after); full 9,971-update training took **30.107 / 30.107 s**.
Both full runs ended at 975,104 active points with eight compiled variants
and no overflow. These comparisons use the explicit 10k benchmark settings
on both versions, so restoring the defaults does not change the comparison.
Full-run timings are one trial per version. See the
[validation record](benchmarks/results/structure_config_validation.json) for
coverage counts, fixed-count repetitions, full-training reports and finite
parameter/occupancy checks.

### Optax optimizer

The production trainer and fixed-count benchmark use Optax by default.
`--optimizer cute` remains available for controlled performance comparisons.
Both optimizers update the same NNX parameters under `nnx.jit` and preserve
fixed-capacity moments, slot resets, spatial reordering and buffer donation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute jaxgs-train \
  /path/to/bicycle --optimizer optax --output bicycle_optax.npz
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/fixed_model.py /path/to/bicycle /path/to/final.ply \
  --backend jaxgs --optimizer optax --output fixed_optax.json
```

`training.optimizer.create_adam_transform()` is an Optax `GradientTransformationExtraArgs`
using `optax.tree` moment updates and `optax.apply_updates`. It keeps LiteGS's
Adam rule: beta1=0.9, beta2=0.999, epsilon=1e-15, no bias correction, the same
per-field learning rates, and frozen parameters/moments in invisible or free
slots. This is a custom Optax transformation, because
[Optax's built-in Adam](https://optax.readthedocs.io/en/latest/api/generated/optax.scale_by_adam.html)
applies bias correction. Compact gradients are mapped back to pool slots
with masked gathers; undefined gradient capacity is never consumed.

On the RTX 5090 bicycle fixed-count benchmark (975,104 Gaussians), three
interleaved trials measured medians of **3.444 ms/update for CuTe Adam** and
**3.770 ms/update for Optax**: Optax took 9.5% longer. Both use the same
one-warp backward rasterizer and compile the timed step once. These measurements preceded making Optax the default.
The CuTe path remains available with `--optimizer cute`. See the
[comparison](benchmarks/results/bicycle_fixed_optax_comparison.json) and
[GPU traces](benchmarks/results/bicycle_optax_profile.json). The Optax XLA
update kernels traverse the full pool; CuTe updates visible clusters directly.
The traced update kernels took about 0.989 ms and 0.688 ms, respectively.
In a full 9,971-update growth-training run, Optax took 30.37 s versus CuTe's
23.97 s. See the full comparison below. All 60 GPU tests passed, including
changing visibility, slot reuse, poisoned unused gradients and NNX donation.

### Current pipeline review

With Optax in both versions, three interleaved fixed-count trials measured
**3.782 ms/update before and 3.704 ms/update after** skipping projection
pullback arithmetic for splats whose incoming gradients are all zero (2.1%
shorter). Zero gradient outputs are still initialized, and Optax still decays
momentum for visible slots. Depth-only and radius-only gradients are covered
by a separate regression check. See the
[comparison](benchmarks/results/bicycle_optax_default_comparison.json).

The current ten-step GPU trace identifies these main costs:

| Stage | GPU ms/update |
| --- | ---: |
| Rasterization backward | 1.156 |
| Optax parameter/moment updates | 0.990 |
| Rasterization forward | 0.490 |
| Pair counting and emission | 0.237 |
| Projection backward | 0.193 |
| Projection forward | 0.152 |

Projection backward previously took 0.252 ms in the matched trace. Stage
times are single traces; use the interleaved whole-step results for speed
comparisons. The remaining largest opportunities are rasterization backward
and Optax's full-capacity SH/moment traffic. Forward rasterization and pair
emission are the next targets. These are priorities for further work, not
promised speedups.

Two approaches were measured and reverted: static-prefix Optax branches
were slower despite preserved buffer donation; distributing raster gradient
atomics over warp lanes also slowed the complete step. The
[profile and experiment record](benchmarks/results/bicycle_optax_default_profile.json)
contains the measurements. Pool capacity, gradient rules and rasterization
arithmetic remain unchanged.

The [2026-09-30 source and algorithm review](benchmarks/results/static_performance_audit_20260930.md)
records CPU checks and small CuTe/native correctness checks while the GPU is
shared. It identifies a screen-edge culling mismatch, a stable-partition
replacement for densification's second sort, and remaining full-capacity
SH/pair work. The screen-edge mismatch has since been fixed: projection uses
LiteGS's coarse center bounds, and reference/diagnostic renderers preserve
contributions beyond 3 sigma. The [edge validation record](benchmarks/results/edge_support_validation_20260930.json)
compares four borders, isotropic/rotated anisotropic Gaussians, native pair
tables, RGB and gradients. GPU performance will be remeasured when it is available
exclusively; the other review items remain optimization candidates.

A fresh full-training comparison with Optax measured **30.459 s before and
30.097 s after** (1.2% shorter) for 9,971 updates. Both finished with 975,104
active Gaussians in a one-million-slot pool and eight compiled variants,
without overflow or non-finite parameters. Mean PSNR over 25 held-out views
was 25.062 / 25.089 dB. These are single full runs with slightly different
floating-point growth trajectories. The
[full comparison](benchmarks/results/bicycle_optax_default_full_comparison.json)
and [validation record](benchmarks/results/bicycle_optax_default_validation.json)
include checkpoint checks and the 61 GPU test cases (the new fixture was
corrected and retested separately).

### Load-balanced rasterization and binning

Heavy tiles, not total work, set the packed rasterizer's time: on the fixed
bicycle view the single heaviest tile (2,732 pairs) alone took 294 µs of a
595 µs forward pass. Tiles now launch heaviest first from a one-block counting
sort, and each single-warp block stages 32 splats at a time in shared memory
while its lanes load the next batch (see [Pipeline](#pipeline)). Pair emission
walks Gaussians in depth order and hands large Gaussians to a whole warp.
Forward outputs are bit-identical to the previous kernels, as are the sorted
pair tables; the counting and emitting kernels now share explicitly rounded
ellipse slices, which also fixed one Gaussian whose previous count exceeded its
emitted pairs. LiteGS parity is unchanged.

Exclusive RTX 5090 measurements with the protocol of the
[exclusive comparison](benchmarks/results/exclusive_performance_20260930.md)
(1,000,064 points, 8M pairs; 30k default training with an evaluation split):

| Measurement | Before | After |
| --- | ---: | ---: |
| Fixed-count update, Optax (median of 3) | 3.851 ms | **2.980 ms** |
| Fixed-count update, CuTe Adam (median of 3) | 3.514 ms | **2.652 ms** |
| Forward kernels / backward kernels | 616 / 1,227 µs | 272 / 872 µs |
| Pair emission | 240 µs | 100 µs |
| 30k training, Optax | 89.43 s | **74.48 / 74.06 s** |
| 30k training, CuTe Adam | 71.05 s | **55.41 s** |

LiteGS recorded 3.837 ms per fixed-count update under the same protocol.
Held-out PSNR was 25.428 dB before and 25.465 / 25.444 dB after with Optax,
within the 25.426–25.510 dB spread of earlier runs. The
[kernel balance record](benchmarks/results/kernel_balance_20260930.md) contains
the analysis, rejected variants, sanitizer checks and raw timings. The dense
Optax update (about 1 ms per update) is now the largest remaining cost.

### CuTe integration

The call structure follows NVIDIA's
[CuTe/JAX examples](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/dsl_tutorials/jax)
and the [JAX CuTe guide](https://docs.jax.dev/en/latest/401/cute-dsl.html).
`kernels/` contains local Python modules for projection, rasterization, loss,
sparse Adam and slot allocation. CUTLASS DSL itself is supplied by the `cute`
dependency extra.

| Layer | API | Example in jaxgs |
| --- | --- | --- |
| GPU computation | `@cute.kernel` | `kernels/projection.py`, `kernels/packed_rasterize.py` |
| Kernel launch (grid, block, stream) | `@cute.jit` | `launch_projection`, `launch_forward` |
| JAX/XLA binding | `cutlass.jax.cutlass_call` | `kernels/projector.py`, `kernels/packed_rasterizer.py` |
| Training and gradients | `nnx.jit`, custom VJP / explicit pullback | `training/trainer.py` and the JAX bindings |

`cutlass_call` compiles the launcher during JAX lowering and lets XLA invoke
it on its CUDA stream. Bindings declare output shapes/dtypes and use static
tensor layouts; sparse Adam also declares input/output aliases for donation.
The directory move preserves these bindings, GPU calculations and launch
parameters. Source licenses remain alongside the translated kernels.

The [JAX/CuTe performance review](benchmarks/results/cute_dsl_review_20260930.md)
records RGB pullback specialization, empty-cluster CTA handling, release kernel
resource checks and GPU correctness tests. Training timing awaits an exclusive GPU.

The [layout migration checks](benchmarks/results/kernels_layout_migration.json)
passed all 54 tests on the GPU and verified unchanged calculation bodies in
all 20 kernel/binding modules. Three interleaved fixed-count trials on the same
975,104-point bicycle model measured 3.566 ms before and 3.548 ms after the
move, with one compiled step per trial. This shows no observed regression;
the small timing difference is not attributed to a kernel optimization.

## Train

Input scenes are undistorted COLMAP text or binary reconstructions with
`sparse/0/{cameras,images,points3D}.{txt,bin}` and an `images/` directory.
`PINHOLE` and `SIMPLE_PINHOLE` cameras are supported. Initial Gaussian scales
use the RMS distance to the three nearest sparse points.

Both trainers load RGB images through Grain `MapDataset`, with four CPU decode
threads and a prefetch buffer of eight images. Decoding and resizing produce
host uint8 arrays; JAX transfers run in the consuming thread. The production
trainer preloads the selected images onto the GPU once, preserving its seeded
epoch permutations and avoiding image I/O and repeated image uploads in the
training loop. The small-scene trainer streams a bounded number of images in
its original cyclic view order. Iterators close when training/loading exits.
The production JSON report records `image_load_seconds` separately from
compilation and the training loop. The pipeline follows Grain's
[threaded MapDataset prefetch API](https://google-grain.readthedocs.io/en/stable/tutorials/dataset_basic_tutorial.html).

On bicycle / RTX 5090, three interleaved loading trials measured a median of
0.858 s for serial decoding plus GPU preload and 0.442 s with Grain (48.4%
shorter). All 169 training images were byte-identical and in the same order.
A full 9,971-step run took 24.339 s before and 24.337 s after this change;
training performance was unchanged, with eight compiled variants and no
overflow. Nine targeted GPU tests passed, covering the data pipeline and both
small-scene training backends. The [Grain migration report](benchmarks/results/grain_migration.json)
contains loading trials, full training histories and checkpoint checks.

To reproduce the historical full-scene comparison with a million-slot pool:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute jaxgs-train \
  /path/to/bicycle --config benchmarks/configs/bicycle_10k.toml \
  --output bicycle.npz
```

The production trainer saves a JSON report beside the checkpoint, using the same
stem. If the checkpoint filename itself ends in `.json`, the report uses
`.report.json` to keep both files (for example, `model.json` and `model.report.json`).

This uses LiteGS's every-eighth-image evaluation split, epoch-based target
point count, weighted fragment-error variance sampling, clone/split and
weight-based pruning, opacity decay, per-parameter learning rates, sparse Adam
without bias correction, and SH degree activation every five epochs.
CuTe collects fragment count/weight in the forward pass and LiteGS
partial-sum error statistics in the backward pass; statistics are stopped
before densification. The production rasterizer uses LiteGS half2 arithmetic,
8x16 tiles and an opacity cutoff. L1+SSIM uses its fused separable CuTe kernels. Slot allocation uses a fixed-size prefix
sum and parallel CuTe writes; new slots have zero Adam state.

The original small-scene trainer remains available:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute jaxgs-train-reference \
  /path/to/scene --backend cute --capacity 2048 --downsample 8 \
  --steps 1000 --output gaussians.npz
```

`--tile-size`, `--sh-degree`, and `--max-visibility-pairs` set static
compilation parameters for CuTe training. Their defaults are 16, 3, and
`16 * capacity`. The last value is the capacity of the global list of
Gaussian/tile pairs; increase it if training reports pair overflow.
`--max-gaussians-per-tile` applies to the small-scene reference path and the
legacy capacity sweep.
When the sparse cloud exceeds `--capacity`, training seeds half the slots so
densification can add points. `--initial-points` overrides that count.
`--densify-threshold` controls the screen-space gradient cutoff (default `1e-6`).

The reference CLI defaults to CuTe and a 1024-slot pool. The library
default `CapacityConfig.max_gaussians` is `1 << 20`; allocate it only when the
scene and GPU have enough memory. CuTe training stops on global pair overflow.

## Pipeline

The bound production step culls world-space clusters, compacts visible cluster IDs,
projects those clusters, builds a visibility table, renders, evaluates fused
L1+SSIM, and applies Optax Adam to visible live slots. Projection
gradients stay in visible-cluster order until Adam maps them to pool slots;
unused gradient capacity is neither cleared nor read. The general projection
custom VJP still returns dense gradients for ordinary autodiff. Parameter and
moment buffers keep fixed capacity; JIT buffer donation allows in-place GPU
updates. The full trainer accumulates overflow on device and checks it at
each epoch boundary, without synchronizing after every update.

The Optax path stores only active SH coefficients in compact projection-gradient
buffers. It restores a zero gradient tail before the update, so higher-order SH
moments still decay and update visible parameters. Parameter and moment shapes
remain unchanged. The packed training pullback combines the symmetric conic
off-diagonal contributions into one atomic add; the general rasterizer VJP
retains its matrix-gradient convention.

Binning ports LiteGS's `binning.cu` and `speedy_splat.cuh`: count ellipse slices,
sort Gaussians by depth, prefix-sum their counts, emit Gaussian/tile pairs,
stably sort by tile, and build tile ranges. A fixed global pair arena preserves
JAX shapes without limiting the number of Gaussians in any individual tile.
The count, emit and tile range kernels use CuTe; prefix sums and stable sorting
use XLA. Tile sort keys use uint16 when the tile count and padding sentinel fit,
otherwise uint32. Gaussian IDs and tile offsets stay int32.
Emission visits Gaussians in depth order, so neighboring lanes write neighboring
prefix-sum segments. A Gaussian with more than 16 pairs is emitted by its whole
warp: lanes compute 32 slice rows at a time and write pairs with coalesced stores.
Counting and emission evaluate the same explicitly rounded ellipse slices, so each
Gaussian writes exactly the pairs it counted.
Production tile sorting uses the full fixed arena. Exclusive profiling found
that runtime selection of a sorted prefix added a device-to-host counter copy
on each update; that variant was withdrawn to retain asynchronous training.
Gaussian depth sorting also retains the full Gaussian capacity.

The packed rasterizer ports LiteGS's `raster.cu`: each single-warp block
processes one tile. Tiles launch heaviest first: a one-block counting sort orders
them by pair count for the forward pass, and by the pairs before each tile's last
contributor (reported by the forward pass) for the backward pass. A warp stages
32 splats at a time in shared memory while each lane loads one splat of the next
batch; the forward pass composites two splats per loop iteration. It uses
vertical pixel pairs, half2 FMA, forward differences, transmittance
scaled by 128, and warp reductions before global gradient adds. RG/BA use
paired half2 reductions; geometric gradients use LiteGS's shared-exponent
integer reductions. Noncontributing splats skip reductions and atomic adds. Backward saves
only final transmittance and the last record per pixel. Last indices use int32
to avoid the source's uint16 counter limit. `rasterize_packed_cute_vjp` exposes
RGB autodiff; the training pullback additionally returns fragment statistics.

In the small-scene trainer, `gradient_stats` are screen-space mean gradient norms;
the training loop averages them over views in which each Gaussian enters the
visibility table.
`densify_step` stops their gradient, selects a fixed number of parents, and
writes children into free slots while clearing those slots' Adam state.
`prune_step` and `reset_opacity` also keep shapes unchanged.
At initialization and after each densification interval, `reorder_gaussians`
groups live Gaussians by Morton order and reorders their Adam state with them.
Production densification keeps weighted parent sampling, then groups split and
clone candidates with a stable CuTe partition. Child generation selects a static
capacity bucket while returned model arrays and Adam state retain full capacity.
Partitionable Threefry preserves the original random prefix; other PRNG settings
use full-capacity generation. The
[algorithm optimization validation](benchmarks/results/algorithm_optimization_20260930.md)
records correctness, compiler memory plans and the limits of shared-GPU probes.
The subsequent [exclusive GPU comparison](benchmarks/results/exclusive_performance_20260930.md)
measured unchanged overall training throughput, 8–24% faster densification,
and removed a runtime sorting branch that introduced a per-step counter read.
The default near plane is 0.2 scene units. Binning applies LiteGS's opacity
threshold of 1/255, NDC bounds of +/-1.3, and positive-definite conic check.

The original `jaxgs-train-reference --backend reference` path uses JAX projection,
bounded per-tile binning and rasterization for small correctness checks.
Its `--backend cute` path uses the float32 diagnostic rasterizer, including
depth and alpha pullbacks. The JAX reference supports rectangular tiles;
the float32 CuTe diagnostic interfaces require square tiles. The production
`jaxgs-train` path uses the packed RGB renderer. Both geometry pullbacks compute xyz, log scale, rotation,
opacity and SH gradients in CuTe. JAX manages the fixed pool and training
loop and Optax parameter updates; the fused loss executes CuTe kernels.

## Fixed-count comparison

`benchmarks/fixed_model.py` pads both backends with the same tail-point copies
to complete 128-point clusters. Reports distinguish the PLY's `input_count`
from the actual rendered `count` (1,000,000 input points become 1,000,064).
Fixed-count benchmarking and evaluation both use LiteGS's `resolution=-1`
convention: cap image width at 1600 pixels and scale camera intrinsics accordingly.
JAX evaluation infers SH degree 0–3 from the checkpoint's coefficient count.

RTX 5090, the **same LiteGS PLY containing 975,104 Gaussians**, one 1237x822
bicycle view, SH degree 3, 10 warmup updates followed by 200 sequential training
updates. Compilation is excluded. JAX donates buffers and compiles the timed
step once. These measurements include forward, loss, backward and Adam.
The NNX migration comparison interleaves the pre-migration and NNX versions,
repeating each three times. LiteGS numbers come from the preceding kernel
comparison; earlier stages include single runs.

| Implementation / migration stage | ms / update |
| --- | ---: |
| LiteGS, recorded median | **3.77** |
| jaxgs before these kernel ports | 10.87 |
| + fused L1+SSIM | 9.36 |
| + half2 rasterization | 6.95 |
| + LiteGS 8x16 tiles and AccuTile binning | 4.89 |
| + visible-cluster projection and sparse Adam | 4.43 |
| + specialized SH, native reductions, compact gradients, coalesced caches | 3.76 |
| + vector parameter loads and direct half2 broadcasts | 3.71 |
| + tile boundary scan, vector parameter stores, SH transpose padding | 3.62 |
| + narrow tile sort keys | 3.55 |
| + NNX model and nnx.jit | **3.54** |
| + one warp per backward block, CuTe Adam | **3.44** |
| Optax, before zero-gradient projection skip | 3.78 |
| Optax, with zero-gradient projection skip | **3.70** |

The latest [interleaved comparison](benchmarks/results/bicycle_fixed_optax_comparison.json)
measured 3.554 ms before the backward launch change and 3.444 ms after it,
a 3.1% reduction. Each warp still handles one tile and uses the same reductions;
the backward launch changes from four warps per block to one. Optax was measured
separately on that same launch configuration, as described above.

The [NNX migration comparison](benchmarks/results/bicycle_fixed_nnx_comparison.json)
measured 3.542 ms before migration and 3.539 ms with NNX (medians), effectively
unchanged: -0.09%. Ranges are 3.529–3.552 ms and 3.535–3.550 ms, respectively.
Each timed step compiled once. All CuTe calculations are unchanged.
The [donation trace](benchmarks/results/nnx_donation_profile.json) records the
initial argument-order regression and its fix: ordering state/statistics before
the model removes 21 large GPU copies per step (about 1.13 ms).

The preceding narrow-key comparison measured medians of 3.634 ms with 32-bit
keys, 3.554 ms with narrow keys and 3.774 ms for LiteGS: approximately 2.2% and
5.8% shorter, respectively.
LiteGS restricts radix sorting to the used tile bits. Narrowing JAX's tile keys
reduces the bicycle tile sort from four radix passes to two; its main sorting
kernels take about 0.047 ms, versus 0.088 ms with 32-bit keys. Gaussian IDs,
parameter precision, stable depth order and arena capacity are unchanged.

LiteGS's tile boundary scan now replaces the repeated search kernels used to
build tile offsets. Empty tiles retain contiguous start/end offsets and the
fixed pair arena retains its padding sentinel. The 32-byte PackedParams use
vector stores as well as loads. Padding the SH shared-memory transpose stride
from 128 to 129 avoids bank conflicts when reading coefficients for one point.

In GPU traces, tile range construction and its related copies took about
0.08 ms before the change; the new range kernel takes about 0.008 ms. Parameter
packing decreased from 0.072 to 0.057 ms, and projection backward from 0.271 to
0.254 ms. These stage timings are single traces; the table uses repeated whole
training-step timings. Adam block reordering, vectorized SH Adam and fusing
parameter packing into projection were measured and reverted because they did
not improve step time.

JIT does not fuse across opaque CuTe FFI calls, so kernel arithmetic and memory
access still matter. This is one model and view, not a multi-scene performance
claim. All trials are retained in the
[narrow-key comparison](benchmarks/results/bicycle_fixed_keys_comparison.json).
The [previous pipeline comparison](benchmarks/results/bicycle_fixed_pipeline_comparison.json),
[vector-load comparison](benchmarks/results/bicycle_fixed_loads_comparison.json)
and [earlier comparison](benchmarks/results/bicycle_fixed_optimized_comparison.json)
are retained for context.

## Full training comparison

RTX 5090, bicycle `images_4` at 1237x822, 169 training views, 25 held-out views,
59 epochs / 9,971 updates, target one million Gaussians:

| Implementation | Training loop | Final Gaussians | Held-out PSNR |
| --- | ---: | ---: | ---: |
| LiteGS | 31.01 s | 975,104 | 25.047 dB |
| jaxgs NNX migration, recorded | 24.34 s | 975,104 | 25.083 dB |
| jaxgs CuTe Adam comparison, recorded | **23.97 s** | 975,104 | **25.098 dB** |
| jaxgs initial Optax integration, recorded | 30.37 s | 975,232 | 25.021 dB |
| jaxgs current, Optax | 30.10 s | 975,104 | 25.089 dB |

The initial [full Optax comparison](benchmarks/results/bicycle_optax_full_comparison.json)
uses the same one-warp backward launch for both optimizers. Each completed
9,971 updates with a one-million-slot pool, no overflow, finite checkpoint
parameters and exactly eight compiled variants. Optax took 26.7% longer.
The full-capacity XLA optimizer traverses free slots during early growth as
well as live slots. These are single full runs; floating-point rounding changes
the growth trajectory, so final active counts and PSNR are not identical.
The fixed-count comparison above uses exactly the same 975,104-point PLY.

The NNX migration was also checked with a fresh pre-migration run: 24.40 s
before, 24.34 s after, a 0.28% difference. Together with the repeated fixed-count
measurements, this shows no observed performance regression in this scene.
The [migration report](benchmarks/results/nnx_migration.json) contains both
full runs, checkpoint checks and 25-view evaluation results (25.055 dB before,
25.083 dB after).
The NNX full run is about 21.5% shorter than the recorded LiteGS run, with
similar held-out PSNR. These full growth-training measurements are single runs;
floating-point reduction and densification trajectories vary. The preceding
narrow-key run took 24.20 s with 25.082 dB PSNR. Earlier jaxgs versions took
26.20 s, 26.92 s, 27.14 s, 31.97 s and 97.56 s. Training-loop timings exclude initialization
and final checkpoint write, and include densification, pruning, opacity decay
and spatial refinement.
NNX/JAX warmup took another 9.52 s, reported separately. The eight variants
(four SH degrees, with/without statistics) were compiled during warmup; growing
the pool did not add compiled variants. The 10,000 requested
iterations become 59 complete epochs, as in LiteGS; its growth schedule stops
short of exactly one million points. Evaluation uses each implementation's own
renderer and the same held-out image names.

The NNX migration run had no overflow or non-finite loss; all final parameters were
finite. Peak visibility storage was 3,292,097 pairs within the eight-million-pair
arena. The pool retained exactly one million slots throughout training.

Remaining implementation differences: world-space bounds/frustum masks,
prefix sums and stable sorting use JAX/XLA. Buffers have fixed capacity with
valid compact prefixes. Reduction rounding, partial-tile masks and RNG
implementations differ. The analytic SH view-direction contribution to xyz
is retained; LiteGS's `activate_backward_kernel` omits that contribution.
These runs follow the same training protocol but are not numerically identical
trajectories.

## Verify and benchmark

```bash
JAX_PLATFORMS=cpu uv run pytest -q
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute pytest -q
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/capacity_sweep.py --scene /path/to/colmap --capacity 4096 \
  --downsample 8 --include-backward --repeats 100
```

The capacity sweep measures the earlier bounded-tile kernels for reference
and regression checks. Its [archived 4096-point results](benchmarks/results/bicycle_4096_rtx5090.json)
predate the global pair table and do not predict current training speed.
The [million-slot comparison record](benchmarks/results/bicycle_1m_comparison_rtx5090.json)
contains the LiteGS run and the original overflow diagnosis.
The [current training history](benchmarks/results/bicycle_jaxgs_nnx_training.json)
records every epoch. Held-out scores are available for
[jaxgs](benchmarks/results/bicycle_jaxgs_nnx_quality.json) and
[LiteGS](benchmarks/results/bicycle_litegs_quality.json).
The NNX migration passed all 50 tests in the GPU environment, including model
state/parameter filtering, compilation reuse, donated parameter/moment buffer
addresses, and parity against the array training step. Existing checks cover
empty/gapped/full-capacity tile ranges, unsigned keys above 32767 and the
16/32-bit key boundary, and the
[pre-migration direct LiteGS parity check](benchmarks/results/packed_parity_narrow_keys.json)
for 8x8, 8x16 and 16x16 tiles.
The [Optax validation](benchmarks/results/optax_validation.json) records the
current 60-test GPU run. The latest
[native parity check](benchmarks/results/packed_parity_backward_warp1.json)
also covers the one-warp backward launch for all three tile shapes.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/evaluate_protocol.py /path/to/bicycle bicycle.npz \
  --backend jaxgs --output quality.json
/path/to/LiteGS/.venv/bin/python benchmarks/evaluate_protocol.py \
  /path/to/LiteGS-compatible/bicycle /path/to/point_cloud.ply \
  --backend litegs --litegs-root /path/to/LiteGS --output litegs_quality.json
```

To reproduce the fixed-count comparison and direct native-kernel parity checks:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/fixed_model.py /path/to/bicycle /path/to/final.ply \
  --backend jaxgs --output fixed_jaxgs.json
/path/to/LiteGS/.venv/bin/python benchmarks/fixed_model.py \
  /path/to/LiteGS-compatible/bicycle /path/to/final.ply \
  --backend litegs --litegs-root /path/to/LiteGS --output fixed_litegs.json
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/packed_parity.py --litegs-root /path/to/LiteGS \
  --litegs-python /path/to/LiteGS/.venv/bin/python
```

[Native parity results](benchmarks/results/packed_parity_native_redux.json) cover square and
8x16 tiles: Gaussian/tile pairs and depth order match exactly, as do
transmittance and fragment counts. Relative L2 differences in geometry/color
pullbacks and squared fragment-error statistics are below 0.5%.

The older float32 fixed-state probe remains available for diagnostics (it
reads actual image dimensions and channel-major PLY SH layout):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/sorted_pipeline.py /path/to/bicycle --images images_4 \
  --ply /path/to/LiteGS/point_cloud/finish/point_cloud.ply \
  --capacity 1000000 --max-visibility-pairs 4000000 --all-views \
  --output probe.json
```

Omit `--ply` to use the scene's complete sparse point cloud. The
benchmark repeats a forward/loss/backward/Adam step from the same input state;
it does not accumulate 20 updates or measure end-to-end training. Historical measurements before the packed port are
stored for [54,275 initial points](benchmarks/results/bicycle_54275_sorted_rtx5090.json)
and [975,104 imported points](benchmarks/results/bicycle_975104_sorted_rtx5090.json).

## Source attribution

The CuTe packed rasterizer, sparse Adam and AccuTile binning are translations
of LiteGS code, with its [license retained](src/jaxgs/kernels/LICENSE.LiteGS).
AccuTile includes work from speedy-splat and gaussian-splatting.
The fused L1+SSIM kernels retain the original
[MIT notice](src/jaxgs/kernels/LICENSE.fused_ssim).
