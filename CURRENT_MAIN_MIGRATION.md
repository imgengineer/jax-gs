# gsplat current-main migration

The active upstream target is `nerfstudio-project/gsplat` `main` at commit
`2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c` (2026-07-24). The branch is
refreshed only at an explicit target-upgrade boundary; an in-progress phase
never silently follows a moving branch.

The implementation remains pure JAX with Flax NNX for stateful modules. Host
NumPy is allowed for serialization, deterministic preprocessing, and reference
test data, but not as the differentiable implementation of an upstream op.
Readability and subsystem boundaries take priority over kernel performance
during this migration.

## Acceptance rule

A subsystem is complete only when all of the following hold:

1. Its public modules, symbols, signatures, defaults, and documented error
   boundaries are inventoried against the pinned commit.
2. Dynamic PyTorch/CUDA outputs have an explicit JAX static-shape contract
   (`valid_count`, capacity, and overflow where needed).
3. Forward values, gradients where applicable, leading batch dimensions, and
   `jax.jit` are covered by focused tests.
4. The full existing CPU suite passes. The same pure-JAX programs are then run
   serially on GPU through `scripts/test_safe.sh` before the phase is marked
   complete.

## Phases

| Phase | Subsystems | Upstream surface | Status |
|---|---|---|---|
| 0 | Target contract and delta inventory | package/module/API/test manifest at the pinned commit | Complete |
| 1 | Core projection and rasterization | renderer configs, render-mode helpers, dense/packed projection, 3DGS/2DGS/3DGUT high-level rendering, extra signals, custom rays, hit distance, normals | Implementation complete; current full CPU acceptance passes, final GPU rerun pending |
| 2 | Sparse rasterization and visibility | sparse tile layout/intersections/pixels, contributing counts and IDs, top-contributor queries, sparse-gradient contracts | Implementation complete with dense-storage row-selective optimizer semantics; current full CPU acceptance passes, final GPU rerun pending |
| 3a | Structured camera sensors | camera kernels, functional APIs, NNX models/frames, poses, rolling shutter, and windshield distortion | Complete |
| 3b | Structured and legacy LiDAR | sensor APIs plus root models, preprocessing, angular tiling/intersections/rays, UT projection, and eval3d rendering | Complete |
| 3c | Root camera compatibility | upstream root camera-wrapper facade and high-level external-distortion plumbing | Complete |
| 4a | Geometry | `geometry.functional` quaternion, SE(3), packed-track, and trajectory operators | Complete |
| 4b | Scene | `scene` representations, transforms, normalization, scene-level state, and fixed-slot topology/resize/compact synchronization | Complete |
| 4c | Stage | `stage` orchestration and stage-level configuration/state | Complete |
| 4d | Dynamics | `contrib.dynamic`, deformation, HexPlane, and regulation | Complete |
| 5a | Losses and regularization | current loss surface, fused Gaussian losses, color correction, and occlusion regularizers | Complete |
| 5b | Compression and export | PNG/NPZ/K-means compression, spatial sorting, PLY/splat import and export | Complete |
| 5c | Experimental inference | packed-scene functional rendering and reusable inference renderer | Complete |
| 6 | Distributed and training integration | distributed renderer validation/collectives, schedulers, strategies, implemented trainer slices, and final compatibility audit | In progress: trainer P0/P1, current-main scene normalization, camera-pose/appearance integration, exact checkpoint/resume, and a dense-SH Gaussian-sharded device step with globally preflighted owner-local refinement commits and indivisible shard checkpoints are implemented; host-distributed bucket growth, data/eval orchestration, resharding, performance work, and final GPU acceptance remain open |

Phases describe dependency order, not monolithic patches. Each phase is split
into independently reviewable slices, and unsupported combinations raise at
their public boundary instead of being silently ignored.

## Static-shape policy

Upstream runtime-length packed arrays become fixed-capacity buffers. Only the
first `valid_count` entries are meaningful; `overflow=True` means the result is
truncated and must not be used for training or final evaluation. A capacity or
image/tile/channel shape change is a deliberate JIT specialization boundary.

Renderer policy objects select semantics, not implementation performance. Until
separate kernels are justified, multiple upstream renderer policies may share
the same readable pure-JAX implementation when their numerical contract is the
same.

## Pure-JAX rasterization policy

Projection, visible-prefix packing, tile-intersection construction, sorting,
offset generation, compositing, and gradients are expressed with JAX
primitives. `backend="auto"`, `"jax"`, and `"intersections"` therefore select
the same numerical implementation on CPU and GPU; `reference` remains the
full-bucket JAX comparison path. There is no device-dependent kernel dispatch.

Current-main `isect_tiles` uses opacity-aware SNUGBOX + AccuTile when conics and
opacities are provided and falls back to conservative AABB candidates
otherwise. AccuTile is an ellipse/tile intersection algorithm from the
gsplat/SpeedySplat lineage, not NVIDIA cuTile; its count/emit walk is implemented
in pure JAX here. Existing configuration files that name the removed `cutile`
backend are accepted at the configuration boundary, warned, and normalized to
`jax`; the legacy value never selects a separate runtime.

## Phase 0 delta inventory

Compared with v1.5.3, the pinned target adds or substantially changes these
families:

- renderer policy objects, hit-distance modes, custom eval3d rays, normals, and
  separately returned extra signals;
- sparse tile construction/rasterization plus visibility and contributor APIs;
- unified camera wrappers, external distortion, structured spinning LiDAR, and
  sensor models;
- geometry, scene, stage, dynamic-scene, and experimental rendering packages;
- expanded regularizers/losses, fused Gaussian losses, color correction, and
  scene SH compression;
- capability queries and revised distributed/training orchestration.

The detailed source of truth for names and behavior is the pinned upstream
tree. `COMPATIBILITY.md` records current implemented adaptations; this file
records migration order and phase exit criteria.

Unless a paragraph gives an explicit date, all test counts below are historical
phase-exit snapshots recorded before the latest 2026-07-27 integration changes;
their exact run timestamps were not retained. They are not the final acceptance
count for the current implementation slice; the current CPU acceptance and
pending GPU status are recorded in the phase-6 section below.

## Phase 1 implementation contract

Phase 1 adds the current-main renderer policy classes and render-mode queries,
hit-distance modes (`d`, `Ed`, `RGB-d`, `RGB-Ed`), custom eval3d rays,
accumulated normals, extra signal channels (including SH evaluation), and the
low-level eval3d debug outputs for last intersection IDs and sample counts.
`RendererConfig_MixedBatch` and `RendererConfig_ParallelBatch` deliberately
share the same readable pure-JAX numerical implementation; their validation and
observable results match while performance policy remains out of scope.

At the end of phase 1, LiDAR and external-distortion arguments were present at
the current-main API boundary and raised a phase-3 `NotImplementedError` rather
than being silently ignored. LiDAR support was subsequently completed in phase
3b, and high-level external-distortion plumbing in phase 3c. Sparse-gradient
and contributor behavior belongs to phase 2.

High-level 3DGS and 2DGS now expose fixed-capacity packed metadata across
leading batches. Their explicit zero-valued JAX probes are strict forward
no-ops. The 3DGS probe exposes the projected-means VJP and accumulates true
AbsGrad by taking componentwise absolute Gaussian×pixel cotangents before
summing pixels/cameras. The 2DGS public `gradient_2dgs` remains forward-zero;
its signed probe exposes the distinct current-main densification VJP from the z
entries of the first two ray-transform rows scaled by `w_M.z`, and its AbsGrad
probe applies absolute value before the pixel reduction. Dense, packed,
leading-batch, reference/intersection, and `jax.jit` paths retain identical
forward values. Eval3d and distributed 3D AbsGrad remain explicit unsupported
combinations.

## Phase 2 implementation contract

Sparse tile layout, intersection, pixel compositing, contributor counts,
all-contributor ID/weight queries, and top-contributor queries are implemented.
Runtime active-tile, intersection, and all-contributor lists are represented by
padded result objects with valid/required counts and explicit overflow. The
default capacities are complete, while optional smaller capacities are an
explicit memory tradeoff. With JAX x64 disabled, pixel masks use 32-bit words
and their cumsums/maps use int32; downstream JAX sparse APIs consume that
representation directly.

Sparse pixel compositing is ordinarily differentiable and is tested against a
dense gather for values and gradients. It also implements true AbsGrad with the
same independent zero-probe contract as dense pixel compositing. Upstream
visibility queries remain non-differentiable. `sparse_grad=True` requires the
unbatched fixed-capacity `packed=True` form and rejects distributed, UT, Eval3D,
and FTheta's implicit UT path. JAX renderer gradients, parameters, and Adam
moments remain dense arrays. Sparse training applies bias-corrected Optax Adam
only to the union of visible rows; `visible_adam` is a separate uncorrected
current-main SelectiveAdam path. Both leave hidden rows/moments unchanged and
advance global counters. This is a semantic adaptation, not a sparse/COO storage
or memory claim. The compatibility class intentionally takes
`(GaussianModel, OptimizerConfig)`, exposes `.step` as the NNX/Optax counter,
and applies gradients through `update(model, grads, visible_mask)` plus
`reset_slots(mask)`; it does not reproduce PyTorch/source's optimizer-object
`step(visibility)` call after mutable autograd gradients.

## Completed phase 3 contract

The structured sensor package implements pure-JAX camera and LiDAR kernels,
functional transforms, pose interpolation, rolling shutter, windshield
distortion, and Flax NNX model/frame wrappers. Camera projection, inverse
projection, eager execution, `jax.jit`, gradients, and leading batches are
covered independently from the root rasterization facade.

Both current-main LiDAR surfaces are implemented. The structured
`jax_gs.sensors` API follows its native `[elevation, azimuth]` angle order,
while the root legacy-compatible API follows `[azimuth, elevation]`. They stay
separate and require explicit conversion at a caller boundary; no implicit
axis swap is performed. Host NumPy is used only for deterministic tiling/CDF
preprocessing. Runtime intersections remain pure JAX and use padded buffers
with `valid_count` and `overflow`.

High-level LiDAR rasterization requires `camera_model="lidar"`, `with_ut=True`,
and `with_eval3d=True`. Sensor rows and columns override generic image
dimensions, angular tiles map back to the exact structured sensor elements,
and opacity is evaluated along the corresponding world-space scan rays,
including start/end pose interpolation. The current implementation is the
readable pure-JAX reference path and makes no specialized-kernel performance
claim.

The root camera facade implements the current-main factory and projection,
inverse projection, shutter timing, shutter-pose world-ray, and shutter-pose
world-point methods. Root bivariate windshield parameters retain upstream's
independent triangular polynomials of orders zero through five. External
distortion is connected to UT projection, orthographic hemisphere mapping, and
eval3d ray construction; it requires `with_ut=True` and is intentionally
invalid for LiDAR.

Phase 3 adds 22 focused root external-distortion/wrapper cases in addition to
the 107 structured-sensor and legacy-LiDAR cases. After phase 3c, the full CPU
suite reports 328 passed, one skipped because the local Mip-NeRF360 stump
dataset is unavailable, and 26 deliberately deselected.

## Completed phase 4a contract

The current-main `geometry` package is implemented with the same public module
layout and functional exports. Quaternion operators use upstream's `xyzw`
storage convention and are deliberately isolated from the older `wxyz` math
and structured-sensor helpers. Safe normalization, Hamilton composition,
vector rotation, matrix conversion, axis-angle conversion, shortest-hemisphere
LERP/SLERP, angular distance, and SO(3) manifold interpolation are pure JAX and
support leading batch dimensions where upstream does.

SE(3) pose transforms retain upstream's exact two-dimensional row contract.
Matrix conversion uses the same Shepperd branch ordering; packed pose tracks
retain lower-bound behavior for duplicate timestamps, endpoint clamping, and
identity no-ops for invalid ranges. Two-pose trajectories preserve unordered
keyframe spans, extrapolation, equal-time pose-zero behavior, and strict
out-of-bounds flags. All functional operations participate directly in JAX
autodiff instead of exposing PyTorch `autograd.Function` classes.

The PyTorch-only requirement for an explicit `quat_identity(device=...)` is
adapted to JAX's default device while still accepting an explicit JAX device.
When JAX x64 is disabled, packed indices and integer timestamp arithmetic use
the safe available int32/float32 types; applications requiring nanosecond-scale
int64 timestamp precision must enable JAX x64 before creating arrays. Phase 4a
adds 51 focused cases. Its full-suite exit reports 379 passed, one unavailable
local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 4b contract

The current-main `scene` package now exposes `Scene`, `GaussianScene`,
`GaussianInferenceScene`, `SHCompressionMode`, and the lazy functional packer
under the same module hierarchy. `GaussianScene` uses `nnx.Dict` and
`nnx.Param` as the Flax NNX equivalent of a PyTorch `ParameterDict`; the first
NNX component retains object identity, later components append matching rows,
and component membership plus arbitrary signal arrays stay aligned through all
topology hooks. State dictionaries preserve trainable versus non-trainable
leaves and component metadata.

Inference packing preserves the upstream layouts: planar float32 means,
float16 `[wxyz, scale, opacity]` rows, aligned RGB, float32 SH0–2, and the
three SH3 layouts. FP16 lanes clamp at 65504 and emit the scene-level warning;
activated tensor construction validates finite values, positive scales,
bounded opacities, and unit quaternions. The packer uses `stop_gradient` to
retain upstream's explicit inference/no-autograd boundary. CUDA contiguity is
not a JAX concept, so the pure-JAX path accepts normal JAX arrays on any JAX
device instead of reproducing that backend-only restriction.

Phase 4b adds 33 focused cases. Its full-suite exit reports 412 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 4c contract

The current-main `stage` package is intentionally a small orchestration layer:
it registers each `GaussianScene` with its render callable, preserves scene-id
insertion order, resolves scenes by id, and dispatches rendering exactly as
`render_fn(splats=scene.splats, **kwargs)`. Duplicate ids and unknown-scene
lookups fail at the registry boundary, while renderer return values are passed
through without imposing an arity or result type.

Phase 4c adds 8 focused cases. Its full-suite exit reports 420 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 4d contract

The experimental `contrib.dynamic` hierarchy now exposes the current-main
`DeformNetwork`, `HexPlaneField`, `DynamicStrategy`, and HexPlane regularizers;
`DeformationTable` retains its explicit back-compat import paths without being
added to the package `__all__`. Trainable networks and feature planes are Flax
NNX modules. Their random initialization accepts explicit `nnx.Rngs`, with a
deterministic default only for source-compatible construction. Zero-initialized
deformation heads preserve the exact identity-at-construction behavior and the
same initial trunk-gradient boundary.

HexPlane retains the six coordinate-pair planes, spatial-only multires scaling,
sign-reversed upstream AABB normalization, temporal-plane initialization at
one, and border-padded bilinear sampling. The pure-JAX sampler also preserves
the upstream private 3D/trilinear path. Plane/time second differences and time
L1 regularization remain direct differentiable JAX expressions.

`DynamicStrategy` follows this repository's fixed physical-capacity contract
instead of resizing a runtime-length tensor. Its NNX state carries a boolean
mask with the same leading capacity as the model: pruned slots clear their
flag, duplicate/split allocations inherit their selected parent's flag, and
bucket resize plus active-prefix compaction preserve row alignment. The public
pre/post-backward hooks accept explicit JAX gradient arrays. Phase 6 now wires
the same fixed-slot transactions into `GaussianScene` splats, component indices,
arbitrary signals, and Dynamic masks, including resize and compaction.

Phase 4d adds 38 focused cases. Its full-suite exit reports 458 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 5a contract

The current-main loss module is implemented in pure JAX, including NCHW SSIM,
depth and masked photometric losses, LiDAR loss dispatch, standard elementwise
losses, scheduling/reduction helpers, bilateral-grid penalties, and unreduced
Gaussian regularizers. In particular, `l1_loss` and `mse_loss` now follow the
current-main contract and return elementwise arrays; repository training and
PSNR code perform their reductions explicitly. Existing channel-last SSIM and
depth/normal helpers remain available under their established local names.

`FusedGaussianLosses` is an NNX call facade over the four differentiable JAX
regularizers. It reports `has_losses() == True` and preserves the fused API and
output tuple, but makes no kernel-fusion or performance claim. Quadratic and
affine color correction use JAX least squares, while targeted TV, mask dilation,
and invisible-mask union retain their current-main numerical and validation
contracts. PNG mask loading is the only host-side path in this slice.

Phase 5a adds 39 focused cases. Its full-suite exit reports 497 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 5b contract

The current-main compression import hierarchy now includes
`compression.png_compression` and `compression.sort`. `PngCompression` retains
the upstream `meta.json` layout, 16-bit log-position PNGs, 8-bit parameter PNGs,
quantized SH codebook, NPZ passthrough fields, square cropping, and normalized
quaternions. The existing byte-exact `GaussianModel` transport remains an
explicit local extension.

PLAS is an optional PyTorch/CUDA compression optimizer rather than an on-disk
semantic requirement. This port therefore uses deterministic host-side xyz
sorting and K-means, preserves aligned rows and the perfect-square boundary,
and makes no PLAS compression-ratio claim. PLY loading follows the INRIA
channel-major property convention and returns float32 JAX arrays in the
basis-major `(N, K - 1, 3)` layout, including degree-zero files. `plyfile`
remains a lazily imported runtime option and is installed in the development
group so the round-trip contract is exercised without adding it to core JAX
dependencies.

Phase 5b adds 5 focused cases. Its full-suite exit reports 502 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Completed phase 5c contract

The complete public `experimental.render` hierarchy is available, including
`RenderReturn`, the stateless `rasterize_gaussian_inference_scene` and
`render_scene` entry points, the raw packed-array operator, and the reusable
Flax NNX `GaussianInferenceRenderer`. The stateless surface accepts exactly one
pinhole camera, validates the current-main request subset, returns float32 RGB
plus alpha, and tags dispatcher results. The stateful surface prepares its SH
codec once, guards scene-count mutation, supports SH-degree reduction, manages
release/context-manager state and a frame cache, and returns float16 RGBT.

Packed means/QSO/RGB/SH0–3 inputs are decoded directly into the existing
readable pure-JAX rasterizer. The 32B and 16B SH modes reproduce the observable
YCoCg scale, gamma, quantization, and decode numerics; because performance and
binary CUDA storage are out of scope, they do not materialize the intermediate
256/128-bit streams. The 16B path drops higher-order chroma as upstream does.
Core packed rendering is covered under `jax.jit`.

JAX has neither PyTorch's process-global grad mode nor writable array buffers.
The renderer therefore applies `stop_gradient` at its packed inference boundary.
An `out=RenderReturn` retains container identity while its immutable frame and
alpha arrays are replaced with the results; callers may use JAX buffer donation
at their own compiled boundary when storage reuse matters. The numerical path
makes no fused-CUDA performance claim.

Phase 5c adds 25 focused cases. Its full-suite exit reports 527 passed, one
unavailable local-dataset skip, and 26 deliberately deselected tests.

## Phase 6 implementation status

The implemented phase-6 package paths provide multi-frame depth unprojection
and chunked KNN initialization, with an optional JAX PRNG key for bounded
sampling. Point-cloud model construction sets each active isotropic scale to
the RMS distance over up to three nearest neighbours times `initial_scale`,
whose `ModelConfig` default is 1.0. The fresh 2DGS profile uses near/far
0.2/200, prune opacity 0.05, and `gradient_2dgs` as its strategy key. The fresh
3DGS MCMC profile maps current-main `init_opa=0.5` and `init_scale=0.1` to the
local model fields and enables 0.01 opacity/scale regularization. The `training`,
`strategy`, and `optimizers` modules now expose the upstream package paths,
including `TwoStageScheduler`, `Strategy`, `DefaultStrategy`, `MCMCStrategy`,
fixed-capacity strategy operations, and `SelectiveAdam`.
Sensor compatibility includes the common tensor helpers, LiDAR and projective
dispatch tables, and the completed model re-export hierarchy. JAX profiler
annotations back the `trace` facade.

The strategy package exposes current-main-style `step_pre_backward` and
`step_post_backward` hooks while retaining this repository's fixed physical
capacity. Signed densification statistics come from explicit
`<key>_gradient`; true compositor AbsGrad comes from `<key>_absgrad`, replacing
PyTorch's mutable tensor-side field. Both accept exact dense `[C, N, 2]` or
padded `[P, 2]` statistics, while a legacy capacity-shaped signed gradient
remains accepted. Duplicate, split, remove, opacity reset, relocation,
sample-add, and MCMC position perturbation keep model, optimizer, strategy,
Scene sidecar, and Dynamic-mask rows aligned. When one original parent satisfies
both current-main growth conditions, the plan emits both a duplicate and a
split from the original snapshot; the event cap preserves duplicate-before-split
ordering, and the retained original plus split child receive independent split
samples and revised opacity. Insufficient capacity skips all of those states
plus statistics and MCMC noise atomically.
The trainer performs scheduled MCMC capacity preflight before its commit, so an
overflow also leaves optimizer step/model updates untouched.

The unified trainer supports both `model_type="3dgs"` and `"2dgs"`, and connects
signed or true-AbsGrad probes directly to the Default strategy lifecycle. Its
default `patch_size=None` path trains on full images, including non-square H/W;
the scene-derived dimensions pass through renderer calls, densification stats,
rectangular tile/intersection capacity, and initial plus bucket-growth memory
preflight. Full-image training requires one shared shape across the training
split. An explicit scalar `patch_size` retains random square-patch training and
is the fallback for mixed-resolution inputs. Grain constructs the training
stream as `shuffle → repeat → batch`, so epoch tails remain visible, small
scenes do not stall under `drop_remainder`, and resume fast-forward stays
deterministic.

Fresh COLMAP training now follows the pinned example's complete spatial
normalization: average camera-up alignment, focus-point median centering,
median camera-distance scaling, point-cloud principal-axis alignment, and the
final upside-down correction. `normalize_world_space=False` selects identity;
`global_scale` affects the derived `1.1`-margin scene scale rather than the
coordinate transform. These dataset operations are deterministic host NumPy
preprocessing; projection, optimization, and gradients remain pure JAX.

For 2DGS the densification path is the forward-zero ray-transform VJP, not a
projected-means proxy. The branch implements packed sparse training plus
upstream normal-consistency and distortion losses with their strict
start-iteration boundaries. The 3DGS branch additionally supports
visibility-selective Adam without requiring packed rendering and adds
active-only scalar regularization: raw opacity logits and log-scales are
transformed by sigmoid/exp, averaged over `active_mask`, and weighted by
`opacity_reg`/`scale_reg`,
and exposed through CLI flags and loss metrics. 2DGS rejects either scalar
weight when nonzero. MCMC 3DGS supports UT/Eval3D by skipping screen statistics
it does not consume. Sparse training uses bias-corrected Adam; visible Adam uses
uncorrected current-main SelectiveAdam. Both retain dense model/optimizer
storage, preserve hidden rows/moments, and advance global counters; neither is a
COO memory optimization. Unsupported combinations fail at configuration
validation: sparse gradients require unbatched packed rendering and reject
distributed/UT/eval3d/FTheta, Default Eval3D remains open, 2DGS rejects visible
Adam, and 2DGS packed training rejects the reference backend because it lacks
the required cross projection/intersection metadata.

Camera-pose optimization is now integrated into the unified trainer.
`jax_gs.training.pose.CameraOptModule` stores one translation + 6D rotation
delta per image, retains upstream's random constructor initialization, uses the
trainer's explicit `zero_init()` identity start, and right-multiplies the
resulting local transform onto camera-to-world matrices;
`rotation_6d_to_matrix` follows current-main's row convention. `TrainConfig`
and the CLI expose `pose_opt`, `pose_opt_lr`, `pose_opt_reg`, and `pose_noise`.
Both real 3DGS and 2DGS train steps apply fixed stop-gradient pose noise first
and the trainable local adjustment second. Pose owns an independent Adam with a
batch-size-scaled learning rate that decays exponentially to one percent and
coupled L2 regularization. The Gaussian and pose updates share one overflow
predicate, so skipped steps and the replayed suffix after capacity growth keep
their model and optimizer states aligned. Orbax stores the pose module and
optimizer; the manifest records the training camera count and exact image-name
ordering, which resume validates before restore. Pose resume requires the same
total `steps`, so rebuilding the exponential learning-rate horizon cannot
silently raise the resumed learning rate.

Appearance optimization is now integrated into the same unified trainer.
`TrainConfig` and the CLI expose current-main's `app_opt=False`,
`app_embed_dim=16`, `app_opt_lr=1e-3`, and `app_opt_reg=1e-6` controls. Enabling
it changes the Gaussian color state rather than layering parameters on top of
SH: the model owns fixed 32D `features` and three base `colors` logits, and does
not own `sh0`/`sh_rest`. The two representations are constructor- and
checkpoint-validated as mutually exclusive. Features and colors have separate
Gaussian optimizer labels but both use `sh0_lr`; inactive-row masking and the
existing dense-storage sparse/visible update semantics still apply.

`jax_gs.training.appearance.AppearanceOptModule` combines the split-local
training image embedding, Gaussian features, and SH direction bases to predict
`[C, N, 3]` color-logit corrections. Its output layer is zero-initialized.
Directions use the pose-adjusted camera origins, and direct RGB is
`sigmoid(colors + correction)`, so both real 3DGS and 2DGS rasterizers receive
`sh_degree=None`. The embedding Adam uses
`app_opt_lr * sqrt(batch_size) * 10` plus coupled `app_opt_reg`; the color-head
Adam uses `app_opt_lr * sqrt(batch_size)` without decay. Gaussian, pose, and
appearance updates share a single overflow predicate, including skipped steps
and the replayed suffix after bucket growth.

The Gaussian feature/color groups retain this repository's existing direct
`sh0_lr` semantics. Every Gaussian optimizer group now uses current-main's
effective batch `B=batch_size*world_size`: learning rates scale by `sqrt(B)`,
epsilon by its inverse, and beta1/beta2 by the upstream linear rule. The means
rate alone also receives `scene_scale`. The host trainer supplies its normalized
camera extent with current-main's 1.1 margin and exact resume rejects a changed
batch size. `B>10` fails early because the upstream beta1 formula becomes
negative. The appearance module optimizer keeps its distinct scaling described
above.

Default duplicate/split and MCMC relocate/birth copy the active appearance
representation from the same parent snapshot. Bucket resize and active-prefix
compaction keep features, colors, and their Gaussian optimizer moments aligned;
atomic commit/replay also preserves the separate appearance module and optimizer
state. Orbax checkpoint format v6 retains v5's model color mode, appearance
module/optimizer, feature dimension, training-camera count, and exact image-name
ordering, and supports an optional scene component. Trainer-generated saves add
the exact world-to-training matrix plus final scene scale.
Strict resume validates those fields together with appearance config, SH
degree, batch size, normalization mode, and global scale before an exact
restore. Resume and CLI rendering reuse the stored matrix instead of rerunning
PCA. When the scene component is absent, including in v1-v5 and generic v6
checkpoints, the legacy camera-center mean/max-extent transform is intentionally
retained so existing Gaussian coordinates are not reinterpreted.

This intentionally closes the pinned upstream example's incomplete appearance
restore path instead of reproducing its lost module/optimizer state.

Held-out evaluation and CLI rendering use `embed_ids=None`, which means a zero
camera embedding, and scheduled evaluation uses the configured full direction
basis degree. CLI export follows current-main's canonical bake: zero embedding,
zero direction, and the configured full degree are evaluated once, then the RGB
result is converted to degree-zero SH with an empty `sh_rest`. The generic
exporter rejects raw features/colors. Point-cloud initialization uses a stable
clipped inverse sigmoid for base color logits; unlike upstream's direct
`torch.logit(rgb)`, exact zero/one inputs therefore remain finite.

This trainer is still not the complete current-main training flow. The
appearance slice is single-process, as is the broader optimizer orchestration;
unified `train()` explicitly rejects `jax.process_count() != 1` rather than
silently producing divergent per-process states. Complete multi-process/multi-
host training, sharded optimizer state, and performance specialization remain
open.

Root `rasterization(distributed=True)` is routed to the distributed
implementation. The single-rank path reproduces the local forward values,
metadata, and gradients without a collective. Multi-shard execution uses a
bound named JAX axis and equal padded shards; the readable implementation
gathers/replicates the scene on each rank and does not support a Gaussian leading
batch. The current-main-style `distributed.cli` runs the worker once in each
externally launched JAX process; process creation belongs to `jax.distributed`,
MPI, Slurm, or another launcher, while single-process multi-device mapping
belongs to `pmap`/`shard_map`. Input capture and replay are available through
`jax_gs.profile`: captured arrays are host NumPy values in standard pickle
payloads and are restored with `load_capture`. The default `.pt` suffix is kept
for upstream discoverability, but these files are not `torch.save` archives.
Replay, gradients, timing, and trace collection use JAX rather than reproducing
PyTorch/CUDA profiler presets or kernel-family assertions.

`training.make_distributed_train_step()` now extends those collectives through
the first current-main Gaussian-sharded optimizer commit. Inside a bound
`nnx.pmap`, every rank owns an equal-capacity Gaussian/Adam shard and one local
camera batch. The differentiable gather transpose sums rank-local photometric
gradients; it is deliberately not followed by `pmean`. Optimizer metadata binds
the local batch, world size, scene scale, and full optimizer config, while the
factory requires `optimizer.max_steps == TrainConfig.steps`. Global visibility
is reduced in global Gaussian coordinates before slicing the owner segment.
Signed densification statistics use an independent
local-camera/global-Gaussian probe: each rank first normalizes and takes the L2
norm for every visible camera/Gaussian pair, then `psum/psum/pmax` combines
gradient sums, counts, and radii before the owner segment is selected. Current
and prior sticky overflow state are synchronized, and any rank mismatch in
optimizer step or SH degree makes the complete mapped update, including
strategy statistics, a no-op. Tests cover the sum-vs-mean Adam moment,
opposite-direction signed screen gradients, asymmetric visibility/radii and
overflow, named vmap, and a real two-virtual-CPU `nnx.pmap` call.

The next slice turns those statistics into an owner-local refinement plan and a
global preflight. A refinement schedule
is now accepted. On every step the shard runs the ordinary `DefaultStrategy`
plan over its own `[L]` rows, using the train step's `scene_scale` rather than
`StrategyState`'s per-rank copy so no rank can score growth or pruning against
a different threshold. Owner-local decision arrays are never reduced; only the
plan's scalars cross ranks, as `psum` on the planned new and pruned counts,
`pmax` on the per-shard required capacity, and `pmax` on the capacity-overflow
flag. The reduced results are reported as `refine_scheduled`,
`reset_scheduled`, `refine_planned_new_count`,
`refine_planned_pruned_count`, `refine_required_capacity`, and
`refine_capacity_overflow`; the counts are zero on steps with no scheduled
refinement. A planned overflow on any single rank joins the existing overflow
atomics, so the model, optimizer, and statistics commit is skipped on every
rank and the step is replayable once a host grows all shards. Because upstream
runs its post-backward strategy callback on rank-local parameters, this
owner-local plan is the faithful decomposition; the fixed-capacity
`max_new_per_refine` bound, however, applies per shard rather than globally.
Tests cover an owner-only duplicate, a same-parent duplicate plus split,
prune counts summed across ranks, inactive padding, the scene-scale source,
single-rank overflow atomics,
and named `nnx.vmap` versus two-virtual-CPU `nnx.pmap` agreement.

The slice after it commits the transaction. Once the update branch has stepped
Adam and accumulated this step's statistics, every owner runs the ordinary
`DefaultStrategy.refine` on its own rows and then its scheduled opacity reset,
which is current-main's post-optimizer callback order. The commit deliberately
recomputes its events from that post-update state instead of replaying the
pre-update preflight, because upstream decides on post-update parameters and
post-accumulation statistics; the preflight remains the host's growth signal,
and the two can legitimately disagree when this step's statistics move a
threshold. Parameters, Adam moments, and strategy statistics stay owner-local,
so `_default_refine`'s existing fixed-slot assignment, moment reset, and
statistics clearing carry over unchanged. An owner whose recomputed events no
longer fit its free slots keeps its rows and statistics, skips its opacity
reset, and reports through `refine_commit_overflow` while other owners still
commit, exactly as independent per-rank upstream strategies would. The
committed counters cross ranks only after both update branches rejoin, so no
collective ever runs inside a conditional. Tests cover a committed owner-only
duplicate, a committed same-parent duplicate plus split with its Adam moment
reset, pruning, a committed opacity reset, the deliberate plan/commit
divergence, single-owner commit overflow with a diverging reset, equality with
a single-process `DefaultStrategy.refine` on the same shard, and `nnx.pmap`
agreement.

The third slice persists that world. `save_distributed_checkpoint` writes the
stacked model, optimizer, `StrategyState`, and `TrainingSafetyState` as one
indivisible shard set, because a shard's parameters, Adam moments, statistics,
and sticky overflow state are only consistent together. Saving is refused when
the nodes are not stacked over one common world axis or when the shards
disagree on the optimizer step, so an inconsistent world cannot be written. The
manifest records world size, per-shard and global capacity, per-shard active
counts and prefix layout, and a configuration fingerprint;
`load_distributed_checkpoint_manifest` exposes it because a host must allocate
matching stacked targets before it can restore. `restore_distributed_checkpoint`
accepts only an exact same-world-size, same-shard-capacity resume and validates
the color mode and, when given a config, the fingerprint. Resharding a saved
world across a different rank count needs model, optimizer, and statistics to
move together and is deliberately rejected. The distributed and single-process
entry points refuse each other's artifacts. Tests cover the round trip through
fresh shards, manifest contents, resharding and capacity rejection, config
mismatch, cross-kind rejection, unsharded and step-divergent saves, and a
restored world that keeps training with an aligned schedule.

These slices are intentionally dense SH pinhole 3DGS. Their photometric and
active/visible metrics are rank-local, whereas overflow, intersection, and
refinement diagnostics are global. Physical shard capacity never changes inside
the step and no Scene/Dynamic sidecar is carried, so bucket growth, sidecar
lineage, and resharding stay host work.
Upstream current-main itself rejects distributed AbsGrad. Pose/appearance,
packed/sparse/visible Adam, UT/Eval3D, 2DGS, host data
sharding, eval, launch, and performance specialization remain later
slices. Unified `train()` consequently continues to reject multiple JAX
processes.

The means optimizer receives the scene scale derived after current-main spatial
normalization. The 4x4 transform and the final `1.1 * global_scale` camera
extent are persisted by trainer-generated checkpoint format v6 saves, making
fresh training, resume, scheduled evaluation, and CLI rendering share one
coordinate system. Generic v6 saves may omit that optional scene component.

The pinned-tree inventory covers the backend-independent production public
Python modules, symbols, import paths, defaults, and validation boundaries. It
is an inventory, not a claim that every behavioral branch has passed the phase
acceptance rule above. Upstream `_helper.py` is test support;
PyTorch autograd wrapper classes, `_lazy_backend.py`, the root `cuda` extension
wrapper package, native CUDA `_backend.py` loaders, and CUDA build modules are
not fabricated in a pure-JAX package. Their observable numerical operations are
exposed directly as differentiable JAX functions.

At a historical phase-6 checkpoint before the latest 2026-07-27 integration
changes (exact run timestamp not retained), four focused test files reported 42
passes. That checkpoint's forced-CPU non-resource-heavy suite reported 552
passed, 21 expected CUDA/backend or local-dataset skips, and 26 deliberately
deselected tests, with one existing Orbax restore warning. Its
`RUN_GPU_TESTS=1 scripts/test_safe.sh` run completed with exit status zero: the
fresh-process CPU/resource-heavy passes succeeded and all 53 collected
GPU-process cases passed serially on `CudaDevice(id=0)`. These numbers predate
the AbsGrad, sparse-training, topology, distributed-root, overlap, and trainer
P0/P1 slices.

The Default public AbsGrad hook, Scene/Dynamic topology transactions, true
3DGS/2DGS densification VJPs, leading-batch 2DGS packed metadata, root
distributed renderer routing, MCMC UT/Eval3D commits, and the two distinct
row-selective optimizer semantics are implemented. Default full-image training,
rectangular H/W propagation, the current-main 2DGS/MCMC profiles, and active-only
3D scalar regularization are also implemented. Camera-pose config/CLI,
3DGS/2DGS train-step integration, independent Adam, atomic overflow replay, and
manifest-validated checkpoint/resume are complete. Appearance config/CLI,
mutually exclusive Gaussian features/colors, real 3DGS/2DGS training,
independent parameter groups, topology transactions, trainer-generated v6
scene-aware exact resume,
zero-embedding evaluation, and canonical export bake are also complete. Phase
6 also has a tested dense-SH Gaussian-sharded device train step whose
refinement schedule is globally preflighted and committed owner-locally at
fixed shard capacity, plus indivisible same-world-size shard checkpoints; it
still requires host-distributed bucket growth and data/eval orchestration,
resharding,
performance work, and the full serial GPU acceptance rerun.
Deliberate boundaries also include dense storage underneath sparse-gradient
semantics, the JAX-specific `SelectiveAdam.update(...)` call surface, explicit
zero probes instead of mutable `.absgrad`, FTheta sparse rejection, external
process launch/equal padded renderer shards, no distributed Gaussian leading
batch, fixed-capacity distributed refinement with a per-shard
`max_new_per_refine` bound, and rejection of 2DGS
reference+packed training where cross metadata is unavailable.

On 2026-07-27, the merged Phase-6 trainer P0/P1 focused selection reports 118
passes (117 from the merged selection plus one targeted fix verification). This
is a focused regression count rather than the full-suite result below.

Also on 2026-07-27, the then-standalone camera-pose and appearance module
selection reported 14 passes. This historical count predates unified appearance
training and does not replace the current full-suite result below.

The distributed shard-checkpoint 2026-07-29 forced-CPU non-resource
acceptance
reported `858 passed, 1 skipped, 38 deselected`; the warnings were four known
Orbax restore sharding warnings and the only skip was the unavailable optional
local Mip-NeRF360 stump dataset. Five fresh-process resource-heavy selections
passed `19+5+9+3+2=38` cases: 19 high-level 2DGS, 5 low-level 2DGS, 9 Eval3D,
3 sparse rasterization, and 2 visibility cases. The current slice therefore has
896 passing CPU cases in total.

Two full-script attempts invoked through
`RUN_GPU_TESTS=1 RUN_RESOURCE_HEAVY_GPU_TESTS=1 scripts/test_safe.sh` passed all
CPU preflight groups and then 21 and 32 isolated CUDA cases respectively. The
per-case safety checks stopped the attempts when transient `libuv-worker` and
kernel-journal D-state threads appeared. No `ALLOW_D_STATE_GPU_TESTS` override
was used, and no complete current-slice GPU pass is claimed.

On 2026-07-26, a follow-up signature audit checked 188 backend-independent
upstream public function definitions. Public parameter names and ordering now
match with zero findings; JAX-only keys/axis/capacity arguments remain trailing
extensions. This slice corrected keyword calls for the signed log transforms,
normalized quaternion conversion, exporter Morton/rotation helpers, and sensor
quaternion-layout helpers. Its adjacent regression set reports 62 passed.

On 2026-07-27, the follow-up class audit inventoried 66 backend-independent
public classes and compared 34 constructor/dataclass contracts. The 23 excluded
symbols are the documented PyTorch autograd wrappers plus the Optax
`SelectiveAdam` constructor adaptation; all remaining constructor field order,
defaults, and the eight upstream constructor validation boundaries have zero
findings. `DefaultStrategy` and `MCMCStrategy` now use the upstream positional
field order while retaining `config=` as a trailing JAX extension and accepting
a lone positional `StrategyConfig` for existing local callers. The adjacent
strategy/training/storage regression set reports 63 passed.

Also on 2026-07-27, the public class-method audit compared 102 callable methods
and 22 properties after excluding the 22 documented PyTorch autograd wrapper
classes. Ninety-three methods match exactly, three retain only trailing JAX
extensions, no upstream method or property is missing, and the remaining six
differences are the documented fixed-capacity strategy and NNX/Optax
`SelectiveAdam` adaptations. Ten public structured-return dataclasses retain
upstream field order and optional defaults with zero findings. This slice
restored `PngCompression`'s `compress_dir=` keyword and `compress() -> None`
contract, current-main `timeit` dunder keywords and exit return, and explicit
`forward()` entry points on the four sensor NNX model/frame classes. Its
adjacent compression/profile/sensor/return-contract regression set reports 118
passed.

The subsequent rasterization consolidation removed the optional specialized
kernel dependency and its four implementation modules. Stable visible packing,
AABB rank mapping, AccuTile count/emit, tuple sorting, offset construction, and
front-to-back compositing now use JAX on every device; ordinary reverse-mode
gradients flow directly through the compositor, while a narrow custom-VJP probe
boundary accumulates true per-pixel AbsGrad without changing forward values or
signed gradients. The public low-level
`isect_tiles` surface now consumes `conics` and `opacities` for current-main
AccuTile semantics instead of treating them as compatibility-only inputs.
Backend selector fields remain in `RasterizationConfig` so saved training
configuration retains its schema, while removed legacy backend values normalize
to `jax`. Historical specialized-backend timing is not evidence for this
implementation and is intentionally excluded from the current compatibility
claim.
