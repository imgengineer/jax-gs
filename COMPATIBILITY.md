# gsplat compatibility contract

The active compatibility target is gsplat `main`, pinned at commit
`2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c` (2026-07-24). The port started
from the historical `v1.5.3` baseline at
`937e29912570c372bed6747a5c9bf85fed877bae`; the table below records the
implemented public surface and its pure-JAX adaptations. Rows marked as
implemented do not override the deliberate behavioral boundaries listed below.
See [`CURRENT_MAIN_MIGRATION.md`](CURRENT_MAIN_MIGRATION.md) for phase scope,
status, and acceptance criteria.

| Implemented surface | JAX implementation | Static-shape adaptation |
|---|---|---|
| `rasterization` | `jax_gs.rasterization` | Dense projection within the current physical bucket, then fixed-capacity pure-JAX visible packing, intersections, sorting, and compositing on every JAX device |
| RGB/D/ED modes, SH, background, leading batch dimensions and batch cameras | Implemented | Arbitrary leading batches are flattened through `lax.map`; cameras are processed serially and all modes share the pure-JAX compositor |
| current-main renderer configs and render-mode queries | Implemented | MixedBatch and ParallelBatch share one pure-JAX numerical path; ParallelBatch retains the eval3d-only validation boundary |
| hit-distance modes, custom eval3d rays and accumulated normals | Implemented | Fixed tile candidate buffers; normals and hit distance are evaluated from the exact world ray/Gaussian transform |
| current-main extra signals and SH extra signals | Implemented | Rendered with the same compositing weights and returned separately as `info["render_extra_signals"]` |
| `rasterize_to_pixels_eval3d_extra` debug outputs | Implemented | Static full-image `last_ids` and `sample_counts`; optional outputs are represented by `None`, not runtime-varying array tuples |
| sorted intersections → feature/alpha compositing | Implemented in pure JAX | Ordinary JAX autodiff; arbitrary feature count and supported tile sizes use the same readable numerical path |
| classic / antialiased | Implemented | Antialias compensation remains differentiable |
| pinhole / ortho / fisheye | Implemented | Fixed camera variant per compiled call |
| UT / FTheta / distortion / rolling shutter | Implemented | Seven sigma points in `ut_chunk_size` static chunks |
| structured camera sensor kernels, functional APIs, and NNX models/frames | Implemented | Pure-JAX projection/inverse projection, pose interpolation, rolling shutter, and windshield distortion; fixed model/state structure per compiled call; data-container `forward()` entry points retain the upstream contract |
| structured LiDAR sensor kernels, functional APIs, and NNX models/frames | Implemented | Native sensor angle order is `[elevation, azimuth]`; row/column structure is static under JIT; model/frame `forward()` entry points retain the upstream contract |
| current-main sensor tensor helpers, dispatch tables, and model re-exports | Implemented in pure JAX | Device/dtype and quaternion-layout helpers retain upstream keyword names under JAX semantics; LiDAR dispatch and all seven projective operations cover the registered projection/distortion Cartesian product |
| root structured-spinning LiDAR parameters, preprocessing, and model | Implemented | Legacy-compatible angle order is `[azimuth, elevation]`; deterministic host preprocessing builds column maps, CDFs, and tile-element maps |
| root `isect_tiles_lidar` and angular tile helpers | Implemented | Runtime intersections use fixed-capacity `PaddedIntersections` with `valid_count` and `overflow`, including periodic azimuth bounds |
| high-level LiDAR UT/eval3d rasterization | Implemented in pure JAX | Requires `camera_model="lidar"`, UT, and eval3d; sensor dimensions override generic image dimensions and angular tiles resolve exact scan rays |
| root camera wrapper factory and model facade | Implemented in pure JAX | Type-erased pinhole, ortho, fisheye, FTheta, and LiDAR models expose projection, inverse projection, shutter timing, and shutter-pose world transforms |
| root bivariate windshield external distortion | Implemented in pure JAX | Separate forward/inverse triangular polynomials support orders 0–5; high-level rendering requires UT, and eval3d applies inverse distortion while constructing exact rays |
| current-main `geometry.functional` quaternion operators | Implemented in pure JAX | Upstream `xyzw` convention is isolated from existing `wxyz` helpers; exact paired batch shapes, safe zero normalization, JIT, and gradients are covered |
| current-main SE(3), packed-track, and trajectory operators | Implemented in pure JAX | Shepperd matrix branches, duplicate-time lower bounds, endpoint clamping, invalid-range identity no-ops, unordered two-pose spans, extrapolation, and OOB flags are retained |
| current-main `Scene` and NNX `GaussianScene` | Implemented | `nnx.Dict`/`nnx.Param` replace `ParameterDict`; component membership, signal sidecars, topology hooks, and state round-trips remain row-aligned |
| current-main `GaussianInferenceScene` and functional packer | Implemented in pure JAX | Exact planar/QSO/RGB/SH0–3 layouts, FP16 clamping, activation checks, multi-component rules, and release/get behavior; packing is explicitly `stop_gradient` |
| current-main experimental stateless inference rendering | Implemented in pure JAX | Single-pinhole request subset, packed scene decode, float32 RGB/alpha, dispatcher tagging, validation, JIT core, and explicit `stop_gradient` boundary |
| current-main NNX `GaussianInferenceRenderer` | Implemented | Reusable SH-codec state, scene mutation guard, resolution-aware frame cache, lifecycle/context manager, SH-degree override, and float16 RGBT output |
| experimental SH 32B/16B renderer codecs | Implemented in pure JAX | YCoCg percentile scales and quantize/decode numerics are retained without materializing CUDA-only bitstreams; 16B drops higher-order chroma |
| current-main `Stage` scene/render registry | Implemented | Insertion-ordered scene ids and exact `render_fn(splats=..., **kwargs)` dispatch; renderer results pass through unchanged |
| current-main `DeformNetwork`, `DeformationTable`, and `HexPlaneField` | Implemented with Flax NNX and pure JAX | Explicit `nnx.Rngs` is supported with a deterministic compatibility default; zero heads, six-plane layouts, temporal-one initialization, border sampling, JIT, and gradients are retained |
| current-main HexPlane regularizers | Implemented in pure JAX | Spatial/time second differences and temporal L1 operate directly on NNX plane arrays and remain differentiable |
| current-main `DynamicStrategy` | Implemented on the existing fixed-slot strategy | `dynamic_mask` has physical-capacity shape; prune clears flags, split/duplicate children inherit the original parent flag, and Scene-aware topology plus compaction/bucket resize preserve row alignment |
| current-main losses, LiDAR losses, and reductions | Implemented in pure JAX | `l1_loss`/`mse_loss` use the current unreduced contract; masking and scalar reductions are explicit, with JIT-safe fixed-shape masked arithmetic |
| current-main `FusedGaussianLosses` and `has_losses()` | Implemented with an NNX pure-JAX facade | Same four unreduced outputs and gradients; availability is true, but no CUDA-style kernel-fusion performance claim |
| current-main color correction and occlusion regularizers | Implemented in pure JAX | JAX least squares, targeted TV, and max-pool dilation are differentiable; invisible PNG-mask loading remains host I/O |
| eval3d | Implemented in pure JAX | UT supplies conservative image-space candidates; opacity is evaluated from the exact world-ray/Gaussian distance and differentiated by JAX |
| `rasterization_2dgs` | Implemented in pure JAX | Fixed tile/intersection capacities; `gradient_2dgs` remains forward-zero while explicit signed/AbsGrad probes expose the current-main ray-transform densification VJP without changing rendered outputs |
| covariance, quaternion, SH and projection primitives | Implemented | Pure JAX, grad/vmap/jit tested |
| packed `fully_fused_projection` | Implemented | `PaddedProjection` unpacks as the nine public values and adds count/overflow metadata |
| `isect_tiles` | Implemented | `PaddedIntersections` + `valid_count` + `overflow`; normal EWA supports opacity-aware AccuTile, with AABB fallback |
| `isect_offset_encode` | Implemented | Ignores padded tail by valid count |
| `rasterize_to_indices_in_range` | Implemented | Fixed output capacity |
| `accumulate`, `rasterize_to_pixels` | Implemented | Pure-JAX bounded compositing; true AbsGrad is exposed through an explicit zero-valued probe, and padded metadata is required to detect truncation |
| 2DGS low-level projection/accumulate/indices/pixels | Implemented | `PaddedProjection2DGS` and padded hit buffers preserve static shapes; pixel rendering accepts dense and packed layouts |
| `rasterize_to_pixels_eval3d` | Implemented | Consumes padded sorted intersections and evaluates exact world-space responses for pinhole, OpenCV, fisheye, FTheta, rolling-shutter, and structured LiDAR scan rays |
| current-main `build_sparse_tile_layout` | Implemented | Active tiles use a padded prefix plus counts/overflow; with x64 disabled, masks use uint32 words and cumsum/pixel-map use int32 |
| current-main `isect_tiles_sparse` | Implemented | Complete Cartesian capacity by default; optional smaller capacity reports exact required count and overflow |
| current-main `rasterize_to_pixels_sparse` | Implemented | Decodes the static uint32 tile layout, scans the complete retained tile range, and supports ordinary gradients plus true AbsGrad through an explicit zero-valued probe |
| current-main dense/sparse contributor and top-contributor queries | Implemented | Runtime maximum contributor lists become `PaddedContributors` with per-pixel counts, required capacity, and overflow metadata |
| current-main `strategy` package, lifecycle, `DefaultStrategy`, and `MCMCStrategy` | Integrated on fixed-capacity state | Public hooks consume exact signed `<key>_gradient` or true `<key>_absgrad`; same-parent duplicate+split and Scene/Dynamic topology preserve alignment; scheduled MCMC capacity overflow atomically skips optimizer/model/stats/refine/noise commits |
| `compute_relocation` | Implemented | Pure-JAX Equation 9 primitive; fixed-slot MCMC applies it to donors, relocated slots, and births |
| current-main `optimizers` package and `SelectiveAdam` | Implemented with a JAX call-surface adaptation | Gaussian groups use current-main global-batch LR/epsilon/beta scaling and means-only scene-scale multiplication; sparse-gradient row selection uses bias-corrected Optax Adam, while `visible_adam`/`SelectiveAdam` uses current-main's uncorrected moments; the wrapper exposes `update(model, grads, visible_mask)` plus a step counter instead of PyTorch's post-autograd `step(visibility)` |
| current-main `init_utils` and point-cloud scale initialization | Implemented in pure JAX | Multi-frame depth unprojection plus chunked KNN; `ModelConfig.initial_scale` defaults to 1.0 and active point-cloud scales are RMS distance to up to three nearest neighbours times that multiplier |
| current-main trainer P0/P1 profiles and scalar regularization | Implemented trainer slice | Full-image is the default and carries rectangular H/W through rendering, stats, capacities, and memory checks; explicit square patches remain supported; Grain training uses `shuffle → repeat → batch`; 2DGS/MCMC profiles and active-only 3D opacity/scale regularizers are wired through config, CLI, and metrics |
| current-main COLMAP trainer world normalization | Implemented as deterministic host preprocessing | Camera up alignment, focus/median centering and scaling, point-cloud PCA, and the upside-down correction match the pinned example; trainer-generated format-v6 checkpoints persist the exact 4×4 transform and scene scale for resume/render, while the differentiable runtime remains pure JAX |
| `training.pose.CameraOptModule` and `training.appearance.AppearanceOptModule` | Integrated in the single-process trainer | Both have config/CLI ownership, independent NNX/Optax state, real 3DGS/2DGS train-step wiring, shared atomic overflow replay, and exact checkpoint/resume; appearance replaces SH parameters with per-Gaussian 32D features plus base color logits and emits direct RGB |
| current-main `TwoStageScheduler` | Implemented | Pure-Python/Optax-compatible two-stage factor and step object with upstream boundary/default behavior |
| current-main `PngCompression` and `sort_splats` import hierarchy | Implemented | `compress_dir=` and the `compress() -> None` contract match upstream; the `meta.json`/PNG/SH-codebook schema uses deterministic host xyz sorting and K-means; PLAS optimization is not reproduced; `GaussianModel` and explicit `image_width` retain the byte-exact transport form |
| `export_splats` and `load_ply_to_splats` | Implemented | Upstream bytes-returning `ply`, `splat`, and Supersplat `ply_compressed` formats plus float32 JAX PLY loading; CLI export canonically bakes appearance with zero camera embedding/direction and the configured full direction-basis degree into degree-zero SH before using these formats; raw features/colors are rejected by the generic exporter |
| `utils` geometry/transforms/deprecated PLY helper | Implemented | Pure JAX for differentiable math, including upstream `x`/`y`/`quat` keyword signatures; host NumPy only for PLY serialization |
| current-main `trace` and `profile` | Implemented for JAX | `jax.profiler.TraceAnnotation`, synchronized timing with current-main dunder keywords/returns, environment-driven input capture, override parsing, and forward/gradient replay; capture payloads use pickle/NumPy and `load_capture` |
| current-main distributed renderer helpers, root routing, `cli`, and fixed-topology training | Renderer plus first device-training slice implemented | `rasterization(distributed=True)` supports an exact single-rank path and named-axis equal-capacity padded multi-shard gather; `make_distributed_train_step()` supports dense SH pinhole 3DGS with fixed topology, global overflow atomics, owner-correct visibility and signed densification statistics, and current-main Gaussian gradient/Adam scaling. Host data/topology/checkpoint/eval and multi-process orchestration remain open |
| current-main capability queries | Implemented | Report pure-JAX subsystem availability rather than CUDA compile flags; `has_camera_wrappers()` and `has_losses()` are true for their completed facades |

## Deliberate behavioral boundaries and pending acceptance

- PyTorch's mutable post-backward `.absgrad` attribute is replaced by an
  independent zero-valued JAX probe that is a strict forward no-op. Low-level
  dense/sparse pixels and high-level 3DGS use projected-means cotangents; 2DGS
  keeps `gradient_2dgs` forward-zero and exposes its distinct ray-transform
  densification VJP. In both cases AbsGrad takes componentwise absolute local
  pixel contributions before reduction. Training and the Default public hook
  consume the explicit result; 3D eval3d and distributed AbsGrad remain
  unsupported combinations.
- `sparse_grad=True` requires unbatched `packed=True`. The 3D path additionally
  rejects distributed, UT, eval3d, and FTheta because FTheta implicitly uses UT.
  Renderer parameter gradients, model arrays, and optimizer moments retain
  dense fixed-bucket storage. Sparse-gradient row selection uses bias-corrected
  Adam, while `visible_adam` matches uncorrected current-main SelectiveAdam;
  neither path claims sparse/COO memory savings.
- 2DGS packed sparse training is implemented. Packed 2DGS training with the
  `reference` backend is rejected because that path does not expose the needed
  cross projection/intersection metadata. `visible_adam` is a 3DGS training
  option and is intentionally rejected for 2DGS.
- MCMC 3DGS training supports UT and Eval3D by omitting screen densification
  statistics that MCMC does not consume. Its scheduled capacity preflight makes
  overflow a whole-commit no-op, including Gaussian/pose/appearance optimizer
  steps, model, pose/appearance state, strategy statistics, refine, and position
  noise; replay after capacity growth preserves that alignment. Default+Eval3D
  remains unsupported.
- Default full-image training is implemented, including non-square images;
  renderer calls, densification statistics, tile/intersection capacities, and
  initial/growth memory preflight use independent height and width. All images
  in the full-image training split must have one shared shape. Explicit square
  patches remain supported and are required for mixed-resolution training.
  Camera-pose and appearance config/CLI, optimizer ownership, 3DGS/2DGS
  train-step invocation, atomic overflow replay, and checkpoint/resume are
  integrated in the single-process trainer. Appearance evaluation without a
  training image id uses the zero embedding and the configured full direction
  basis degree.
- Gaussian means LR consumes the final training-coordinate scene scale. Fresh
  training uses current-main's focus/median camera normalization, point-cloud
  principal-axis alignment, upside-down correction, and `1.1 * global_scale`
  extent. Trainer-generated format-v6 checkpoints persist the exact transform
  and scale, so resume and CLI rendering do not recompute PCA. Any checkpoint
  without a scene component, including v1-v5 and generic v6 saves, deliberately
  retains the legacy camera-center mean/max-extent transform that produced its
  Gaussian coordinates.
- Root distributed rendering supports single rank and a bound named axis with
  equal padded shards. The separate `make_distributed_train_step()` now covers
  a fixed-topology, dense-SH, pinhole 3DGS device step inside `nnx.pmap`: each
  rank owns a Gaussian shard and local camera batch, gather transpose sums the
  rank-local photometric gradients, visibility is reduced in global Gaussian
  coordinates before owner slicing, and signed screen statistics preserve the
  local-camera dimension until each visible gradient has been normalized and
  reduced by `sum/sum/max` into its Gaussian owner. Overflow or rank
  step/SH-degree mismatch atomically skips model, optimizer, and statistics on
  every shard. The optimizer is required to carry matching
  batch/world/scene-scale/config metadata. This slice does not include a
  launcher, Gaussian leading batch, pose/appearance, dynamic topology, host
  data sharding, checkpoint/eval, or complete multi-process orchestration.
  Rank-local photometric/active metrics remain local while
  overflow/intersection diagnostics are global. Startup belongs to
  `jax.distributed`, MPI, Slurm, or another external launcher, and unified
  `train()` still rejects `jax.process_count() != 1`.
- Leading-batch 3DGS and 2DGS packed metadata, public signed/AbsGrad strategy
  hooks, Scene/Dynamic fixed-slot topology, and current-main COLMAP
  normalization are implemented. The 2026-07-29 distributed signed-statistics
  forced-CPU acceptance reported `839 passed, 1 skipped, 38 deselected`; five
  fresh-process resource-heavy groups then passed `19+5+9+3+2=38` cases, for
  877 passing cases in total.
  The only skip is the unavailable optional local Mip-NeRF360 stump dataset.
  Two full GPU-safe-script attempts passed 21 and 32 isolated CUDA cases before
  transient `libuv-worker` and kernel-journal D-state preflights stopped them.
  No risk override was used and no complete current-slice GPU pass is claimed.

## Rasterizer backend contract

`RasterizationConfig.backend` defaults to `"auto"`:

- `auto`, `jax`, and `intersections` select the fixed-capacity pure-JAX
  sorted-intersections compositor on every JAX device.
- `reference` forces the original JAX tile-by-capacity scan and is retained for
  debugging and numerical comparisons, not performance.

Visible packing, AABB rank mapping, opacity-aware AccuTile count/direct emit,
tuple sorting, offset construction, forward compositing, and gradients are all
ordinary JAX programs. `intersection_backend="auto"` and `"jax"` are equivalent;
so are `sort_backend="auto"` and `"jax"`. Sorting operates on the runtime
`valid_count` prefix within a static-capacity buffer; changing capacity remains
a JIT-shape change.

`intersection_mode="auto"` selects opacity-aware AccuTile for normal EWA 3DGS;
UT, nonlinear projection, and 2DGS use conservative AABB candidates. AccuTile
is the conservative ellipse/tile intersection algorithm used by current gsplat
and SpeedySplat; it is unrelated to NVIDIA cuTile and is implemented here in
pure JAX.

For existing configuration files, the removed `"cutile"` values in `backend`,
`intersection_backend`, and `sort_backend` are accepted only as deprecated
aliases and normalized to `"jax"`. Older `cuda_ffi` and `pallas` values are also
migrated while loading `TrainConfig`. These aliases do not load another
runtime, and the CLI exposes only the current pure-JAX choices.

Two overflow conditions have different meanings:

- `intersection_overflow=True` means the global padded intersection list was
  truncated. The output is not suitable for training or final evaluation.
- `visible_overflow=True` means the projected visible population exceeded the
  same fixed bound before SH/intersection work. It is folded into
  `intersection_overflow` so training cannot consume truncated gradients.
- `max_gaussians_per_tile` controls a temporary pure-JAX compositor chunk.
  Exceeding it sets `candidate_limit_exceeded` for diagnostics but does not
  truncate. `tile_overflow=True` is reserved for an inconsistent/incomplete
  input intersection buffer whose physical storage cannot cover a tile list.

Training uses an NNX conditional commit plus a device-resident sticky flag: a
truncating overflow skips model, optimizer, and strategy-state updates on that
step and every subsequent dispatched step. The host checks the flag at the
union of logging, refine, reset, checkpoint, evaluation, and final boundaries,
always before any host-controlled mutation or save. CLI rendering performs a
memory preflight and `--strict-overflow` rejects truncated output before it is
written.

## Deliberate semantic differences

JAX cannot return a runtime-length `nnz` array from a normal jitted function.
Consequently, packed projection/intersection/pixel-hit APIs use fixed buffers.
The first `valid_count` entries are meaningful; `overflow=True` means the
configured capacity was insufficient. Cropping such buffers to dynamic arrays
belongs outside JIT.

The two upstream LiDAR surfaces intentionally retain their distinct angular
layouts: `jax_gs.sensors` uses `[elevation, azimuth]`, while the root
legacy-compatible API uses `[azimuth, elevation]`. Callers crossing that
boundary must swap axes explicitly. LiDAR CDF and tile-element maps are
deterministic host-preprocessed state; differentiable projection,
intersections, scan-pose rays, and eval3d compositing are pure JAX. Angular
intersection lists follow the same padded-capacity contract as image tiles.

Root and structured-sensor windshield parameters also remain distinct. The
root renderer uses four independent triangular arrays of 1, 3, 6, 10, 15, or
21 coefficients (orders 0–5). The structured sensor kernel keeps its packed
42-value horizontal-degree-2/vertical-degree-4 layout. Both are pure JAX, but
callers must convert explicitly rather than relying on a lossy implicit
adapter.

The current-main geometry package uses `xyzw` quaternions. Existing root math
and structured sensor pose helpers use `wxyz`; callers crossing those package
boundaries must reorder components explicitly. PyTorch's required
`quat_identity(device=...)` keyword is optional in JAX because JAX has a
defined default device, while an explicit `jax.Device` is still accepted.
PyTorch autograd wrapper classes and the CUDA backend loader are not reproduced:
the matching functional operators are directly differentiable JAX programs.
The root `_lazy_backend.py` and `cuda` extension-wrapper package, plus native
geometry, scene, sensor, and experimental `_backend.py`/`cuda.build` modules,
are PyTorch/CUDA implementation details and are intentionally absent rather
than exposed as non-functional compatibility shims.
With the project-default x64-disabled configuration, packed track indices and
integer time calculations use int32/float32. Enable JAX x64 before array
creation when exact int64 timestamp spans are part of the application contract.

Scene component mutation is an initialization/topology boundary, not a jitted
runtime operation: appending or filtering components changes row shapes just as
the upstream `ParameterDict` implementation does. `nnx.Dict` is the supported
live trainable container; plain mappings are accepted and their arrays are
wrapped in `nnx.Param`. JAX arrays do not carry PyTorch's `requires_grad` bit,
so inference construction has no detach warning to reproduce; instead, packed
outputs have an explicit `stop_gradient` boundary. CUDA contiguity checks are
also omitted because JAX owns physical layouts, while shapes, dtypes, packing,
and value validation remain enforced.

Current-main compression sorting normally delegates to optional PLAS. The JAX
port instead uses a deterministic host xyz ordering: decoded Gaussian rows and
the directory schema remain compatible, but compressed file sizes need not
match PLAS. PLY parsing lazily imports `plyfile`; it is a development dependency
for contract tests rather than a core JAX runtime dependency.

Experimental inference uses immutable JAX arrays. `out=RenderReturn` therefore
preserves the Python container but replaces its frame/alpha arrays instead of
writing into an existing tensor allocation. There is no global `no_grad` mode;
the packed operator uses `stop_gradient`, so autodiff observes the same
inference-only boundary. Stateful RGBT caching is an NNX cache, while actual
allocation reuse is left to JAX compilation and optional caller-side donation.
The 32B/16B codecs perform the same lossy numerical transform without emitting
the CUDA implementation's transient packed bitstream.

Sparse tile layouts follow the same rule. `active_tiles`, per-active-tile
bitmasks/cumsums, and sparse intersections use padded capacities with valid and
required counts. Because this project deliberately leaves JAX x64 disabled,
`tile_pixel_mask` uses 32-bit raster-order words instead of gsplat's 64-bit
words, while `tile_pixel_cumsum` and `pixel_map` use int32 instead of int64.
Public sparse consumers decode this representation directly; callers must not
reinterpret the words as upstream CUDA buffers.

`ModelConfig.capacity` defaults to a logical maximum of 1,000,000 slots; it is
not fully preallocated at startup and may be configured up to the enforced
10,000,000-slot limit. The physical parameter, optimizer and
strategy-state bucket ladder starts at 65,536 slots by default and doubles as
needed, capped by that logical maximum. The actual initial bucket is the
smallest one covering the initial population; an exact read-only refinement
plan decides whether the next refinement needs a larger bucket.
Densification, pruning and relocation update `active_mask` and slot contents
without changing shapes inside a bucket, so an active-count change does not
cause recompilation. Full-state compaction is deferred until checkpoint save,
rather than being performed after every refinement. Crossing a bucket boundary
grows the complete training state; each affected jitted
function compiles once for the new static shape and then reuses it. This is a
shape-stability mechanism, not sparse execution: inactive storage inside the
current bucket still contributes to parameter/optimizer memory and dense
projection work. Projection output is subsequently packed to a bound no larger
than the intersection capacity, so SH evaluation, sorting and the compositor
scale with retained visibility rather than the full storage bucket.

Current-main strategy hooks cannot read mutable `.grad`/`.absgrad` fields from
immutable JAX arrays. `step_post_backward` therefore receives either the
configured `<key>_gradient` or, when `absgrad=True`, `<key>_absgrad`, in exact
dense `[C, N, 2]` or padded packed `[P, 2]` form. The earlier capacity-shaped
signed gradient remains a legacy compatibility form. Same-parent
duplicate+split plans emit the duplicate before the split and operate from the
original parent snapshot; the retained original and split child receive
independent split samples and revised opacity. Topology operations fail before
mutation when the physical bucket has insufficient inactive slots; that
overflow atomically preserves the model, optimizer, statistics, Scene/Dynamic
sidecars, and MCMC noise. Scene transactions and training-state
resize/active-prefix compaction apply the same
row mapping to splats, component indices, signals, and dynamic masks.
`DefaultStrategy` and `MCMCStrategy` retain current-main positional and named
constructor fields. A local `StrategyConfig` is a trailing keyword-only
extension; a lone positional config is still recognized for existing jax-gs
callers, and an explicit config is the source of truth. Scheduled MCMC
refinement performs a device-side
capacity preflight before committing optimizer/model state. On overflow the
optimizer step, model, statistics, refine, and position noise remain unchanged;
UT/Eval3D MCMC training skips screen statistics that the strategy does not use.

Strategy initialization takes an explicit physical capacity because JAX state
is allocated at a fixed shape, and its lifecycle hooks accept explicit
gradients instead of reading mutable PyTorch `.grad` fields. The local
`SelectiveAdam` constructor takes a `GaussianModel` plus `OptimizerConfig` so it
can allocate dense NNX/Optax state. Its `.step` is a readable counter, while
`update(model, grads, visible_mask=None)` performs the row-masked update and
`reset_slots(mask)` supports topology changes; this intentionally replaces
PyTorch/source's post-autograd mutating `step(visibility)` call. These are the
explicit source-call adaptations retained by the current-main audit, not a
claim of an exact PyTorch optimizer-object interface.

`multi_frame_depth_unprojection` and `knn_scale_init` deliberately run at an
eager initialization boundary because their output population is data
dependent. Point-cloud construction initializes each active isotropic scale as
the RMS distance to its three nearest neighbours times `initial_scale`, whose
`ModelConfig` default is 1.0; inputs with fewer than four points use all
available neighbours, and the single-point fallback is `initial_scale`.
Bounded sampling accepts an explicit JAX PRNG key; the compatibility default
uses a fixed key for reproducible initialization. A fresh 2DGS profile selects
near/far 0.2/200, `prune_opacity=0.05`, and
`key_for_gradient="gradient_2dgs"`. A fresh 3DGS MCMC profile maps upstream
`init_opa=0.5`/`init_scale=0.1` to `initial_opacity`/`initial_scale` and enables
0.01 opacity and scale regularization. Loading a config/checkpoint preserves
its stored profile except for explicit CLI overrides.

Training defaults to `patch_size=None`, meaning full-image rendering. The scene
supplies a static `(height, width)` shared by all training images; rectangular
dimensions pass unchanged through the renderer, screen-stat normalization,
tile-grid/intersection capacity, initial memory check, and bucket-growth memory
check. A scalar `patch_size` continues to select a random square patch and is
the documented fallback for mixed-resolution splits. Standalone full-image
`estimate-memory` calls require explicit `image_height` and `image_width`; patch
configs derive both from `patch_size`. The Grain training pipeline applies
`shuffle → repeat → batch`, so epoch tails continue into the next batch, small
scenes do not stall under `drop_remainder`, and resume fast-forward is
deterministic.

The 3DGS scalar regularizers consume the model's raw opacity logits and raw
log-scales, apply sigmoid and exp respectively, and average only rows selected
by `active_mask`. Their weighted contributions are configured through
`opacity_reg`/`scale_reg`, exposed as `--opacity-reg`/`--scale-reg`, and reported
as `opacity_reg_loss`/`scale_reg_loss` metrics. Non-negative weights are
required, and 2DGS rejects either weight when nonzero because its normal and
distortion losses are the supported model-specific regularizers.

`jax_gs.training.pose` exposes `CameraOptModule` and
`rotation_6d_to_matrix`. Each image owns a nine-value embedding (translation
plus a 6D row-rotation delta). Construction retains upstream's random embedding
initialization, while the explicit `zero_init()` call used by the trainer makes
the transform an identity. The resulting transform is right-multiplied onto
camera-to-world matrices, so translations remain in the local camera frame.
`TrainConfig` and the CLI expose `pose_opt`, `pose_opt_lr`, `pose_opt_reg`, and
`pose_noise`. The real 3DGS and 2DGS train steps apply a fixed stop-gradient
noise transform first and the trainable local adjustment second. Pose owns an
independent Adam whose initial learning rate is batch-size scaled and decays
exponentially to one percent over training; `pose_opt_reg` is coupled L2. The
same overflow predicate atomically skips Gaussian and pose updates, including
the replayed suffix after capacity growth. Orbax saves the pose module and
optimizer; its manifest records camera count and exact image-name ordering,
which resume validates before restoring state. Pose resume also requires the
same total `steps`, preventing an implicit jump from rebuilding the exponential
learning-rate horizon.

`jax_gs.training.appearance.AppearanceOptModule` is integrated into the unified
3DGS/2DGS trainer. `TrainConfig` and the CLI expose `app_opt`,
`app_embed_dim`, `app_opt_lr`, and `app_opt_reg`, with current-main defaults
`False`, 16, `1e-3`, and `1e-6`. The Gaussian color representation is strictly
exclusive: normal models own `sh0`/`sh_rest`, while appearance models own fixed
32D `features` plus three base color logits and have no SH parameter leaves.
Point-cloud appearance features are random on active rows and zero-padded to
the physical bucket. RGB is converted to a finite clipped inverse sigmoid; this
deliberately avoids the infinities that upstream's direct `torch.logit(rgb)` can
produce for exact zero or one.

The MLP combines the split-local training image embedding, each Gaussian
feature, and SH direction bases, and returns `[C, N, 3]` color-logit
corrections. Its final layer starts at zero. Training computes directions from
the pose-adjusted camera origins, evaluates
`sigmoid(colors + correction)`, and passes direct RGB to both real 3DGS and
2DGS rasterizers with `sh_degree=None`. The Gaussian `features` and `colors`
have separate parameter labels but both use the configured `sh0_lr`. The
appearance optimizer uses Adam at `app_opt_lr * sqrt(batch_size) * 10` for the
embedding with coupled `app_opt_reg`, and at
`app_opt_lr * sqrt(batch_size)` without decay for the color head. Gaussian,
pose, and appearance updates share the same overflow predicate and replay
boundary.

The feature/color Gaussian groups intentionally inherit this repository's
existing `sh0_lr` semantics. All Gaussian optimizer groups now apply
current-main's effective-batch scaling: for
`B=batch_size*world_size`, learning rates use `sqrt(B)`, epsilon uses its
inverse, and both betas use the upstream linear rule. The means group alone is
also multiplied by the training-coordinate scene scale. Effective batches over
ten are rejected because the upstream beta1 formula becomes invalid. The
appearance module optimizer retains its separate scaling described above.

Duplicate, split, MCMC relocate/birth, bucket resize, and active-prefix compact
copy or reorder appearance rows and their Gaussian optimizer moments exactly as
they do the SH representation. Checkpoint format v6 retains v5's model color
mode, appearance feature dimension, camera count, exact training image-name
ordering, module, and optimizer state, and supports an optional scene component;
trainer-generated saves include the exact transform and scene scale. Strict
resume validates that manifest plus the appearance config,
SH degree, batch size, normalization mode, and global scale before restoring;
model-only restore retains the Gaussian features/colors while intentionally
omitting the optional appearance module state. Formats v1-v5 remain readable,
including v4 pose state and v5 appearance state.

Unlike the pinned upstream example's incomplete appearance restore path, the
local v5/v6 contract deliberately restores both appearance module and optimizer
state so uninterrupted and resumed training share the same state trajectory.

Held-out evaluation, CLI rendering, and other calls without a training image id
use a zero camera embedding; scheduled evaluation passes the configured full
direction-basis degree. CLI export performs the current-main canonical bake
with zero embedding, zero direction, and that full degree, converts the result
to degree-zero SH, and discards raw features/colors from the exported model.
This creates one portable canonical appearance and does not encode per-image
embedding variation.

`sparse_grad=True` follows current-main's unbatched `packed=True` public
contract and rejects distributed, UT, Eval3D, and the implicit FTheta UT path.
Projection/rasterizer parameter gradients and optimizer storage stay dense over
the physical Gaussian bucket, so metadata continues to report
`sparse_grad_is_dense=True`. The sparse-training adapter applies normal
bias-corrected Optax Adam only to the packed visible-ID union; `visible_adam`
instead uses current-main SelectiveAdam's uncorrected first/second moments.
Hidden rows and moments remain unchanged while global counters advance.

True AbsGrad is available when the caller supplies an independent zero-valued
probe. For 3DGS the signed means gradient remains unchanged and the AbsGrad
probe sums componentwise absolute per-pixel projected-means contributions. For
2DGS, the public `gradient_2dgs` value remains zero in the forward pass; signed
and AbsGrad probes expose the separate VJP defined by the z entries of the first
two ray-transform rows scaled by `w_M.z`, with absolute values applied before
the pixel sum. Reference/intersection, dense/packed, leading-batch, and jitted
paths retain the same forward values. Metadata reports whether the functional
probe was enabled rather than pretending that an immutable JAX array acquired a
mutable `.absgrad` field.

The low-level 2DGS projection returns `PaddedProjection2DGS` when
`packed=True`. It iterates as current-main's nine public packed values,
including int32 batch-camera `indptr`, and adds `valid_count`/`overflow`; the
low-level pixel renderer consumes this fixed buffer directly. It is
intentionally static rather than runtime-ragged.

Root `rasterization(distributed=True)` routes to the distributed implementation.
The single-rank case returns the local forward values, metadata, and gradients
without a collective. A bound named axis gathers equal fixed-capacity padded
shards and remains differentiable; the readable implementation currently
replicates the gathered scene on each rank and does not accept a Gaussian
leading batch. `distributed.cli` does not reproduce
`torch.multiprocessing.spawn`: multi-host or multi-process jobs must be launched
before entry through `jax.distributed`, MPI, Slurm, or another process manager,
and local multi-device mapping belongs inside `pmap`/`shard_map`.

`training.make_distributed_train_step()` builds on that renderer contract for
the first readable device-side training slice. It supports dense SH pinhole
3DGS with fixed topology and equal physical shard capacities. Gaussian
photometric gradients are the sum of rank-local mean losses, matching current
main; local Gaussian regularizers remain owner-local. Global visibility is
reduced before owner slicing, and both new and sticky overflow state are
collectively synchronized. An optimizer step or SH-degree mismatch also makes
the whole mapped update a no-op. Optimizers must be built with matching local
batch, world size, scene scale, and optimizer config, and the train and
optimizer schedule horizons must agree. A two-virtual-CPU `nnx.pmap` smoke test
covers the real collective boundary. Pose/appearance, screen densification
statistics, topology changes, distributed checkpoints/eval, host input
sharding, launch, and performance specialization remain separate later slices;
unified `train()` therefore remains single-process.

Profile captures keep current-main's default `.pt` suffix but contain standard
Python pickle payloads with host NumPy arrays, not `torch.save` archives. They
must be restored with `jax_gs.profile.load_capture`. Replay and tracing use JAX
forward/gradient functions and JAX profiler annotations; CUDA/PyTorch profiler
presets and native kernel-family expectations are not claimed.

The high-level `with_eval3d=True` route uses nonlinear UT projection only to
build conservative tile candidates. Per-pixel opacity uses gsplat's
world-space ray distance equation. `info["eval3d_world_space"]` identifies this
route; the legacy `eval3d_ewa_approximation` key remains present and false.

The compositor follows gsplat's front-to-back alpha equations. Ordinary output
gradients use direct JAX autodiff; true AbsGrad uses only the explicit probe
boundary described above. Device-specific XLA lowering and the upstream CUDA
reduction order can still produce small floating-point differences; the
implementation promises tolerance-based agreement, not bitwise identity.

## Measured performance status

No historical specialized-backend timing is presented as a measurement of the
current pure-JAX renderer. The migration currently prioritizes API parity,
numerical behavior, implemented training slices, and readability. Local benchmarks
remain available for regression work, but a publishable comparison must be
rerun on the current code and report device, JAX/XLA version, physical and
intersection capacities, image shape, warmup policy, and whether gradients are
included. None of the compatibility claims in this document imply performance
equivalence with upstream gsplat.

## Memory-safety contract

The package defaults `XLA_PYTHON_CLIENT_PREALLOCATE` to `false` when the user has
not set it. It also defaults XLA GPU compilation to one worker;
this trades slower cold compilation for lower host-memory pressure and avoids
the observed CUDA 13.3 `ptxas` multi-worker crashes. Explicit user environment
settings take precedence. Training estimates the actual initial physical
bucket, scales dense projection/UT estimates by camera batch size, derives the
rectangular tile grid and intersection workspace from independent training H/W,
and before bucket growth checks the transient peak where old and new training
states coexist. Standalone full-image estimation requires explicit H/W, while
the train loop obtains them from scene metadata. Training-time evaluation and
full-resolution CLI rendering include live allocations in their 70% preflight;
training also stops above 85% runtime usage. Default rasterization limits are
`tile_size=16`, `max_gaussians_per_tile=512`, and `tile_batch_size=4`.

Host data/KD-tree concurrency defaults to four workers. Larger positive values
are accepted; values above the current CPU affinity emit an oversubscription
warning. Grain read-ahead is bounded to eight elements. GPU code generation
stays single-worker independently of data loading. Direct initialization
from a ten-million-point cloud can still have a large host KD-tree/staging peak;
bucket growth from a smaller initial cloud is the safer normal path.

The estimates are safeguards rather than exact allocator predictions. Large
`max_intersections`, image resolutions, tile batches, UT chunks or compiler
temporaries can still be expensive. The benchmark therefore disables backward
by default and applies additional small-shape limits unless `--allow-unsafe` is
explicitly requested.
