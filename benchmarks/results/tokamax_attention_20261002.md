# Tokamax attention strategies applied to packed rasterization

2026-10-02. These are correctness-validated **experimental patches**, retained in
[the combined record](tokamax_attention_20261002.json). They are not enabled in
production. The GPU is occupied by another training job, and the user confirmed
that exclusive access is unavailable. No new kernel, complete-update or full
training timing result is reported.

## What transfers from the attention implementation

The source review uses [Tokamax commit
47d3d663](https://github.com/openxla/tokamax/tree/47d3d6632226298a4cf017c3a0bb997c27d1f2b5),
dated 2026-10-01, together with the current official JAX and Flax documentation.
Tokamax is not added as a dependency.

- Its [SM100 forward kernel](https://github.com/openxla/tokamax/blob/47d3d6632226298a4cf017c3a0bb997c27d1f2b5/tokamax/_src/ops/attention/pallas_mosaic_gpu_kernel_sm100.py)
  bounds the work using valid key ranges and separates interior iterations from
  causal boundary iterations. The raster prototypes use cached pixel termination
  indices and forward fragment validity to bypass unnecessary group work.
- Its [SM90 attention pullback](https://github.com/openxla/tokamax/blob/47d3d6632226298a4cf017c3a0bb997c27d1f2b5/tokamax/_src/ops/attention/pallas_mosaic_gpu_vjp_kernel_sm90.py)
  keeps normalization reciprocals and loop-invariant masks outside the pipeline.
  Raster gradient normalization already computes its reciprocal once per tile;
  reverse transmittance depends on each Gaussian alpha and cannot use that
  particular transformation.
- [JAX's pipeline guide](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)
  explains how transfer stages overlap computation, and why asynchronous readers
  must finish before a shared buffer is reused. The current rasterizer already
  stages 32 records and loads the next batch into registers before compositing.
  Adding stages without measurement would increase register/shared-memory costs.
- Attention split-K combines partial softmax results. Splitting a raster depth
  chain would require transmittance and accumulated-color boundary states;
  half2 rounding and early termination prevent assuming the same equivalence.
  These prototypes preserve the original pixel and gradient accumulation order.

These are applications of the attention strategies to the raster algorithm,
not claims that Tokamax benchmarks predict raster performance. Its current
[Mosaic attention dispatch](https://github.com/openxla/tokamax/blob/47d3d6632226298a4cf017c3a0bb997c27d1f2b5/tokamax/_src/ops/attention/pallas_mosaic_gpu.py)
selects SM90/SM100 kernels; this machine's RTX 5090 uses SM120. The prototypes
continue using CuTe.

## Prototypes and correctness

The first prototype computes a warp maximum last index for each 64-pixel group
once. It skips exponentials and gradient work beyond that boundary while still
advancing the original quadratic recurrence. All 26 contribution/staging cases
pass. Its patch is stored as `candidates.range_bounds.patch` in the JSON.

The second prototype records one contribution bit per pair **and pixel group**.
A warp OR collects all group flags once in forward. Backward stages pairs in
the union, visits them in the original reverse order, and processes only groups
whose bits are set. A set bit certifies that forward had a valid fragment before
that pixel's last index, so backward also eliminates its repeated per-group warp
validity votes. Individual pixel masks and the original half2 accumulation and
squared partial-gradient statistic remain intact.

Tile `t`, 32-pair batch `b`, group `i` uses word
`(tile_offsets[t] // 32 + t + b) * G + i`, where `G = tile pixels // 64`.
Only batches before `ceil(backward_work[t] / 32)` are initialized. The final
batch is masked at the maximum last index. Pure RGB rendering still compiles
out contribution recording. With 8M pair capacity and 8,034 tiles, 8x16 tiles
increase bitmap allocation from **1,032,136 to 2,064,272 bytes**.

Verification for the group prototype includes:

- 24 independent pixel-oracle cases: all four tile shapes, statistics on/off,
  alternating groups and holes, empty/opaque cases, 31/32/33/65-pair spans,
  partial image tiles, poisoned unused words/bits, and original backward
  gradient comparisons at `rtol=1e-5, atol=1e-8`.
- 62 existing parameter-pullback, cross-batch VJP and edge-support regressions,
  plus 24 existing contribution/poisoned-cache cases adapted to the new layout.
  The final group candidate passes 110 pytest cases in these groups.
- Six NNX training comparisons: Optax/Muon/CuTe with SH0/statistics off and
  SH3/statistics on, eight consecutive updates starting at step 17, nonzero
  incoming statistics, and a custom learning-rate horizon. All 138 arrays pass
  `rtol=2e-5, atol=2e-6`; maximum relative L2 difference is 2.32e-8. All 15
  parameter/Adam buffers retain their addresses and each variant caches one JIT
  signature. Incoming overflow and peak-count state are preserved.
- CUDA memcheck: all 24 oracle cases pass, with zero device errors. As in the
  preceding contribution-pruning validation, `--report-api-errors no` suppresses
  unrelated symbol-query API diagnostics; device instrumentation has no filters.

## Real-scene work counts

Four bicycle `images_4` views use the ordinary COLMAP initialization: 54,275
points padded to 54,400 slots, SH3, 8x16 tiles, 128-slot clusters and an 8M pair
arena. This is an **untrained model**, not the previous million-point checkpoint.
Each view is checked with statistics both off and on. RGB, final transmittance,
last indices, work counts and visible packed parameters are bitwise equal.
Packed records for invisible Gaussians are undefined and are excluded from that
comparison. The group bitmap's union matches the original bitmap.

| View | Pair count | Original exponent groups | Group prototype | Fewer exponent groups |
| --- | ---: | ---: | ---: | ---: |
| 0 | 792,200 | 1,538,850 | 1,495,680 | 2.81% |
| 48 | 738,111 | 1,434,102 | 1,393,476 | 2.83% |
| 96 | 712,221 | 1,383,564 | 1,345,731 | 2.73% |
| 144 | 522,563 | 1,012,344 | 981,049 | 3.09% |

An exponent group is one half2 exponential instruction executed by each of the
32 lanes. These counts describe algorithmic work, **not a speedup percentage**.
Field-gradient relative L2 differences are at most 1.844e-6; forward-statistic
and squared-gradient differences are below 1e-6. Floating-point atomics retain
their existing nondeterminism. Full training and held-out quality are not tested
for this candidate.

The extra bitmap storage, forward flag bookkeeping and control flow may offset
the saved calculations. Exclusive measurements must compare both raster kernels
and the complete donated NNX update, followed by alternating full-training runs
and held-out evaluation, before selecting a production implementation.

## JAX / NNX review and reproduction

The current [NNX transform documentation](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html)
and [`jit_partial` source](https://flax.readthedocs.io/en/latest/_modules/flax/nnx/transforms/compilation.html)
confirm the project's existing fixed-state tree-mode binding: flatten the state
structure once, retain Variable identities, update their array values, and donate
buffers. [JAX's CuTe guide](https://docs.jax.dev/en/latest/401/cute-dsl.html)
documents the existing `cutlass_call`/static-shape integration. These paths need
no additional refactor. Timing must wait for GPU completion, following
[JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html).

The JSON stores both unmerged patches, baseline/candidate source hashes,
standalone oracle, real-scene and NNX-check scripts, comparison results and
checker summaries. Reconstruct the group experiment from the recorded baseline:

```bash
work_dir=$(mktemp -d /tmp/jaxgs-tokamax-replay-XXXXXX)
mkdir -p "$work_dir/baseline"
git archive 64d2b1aaf7dec436ce7384c980498c0d4d8c3c6f src tests | tar -x -C "$work_dir/baseline"
.venv/bin/python - "$work_dir" <<'PY'
import json
from pathlib import Path
import shutil
import subprocess
import sys

work = Path(sys.argv[1])
record = json.loads(Path("benchmarks/results/tokamax_attention_20261002.json").read_text())
shutil.copytree(work / "baseline/src", work / "groups/src")
shutil.copytree(work / "baseline/tests", work / "groups/tests")
patch = work / "group.patch"
patch.write_text(record["candidates"]["group_contributions"]["patch"])
subprocess.run(["git", "apply", str(patch)], cwd=work / "groups", check=True)
for name, source in record["reproduction_scripts"].items():
    (work / name).write_text(source)
PY
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$work_dir/baseline/src" \
  .venv/bin/python -m pytest "$work_dir/test_group_masks.py" -q
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$work_dir/groups/src" \
  .venv/bin/python -m pytest tests/test_packed.py \
  tests/test_work_balance.py::test_packed_staging_across_batches_matches_reference \
  tests/test_edge_support.py "$work_dir/groups/tests/test_contribution_pruning.py" -q
```

`check_scene.py` expects the local bicycle path recorded in its source.
Run `check_training.py OUTPUT.npz` in separate baseline/group processes and
compare the outputs with the recorded tolerances. The production source matches
its baseline snapshot. This round retains the combined records and removes the
owned temporary clone, source copies, scripts, state archives and logs.
