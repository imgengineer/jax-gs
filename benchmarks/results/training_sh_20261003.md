# Deferred SH gradients during training — 2026-10-03

Reconstructing spherical-harmonic (SH) gradients inside the Optax update
reduces the measured training loop by **5.5% at 10k** and **6.1% at 30k**.
Compilation adds about 0.8 s: image preload, warmup and training combined are
**0.5% longer at 10k** and **3.2% shorter at 30k**. The short default run mainly
benefits in update throughput and temporary storage, with little change in
startup-inclusive time.

[Raw reports, patches and reproduction scripts](training_sh_20261003.json)
retain every training history, held-out view score, fixed-update measurement,
GPU process observation and source/model hash. The baseline is
`efbfb4a14332da76c48ee98052c385d58724d3cf`.

## Change and measured bottleneck

An SH3 gradient is an outer product: 16 direction-dependent basis values times
three masked color cotangents. The projection pullback now writes those three
float32 cotangents, and the optimizer reconstructs the coefficients using the
original position and camera center. The ordinary Optax transformation still
receives full coefficient gradients and handles its own parameter/state
updates. Inactive SH bands receive zero gradients while their moments continue
to decay. Optax, Muon and the dense fallback are supported. SH0 keeps its
existing path because it already has only three gradient values.

This avoids materializing a wide intermediate between two kernels. At a
one-million-slot capacity, its SH3 storage falls from **192 MB to 12 MB**.
Position gradients still include the derivative through SH viewing direction.
The iteration schedule, camera order, learning rates, densification, pruning,
opacity decay and rasterization are unchanged. Reconstructing gradients in
another compiler can change float32 rounding; exact training trajectories
are not promised.

The [previous prototype](performance_sweep_20261002.md)
summed coefficient arrays masked with zeros. Replacing those additions with
disjoint selections makes a substantial difference in the Pallas update.
Diagnostic GPU traces, each covering 20 updates of view 0, show:

| Kernel | Baseline | Original masked sums | Disjoint selection |
| --- | ---: | ---: | ---: |
| Projection backward | 181.96 µs | 100.84 µs | 100.92 µs |
| Visible-cluster Optax | 677.97 µs | 689.01 µs | 603.86 µs |

These traces locate the savings; the repeated comparisons below measure the
final implementation. The initial masked-sum prototype reduced complete
updates by only 3.3–3.5% across four views. A concatenation variant could not
compile with the installed Pallas Triton lowering. A binary-stack variant
compiled but took 81.8 ms/update in its single-view screen, so it was rejected.
Their patches and measurements are archived, with no dependency modifications.

## Complete training and quality

RTX 5090, driver 615.71.09, Ryzen 9 9950X3D, Python 3.12.13, JAX 0.11.2,
Flax 0.12.10, Optax 0.2.8 and CUTLASS DSL 4.8.0. Bicycle uses `images_4` at
1237×822, 169 training views and 25 held-out views. Configuration overrides
only the image folder, resolution factor 1, evaluation split and requested
iteration budget. All other defaults remain: Optax, 8×16 tiles, one million
slots, eight million visibility pairs, seed 0 and a 30k position-LR horizon.

Each budget has three paired runs in alternating order, with a fresh process
and initialization each time. Training/evaluation and fixed-update jobs run
sequentially. Every job checks GPU availability first and polls other compute
PIDs approximately every 0.5 s; none was observed during these measurements.
Both sets of checkpoints use the frozen baseline renderer for evaluation.

Training time includes density control, opacity decay, spatial reordering and
epoch reports. It excludes initialization, image preload, compilation and
checkpoint writing. The additional stage sum includes image preload and
compilation/warmup, but still excludes point initialization and checkpoint
writing; it is not the complete CLI wall time. Monitor wall times have
0.5-second polling granularity and are retained only as diagnostics.

| Requested steps | Actual updates | Baseline loop | Candidate loop | Less time | Baseline PSNR | Candidate PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 9,971 | 12.517 s | 11.825 s | 5.5% | 24.572 dB | 24.547 dB |
| 30,000 | 29,913 | 41.101 s | 38.599 s | 6.1% | 25.464 dB | 25.479 dB |

Times are medians of three runs; PSNR is the mean of three 25-view means.
All six paired training-loop comparisons were faster. Final active counts
were 975,104 at 10k and 993,152 at 30k for both implementations, with seven
compiled training variants and no visibility overflow.

| Image preload + warmup + training | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 10k | 25.967 s | 26.095 s | 0.5% longer |
| 30k | 54.594 s | 52.830 s | 3.2% shorter |

Mean PSNR changed by −0.026 dB at 10k and +0.015 dB at 30k. CUDA atomic
reductions already make repeated same-seed training vary. The small score
differences and three repeats on one scene do not establish general quality
equivalence or improved reconstruction quality.

## Fixed-model updates and compiled memory

Each view and process starts from the same 975,104-point PLY and fresh Adam
state. SH3, scene radius 1, step 30,000 onward, 30 warmup updates and 500 timed
updates are used, without density control. Each block synchronizes before
and after timing. The table is the median of three block-average update times,
with alternating baseline/candidate order.

| COLMAP view | Baseline | Candidate | Less time |
| --- | ---: | ---: | ---: |
| 0 | 1.9911 ms | 1.8341 ms | 7.9% |
| 48 | 1.8548 ms | 1.7108 ms | 7.8% |
| 96 | 1.7897 ms | 1.6440 ms | 8.1% |
| 144 | 1.7526 ms | 1.6140 ms | 7.9% |

Every paired sample improved. All 15 parameter/moment buffer addresses were
preserved, each bound NNX step reused one JIT entry, and no overflow occurred.
A separate single-round statistics-enabled probe at view 0 measured
2.1597 → 2.0035 ms/update; it is not a repeated measurement.

For the actual donated NNX executable, ordinary-update temporary memory fell
from **281,840,256 to 186,990,720 bytes**, a reduction of **94,849,536 bytes
(33.7%)**. All four views reported the same sizes. Argument/output sizes and
711,825,924 aliased bytes were unchanged. This is the compiler's temporary
allocation estimate, not total training-process VRAM. The latter also includes
model/optimizer state, preloaded images and allocator/runtime storage.

## Correctness

The final candidate's eight-update comparison covers Optax, Muon and CuTe,
SH0 without statistics and SH3 with statistics, partial clusters, donation
and existing overflow/peak-pair state. All **138 state arrays** pass the
existing `rtol=2e-5, atol=2e-6`; 77 are bit-exact. Maximum absolute difference
is `1.193e-7` and maximum relative L2 difference is `1.294e-7`.

New equation checks compare the reconstruction with independent JAX automatic
differentiation for SH0–3, ordinary/padded channel layouts, zero direction,
inactive rows and existing momentum. All eight also pass on CPU. Ten optimizer
cases cover visible/free slots, undefined compact tails, empty visibility and
the dense fallback. Existing compact-projection tests additionally compare
the color-only pullback with full gradients for all SH degrees, empty views
and partial clusters.

The full regression passed in 651.71 s with **99.77% Python line and branch
coverage** (2,264 statements, 326 branches). All **444 project tests** passed;
the invocation collected 452 cases because it also discovered eight archived
SH equation checks in the ignored experiment directory. That directory has
since been cleaned. The explicit `pytest tests` command below avoids this
extra collection.

CUDA Compute Sanitizer memcheck passed **26 compact-projection and deferred
optimizer cases**, including all SH degrees, empty visibility, partial
clusters and the dense fallback, with **zero device-memory errors**. API error
reporting was disabled (`--report-api-errors no`); device-memory checks were
not filtered. Exact commands and complete logs are in the raw record.

## Reproduce

The JSON contains source hashes, the candidate patch, exact commands in each
monitor record and the scripts used. Export the baseline source, apply the
candidate patch to a second copy and extract the scripts into their parent:

```bash
uv run python - <<'PY'
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

record = json.loads(Path("benchmarks/results/training_sh_20261003.json").read_text())
work = Path("output/training_sh_reproduce")
work.mkdir(parents=True)
archive = subprocess.check_output(["git", "archive", record["baseline_revision"], "src"])
with tarfile.open(fileobj=io.BytesIO(archive)) as data:
    data.extractall(work / "baseline", filter="data")
shutil.copytree(work / "baseline", work / "candidate")
subprocess.run(["patch", "--batch", "-p1"], cwd=work / "candidate",
               input=record["source"]["candidate_patch"], text=True, check=True)
for name, source in record["reproduction_scripts"].items():
    (work / name).write_text(source)
(work / "training.toml").write_text(record["protocol"]["config"])
PY

PYTHONDONTWRITEBYTECODE=1 uv run python output/training_sh_reproduce/full_training.py
```

Update the dataset path in the scripts for another machine. Fixed-update
scripts also require the same PLY on both sides; its hash is recorded and
the model is not bundled. Run GPU jobs sequentially during an exclusive
window. The project regression command is `uv run pytest tests -q
--cov=jaxgs --cov-report=term-missing`.
