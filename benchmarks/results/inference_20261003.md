# RGB-only forward prototype — 2026-10-03

The candidate on `perf/inference-forward` passed correctness checks and the
subsequent [exclusive performance comparison](inference_exclusive_20261003.md),
including frame readback and JPEG encoding. It is accepted for `main`. The
measurements below document the earlier prototype screening, when a sustained
exclusive GPU window was unavailable; use the follow-up for final timings.

[Experiment data and reproduction scripts](inference_20261003.json) retain the
selected prototype's raw samples, all prototypes' round summaries and patches,
model hashes, validation results and compiled memory estimates. The baseline is
`1d4c26223fea53d8ba7178de37af2f9424e7d445`.

## Motivation and candidate

On the native XiaoChe model, a diagnostic baseline GPU trace attributed an
average 4.612 ms at 640×360 and 1.954 ms at 1280×720 to rasterization; projection
took about 0.201 ms. Each trace covers 20 frames after 30 warmup calls. These
traces identify a hotspot; they did not continuously monitor other GPU jobs.

[Balanced3DGS](https://arxiv.org/html/2412.17378v1) explores finer raster work
partitioning and workload-dependent kernel selection, including the overhead
of partitioning already balanced work.
[FasterGS](https://github.com/GoogolplexGoodenough/FasterGS) explores workload
scheduling and compositing separate depth buckets. These ideas motivated a
smaller pixel-group prototype and specialization of the existing forward path.
No external implementation was copied in this change.

The retained candidate specializes the existing RGB-only path:

- Skip last-contributor updates, tile maxima and writes of backward state.
  The JAX binding supplies scalar placeholders for unused pixel/tile caches.
- Expand each raster iteration from two to eight consecutive Gaussians,
  exposing more independent exponential calculations to the compiler.
  Per-pixel half2 compositing order and arithmetic remain the same.
- Keep the training path at two Gaussians per iteration, with its full caches,
  contribution bits and fragment statistics. Public calling conventions stay
  the same, including differentiation through the renderer.

No depth-bucket regrouping was introduced: it would change the grouping of
finite-precision compositing operations and require separate quality checks.

## Prototype screening

The environment matches the [viewer record](viewer_20261002.md): RTX 5090,
driver 615.71.09, Ryzen 9 9950X3D, Python 3.12.13, JAX 0.11.2 and CUTLASS DSL
4.8.0. Both models use SH3, 8×16 tiles and 8,000,000 visibility pairs. Bicycle
has 975,104 Gaussians; XiaoChe has 982,912.

Each comparison uses eight nearby horizontal camera offsets, three alternating
baseline/candidate rounds and 100 synchronized calls per variant per round,
after 30 warmup calls. All presets preserve 16:9 viewing rays. The reported
statistic is the median of the three round medians. Timing covers the compiled
render and uint8 conversion, excluding readback, JPEG, network, browser and
compilation. All 64 warmup images per prototype matched exactly as uint8 RGB,
with no visibility overflow and four JIT entries per renderer.

| Prototype | GPU process observations | Interpretation |
| --- | --- | --- |
| Split each tile into 64-pixel warp groups | Another process in 78/80 snapshots | Image checks passed; timing is contaminated and cannot establish performance. |
| Skip backward bookkeeping, unroll 2 | No other compute PID in 63 snapshots | Small preliminary reductions; retains the original iteration width. |
| Skip backward bookkeeping, unroll 4 | No other compute PID in 60 snapshots | Promising preliminary raster improvement. |
| Skip backward bookkeeping, unroll 8 | No other compute PID in 58 snapshots | Retained for further evaluation. |
| Skip backward bookkeeping, unroll 16 | No other compute PID in 61 snapshots | Similar preliminary results; separate runs do not establish superiority over 8. |

Process sampling occurred approximately every 0.5 s. Observing no competing
compute PID during a short run does not reserve the GPU for the final study.
Unroll 8 limits expansion relative to 16 while retaining the observed benefit;
it is not claimed to be optimal for every scene or GPU.

Screening the unroll-8 kernel changes with the original JAX output shapes
produced these full-render call times in milliseconds:

| Model | Resolution | Baseline | Prototype | Less time |
| --- | --- | ---: | ---: | ---: |
| bicycle | 640×360 | 0.743 | 0.656 | 11.7% |
| bicycle | 640×480 | 0.718 | 0.645 | 10.2% |
| bicycle | 1280×720 | 0.688 | 0.632 | 8.2% |
| bicycle | 1920×1080 | 0.785 | 0.753 | 4.1% |
| XiaoChe | 640×360 | 5.233 | 3.994 | 23.7% |
| XiaoChe | 640×480 | 4.012 | 3.057 | 23.8% |
| XiaoChe | 1280×720 | 2.261 | 1.774 | 21.5% |
| XiaoChe | 1920×1080 | 1.700 | 1.401 | 17.6% |

These runs still allocated the baseline cache shapes. Scalar placeholder
allocation was added afterward and received correctness and memory checks.
The table describes the archived kernel prototype; the complete candidate is
measured in the [exclusive follow-up](inference_exclusive_20261003.md).

## Correctness and memory

- **89 float RGB images matched the baseline bit for bit:** two models × four
  presets × eight orbit/FOV poses, plus 25 bicycle COLMAP poses at 1237×822.
  All images were finite and reported no visibility overflow. This tests a
  broader camera path than the nearby poses used for screening.
- **42 targeted tests passed**, covering the viewer, contribution pruning and
  inference. New tests cover all four supported tile shapes, partial image
  tiles, multiple staging batches, a partial final batch and small opacities;
  they compare RGB-only output with the training forward output and verify
  the cache shapes.
- **426 full-suite tests passed in 663.66 s**, with **99.77% Python line and
  branch coverage**: 2,236 statements and 322 branches. GPU DSL bodies are
  excluded from Python coverage.
- **CUDA memcheck passed all four inference cases with zero device errors.**
  API error reporting was disabled with `--report-api-errors no`; device
  memory checks remained enabled. This is not a claim about every CUDA API
  diagnostic or race detection.

At 1920×1080, the three unused backward outputs shrink logically from
16,653,600 bytes to 12 bytes. However, JAX's compiled full-frame temporary-memory
estimate was **unchanged in all eight model/resolution comparisons**, around
184–185 MB. Buffer reuse and other stages determine that peak. No reduction in
peak VRAM is claimed. All argument/output/temporary/alias byte counts are
retained in the JSON record.

## Reproduce and finish measurement

On this candidate branch, extract the archived local scripts and baseline
modules without replacing the installed candidate:

```bash
uv run python - <<'PY'
import json
import subprocess
from pathlib import Path

record = json.loads(Path("benchmarks/results/inference_20261003.json").read_text())
root = Path("output/optimization_20261003")
root.mkdir(parents=True, exist_ok=True)
for stem in ("packed_rasterize", "packed_rasterizer"):
    source = subprocess.check_output([
        "git", "show", f"{record['baseline_revision']}:src/jaxgs/kernels/{stem}.py"
    ])
    (root / f"{stem}_baseline.py").write_bytes(source)
for name, source in record["reproduction_scripts"].items():
    (root / name).write_text(source)
PY

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  output/optimization_20261003/validate_models.py
```

The scripts contain the two local model/dataset paths; change those paths for
another machine. Models are identified by SHA-256 in the record and are not
bundled. `compare.py` switches both the kernel and its JAX output-shape wrapper
before compiling each variant. The archived `compare_prototypes.py` records
the original kernel-only procedure and requires the baseline wrapper; using
that procedure with scalar cache allocations would give the baseline kernel
undersized output buffers.

The planned exclusive comparison is complete. The
[follow-up record](inference_exclusive_20261003.md#reproduce) contains the final
five-round moving/fixed JIT and readback/JPEG commands and results.
