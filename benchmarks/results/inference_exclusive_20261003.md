# RGB-only forward: exclusive comparison — 2026-10-03

The RGB-only specialization in `254cf96` passed the planned exclusive GPU
comparison and is accepted for `main`. Across the four tested presets, moving
JIT render latency fell by **4.6–10.3% for bicycle** and **17.9–23.7% for
XiaoChe**. All five paired rounds improved at each resolution. Fixed-camera
and readback/JPEG comparisons also improved in all five rounds.

[Raw samples and reproduction scripts](inference_exclusive_20261003.json)
contain every timed call, round median/p95, source and model hashes, GPU process
observations and summaries. The baseline is
`1d4c26223fea53d8ba7178de37af2f9424e7d445`; the measured candidate is
`254cf96a9b916ff60517bf8071698a415a03d017`, including scalar cache allocations.
The [prototype record](inference_20261003.md) explains the changes and earlier
experiments.

## Method

- RTX 5090, driver 615.71.09, Ryzen 9 9950X3D, Python 3.12.13, JAX 0.11.2,
  CUTLASS DSL 4.8.0 and viser 1.1.1. Both models use full SH3, 8×16 raster
  tiles and 8,000,000 visibility pairs.
- The user provided an exclusive GPU window. The initial check showed 0%
  utilization, 18 MiB allocated and no compute processes. Neither run observed
  another compute PID: **104 snapshots** for moving JIT and **152 snapshots**
  for fixed JIT/readback/JPEG, approximately every 0.5 s. No other GPU workload
  was launched during measurement.
- Baseline and candidate share model buffers and camera inputs in each
  process. Both the kernel and JAX output-shape wrapper are selected before
  compiling each variant. All four resolutions reuse compiled executables.
- Five rounds alternate baseline/candidate order. Each variant receives 30
  warmup calls before every timed round. Moving JIT uses 200 timed calls per
  round; fixed JIT and readback/JPEG each use 100.
- Moving cameras use eight horizontal displacements of up to 1% of the
  look-at distance. All texture presets retain the same 16:9 viewing rays.
  Fixed measurements use camera zero of that path. Bicycle starts at its
  first COLMAP camera; XiaoChe uses automatic bounds fitting.
- The tables report the **median of five round medians**, in milliseconds.
  JIT timing synchronizes the full render and uint8 conversion. Frame
  preparation adds device readback and viser's JPEG encoder at quality 90,
  using the same calls as [benchmarks/viewer.py](../viewer.py). Compilation,
  network delivery and browser decoding/display are excluded.

## Compiled rendering

Each arrow shows baseline → candidate. Bicycle has 975,104 Gaussians and
XiaoChe has 982,912. These are comparisons of two versions of this project's
renderer, including the native LiteGS model as an input scene.

| Model | Resolution | Moving JIT, ms | Less time | Fixed JIT, ms |
| --- | --- | ---: | ---: | ---: |
| bicycle | 640×360 | 0.744 → 0.668 | 10.2% | 0.728 → 0.649 |
| bicycle | 640×480 | 0.732 → 0.657 | 10.3% | 0.724 → 0.648 |
| bicycle | 1280×720 | 0.690 → 0.643 | 6.8% | 0.687 → 0.636 |
| bicycle | 1920×1080 | 0.797 → 0.760 | 4.6% | 0.796 → 0.762 |
| XiaoChe | 640×360 | 5.280 → 4.031 | 23.7% | 5.337 → 4.066 |
| XiaoChe | 640×480 | 4.119 → 3.168 | 23.1% | 4.245 → 3.256 |
| XiaoChe | 1280×720 | 2.284 → 1.779 | 22.1% | 2.588 → 2.016 |
| XiaoChe | 1920×1080 | 1.703 → 1.398 | 17.9% | 1.675 → 1.391 |

Both revisions already use JIT, so this comparison measures the effect of the
RGB-only raster specialization. Fixed rendering retains dynamic model/camera
arguments and executes the renderer every call. Different camera paths can
have different costs, as the XiaoChe moving/fixed 720p timings illustrate.

## Render, readback and JPEG

| Model | Resolution | Baseline, ms | Candidate, ms | Less time |
| --- | --- | ---: | ---: | ---: |
| bicycle | 640×360 | 1.490 | 1.393 | 6.5% |
| bicycle | 640×480 | 1.639 | 1.565 | 4.5% |
| bicycle | 1280×720 | 2.822 | 2.768 | 1.9% |
| bicycle | 1920×1080 | 5.077 | 5.050 | 0.5% |
| XiaoChe | 640×360 | 6.179 | 4.983 | 19.4% |
| XiaoChe | 640×480 | 5.196 | 4.193 | 19.3% |
| XiaoChe | 1280×720 | 4.351 | 3.701 | 14.9% |
| XiaoChe | 1920×1080 | 5.490 | 5.147 | 6.2% |

Readback and JPEG costs limit how much a faster renderer improves frame
preparation, particularly for bicycle at high resolution. The 0.5% bicycle
1080p change is small: paired-round reductions ranged from 0.13% to 1.23%.
These are frame-preparation latencies on two local paths and one GPU, not
browser FPS or a general scene benchmark.

## Validation

Both runs checked all 64 model/resolution/pose RGB pairs for exact uint8
equality and no visibility overflow. The frame run additionally checked
**64 byte-identical JPEG pairs**. Each renderer retained four JIT entries.

The measured source hashes are identical to the previously validated
candidate: **89 bit-exact float RGB images, 426 passing tests, 99.77% Python
line/branch coverage and four CUDA memcheck cases with zero device errors**.
There were no production-code changes between that regression and this
performance run. The checker scope and compiled memory estimates remain in
the [prototype validation](inference_20261003.md#correctness-and-memory).
The allocation changes did not lower the measured compiled temporary-memory
peak; no peak VRAM reduction is claimed.

## Reproduce

Extract the baseline modules and local benchmark scripts on the candidate or
its merged revision:

```bash
uv run python - <<'PY'
import json
import subprocess
from pathlib import Path

record = json.loads(Path("benchmarks/results/inference_exclusive_20261003.json").read_text())
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
  output/optimization_20261003/compare.py --steps 200 --rounds 5 \
  --output output/optimization_20261003/final_comparison.json

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  output/optimization_20261003/compare_frames.py --steps 100 --rounds 5 \
  --output output/optimization_20261003/final_frames.json
```

Run the two commands sequentially during an exclusive window. Update the
model/dataset paths in the extracted scripts for another machine; the two
PLY files are identified by SHA-256 and are not bundled.
