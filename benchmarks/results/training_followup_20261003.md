# Training compilation and forward scheduling — 2026-10-03

This follows the [deferred SH update](training_sh_20261003.md), with baseline
`71f4ca5682ffdb0396cb8c931a8d5e6949e40660`. The retained changes target both
startup compilation and the training loop. Training steps, image order,
learning rates, density control, parameter/state precision and checkpoint
formats keep their existing behavior.

## Changes

All 22 production `cutlass_call` sites now supply their module-level launch
function as `compile_key`. The installed CUTLASS DSL 4.8.0 otherwise generates
frontend IR to compute a content key on each cache lookup, then traces again
on a backend cache miss. The explicit key avoids the first trace. CuTe still
includes shapes, dtypes, tensor specifications, aliases, static arguments,
compiler options and environment in the key. Kernel code and module constants
are treated as immutable within a process; specialization choices remain
explicit keyword arguments. This uses the library API and changes no dependency.

Ten small projection calls over two image sizes and SH0–3 produced ten IR
key-generation traces with the baseline and zero with the candidate. The
regression also compares all projection outputs bit-for-bit against automatic
content keys, revisiting SH0 after higher degrees to check cache isolation.
This optimization changes compilation work, not GPU work in an already
compiled update. The API was previously identified in the
[CuTe review](cute_dsl_review_20260930.md).

Training raster forward now composites four splats per outer iteration,
overlapping independent exponentials and sharing the loop's active-pixel vote.
The per-pixel compositing order and thresholds are preserved. Contribution
bits and last indices still bound the backward pass. Statistics reduce four
splats together, so floating-point accumulation order can change. RGB-only
rendering retains its eight-splat loop.

## Measurement protocol

RTX 5090, driver 615.71.09, Ryzen 9 9950X3D, Python 3.12.13, JAX 0.11.2,
Flax 0.12.10, Optax 0.2.8 and CUTLASS DSL 4.8.0. Bicycle uses `images_4` at
1237×822, 169 training views and 25 held-out views. Only image folder,
resolution factor 1, evaluation split and iteration budget override defaults.
The optimizer is Optax, with 8×16 tiles, one million slots, eight million
visibility pairs, seed 0 and a 30k position learning-rate horizon.

Three fresh-process pairs completed at each of 10k and 30k in alternating
baseline/candidate order. The first two 10k pairs were measured in the initial
window. The third pair was rerun in a later exclusive window, together with all
30k and fixed-model runs. Candidate source is commit
`1ec157defe79308884a8e0ffe6e4d73e021ec932`; source hashes, dependency versions
and the common fixed-model hash match the initial record.

The same frozen baseline renderer evaluates every accepted checkpoint. All
performance jobs run sequentially; a monitor checks for other GPU compute
processes before launch and approximately every 0.5 seconds during each job.
All 28 completion-window jobs (22 main measurements and six additional
cache-only control jobs) returned success with zero foreign-process observations.

Loop timing includes density control, opacity decay, spatial reordering and
epoch reports. It excludes initialization, image preload, precompilation and
checkpoint writing. The additional stage sum includes image preload, warmup
and the loop, but excludes point initialization and checkpoint writing.
Monitor wall time has 0.5-second polling granularity and is diagnostic only.

## Completed training comparisons

| Budget | Stage | Baseline median | Candidate median | Less time |
| --- | --- | ---: | ---: | ---: |
| 10k | Precompilation / warmup | 14.112 s | 10.392 s | 26.4% |
| 10k | Training loop | 11.853 s | 11.694 s | 1.3% |
| 10k | Image preload + warmup + loop | 26.402 s | 22.544 s | 14.6% |
| 30k | Precompilation / warmup | 13.878 s | 10.369 s | 25.3% |
| 30k | Training loop | 38.667 s | 38.060 s | 1.6% |
| 30k | Image preload + warmup + loop | 52.980 s | 48.832 s | 7.8% |

Each median has **three samples**. All six paired loop comparisons improved
by 1.22–1.94%. Each run reused seven compiled variants with donation enabled,
no visibility overflow and finite loss. The 10k runs performed 9,971 updates;
point counts were 975,104, 975,232 and 975,104 for pairs 1–3, matching within
each pair. The 30k runs performed 29,913 updates and all ended at 993,152 points.

| Budget | Pair | Baseline loop | Candidate loop | Baseline PSNR | Candidate PSNR |
| --- | --- | ---: | ---: | ---: | ---: |
| 10k | 1 | 11.839 s | 11.664 s | 24.565 dB | 24.570 dB |
| 10k | 2 | 11.853 s | 11.708 s | 24.534 dB | 24.557 dB |
| 10k | 3 | 11.855 s | 11.694 s | 24.557 dB | 24.558 dB |
| 30k | 1 | 38.760 s | 38.007 s | 25.536 dB | 25.438 dB |
| 30k | 2 | 38.546 s | 38.060 s | 25.457 dB | 25.399 dB |
| 30k | 3 | 38.667 s | 38.113 s | 25.450 dB | 25.473 dB |

Mean held-out PSNR changes from 24.552 to 24.562 dB at 10k (+0.010 dB), and
from 25.481 to 25.437 dB at 30k (−0.044 dB). Each score averages all 25 held-out
views. At 30k, baseline run means span 25.450–25.536 dB and candidate means
25.399–25.473 dB. Two pairs favor the baseline; the third favors the candidate.

The 30k per-view means decrease on 14 of 25 views. Differences range from
−0.922 to +0.184 dB; the largest decrease is on `_DSC8768.JPG` (23.721 to
22.799 dB). A small scene-wide mean difference therefore does not imply small
differences on every view. These three repeats on one scene do not establish
long-run quality equivalence, despite passing short-run numerical regressions.

The original third baseline job encountered a foreign `jimm` GPU process.
Its report is retained separately and excluded; the replacement pair above
completed without interference. No planned main comparison remains deferred.

## Fixed-model training updates

The same 975,104-point PLY starts every view with fresh Adam state. Three
alternating process pairs cover views 0, 48, 96 and 144, at SH3 with statistics
off and no densification. Each view has 30 warmup updates followed by a block
of 500 complete NNX training updates. Timing synchronizes at block boundaries;
values below are medians of the three per-update block means.

| View | Baseline | Candidate | Less time |
| --- | ---: | ---: | ---: |
| 0 | 1.837 ms | 1.808 ms | 1.60% |
| 48 | 1.711 ms | 1.693 ms | 1.06% |
| 96 | 1.649 ms | 1.631 ms | 1.12% |
| 144 | 1.617 ms | 1.597 ms | 1.23% |

All 24 view runs preserve the 15 donated parameter/moment addresses, use one
JIT cache entry and pass the overflow check. Compiled temporary storage is
186,990,720 bytes for both implementations. These are complete training
updates, not rendering-only frame times.

## Cache-only control

The lower 30k mean PSNR prompted three additional runs with explicit compilation
keys and the original two-splat rasterizer. This control restores the complete
baseline `packed_rasterize.py`; all other source matches the candidate. It uses
the same scene, settings and frozen baseline evaluator. The control group ran
after the main measurements, so these are not three additional alternating
pairs.

| 30k implementation | Warmup median | Loop median | Stage sum median | Mean PSNR |
| --- | ---: | ---: | ---: | ---: |
| Baseline: automatic keys, two splats | 13.878 s | 38.667 s | 52.980 s | 25.481 dB |
| Control: explicit keys, two splats | 10.295 s | 38.608 s | 49.298 s | 25.431 dB |
| Candidate: explicit keys, four splats | 10.369 s | 38.060 s | 48.832 s | 25.437 dB |

Cache-only PSNR values are 25.417, 25.474 and 25.401 dB. Its mean is 0.050 dB
below the paired baseline group and 0.006 dB below the full candidate. This
control also has per-view differences; its largest mean decrease is 0.631 dB
on `_DSC8752.JPG`. The measurements do not isolate the observed quality
difference to forward unrolling, nor establish long-run quality equivalence.
They support separating the substantial compilation gain from the smaller
training-loop gain. No implementation change was made during this completion
window.

## Correctness

All 115 targeted projection, rasterization, staging, contribution-bitmap and
compilation-cache cases passed with their original tolerances. The new cache
regression compares every projection output bit-for-bit against automatic IR
content keys across image dimensions and SH0–3, and asserts that explicit keys
do not invoke IR key generation.

Eight successive updates cover Optax, Muon and CuTe with SH0/statistics off
and SH3/statistics on, partial clusters, donation, and existing overflow and
peak-pair state. All 138 arrays pass `rtol=2e-5, atol=2e-6`; 74 are bit-exact.
Maximum absolute difference is `1.193e-7`, and maximum relative L2 difference
is `1.729e-8`.

The full suite passed **445 tests** in 667.32 s with **99.77% Python line and
branch coverage** (2,264 statements, 326 branches). GPU DSL bodies are excluded
from Python coverage and checked through their GPU behavior. Correctness runs
after the interference report are not used for performance measurements.

CUDA memcheck, racecheck and synccheck each passed all 28 selected contribution
and staging cases. They reported zero device memory errors, race hazards or
synchronization errors. As in the
[earlier checker run](contribution_pruning_20261002.md), API-error reporting is
disabled with `--report-api-errors no`, and racecheck uses one worker. Device
instrumentation remains enabled without kernel filters. These checks validate
the selected GPU paths; they do not measure performance.

Ruff, formatting and `git diff --check` pass. The raw record retains the two
baseline-only cache-probe setup failures: an unsupported reference keyword and
a conic tolerance inappropriate for the existing approximate projection.
The final standalone probe checks five reference fields, while the new project
regression checks all seven fields bit-for-bit against automatic CuTe keys.
Existing project test tolerances were not changed.

## Rejected prototypes

The local exploration also tried extending SH deferral to the position
derivative. Projection backward then avoids reading SH parameters, which are
already loaded by the optimizer. A direct `jax.vjp` prototype generated
expensive repeated reductions: its Optax kernel took 6.21 ms, versus 0.60 ms
for the baseline. An analytic derivative first contracts color channels and
then performs three basis reductions. Tuning it to eight rows per single-warp
program and four blocks reduced a fixed update to 1.804 ms, versus about
1.834 ms for the baseline.

That small update gain did not pay for compilation in the 10k pilot: its loop
took 11.768 s versus 11.831 s, but warmup took 15.440 s versus 14.699 s.
All 138 checked state arrays passed the original tolerances. The additional
derivative implementation and changed internal gradient contract were not
adopted. No general quality or long-run speed claim is made for this prototype.

Factoring color cotangents out of SH polynomials left Optax timing unchanged
(604.0 versus 604.7 µs in the diagnostic traces). Scheduling changes alone
also failed to improve the baseline. Larger blocks could be much slower.
An eight-splat training forward measured 1.809 ms/update in its single-view
screen, versus 1.798 ms for four. These are screening samples, separate from
the repeated final comparisons.

## Reproduce

The [raw record](training_followup_20261003.json) contains source/model hashes,
candidate and rejected prototype patches, per-epoch histories, per-view
quality, process monitors, validation logs and reproduction scripts.

Export `src/` from the baseline commit into a `baseline/` directory. Copy it
to `candidate/`, apply `source.candidate_patch` with `patch -p1`, and extract
`reproduction_scripts` and `protocol.config` into their parent directory as
Python files and `training.toml`. Run `full_training.py` from the repository
root. Update dataset paths for another machine. `final_fixed_update.py` also
needs the common PLY identified by `fixed_protocol`; the model is not bundled.
Use an exclusive GPU window and run jobs sequentially.

`complete_measurements.py` reproduces the missing third 10k pair, all three
30k pairs and the fixed-model comparisons. For the additional control, copy
`candidate/` to `compile_keys_only/`, restore
`src/jaxgs/kernels/packed_rasterize.py` from `baseline/`, and run
`check_cache_only.py`. The record retains the original partial summary,
completion-window logs and every per-view score.

The project regression command explicitly selects `tests`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests -q \
  --cov=jaxgs --cov-report=term-missing
```
