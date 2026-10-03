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

Two fresh-process 10k pairs completed in alternating order. The same frozen
baseline renderer evaluates every accepted checkpoint. All performance jobs
run sequentially; a monitor checks for other GPU compute processes before
launch and approximately every 0.5 seconds during each job.

Loop timing includes density control, opacity decay, spatial reordering and
epoch reports. It excludes initialization, image preload, precompilation and
checkpoint writing. The additional stage sum includes image preload, warmup
and the loop, but excludes point initialization and checkpoint writing.
Monitor wall time has 0.5-second polling granularity and is diagnostic only.

## Completed 10k comparisons

| Stage | Baseline median | Candidate median | Less time |
| --- | ---: | ---: | ---: |
| Precompilation / warmup | 13.943 s | 10.421 s | 25.3% |
| Training loop | 11.846 s | 11.686 s | 1.3% |
| Image preload + warmup + loop | 26.222 s | 22.568 s | 13.9% |

Each median has **two samples**. Both pairs improved: loop time fell by
1.48% and 1.22%, and the stage sum by 13.25% and 14.61%. Each run performed
9,971 actual updates and reused seven compiled variants, with donation enabled
and no visibility overflow. The first pair ended with 975,104 active points;
the second with 975,232, identical between implementations within each pair.

| Pair | Baseline loop | Candidate loop | Baseline PSNR | Candidate PSNR |
| --- | ---: | ---: | ---: | ---: |
| 1 | 11.839 s | 11.664 s | 24.565 dB | 24.570 dB |
| 2 | 11.853 s | 11.708 s | 24.534 dB | 24.557 dB |

The two-run mean held-out PSNR changed from 24.549 to 24.563 dB (+0.014 dB).
Each score averages 25 held-out views. These two repeats on one scene do not
establish general quality equivalence or better reconstruction quality.

A foreign `jimm` GPU process appeared during the third baseline job. Its
measurements are retained separately and excluded from all summaries above.
Further timing awaits an exclusive GPU window.
The third 10k pair, 30k comparisons and repeated fixed-model measurements are
**deferred**. These are the completed two-pair results, not a completed
three-pair or 30k study.

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

The project regression command explicitly selects `tests`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests -q \
  --cov=jaxgs --cov-report=term-missing
```
