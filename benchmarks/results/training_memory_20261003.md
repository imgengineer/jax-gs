# Optimizer memory and register-pressure study — 2026-10-03

The production baseline is `36764ec248971be886075a51011017ef927381b5`, following
the [completed training comparisons](training_followup_20261003.md). This study
validates three optimizer prototypes and records initial screening results.
**No prototype is enabled by default, and no additional speedup is established.**
The [raw record](training_memory_20261003.json) contains all patches, source
hashes, reports, compiler diagnostics, validation logs and reproduction scripts.

## Evidence and hypotheses

A diagnostic trace of 20 complete training updates puts the visible-cluster
Optax kernel at 603.98 µs per update. Its launch metadata reports **242 registers
per thread** and **16.67% theoretical occupancy**, with 128 threads per block.
These are resource limits reported by the profiler, not measured SM utilization
or proof that register pressure is the sole bottleneck.

The optimizer loads parameters and both Adam moments before updating them.
Two approaches are worth distinguishing:

- Streaming read/write eviction hints may reduce cache pollution while keeping
  the same row-local Optax transformation and launch shape.
- Reducing rows per program from 32 to 16 or 8, while keeping four warps and one
  block, may reduce live values per thread. This also creates more programs;
  neither register reduction nor net speed improvement has been measured yet.

The formulas, precision, moment updates, inactive-band momentum decay and buffer
donation contracts stay the same in these three validated prototypes.

## Initial screens

RTX 5090, driver 615.71.09, JAX 0.11.2, Flax 0.12.10, Optax 0.2.8 and CUTLASS
DSL 4.8.0. Every screen starts from the same 975,104-point PLY and fresh Adam,
using bicycle view 0 at 1237×822, SH3, 8×16 tiles, statistics off, 30 warmups and
500 timed complete NNX updates. Timing synchronizes around the update block.

An initial GPU check stopped the driver when a `jimm` process was present.
Screens resumed after that process exited. Each successful job checked for
other GPU compute processes before launch and approximately every 0.5 seconds;
none was observed. These screens finished before the user requested
correctness-only work. No performance measurements ran after that instruction.

| Prototype | Update time | Change from baseline |
| --- | ---: | ---: |
| Baseline | 1.8012 ms | — |
| Streaming stores | 1.7956 ms | 0.31% less time |
| Streaming loads | 1.8004 ms | 0.04% less time |
| Streaming loads and stores | 1.7879 ms | 0.73% less time |
| Order backward tiles by contributor count | 1.8054 ms | 0.24% more time |

Each row has **one sample on one view**. These small differences need repeated,
alternating comparisons. All five runs preserve 15 parameter/moment buffer
addresses, use one JIT entry and pass the overflow check. Temporary storage
remains 186,990,720 bytes. There are no new full-training or PSNR results.

The streaming-store trace measures the optimizer at 602.42 µs, with the same
242-register footprint. Contributor-count scheduling reduces backward time
from 432.84 to 429.65 µs but increases forward time from 176.67 to 182.74 µs;
the complete update does not improve. That prototype is not adopted, and its
changed internal work-cache meaning has not received full correctness testing.

The `.cg` load prototype cannot compile with the installed JAX lowering:
it accepts `.cg` but indexes a dictionary whose key is `cg`, producing
`KeyError: '.cg'`. No dependency was modified. The combined `.cg`/streaming
prototype was not executed.

## Correctness

The streaming read/write, 16-row and 8-row prototypes each pass the same
**51 existing optimizer, SH pullback and Muon tests** with unchanged tolerances.
The checks cover free and invisible slots, partial clusters, undefined gradient
tails, inactive SH momentum and the dense fallback.

An additional isolated optimizer comparison uses nonzero moments and the
degree sequence **0 → 1 → 2 → 3 → 0**, including a zero viewing direction,
partial clusters and poisoned unused gradient tails. All **95 output arrays**
are bit-exact against the baseline for each of the three prototypes.

Eight-update end-to-end checks for streaming read/write cover Optax, Muon and
CuTe, SH0/statistics off and SH3/statistics on. All **138 state arrays** pass
the existing `rtol=2e-5, atol=2e-6`; 80 are bit-exact. Maximum absolute difference
is `2.385e-7`, and maximum relative L2 difference is `2.302e-8`. All 15 donated
parameter/moment addresses, one JIT entry and existing overflow state are
preserved. This short comparison is not a long-run quality guarantee.

CUDA memcheck passes **17 cases per prototype**, with zero device errors.
Device instrumentation has no kernel filter; `--report-api-errors no` follows
the driver symbol-query limitation documented in the
[earlier checker record](contribution_pruning_20261002.md).
These correctness runs are not timing evidence. The production implementation
and project tests remain identical to the baseline.

## Reproduce and remaining work

Check out the baseline revision, export its `src/` into a `baseline/` directory,
and copy that directory for each variant. Apply each
`source.prototypes.<variant>.patch` with `patch -p1` in its copy. Extract the
`reproduction_scripts` beside those directories and run from the repository
root. `validate_states.py`, `validate_rows.py` and `validate_final.py` reproduce
the numerical and CUDA checks. Fixed-model screens additionally need the common
PLY identified by `protocol.model` and its hash; the model is not bundled.

When an exclusive GPU window is available, compare the three validated
prototypes over repeated runs on four views, and measure the smaller-row
kernels' register usage. Only a repeatable winner warrants matched 10k/30k
training and held-out quality evaluation before changing the default.
