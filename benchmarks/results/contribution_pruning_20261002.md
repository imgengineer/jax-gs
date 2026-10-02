# Contribution pruning, 2026-10-02

Local analysis and measurements on an RTX 5090; no network lookup. The baseline
is the working tree after the preceding arithmetic improvements. Source hashes,
the incremental patch, all timing samples, per-view quality and GPU process
monitors are in [the combined record](contribution_pruning_20261002.json).

## Algorithm

Forward records one bit per tile/Gaussian pair that has at least one valid
fragment. Backward visits only set bits, from highest to lowest index. It skips
parameter loads, exponentials and gradient calculations for zero bits, including
empty 32-pair batches. Pixel grouping, compositing order, half2 reductions and
the squared partial-gradient statistics retain their original formulas. The
existing opacity and transmittance thresholds are unchanged.

A skipped pair has zero alpha at every pixel still participating in the
composite. Pixels that already terminated exclude that pair through their last
indices. Skipping it therefore leaves reverse transmittance, accumulated color
and all cotangents unchanged. Validity is determined from forward fragments;
an opacity gradient that cancels does not suppress a nonzero color gradient.

For tile `t`, the bitmap starts at `tile_offsets[t] // 32 + t`. Adding one padding
word per tile makes adjacent unaligned pair segments disjoint. Backward reads
only the initialized prefix and masks the final word at the maximum pixel last
index. With 8M pair capacity and 8,034 tiles, storage is 1,032,136 bytes, about
1 MiB. Pure RGB rendering compiles out the bit recording and uses a single
unused output word; custom VJP forward and training retain the bitmap.

## Measurements

The fixed input is a fresh baseline checkpoint with 1M capacity and 993,152 live
Gaussians. It uses bicycle `images_4`, SH3, 8x16 tiles, 128-slot clusters and an
8M pair arena. Each kernel measurement uses 30 warmups and six alternating
150-call batches. GPU compute processes were monitored every 0.5 seconds;
accepted measurements had no competing process.

| View | Statistics | Backward before / after, µs | Reduction |
| --- | --- | --- | --- |
| 0 | off | 843.85 / 679.91 | 19.4% |
| 0 | on | 882.99 / 716.53 | 18.9% |
| 48 | off | 707.15 / 588.30 | 16.8% |
| 48 | on | 750.67 / 621.88 | 17.2% |
| 96 | off | 703.04 / 575.12 | 18.2% |
| 96 | on | 748.88 / 604.75 | 19.2% |
| 144 | off | 642.66 / 558.69 | 13.1% |
| 144 | on | 694.58 / 588.61 | 15.3% |

Recording adds approximately 21–25 µs to forward without statistics and
10–13 µs with statistics. At view 0, backward processes 2,488,095 contributing
pairs instead of 2,776,580 pairs; the heaviest tile drops from 2,466 to 1,195
iterations. Pure RGB rendering changed by at most 0.6 µs across the four views.
RGB, final transmittance, last indices and backward prefix counts are bitwise
equal in all eight comparisons. Field-gradient relative L2 error is at most
2.103e-6, and existing statistics comparisons pass without relaxed tolerances.

Complete updates include projection, pair sorting, loss, both pullbacks and
Optax. Each trial starts from the same checkpoint, warms up for 30 donated NNX
updates and times 500 updates at view 0. There are three alternating trials per
implementation, each with one cached JIT signature.

| Trial | Baseline, ms | Final implementation, ms |
| --- | --- | --- |
| 1 | 2.39127 | 2.22709 |
| 2 | 2.39133 | 2.22183 |
| 3 | 2.39548 | 2.22354 |
| Median | 2.39133 | 2.22354 |

The complete update takes **7.02% less time**. This is distinct from the
13–19% reduction in raster backward alone.

Full training uses the 30k defaults with only `images="images_4"` and
`eval=true` overrides, seed 0, 169 training views and 29,913 actual updates.
Timing includes the epoch loop, densification, pruning, opacity decay and
spatial refinement; it excludes preload, compilation and final checkpoint
write. Three baseline and three final-source runs were evaluated on the same
25 held-out views. The first development run was unconditionally replaced by
a final-source run; its original results remain separately in the record.

| Trial | Baseline seconds | Final seconds | Baseline PSNR | Final PSNR |
| --- | --- | --- | --- | --- |
| 1 | 42.9415 | 41.2827 | 25.4969 | 25.4958 |
| 2 | 43.2807 | 41.1963 | 25.4736 | 25.4043 |
| 3 | 43.5413 | 41.3061 | 25.4594 | 25.4000 |

Median training time drops from **43.2807 to 41.2827 seconds (4.62%)**.
Mean held-out PSNR changes from **25.4766 to 25.4334 dB (−0.0432 dB)**. Runs
vary, and floating-point atomic accumulation remains nondeterministic; these
three trials do not establish quality equivalence. Performance measurements
cover this scene and GPU.

## Verification

The regression adds independent opacity/transmittance oracles for all four
supported tile shapes, statistics on/off, holes, empty contributions and opaque
tails. It covers empty tiles, unaligned segment starts, 31/32/33/65-pair spans,
partial image tiles and poisoned unused bitmap words/bits. Equal R/G colors
with opposite incoming cotangents verify that cancelling opacity gradients
retain valid color gradients. Pure RGB rendering matches the cached forward.

The full GPU suite passes **390 tests** in 585.29 seconds. Python coverage is
**100%** for 2,001 statements and 292 branches; the project excludes GPU DSL
bodies from Python coverage and validates them with the GPU tests.

CUDA memcheck passes 26 contribution/staging cases with zero device errors.
Racecheck and synccheck each pass all 24 new contribution cases, with zero
hazards, warnings or synchronization errors. The checker uses
`--report-api-errors no` after both baseline and candidate report the same seven
`cuGetProcAddress_v2` symbol-query errors under default API reporting. The
baseline checker also crashed during that diagnostic. A mixed racecheck run
crashed after the 24 contribution cases; the isolated repeat uses
`--racecheck-num-workers 1`. Device instrumentation remains enabled without
kernel filters. These diagnostics and successful check summaries are retained
in the combined JSON.

Ruff, formatting and `git diff --check` pass. Temporary checkpoints, source
copies and exploratory scripts are removed after verification; only the
implementation, regression tests and combined records remain in the project.
