# Visibility-gated SH evaluation — 2026-10-09

The `perf/sh-visibility` branch skips SH coefficient loads and color evaluation
for Gaussians rejected by the training projection's existing visibility test.
The frozen baseline is `238aa5196fc59f75cd35b17f2daee90fc951084e`.
**Correctness is validated; performance measurement is pending an exclusive GPU
window. This candidate has not been promoted to main.** The GPU was running
BAF-FSplat jobs throughout validation. No shared-GPU timings are treated as
performance evidence.

[Machine-readable validation and reproduction scripts](sh_visibility_20261009.json)
record the source patch, hashes, compiler evidence and state comparison.

## Implementation and scope

`project_with_compact_pullback(..., visible_color_only=True)` sets culled
points' colors to zero inside the listed clusters. The training step enables
this option. It preserves visible colors, geometry, the visibility predicates
and rasterization. Points rejected by visibility have no raster color
cotangents; projection backward already skips rows with no upstream gradient.
Ordinary projection and general VJPs keep the default complete color evaluation.
The undefined tails outside the listed clusters retain their existing contract.

Reviewing [google/spherical-harmonics](https://github.com/google/spherical-harmonics)
confirmed that its low-order
Cartesian polynomials match the approach already used here. No external code or
dependency was added. The compiled forward kernel already shares basis values
across RGB: for example, SASS computes the scaled SH20 value once in R19 and
uses it in all three channel accumulations. An extra basis array is therefore
not introduced.

Offline CuTe compilation with the production tensor shapes and `sm_120a`, followed
by device-binary inspection, gives:

| Active degree | Complete color registers/thread | Visible color registers/thread | Local memory / stack |
| --- | ---: | ---: | ---: |
| SH0 | 40 | 40 | 0 / 0 |
| SH3 | 43 | 44 | 0 / 0 |

SASS places a conditional branch before the SH coefficient loads and routes
culled lanes to three zero-color stores. This confirms skipped device work,
but divergence and the extra register may affect the net time. Compiler
resource counts do not establish a speedup.

## Input-work estimate

A CPU evaluation of the visibility predicates on the existing 975,104-point
bicycle PLY gives the following estimates. The denominator is the points in
frustum-selected clusters, rather than the entire pool. CPU/GPU rounding at
predicate boundaries may differ.

| View | Points in selected clusters | Culled within those clusters | SH evaluations skipped |
| --- | ---: | ---: | ---: |
| 0 | 621,824 | 109,485 | 17.61% |
| 48 | 580,096 | 110,818 | 19.10% |
| 96 | 564,480 | 118,035 | 20.91% |
| 144 | 535,936 | 128,015 | 23.89% |

Each skipped SH3 evaluation omits 48 float32 coefficients, or 192 logical
load bytes. These counts are not measured DRAM traffic or training speedups.

## Correctness

- Eight new cases cover SH0–3, empty cluster lists, partial clusters, offscreen
  points, near/far rejection, low opacity, dead points and clamped color.
  Visible colors, RGB, transmittance and parameter pullbacks match the complete
  color path bitwise. Fragment statistics pass the existing small tolerance
  for atomic reduction ordering.
- All 129 related packed-rendering, CuTe, SH-pullback, backend/interface and
  production-training regressions pass, for 137 distinct passing cases.
- An eight-update NNX comparison covers Optax, Muon and CuTe, SH0/statistics off
  and SH3/statistics on, with partial clusters and offscreen points. All 138
  state arrays pass `rtol=2e-5, atol=2e-6`; 63 are bit-exact. Maximum absolute
  difference is `2.3841858e-7`, and maximum relative L2 difference is
  `2.0780899e-8`. Every run preserves all 15 donated parameter/moment buffers,
  its single JIT entry and the incoming overflow state.
- Compute Sanitizer memcheck passes all eight new cases with zero device-memory
  errors. As in prior studies, CUDA API reporting is disabled with
  `--report-api-errors no`; device-memory checking is enabled.
- Ruff lint/format and `git diff --check` pass.

## Pending performance comparison

Use the same PLY for both source revisions with `benchmarks/fixed_model.py`:
Optax, SH3, 8M pairs, 30 warmup updates and 500 timed updates, views
0/48/96/144. Run three alternating baseline/candidate rounds in separate
processes. Set `PYTHONPATH` to the corresponding revision's `src` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

The raw record contains `run_checked.py`, which checks GPU availability before
launch and polls foreign compute processes every 0.5 seconds; a contaminated
run is rejected. It also contains the state checker and offline compiler
inspection scripts. Extract them beside `baseline/src` and `candidate/src`.
For example, from the repository root:

```bash
.venv/bin/python /tmp/sh-check/run_checked.py baseline baseline_view0 \
  .venv/bin/python benchmarks/fixed_model.py /path/to/bicycle /path/to/model.ply \
  --backend jaxgs --optimizer optax --pairs 8000000 --warmup 30 --steps 500 \
  --view 0 --output /tmp/sh-check/baseline_view0.json
```

If fixed-model timings improve repeatably, follow with matched 10k/30k full
training and held-out quality evaluation before promoting the candidate.
