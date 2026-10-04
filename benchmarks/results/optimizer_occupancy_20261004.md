# Optimizer occupancy and cache-hint follow-up — 2026-10-04

This is the exclusive-GPU follow-up to the [optimizer memory study](training_memory_20261003.md), based on `1800632a2be57b7e870d4a51281443896e219372`. The raw record is [optimizer_occupancy_20261004.json](optimizer_occupancy_20261004.json). It compares the frozen production optimizer with three row-local Pallas/Triton variants and then measures the only repeatable candidate through complete 10k and 30k training runs.

The accepted change adds `eviction_policy="evict_first"` to the visible-cluster optimizer's masked and unmasked loads and its stores. The update arithmetic, masks, aliases, program shape, Optax state transitions and dense fallback are unchanged. The hint tells Triton that this one-pass row working set should not displace longer-lived raster data from cache.

## Fixed-model measurement

The same 975,104-point bicycle PLY starts every run with fresh Adam state. Each variant has three alternating process rounds on views 0, 48, 96 and 144, with 30 warmup updates and 500 timed complete NNX updates. The table reports median milliseconds per update and the reduction in elapsed time relative to the baseline.

| View | Baseline | Rows 16 | Rows 8 | Read/write eviction hints |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.805 ms | 1.805 ms (−0.03%) | 1.820 ms (−0.85%) | **1.791 ms (+0.79%)** |
| 48 | 1.685 ms | 1.692 ms (−0.42%) | 1.702 ms (−1.00%) | **1.670 ms (+0.85%)** |
| 96 | 1.624 ms | 1.631 ms (−0.45%) | 1.640 ms (−1.01%) | **1.612 ms (+0.69%)** |
| 144 | 1.587 ms | 1.598 ms (−0.72%) | 1.613 ms (−1.68%) | **1.573 ms (+0.85%)** |

The optimizer kernel profile moves from 604.0 µs to 591.5 µs per update (2.07%) with the accepted hint. Rows 16 and 8 reduce the reported register count from 242 to 106 and 96 and raise theoretical occupancy from 16.7% to 33.3% and 41.7%, but their extra programs make the complete update slower. They are rejected.

All fixed-model jobs returned successfully, preserved the donated buffers and one-entry JIT cache, and observed no foreign GPU process. The common model hash, environment and complete samples are in the raw JSON record.

## Full training measurement

Three fresh-process pairs per budget use Optax, 8×16 tiles, a one-million-slot pool, seed 0 and a 30k position-learning-rate horizon. Timing covers the training loop, including densification, pruning, opacity decay, spatial refinement and epoch reports; it excludes preload, warmup and final checkpoint writing. Evaluation uses the frozen production JAX renderer on 25 held-out views.

| Budget | Baseline median | Candidate median | Less time | Baseline PSNR median | Candidate PSNR median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 11.699 s | **11.641 s** | **0.50%** | 24.552 dB | 24.549 dB |
| 30k | 38.168 s | **37.768 s** | **1.05%** | 25.467 dB | 25.446 dB |

The small PSNR differences are retained as a limitation: the mean candidate differences are −0.021 dB at 10k and −0.028 dB at 30k across these three runs, while the candidate is bitwise different because the GPU training path is not fully deterministic. The cache hint does not change the mathematical update; longer multi-scene quality validation would still be needed for a stronger equivalence claim.

## Validation and disposition

The production change passes the 51 targeted visible-optimizer, Optax, SH-pullback and Muon tests. Ruff, formatting and `git diff --check` pass. The same stream candidate had already passed the isolated 95-array optimizer comparison, the 138-array end-to-end state check and 17 CUDA memcheck cases documented in the preceding study. Every fixed-model, training and evaluation monitor reported zero foreign GPU compute processes.

The read/write eviction hints are enabled by default in `src/jaxgs/kernels/visible_optax.py`. The smaller-row variants remain archived as rejected experiments. This is a measured single-GPU optimization; it does not establish a universal speedup across GPU generations or scenes.
