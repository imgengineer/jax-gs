| iteration | optimization | latency_ms | correctness | status |
|----------:|:-------------|-----------:|:------------|-------:|
| 0 | cuTile radix-4 baseline, capacity=65,536 | 0.254580 | PASS | keep |
| 1 | radix-5, capacity=65,536 | 0.209315 | PASS | keep |
| 2 | radix-6, capacity=65,536 | 0.279845 | PASS | revert |
| 3 | radix-5 with 128-element blocks | 0.218116 | PASS | revert |
| 4 | occupancy sweep 1/2/4/8; best=2 | 0.209876 | PASS | revert |
| 5 | fuse single-chunk histogram scans | 0.205741 | PASS | keep |
| 6 | compositor scalar forward baseline | 0.130631 | PASS | keep |
| 7 | compositor forward candidate batch=8 | 0.078880 | PASS | keep |
| 8 | compositor scalar backward baseline | 0.755772 | PASS | keep |
| 9 | compositor backward candidate batch=8 | 0.326991 | PASS | keep |
| 10 | compositor candidate batch=4 | 0.101096 | PASS | revert |
| 11 | compositor candidate batch=16 | 0.103105 | PASS | revert |
| 12 | radix sort skips work beyond runtime `valid_count` | — | small cases PASS; multi-chunk not rerun | pending host recovery |
| 13 | AABB map prefix short-circuit, AccuTile finite guards, 65,536/tile safety bound | — | static review only | pending host recovery |

Fresh measurement is paused because even a `JAX_PLATFORMS=cpu` module import
reproduced a glibc `ld.so` relocation assertion and exited with status 127.
Do not attribute the older rows to iterations 12–13 until the host is stable and
the isolated correctness/backward/scale suite has been rerun.
