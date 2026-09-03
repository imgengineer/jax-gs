# 性能实验记录

更新日期：2026-08-28。

## 记录规范

每条可保留的性能结论必须同时记录：硬件与软件环境、数据集/分辨率、物理容量、有效元素数、candidate bound、warmup/hot iterations、同步方式、正确性结果和最终决策。不同协议的绝对延迟不可直接比较；桌面 GPU 的逐调用 wall-time 结论应同时提供 Nsight 或持续排队测量。

## 当前保留结果

环境：Ubuntu 24.04、Python 3.12、CUDA 13、JAX 0.11.0、NVIDIA GeForce RTX 5090。

场景：Mip-NeRF360 Garden，138,766 active Gaussians，640×360，355,211 intersections，524,288 intersection capacity，2,048 max candidates/tile。

测量协议：

- Nsight：30 次 warmup，捕获 100 次 forward，`cudaProfilerApi` range，统计 GPU kernel count/total。
- 逐调用 wall-time：`benchmarks/benchmark_rasterization.py`，50 次 warmup、100 次 hot iteration，每次显式同步。
- 持续排队 wall-time：同一 JIT forward 每个 sample 连续提交 200 次，只同步最后一个输出；共 7 个 sample，并交换 baseline/mega 进程顺序。

| 日期 | 优化 | Baseline | Candidate | 正确性 | 决策 |
| --- | --- | ---: | ---: | --- | --- |
| 2026-08-24 | 32-bin tile-radix bucket prefix 合入 scatter | 41 kernels/frame；约 0.4817 ms median | 39 kernels/frame；约 0.4794 ms median | render/topology/overflow 逐位一致；15 tests passed | keep |
| 2026-08-24 | Tail mega kernel：range + segmented depth sort + sorted outputs + compositor | 39 kernels/frame；0.367 ms Nsight GPU total | 36 kernels/frame；0.359 ms Nsight GPU total | Garden forward 全字段逐位一致；loss 逐位一致；梯度 max-abs ≤ 1.31e-6；31 tests passed | keep |
| 2026-08-24 | Tail mega 持续排队 wall A/B，baseline-first | 0.7124 ms median | 0.6992 ms median | 同上 | keep（+1.88%） |
| 2026-08-24 | Tail mega 持续排队 wall A/B，mega-first | 0.7107 ms median | 0.6978 ms median | 同上 | keep（+1.84%） |
| 2026-08-28 | Uint8 warp radix count + 256-candidate mega compositor batch | tail `237.25 µs`；GPU kernels/frame `377.56 µs`；dynamic SMEM `48,176 B` | tail `233.04 µs`；GPU kernels/frame `373.26 µs`；dynamic SMEM `47,152 B` | 大 segment 稳定排序、257-candidate batch 边界与 fused forward/backward 对拍；27 tests passed | keep（tail +1.77%，GPU kernel total +1.14%；空闲卡 wall A/B 待补） |

逐调用同步、顺序平衡的三组 median-of-medians 为 baseline `0.4562 ms`、mega `0.4541 ms`，改善约 `0.47%`。该协议受动态时钟影响明显，只作为辅助结果。Tail mega kernel 的 Nsight 时间约 `231.1 µs`，使用 920 CTAs、256 threads/CTA、约 48 KiB shared memory。

2026-08-28 的 radix/compositor 结果来自两组反向进程顺序的 Nsight Systems profile；每个进程包含 1 次首次执行、20 次 warmup 与 100 次 hot iteration。两组 tail median 分别为 baseline `237.341/237.150 µs`、candidate `232.926/233.150 µs`；全部 GPU kernel 合计分别为 baseline `377.568/377.545 µs/frame`、candidate `373.209/373.310 µs/frame`。两路均为 920 CTAs、256 threads/CTA、46 registers/thread，kernel 实例数相同。测量时另一个训练进程持续占用 GPU（约 98–99% utilization），因此只把顺序平衡且方差稳定的 kernel 时间用于实现去留，不报告受污染的同步 wall-time；最终空闲卡 wall A/B 仍待补测。

## 历史 cuTile/Pallas 探针

以下结果来自旧环境与旧协议，只保留决策历史，不与当前 CuTe production 数值横向比较。

| iteration | optimization | latency_ms | correctness | status |
| ---: | --- | ---: | --- | --- |
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
| 12 | radix sort skips work beyond runtime `valid_count` | — | small cases PASS；multi-chunk 未复测 | incomplete |
| 13 | AABB map prefix short-circuit、AccuTile finite guards、65,536/tile safety bound | — | static review only | incomplete |

`incomplete` 行不得作为保留实现的性能或正确性证据；如需恢复实验，必须在当前环境重新建立 baseline、完整对拍并记录命令。
