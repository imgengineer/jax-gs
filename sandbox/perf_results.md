# 性能实验记录

更新日期：2026-09-11。

## 记录规范

每条可保留的性能结论必须同时记录：硬件与软件环境、数据集与分辨率、物理容量、有效元素数、intersection capacity、candidate bound、warmup/hot iterations、同步方式、正确性结果和最终决策。不同协议的绝对延迟不可直接比较；桌面 GPU 的逐调用 wall-time 结论应同时提供 profiler 或持续排队测量。

## 当前状态

原生 GPU 流水线已统一为 NVIDIA cuTile Python 1.6。旧原生 kernel 后端的测量结果不再代表当前实现，已从当前基线中移除。

cuTile 已建立下述 synthetic 算法基线；真实场景基线仍待使用固定 NPZ 重建。任何新结果都必须先通过 cuTile 与权威 JAX 路径的投影拓扑、渲染值、overflow metadata 和反向梯度对拍。

## 2026-09-11 cuTile 算法优化基线

环境：NVIDIA GeForce RTX 5090（compute capability 12.0，32,607 MiB），driver 615.71.09，JAX 0.11.0，cuda-tile 1.6.0，`XLA_PYTHON_CLIENT_PREALLOCATE=false`。所有 wall time 都显式同步；场景为 `synthetic(seed=42)`，tile size 16，classic RGB。

保留的算法改动：

- radix histogram、scan 与 scatter 只处理运行时 `valid_count` 前缀，不再对 intersection capacity 的空尾部重复排序；10k Gaussian、320×180、31,379 / 131,072 intersections 的交替 A/B 中位数为 1.775 → 0.774 ms（-56.4%）。
- 每个 radix bucket 的串行 block scan 改为 256-block 局部 scan 加 chunk-total scan。100k Gaussian、640×360、617,419 / 1,048,576 intersections 的 GPU trace 中，scan 总时长为 3.937 → 0.062 ms/帧。
- compositor backward 从 tile 内所有像素的 `max(last_id)` 开始反向遍历，并跳过整块无贡献候选的梯度代数和零值 atomic；gradient atomic 使用 relaxed ordering。相同大场景的最终 backward kernel trace 为 1.170 ms/帧。

最终大场景协议：`--capacity 100000 --active 100000 --resolution 640x360 --max-intersections 1048576 --max-candidates-per-tile 2048 --k 1024 --warmup-iters 10 --hot-iters 30 --backward --backward-iters 30 --allow-unsafe`。实际 intersections 为 617,419（58.88% fill），最大 tile candidates 为 943，无 intersection、candidate 或 tile overflow。

| 指标 | 中位数 | 最小值 | 迭代 |
| --- | ---: | ---: | ---: |
| Forward | 1.237 ms | 1.228 ms | 30 |
| Value and grad | 2.624 ms | 2.565 ms | 30 |

### 1M Gaussian 扩展测试

协议：`--capacity 1000000 --active 1000000 --resolution 640x360 --max-intersections 8388608 --max-candidates-per-tile 16384 --k 16384 --warmup-iters 10 --hot-iters 30 --backward --backward-iters 10 --allow-unsafe`；其余环境与 synthetic 参数同上。实际 intersections 为 6,179,684（73.67% fill），最大 tile candidates 为 8,905，无 intersection、candidate、tile 或 visible overflow。完整 value-and-grad loss 为有限值 0.496095。

| 指标 | 中位数 | 均值 | 最小值 | P90 | 吞吐 | 迭代 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | 11.572 ms | 11.614 ms | 11.330 ms | 11.882 ms | 86.10 FPS | 30 |
| Value and grad | 15.455 ms | 15.446 ms | 14.999 ms | 15.801 ms | 64.74 iter/s | 10 |

benchmark 的保守工作集估算为 4.34 GiB；JAX device allocator 报告的峰值 in-use 为 545.9 MiB、pool 为 898 MiB，进程峰值 RSS 为 1.40 GiB。两种口径覆盖范围不同，不应视为等价的整卡显存测量。首次空编译缓存的 forward cold time 为 5.675 s；正式进程复用编译缓存后，forward / value-and-grad cold time 分别为 856.8 / 969.3 ms，cold time 不纳入热态性能结论。

### 1M 瓶颈复查与第二轮优化

优化前的 20 帧 GPU trace 中，radix sort 为 5.273 ms/帧（47.0%），compositor forward 为 4.081 ms/帧（36.4%），intersection build 为 1.007 ms/帧（9.0%），其余为 0.850 ms/帧（7.6%）。保留下列改动：

- radix 尾轮只实例化剩余有效 bits 对应的 buckets；本场景最后 2-bit pass 的 histogram 从约 456 降到 212 us。
- compositor forward 每 16 个候选检查一次 all-done 状态，并在所有有效像素都达到 transmittance cutoff 后结束该 tile 的候选循环；kernel trace 从 4.087 降到 1.511 ms/帧。单 tile、8,192 个低于 alpha threshold 的透明候选最坏测试中，不检查、逐候选检查和每 16 个候选检查分别为 2.120、3.046 和 2.254 ms，分段检查将纯 reduction 回退从 43.7% 限制到 6.4%。
- intersection capacity 至少为 4,194,304 时 radix 使用 512-block，否则保持 256-block。512-block 在 1M 上约快 3%，但 10k 固定使用时慢 6.6%，因此不做全局替换。

最终协议与上述 1M 协议相同，仅将 `--backward-iters` 提高到 30。实际 intersections、最大 tile candidates、loss 与 overflow metadata 均保持不变。

| 指标 | 优化前中位数 | 优化后中位数 | 优化后均值 | 优化后最小值 | 优化后 P90 | 最终吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | 11.572 ms | 8.560 ms | 8.501 ms | 8.235 ms | 8.734 ms | 117.64 FPS |
| Value and grad | 15.455 ms | 11.918 ms | 12.016 ms | 11.758 ms | 12.331 ms | 83.22 iter/s |

相对第一轮 1M 基线，forward / value-and-grad 中位数分别下降 26.0% / 22.9%。最终 trace 中 radix sort 为 4.770 ms/帧（58.7%），已经是下一阶段的主要瓶颈；compositor forward 降至 1.511 ms/帧。大规模 512-block 变体的 JAX allocator 峰值为约 642 MiB，观察到的首次 forward compile 为 8.60 s；这是用约 96 MiB allocator 峰值和更长冷编译换取热态吞吐。

回退的实验：4-bit radix 从 6.82 ms 回退到 8.27 ms；12-float AccuTile state 相比 16-float 对齐布局没有稳定收益；128-block radix 在 1M 上从 11.572 回退到 12.672 ms。固定 512-block 虽对 1M 有利，但使 10k 从 0.409 回退到 0.436 ms，最终改为按 capacity 静态选择。6,291,456 的紧 intersection capacity 可把 1M 前向进一步压到 8.348 ms，但只留 1.8% headroom，不作为生产默认值。

## 重建命令

```bash
uv run python benchmarks/benchmark_rasterization.py \
  --npz /path/to/scene.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --backend intersections \
  --projection-backend cuda_tile \
  --compositor-backend cuda_tile \
  --intersection-backend cuda_tile \
  --intersection-mode accutile \
  --sort-backend cuda_tile \
  --max-intersections 524288 \
  --max-candidates-per-tile 2048 \
  --backward \
  --allow-unsafe

uv run python benchmarks/autotune_cutile.py \
  --npz /path/to/scene.npz \
  --capacity 138766 \
  --active 138766 \
  --resolution 640x360 \
  --max-intersections 524288
```

## 结果模板

| 日期 | 环境 | 场景 / shape | 协议 | Baseline | Candidate | 正确性 | 决策 |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-09-11 | RTX 5090 / JAX 0.11.0 / cuTile 1.6.0 | synthetic seed 42 / 100k / 640×360 | 上述最终大场景协议 | Fwd 1.237 ms；V+G 2.624 ms | radix valid-prefix + hierarchical scan；backward last-id/zero-work pruning | cuTile 单测、跨 chunk 稳定排序和训练对拍 | 保留 |
| 2026-09-11 | RTX 5090 / JAX 0.11.0 / cuTile 1.6.0 | synthetic seed 42 / 1M / 640×360 | 1M 扩展协议 | Fwd 11.572 ms；V+G 15.455 ms | 同上；8,388,608 intersections / 16,384 candidates | 无 overflow；有限 loss；同一实现已通过 JAX 对拍门禁 | 1M 基线 |
| 2026-09-11 | RTX 5090 / JAX 0.11.0 / cuTile 1.6.0 | synthetic seed 42 / 1M / 640×360 | 第二轮优化协议 | Fwd 8.560 ms；V+G 11.918 ms | mixed-width radix + large-job 512-block；forward 16-candidate all-done check | 32 个 cuTile/JAX/训练对拍通过；无 overflow；loss 不变 | 保留，替代 1M 基线 |
