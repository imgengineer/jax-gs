# 性能实验记录

更新日期：2026-09-12。

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

### 1M 第三轮排序流水线优化

协议保持不变：1M active/capacity、640×360、8,388,608 intersection capacity、16,384 candidate bound、10 次 warmup、30 次前向和 30 次 value-and-grad；实际 intersections 仍为 6,179,684，最大 tile candidates 仍为 8,905，无 overflow，loss 仍为 0.4960950315。

保留三项语义等价改动：

- Gaussian count 的 3,907 个 block sums 从单 kernel 串行遍历改为 256-block 局部 scan 加 16-chunk scan；该阶段在 20 帧 trace 中从 0.265 降至 0.035 ms/帧，整帧 GPU events 从 8.143 降至 7.922 ms。
- AccuTile emission 在流水线路径直接产生 `(uint64 sort key, gaussian id)`，不再先写 `(gaussian id, tile id)` 后由独立 kernel 重新读取并编码；去掉了 0.103 ms/帧的 key preparation，整帧 GPU events 进一步降至 7.838 ms，前向 allocator 峰值从 368,100,864 降至 344,088,064 bytes。
- radix block-local rank 的上界为 511，暂存类型从 int32 收窄为无损的 uint16；radix scatter 从 0.886 降至 0.695 ms/帧，最终整帧 GPU events 为 7.599 ms，相对本轮起点下降 6.7%。

| 指标 | 第二轮中位数 | 第三轮中位数 | 第三轮均值 | 第三轮最小值 | 第三轮 P90 | 最终吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward | 8.560 ms | 7.700 ms | 7.739 ms | 7.668 ms | 7.757 ms | 129.21 FPS |
| Value and grad | 11.918 ms | 11.440 ms | 11.444 ms | 11.311 ms | 11.554 ms | 87.39 iter/s |

相对第二轮最终基线，forward / value-and-grad 中位数分别下降 10.0% / 4.0%。新增的 70,013 项分层前缀测试覆盖 overflow 与非 overflow，cuTile 的稳定排序、畸形前缀和完整 AccuTile/JAX 拓扑对拍均通过。

本轮回退的实验：6-bit × 512 radix 超出 TileIR 编译资源；6-bit × 256 虽少两轮 scatter，但前向中位数 8.884 ms，相对同轮 8.675 ms 基线回退 2.4%；从已有 cumsum 末列提取 bucket totals 使 histogram trace 慢 4.0%；移除无效 lane 的 key padding select 没有可测收益。

## 2026-09-12 代码审查与第四轮优化

审查修复：`_prefix_counts_cutile` 的顶层扫描固定为 256 个 chunk，在直接输入超过 16,777,216 行时会漏计，并读取未写入的后续 chunk base。顶层扫描现按实际 chunk 数向上取 2 的幂。训练配置上限为 10M，不受该边界影响；直接交集 API 需要此修复。缩小层级后的边界回归，以及真实 16,777,217 行的完整 cumulative、required count、overflow / 非 overflow 两种容量验证均通过。

保留的语义等价优化：

- radix histogram 的局部 **扫描中间值**从 int32 改为 uint16；每个 block 的 inclusive count 最多为 512，不会溢出。上一轮只收窄了写入显存的 local-rank buffer，本轮进一步缩小 kernel 内扫描的工作集。
- radix chunk totals 并行扫描，并把 bucket base 融入 chunk prefix，1M 每帧少启动 9 个 kernel，scatter 每项少读取一个定位值。扫描按最多 256 个 chunk 分组，保留大容量支持。
- AccuTile 始终选择 clipped rectangle 的较短轴，count / emission 的静态遍历上限从网格较长边改为较短边；640×360、tile size 16 时从 40 缩短为 23。

环境与上一轮一致：RTX 5090、JAX 0.11.0、cuTile 1.6.0，禁止 JAX 显存预分配。大场景仍为 synthetic(seed=42)、1M active / physical capacity、640×360、8,388,608 intersection capacity、16,384 candidate bound。基线为本轮开始时的第三轮工作区实现；两版在同一进程内分别编译，每版 warmup 10 次，再交替执行 100 次同步调用，交替顺序逐次反转。另交替运行每版 30 组连续排队调用，每组 20 次、组末同步，用每组每调用耗时的中位数对照。桌面 GPU 有周期性长尾，以下报告中位数；不将不同日期、不同协议的绝对耗时直接比较。

| 1M 指标 | 本轮基线 | 优化后 | 耗时下降 |
| --- | ---: | ---: | ---: |
| Forward，逐调用同步 | 7.772 ms | 6.822 ms | 12.2% |
| Value and grad，逐调用同步 | 11.511 ms | 10.518 ms | 8.6% |
| Forward，连续排队 | 7.765 ms | 6.810 ms | 12.3% |
| Value and grad，连续排队 | 12.120 ms | 11.151 ms | 8.0% |

独立运行仓库 benchmark（10 warmup、30 hot、30 backward，均逐调用同步）的最终 forward / value-and-grad 中位数为 6.825 / 10.733 ms；实际 intersections 6,179,684，最大 tile candidates 8,905，所有 overflow 标志为 false，loss 保持 0.49609503149986267。JAX allocator 的 forward / backward 峰值为 344,088,064 / 553,523,712 bytes。20 帧 GPU trace 中，emission 从 0.611 降到 0.400 ms/帧，radix scan 与 bucket-base 合计从 0.113 降到 0.070 ms/帧；全部 GPU events 的平均和为 7.660 → 7.146 ms/帧，包含桌面干扰的长尾，不能与上表中位数视为同一统计量。

图像、alpha 和 overflow / count metadata 与基线逐值一致；完整梯度最大绝对差为 2.794e-9，通过 `atol=1e-6, rtol=1e-4` 检查。小场景另测 synthetic(seed=42)、10k active / capacity、320×180、131,072 intersection capacity、512 candidate bound，同样使用上述交替协议。实际 intersections 为 31,379，最大 tile candidates 为 206，无 overflow。Forward 中位数 0.3168 → 0.3142 ms，value-and-grad 0.8327 → 0.8295 ms；连续排队分别为 0.2559 → 0.2578 ms 和 0.7905 → 0.7849 ms，基本持平。早期 128 candidate bound 的小场景测量会截断候选，不纳入性能结论。

本轮验证：61 项 cuTile / AccuTile / 投影 / compositor / 训练相关测试通过，使用 `-W error::UserWarning`；新增或扩展了顶层 chunk 边界、256 / 512 block、多组 chunk 扫描、整 block 相同键的稳定排序、空输入有效前缀以及横向 / 纵向窄网格对拍。

回退实验：按每个 block 实际最大 span 引入动态循环，虽减少迭代数，却使 1M forward 从 uint16 变体约 7.05 ms 回退至约 7.68 ms；最终保留静态较短边上限。

## 2026-09-12 第五轮：并发训练负载下的打包排序优化

基线为第四轮完成后的工作区快照，不是 Git HEAD。环境仍为 RTX 5090、JAX 0.11.0、cuTile 1.6.0，禁止 JAX 显存预分配。测试期间另一个 `train.py` 持续运行，观察到 GPU 利用率约 96–98%，期间也有短暂波动；未干预该任务。按用户要求在负载下比较相对性能，以下绝对耗时不能与第四轮较空闲时的结果直接比较，也不能推算独占 GPU 延迟。

保留的改动仅在 radix histogram / local rank：把四个 bucket 的计数打包进 uint32 的四个 6-bit 字段，先在 32 元素子组内扫描，再将子组总数展开为 uint64 中的四个 16-bit 字段，扫描子组总数并合成稳定的 block-local rank。子组计数最多 32，block 总数最多 512，均不会跨字段进位。32-bucket、512-block 的主扫描张量从 32,768 降到 16,384 bytes，扫描长度从 512 缩短为 32；这不是整个 kernel 实际寄存器或 shared-memory 用量的测量。排序键、深度精度、稳定性和全局暂存 buffer 接口不变，仍使用 5-bit radix。

协议：synthetic(seed=42)、1M active / physical capacity、640×360、tile size 16、classic RGB、8,388,608 intersection capacity、16,384 candidate bound。两版在同一进程分别编译，复用同一组 device arrays，每版 warmup 10 次；交替执行每版 100 次逐调用同步测量，逐次反转 A/B 顺序；另交替测量每版 30 组连续排队调用，每组 20 次、组末同步。下表为第二次完整复测的中位数。

| 1M 指标 | 第四轮基线 | 打包扫描 | 耗时下降 |
| --- | ---: | ---: | ---: |
| Forward，逐调用同步 | 16.121 ms | 12.121 ms | 24.8% |
| Value and grad，逐调用同步 | 24.594 ms | 20.637 ms | 16.1% |
| Forward，连续排队 | 12.769 ms | 10.129 ms | 20.7% |
| Value and grad，连续排队 | 20.125 ms | 17.256 ms | 14.3% |

第一次完整复测的逐调用 forward 为 16.094 → 12.075 ms（-25.0%），value-and-grad 为 24.635 → 20.626 ms（-16.3%）；连续排队分别下降 23.7% / 11.4%。两次方向一致，但并发负载下的百分比仍有波动。

两次 1M 对拍的图像、alpha、count / overflow metadata 与基线逐值一致；完整 value-and-grad 最大绝对差分别为 2.980e-8 / 3.725e-9，通过 `atol=1e-6, rtol=1e-4`。实际 intersections 6,179,684，最大 tile candidates 8,905，无 intersection、candidate、tile 或 visible overflow。

10k 小场景使用相同交替协议，320×180、131,072 intersection capacity、512 candidate bound，实际 intersections 31,379、最大 tile candidates 206，无 overflow。逐调用 forward 为 2.534 → 2.523 ms，value-and-grad 为 3.067 → 3.053 ms，基本持平；连续排队分别为 0.5946 → 0.5832 ms 和 1.6530 → 1.5926 ms。图像与 alpha 逐值一致，value-and-grad 最大绝对差 5.960e-8。

验证：69 项 cuTile / AccuTile / 投影 / compositor / 训练相关测试通过，启用 `-W error::UserWarning`。新增 8 组 packed radix 参数组合，覆盖 256 / 512 block、2 / 4 / 32 / 64 buckets、空前缀、32 元素子组边界、512 元素 block 边界、部分 block，以及最高字段连续 512 个相同 digit 的稳定排序。Ruff、格式、锁文件和 diff 检查通过。

未保留的实验：双 bucket 打包收益较小；64 / 128 元素子组、4 / 6-bit radix、同 digit block 快路径和 compositor 8×8 子 tile 均未超过最终方案的稳定收益；未打包的 uint8 分层扫描遇到 TileIR 编译失败。生产 compositor 未改动。

本机原始记录与 A/B 脚本保存在 `/tmp/jax-gs-extreme-fOhgjC/`：`ab-pack4-1m-shared.json`、`ab-final-1m-round2.json`、`ab-final-10k.json`、`ab_benchmark.py` 和第四轮 `intersections_baseline.py` 快照；这些临时实验文件不属于仓库交付内容。

## 2026-09-12 第六轮：compositor 反向归约热点

本轮起点已提交并推送为 `cd5196b`（第五轮优化与前期告警修复）。在同一 RTX 5090、JAX 0.11.0、cuTile 1.6.0 环境重新采集 1M synthetic(seed=42) 的前向与 value-and-grad，各 20 帧 warm GPU trace；另一个 `train.py` 仍在运行，观察到 GPU 利用率约 96–98%，未干预该任务。

基线 value-and-grad trace 中，compositor backward 是最大的单一 kernel：GPU event 平均 7.555 ms/帧，占全部 GPU events 时长之和 17.610 ms/帧的约 42.9%；compositor forward 为 3.708 ms/帧。以上 event 时长包含并发调度干扰，不能解释为独占 GPU 的纯计算时间；部分排序 pass 也会被调度明显拉长，最终收益仍以同进程交替 A/B 和连续排队测量为准。

保留的改动只在反向归约：把两个 mean、三个 conic 和一个 opacity 梯度组成八列 tile（两列补零），一次按像素轴归约后以向量 atomic 累加；颜色梯度仍单独归约。补零列使用越界负索引丢弃，保持原有 relaxed atomic ordering。kernel 中的归约 / atomic 调用组从七组变为两组，所需的有效标量累加数量不变。没有新增全局缓冲区或配置项，不改候选顺序、梯度公式、精度、前向合成和排序。

最终协议沿用第五轮：1M active / physical capacity、640×360、tile size 16、classic RGB、8,388,608 intersection capacity、16,384 candidate bound；两版分别编译，复用同一组 device arrays，每版 warmup 10 次、100 次交替逐调用同步、30 组交替连续排队（每组 20 次、组末同步）。最终源码独立复测两轮，另以 10k active / capacity、320×180、131,072 intersection capacity、512 candidate bound 检查小场景。

| 1M value-and-grad 中位数 | `cd5196b` 基线 | 合并归约 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 第一轮，逐调用同步 | 20.542 ms | 16.892 ms | 17.8% |
| 第二轮，逐调用同步 | 19.881 ms | 16.901 ms | 15.0% |
| 第一轮，连续排队 | 16.897 ms | 14.275 ms | 15.5% |
| 第二轮，连续排队 | 17.176 ms | 14.359 ms | 16.4% |

纯前向未修改，逐调用中位数分别为 12.135 → 12.129 ms 和 12.115 → 12.111 ms；连续排队分别为 10.220 → 10.222 ms 和 10.189 → 10.259 ms，基本持平。并发负载会使基线中位数波动，因此报告两次结果，不把某次降幅视为独占 GPU 的固定收益。

两次 1M 对拍的图像、alpha 和 count / overflow metadata 与基线逐值一致；value-and-grad 最大绝对差分别为 4.657e-9 / 5.588e-9，通过 `atol=1e-6, rtol=1e-4`。实际 intersections 6,179,684、最大 tile candidates 8,905，无 intersection、candidate、tile 或 visible overflow。

10k 小场景逐调用 forward 为 2.521 → 2.524 ms，value-and-grad 为 3.053 → 2.794 ms（-8.5%）；连续排队分别为 0.5834 → 0.5833 ms 和 1.6950 → 1.0747 ms。图像、alpha 和 metadata 逐值一致，value-and-grad 最大绝对差 2.328e-10。实际 intersections 31,379、最大 tile candidates 206，无 overflow；排队与逐调用的降幅不同，包含小任务在共享 GPU 下的调度影响。

最终 20 帧 value-and-grad trace 中，compositor backward 的 GPU event 平均由 7.555 降至 3.899 ms/帧（-48.4%），全部 GPU events 的时长之和由 17.610 降至 13.945 ms/帧；同一次 trace 的 compositor forward 为 3.695 ms/帧，与基线 3.708 ms/帧 接近。该比较仍包含并发调度开销，不与逐调用中位数混用。独立仓库 benchmark 的最终 value-and-grad 中位数为 16.793 ms，loss 保持 0.49609503149986267；JAX allocator 的 forward / backward 峰值仍为 344,088,064 / 553,523,712 bytes，与基线完全相同。

验证：81 项 cuTile / AccuTile / 投影 / compositor / 训练相关测试通过，启用 `-W error::UserWarning`。反向对拍扩展至全部支持的 1 / 2 / 3 / 4 / 8 / 16 / 32 channels，覆盖完整与部分 tile，并确保重复 Gaussian 在两个 tile 中均有贡献；比较 mean、conic、color、opacity 和 background 梯度。Ruff、格式、锁文件和 diff 检查通过。

未保留的实验：32 候选批量预取加动态 tile 提取使前向从约 12 增至 41 ms，停止该实验；反向 8×4 像素分块未改善；把颜色也并入 16 列归约未超过六字段分组；额外打包输入属性虽使 1M 前向略快，但相对六字段分组使 10k 连续排队 value-and-grad 从 1.087 增至 1.233 ms，且引入额外缓冲区，未保留。

基准审计：最初的最终复测发现脚本切换基线时覆盖了默认候选函数，实际成为 A/A 测量，已剔除。脚本现提前保存候选函数、断言与基线对象不同，并记录两版实现来源后重跑；此前显式加载独立候选模块的探索实验不受此问题影响。原始脚本、基线快照、trace 和 JSON 保存在本机 `/tmp/jax-gs-hotspot-1uqoJF/`，最终数据为 `ab-verified-1m-round1.json`、`ab-verified-1m-round2.json`、`ab-verified-10k.json` 和 `final.json`，不属于仓库交付内容。

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
