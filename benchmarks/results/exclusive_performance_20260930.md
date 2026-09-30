# GPU 独占性能验证

2026-09-30，RTX 5090，驱动 `615.71.09`，JAX `0.11.2`、Flax `0.12.10`、
Optax `0.2.8`、CUTLASS DSL `4.8.0`。测试串行执行，监控未检测到其他 GPU 计算进程。
原始计时、配置、执行命令、checkpoint 检查和 profiler 汇总见
[JSON 记录](exclusive_performance_20260930.json)。

基线为 `2aeb5c4046fb131ecc0c6e3c512059332558d40b`。
最终版保留稳定候选 partition、子 Gaussian 容量分支、低阶 SH 梯度压缩、
对称 conic 原子加合并。pair 排序的动态容量分支在本次验证后撤回。
渲染默认 CuTe，优化器默认 Optax，模型由 Flax NNX 管理。

## 固定百万输入点的完整更新

输入为同一份 bicycle PLY：保留原有 975,104 个 Gaussian，并重复最后 24,896 行，
得到 1,000,000 个输入点。两端追加相同的 64 个尾部点完成 128-point cluster，
实际渲染点数均为 **1,000,064**。同一 `_DSC8679.JPG`、822×1237 图像、
SH degree 3、空间学习率倍率 1。JAX 使用 Optax，LiteGS 使用其原生 sparse Adam。
更新涵盖 culling、projection、binning、forward、loss、backward 和 Adam。

每版三轮串行测量，30 次 warmup 后计时 300 次连续更新；编译不计入时间。
前三个版本按轮交替执行，撤回排序分支后的最终版本随后重复三轮。
固定点数不包含增密。参数更新产生各自的数值轨迹，训练损失不要求位级相同。

| 版本 | 中位数 ms/步 | 三轮范围 ms/步 |
| --- | ---: | ---: |
| 优化前 jaxgs | 3.853 | 3.841–3.858 |
| 保留动态排序分支的候选版 | 3.862 | 3.861–3.863 |
| 最终 jaxgs | 3.854 | 3.826–3.855 |
| LiteGS | 3.837 | 3.836–3.847 |

最终版相对优化前耗时变化 **+0.018%**，相对 LiteGS 为 **+0.442%**。
这些差异与轮次波动接近，未测得明显的逐步训练提速。

## 完整训练流程

使用相同 bicycle、seed 0、百万容量、8,000,000 pair arena。
算法参数采用项目默认设置，仅模型配置覆盖 `images="images_4"`、`resolution=-1`、
`eval=true`。169 个训练视图，177 个完整 epoch，实际 **29,913 次更新**。
每个版本运行两次；基线和候选版交换过执行顺序，最终版在撤回排序分支后运行。
全部训练均包含 28 次增密以及 pruning、opacity decay、空间重排序；
计时排除数据 preload、warmup 和最终 checkpoint 写入。

| 版本 | 两轮训练秒数 | 平均秒数 | 25 个留出视图平均 PSNR |
| --- | --- | ---: | ---: |
| 优化前 jaxgs | 89.500 / 89.588 | 89.544 | 25.505 dB |
| 动态排序分支候选版 | 88.810 / 89.401 | 89.105 | 25.470 dB |
| 最终 jaxgs | 89.604 / 89.519 | 89.561 | 25.462 dB |

最终版全程耗时变化 **+0.019%**，整体吞吐持平。最终模型均为 993,152 个活跃
Gaussian，JIT cache 均为 8，没有 pair overflow。六份 checkpoint 的参数均有限，
容量为 1,000,000，`free_mask == ~alive` 且 active count 匹配。

最终版两次 PSNR 分别为 **25.426 / 25.498 dB**，基线为 **25.510 / 25.501 dB**。
平均差为 **−0.043 dB**；两次测量不足以判断稳定的质量差异，不能声称质量完全一致。
原子归约和后续增密存在数值轨迹差异。
本次 LiteGS 仅复测固定点数更新，未重新测量原生 LiteGS 的完整 3 万步训练。

## 增密单独计时

固定容量 1,000,000、活跃点 500,000、SH degree 3。
使用相同输入参数、统计和随机 key，预热后交替执行基线与当前函数，
每组各 50 次，记录同步到结果完成的中位数。

| 新生点数 | 基线 ms | 当前 ms | 耗时下降 |
| ---: | ---: | ---: | ---: |
| 0 | 1.669 | 1.274 | 23.7% |
| 4,992 | 1.706 | 1.334 | 21.8% |
| 49,920 | 1.740 | 1.422 | 18.3% |
| 249,984 | 1.787 | 1.641 | 8.1% |

增密实现有明确局部收益。完整训练仅在 29,913 次更新中增密 28 次，
因此该局部收益对全程秒数的影响很小。

## 排序分支撤回与实际热点

候选版的 `lax.switch` 每步产生一次 **4 字节设备到主机的计数读取**。
10 次更新的 GPU trace 记录到 10 次 D2H；显式启用 CONDITIONAL command buffer
仍不能消除这些读取。生产代码恢复为固定 arena 的单一路径稳定排序后，
最终 trace 的 **D2H 次数为 0**。没有保留新的全局 XLA 环境设置。

最终版 profile 包含 10 次更新，每步 GPU kernel 时间总和约 3.806 ms。
这项和式排除了部分拷贝与调度间隙，不作为墙钟迭代时间使用。

| 热点 | 平均 kernel 时间 ms/步 |
| --- | ---: |
| packed rasterization backward | 1.170 |
| SH 参数与 m/v 的 Optax 更新融合 kernel | 0.823 |
| packed rasterization forward | 0.543 |
| Gaussian/tile pair emission | 0.210 |
| projection backward | 0.195 |
| projection forward | 0.156 |

SH 更新归属由编译 HLO 的 `loop_add_select_fusion` 输入确认：
其三个输出均为 `[1,000,064,16,3]`，对应 SH 参数及其 m/v。
SH3 每步训练的主要时间仍在 rasterization 和参数更新。
这次增密和中间缓冲区优化没有显著改变这些主要成本。

## 编译资源与回归

在百万容量、822×1237 目标图像下，SH0/no-statistics 的 XLA 临时显存计划
从 **293,068,344 B** 降至 **202,164,792 B**，减少约 **86.7 MiB / 31%**。
SH3/statistics 仍为 **293,068,344 B**。argument、output 与 alias 字节均保持一致，
alias 为 730,000,004 B。这里是编译器计划，不是实际峰值显存测量。

撤回排序分支后，CuTe binning/rasterization、NNX donation 和 production trainer 的
**88 项相关回归测试通过**；Ruff lint、format check 和 `git diff --check` 通过。

## 复验入口

使用 JSON `execution` 中记录的命令可复验各轮。当前版本的固定点数入口为：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python benchmarks/fixed_model.py \
  /home/lzc/datasets/benchmarks/MipNerf360/bicycle \
  /tmp/jaxgs-performance-20260930/bicycle_1000000.ply \
  --backend jaxgs --optimizer optax --pairs 8000000 \
  --warmup 30 --steps 300 --output /tmp/fixed-jaxgs.json
```

完整训练使用 `/tmp/jaxgs-performance-20260930/bicycle_30k.toml`，其内容为：

```toml
[model]
images = "images_4"
resolution = -1
eval = true
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run jaxgs-train \
  /home/lzc/datasets/benchmarks/MipNerf360/bicycle \
  --config /tmp/jaxgs-performance-20260930/bicycle_30k.toml \
  --seed 0 --output /tmp/training-jaxgs.npz
```
