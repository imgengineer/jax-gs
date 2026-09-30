# CuTe + Optax 代码与算法性能审查

基线：`11d598fe03120642daf7e1f94f5ac30b7fde2807`，2026-09-30。
默认生产路径是 `jaxgs-train` → `nnx.jit` → CuTe projection/binning/rasterization
和 Optax 更新。reference backend 用于诊断。下面保留未修改运行时代码时的审查结果，
以及随后对屏幕边缘问题的修复记录。

GPU 正在共享。本次进行了 CPU 编译/等价性检查，以及两个很小的 CuTe/LiteGS
GPU 正确性检查，没有重新测 GPU 性能。[原始数据](static_performance_audit_20260930.json)
包含具体输入、CPU workload 对比和核验结果。

## 先处理的算法偏差：屏幕边缘预剔除

[projection.py](../../src/jaxgs/kernels/projection.py) 在 projection 内用
`mean ± 3 * sqrt(max_eigenvalue)` 做屏幕边缘剔除。LiteGS
`binning.cu::get_allocate_size_kernel` 使用中心的 ±15% 屏幕边界、深度和 opacity
粗筛，再用 `sqrt(2 * log(alpha * 255))` 的椭圆支持域生成 pair。
高 opacity 时这个支持域可以超过 3σ。

实际 CuTe 反例：256×128 图像，fx=fy=128，cx=128，cy=64；一个 Gaussian 的
z=2，投影均值约 `[-24.100006, 64.5]`，屏幕标准差 8，opacity=0.99，RGB=0.5。
cluster frustum 检查通过，但投影的半径约 24 使其 `visible=False`。

| 路径 | pair 数 | 像素 (x=0, y=64) 的 RGB |
| --- | ---: | --- |
| 修复前 CuTe 生产投影 + binning + packed forward | 0 | [0, 0, 0] |
| 同一投影数据，绕过额外的 3σ 屏幕预剔除 | 4 | [0.0043907, 0.0043907, 0.0043907] |
| LiteGS 原生 get_allocate_size，相同均值/σ/opacity | 4 | 本次只检查 pair 数 |

这是审查时已复现的算法差异，已在后续修复中处理，见下节。性能比较前应让生产可见性筛选遵循原生 binning
条件，并增加边缘、高 opacity、各向异性和裁剪视图的检查。修正后新增的有效 pair
可能增加渲染工作量，因此不能把更激进的剔除算作性能收益。

[reference rasterizer](../../src/jaxgs/reference/rasterizer_jax.py) 还有
`exponent >= -4.5` 截断；reference projection/binning 也使用 3σ。
只比较这些 reference 路径可能共同漏掉同一类贡献。应保留直接原生核验和独立的
alpha 阈值用例。一般可微投影的 radius 输出还有调用者，不能直接删掉。

### 屏幕边缘修复与核验

生产 CuTe projection 和 JAX reference projection 改为中心的 ±15% 屏幕粗筛；
生产的椭圆 pair count/emit、packed forward/backward 继续使用现有原生算法。
3σ radius 输出及其可微接口保留。reference/诊断路径的圆形 tile/cluster 边界
扩展到覆盖 `alpha >= 1/256` 的范围，并去掉逐像素的 `exponent >= -4.5` 截断。
这个保守圆边界仅用于 reference/诊断路径。

新增 [边缘回归测试](../../tests/test_edge_support.py)，修复前 CPU 的九个检查失败，
CuTe packed 回归和完整训练步也失败；其中生产训练步得到零个 pair。修复后该
完整 CuTe + Optax 训练步得到四个 pair，loss、梯度统计与参数更新均恢复。
三个 CuTe 渲染接口的 xyz、log_scale、rotation、opacity、SH 梯度还与单像素
解析贡献的自动微分结果对比，避免依赖共同截断的 reference rasterizer。

使用 [已有原生核验工具](../packed_parity.py) 对左右上下四个边界，以及各向同性和
旋转的各向异性 Gaussian，共八组输入核验。pair 数、Gaussian IDs、tile ranges
完全一致；完整 RGB 和 mean/conic/color/alpha 梯度完全一致。fragment weight
的最大相对 L2 差为约 `1.06e-4`，其余核验结果见
[修复验证记录](edge_support_validation_20260930.json)。各组均无 overflow。

全量 GPU 测试 `165 passed`，Ruff 检查通过。Python 宿主代码覆盖了全部 1535 条语句
和 224 个分支，行/分支覆盖率 100%；CuTe 编译的 kernel/DSL 函数体不计入 Python
覆盖率分母。现有可微 radius、NNX donation 和缓存稳定性检查也包含在这次测试中。

这些是在共享 GPU 上完成的正确性核验。修复恢复了原来漏掉的有效贡献，性能计时
及完整场景训练质量需要在后续 matched benchmark 中重新测量。

## 已有独占测量用于确定优先级

以下来自 [已有 profile](bicycle_optax_default_profile.json)，不是本次的新计时。
条件是 975,104 点、1237×822 单视图、4M pair arena、10 个预热后的完整训练步，
使用历史 10k benchmark 配置。

| 阶段 | GPU ms/step | 源码中的直接原因 |
| --- | ---: | --- |
| raster backward | 1.156 | 每 tile 逆序重建 T，warp 归约，向 Gaussian 全局梯度累加 |
| Optax 更新 | 0.990 | 全容量梯度映射、参数和 m/v 更新；SH 单个 kernel 约 0.802 ms |
| raster forward | 0.490 | tile 内按深度串行合成，每像素对计算 alpha 和 T |
| pair count + emit | 0.237 | 两次椭圆切片计算，emit 的每个线程串行写该 Gaussian 的 tile 列表 |
| projection backward | 0.193 | 几何反向计算、SH 共享内存转置 |
| projection forward | 0.152 | 几何/SH 激活及投影 |

这些阶段的时间不是可直接相加得到的通用训练耗时；不同图像和 densification 阶段
的 workload 不同。旧 profile 也不代表恢复后的 30k 默认协议。

## 优先尝试：densification 的第二次排序

[densify.py](../../src/jaxgs/training/densify.py) 的第一次排序实现加权无放回抽样，
需要保留其抽样语义。第二次排序的键只有三类：选中的 split、选中的 clone、未选中。
而 `selected` 是原序列的一个前缀，因此第二次排序可以改为一次前缀和与稳定 scatter。

设 `i=arange(C)`、`selected=i<raw_budget`：

```python
prefix = jnp.cumsum(selected & split, dtype=jnp.int32)
destination = jnp.where(
    selected,
    jnp.where(split, prefix - 1, prefix[-1] + i - prefix),
    i,
)
order = jnp.zeros_like(i).at[destination].set(i, unique_indices=True)
```

所有 destination 都唯一且处于 `[0, C)`：split 按原序占据第一个区间，clone
占据第二个区间，未选中项保持原位置。CPU 已检查 90 个等价用例，覆盖容量
1/2/127/128/129/1024、空/满预算、全 split/全 clone 和随机分布。
一百万容量的 StableHLO 从一个 sort 变为零个 sort、一个 scatter。

这是最小范围、已验证算法等价的优化候选。它尚未接入生产，也没有 GPU 性能结论。
后续检查真实 densify 输出、slot reuse、Adam 重置和 buffer donation，再测整个
densify 周期；不能仅凭少了一个 sort 就认定端到端更快。

## 最大持续开销：Optax 全容量 SH/moment 流量

[optimizer.py](../../src/jaxgs/training/optimizer.py) 先把紧凑梯度映射回池槽位，
再对五个全容量参数叶子执行 Optax 更新。`active` 控制更新结果与状态冻结，
不会改变这些数组的执行域。`active_degree` 没有传给优化器，因此早期低阶 SH
阶段仍处理 `[C, 16, 3]` 的完整 SH 状态。

在 C=975,104 时，SH 有 46,804,992 个 float32 标量。按参数/gradient/m/v 四次
读取和参数/m/v 三次写入估算，SH 的逻辑流量约 1.31 GB/step，全部参数约
1.61 GB/step。这是逻辑访问估算，缓存、融合和条件加载会影响实际 DRAM 流量。
CPU cost_analysis 也不能替代 GPU 带宽测量。

同一个已训练模型的 169 视图中，平均只通过约 3,739/7,618 个 cluster；
Optax 仍生成全容量参数和状态更新。这支持继续检查稀疏执行与 SH 数据布局。

LiteGS 将 SH DC/rest 作为独立参数。固定形状下分离相应叶子，或给早期从未启用
的高阶 SH 保留独立状态，是可对照的方向，但需要检查合并布局和新增拷贝。
已经启用过的 SH 即使本步梯度为零，也必须保留 Adam momentum 衰减；可见 cluster
不能仅按非零梯度更新。改用现有 CuTe Adam 并不满足本次保留 Optax 的约束。

## 最大渲染开销：反向循环与梯度累加

[packed_rasterize.py](../../src/jaxgs/kernels/packed_rasterize.py) 的 backward 对每个 tile
逆序恢复 transmittance，每个贡献 Gaussian 执行几何/颜色归约，以及 10 次全局 atomic。
旧 profile 是 75 registers/thread、一个 warp/block、估算 occupancy 50%。
时间和这些资源指标不能区分递推依赖、缓存访问与 atomic contention 的各自贡献。

一个可检查的局部候选是 conic 的内部对称表达：forward 只使用 c00/c01/c11，
backward 却把相同 gc01 分别 atomic 到两个 off-diagonal 位置，projection backward
随后再把二者相加。生产路径用一个合并的 off-diagonal 梯度可少一次 atomic/贡献 pair，
但要保留通用 custom_vjp 的矩阵梯度约定，并验证浮点累加顺序与训练质量。
这项候选没有实现或测量；它与已测失败的 lane redistribution 是不同的改动。

## 固定容量不要求对整个 padding arena 排序

[sorted_binning.py](../../src/jaxgs/kernels/sorted_binning.py) 对 C 个 depth 排序，
再对 P_MAX 个 tile key 做稳定排序。[sorted_visibility.py](../../src/jaxgs/kernels/sorted_visibility.py)
先清理 P_MAX 个 pair，最后 tile range kernel 也扫描整个 arena。

历史完整训练记录中的 P_MAX=8M，而各 epoch 的最大有效 pair 数最多为 3,237,850
（40.47%）。旧 4M 单视图 profile 中两个 radix sort 合计约 0.12 ms；这是值得改进
的额外工作，但它比 backward 和 Optax 小。上述占用率不能用于缩小 30k 默认协议
的安全容量，新默认图像更大，视图/生长过程也会改变峰值。

LiteGS 也预分配表，但会使用反馈的 predicted allocation size，并把 radix sort
限制到使用的 tile bits。当前 uint16 tile key 已做过位宽优化。
接下来可检查设备端有效前缀排序/扫描：外部 buffer 形状固定，kernel 只处理有效
长度。`jax.lax.sort` 没有动态有效长度参数，简单加 mask 仍会排序全部 P_MAX；
需要专用实现和稳定深度顺序验证，并避免每步读回 CPU 获取长度。

## RGB 投影反向仍包含零梯度路径

[packed_backward](../../src/jaxgs/kernels/packed_rasterizer.py) 的 depth 输出恒为零，
radius cotangent 也恒为零，但 [projection_backward.py](../../src/jaxgs/kernels/projection_backward.py)
仍读取两者，并执行 radius 的 eigen/discriminant/sqrt 反向运算。
CuTe FFI 对外部 kernel 的计算是不透明的，XLA 无法根据零输入删掉内部算术。

生产 RGB 训练可以使用静态特化跳过这两条梯度路径，并减少零 depth/radius buffer
的清理与读取。普通 custom_vjp 的任意 depth/radius cotangent 仍需要通用 kernel；
必须保留已有的单独 cotangent 和有限差分检查。前向 radius 当前参与可见性条件，
删除前向计算之前要先解决前述筛选差异。

紧凑 projection backward 仍按 C_MAX 发射所有 CTA，每个 CTA 预留 SH shared memory，
随后无条件进入 barrier 和共享内存转置循环。空 CTA 可以尝试按整个 block 的共同
有效性条件跳过工作；不能让部分线程在 barrier 前返回。编译后的指令是否已有相关
消除，以及这种 guard 能节省多少，仍需检查 GPU 编译输出和计时。

## 生长阶段还有按 C_MAX 生成子点的额外工作

当前 densify 对整个 C_MAX 生成 jitter、计算 rotation、读取候选参数，再用
`destination=C_MAX` 丢弃无效写入。历史 1M 容量训练每轮出生约 25k–64k 个点，
大部分随机数/几何生成没有写入子点。LiteGS 只对选出的 split/clone 生成这些值。

可以让 child generation kernel 只处理设备端 budget 的有效前缀，同时保持输出形状
固定。抽样预算包含 pruning 的补偿，不能直接把历史 50k 点当作硬上限。
复用被 prune 的槽位时，parent 数据可能仍被后续 child 读取，原地更新之前应先
缓存所选 parent/child 数据，避免读写竞争。

## 低优先项与已排除的方向

* 10-bit 与 LiteGS 实际默认 21-bit Morton 精度不同。CPU 在同一 frozen model 上
  重排全部 975,104 个活点，并检查 169 视图：平均可见 cluster 从 3738.864 降至
  3730.284，约减少 0.23%。此场景收益有限；更宽的排序键也有成本，暂不优先改。
* Grain 解码和 uint8 GPU preload 在训练计时区间之前；它们不是这条稳态训练路径
  的主要热点。epoch 末的同步用于 overflow/loss 检查，不能把每个 Python 语句
  都视为一次设备同步。
* Flax/Chex/Optax 的 Python 对象构造与静态断言主要发生在 tracing；已有 Chex
  检查前后的 StableHLO 相同，不应靠删除它们优化 GPU 时间。
* `collect_stats=False` 返回的未使用 zeros 可被 XLA 消除，需要结合编译结果判断，
  不能仅按 Python 中看到的临时数组计入设备流量。
* 已测慢并撤回：静态 prefix Optax 分支、把 raster atomics 分配给更多 warp lane、
  简单 gather 改写。重做这些尝试需要新的机制和证据。

建议顺序：先补屏幕边缘算法核验并对齐原生筛选；随后尝试稳定分区替换第二次排序
和 RGB 反向特化；较大改动再处理 child generation、Optax SH 布局与有效前缀排序。
所有性能判断最终以 matched 全训练计时、相同 overflow 约束与质量检查为准。
