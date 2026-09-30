# CuTe 光栅化与 binning 的负载均衡

2026-09-30，RTX 5090，驱动 `615.71.09`，JAX `0.11.2`、Flax `0.12.10`、Optax `0.2.8`、
CUTLASS DSL `4.8.0`。基线是本轮开始时的工作区，即
[独占性能验证](exclusive_performance_20260930.md) 的最终版（`2aeb5c4` 加上当时未提交的
算法优化）。计时均在 GPU 独占时串行执行，每次运行前确认没有其他 GPU 计算进程；
原始数据、命令和逐 epoch 历史见 [JSON 记录](kernel_balance_20260930.json)。

默认配置、训练规则、CuTe 渲染 + Optax 优化器组合、NNX 状态与 checkpoint 格式均不变。

## 结论

| 指标 | 基线 | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| 固定百万点更新，Optax，三轮中位数 | 3.851 ms | **2.980 ms** | −22.6% |
| 固定百万点更新，CuTe Adam，三轮中位数 | 3.514 ms | **2.652 ms** | −24.5% |
| 30k 完整训练，Optax | 89.43 s | **74.48 / 74.06 s** | −17.0% |
| 30k 完整训练，CuTe Adam | 71.05 s | **55.41 s** | −22.0% |

同一固定点数协议下已记录的 LiteGS 为 3.837 ms/步。前向渲染（packed 参数、RGB、
最终透射率、最后贡献者索引）与基线逐位一致，排序后的 pair 表与基线逐位一致。
所有完整训练都得到 993,152 个 Gaussian、8 个编译变体、无 overflow；PSNR 见第 5 节。

## 1. 光栅化：重尾 tile 决定 kernel 时间

基线中 raster backward 约 1.17 ms、forward 约 0.54 ms，占每步约 45%。
同一视图只保留最重的一个 tile（2,732 个 pair）时，前向 kernel 仍需 **294 µs**，
约为完整前向的一半；只保留最重的 64 个 tile 为 481 µs。单个 warp 顺序处理重 tile
的延迟决定了 kernel 时间，而不是总工作量。编译产物还显示：32 字节的 PackedParams
被拆成 7 条标量 `LDG.E`（偏移丢失了对齐信息），每个 splat 先读 id、再读参数，
形成依赖的全局访存链。

[packed_rasterize.py](../../src/jaxgs/kernels/packed_rasterize.py) 的改动：

1. **重 tile 优先**：单 block 计数排序按 log2 分桶（每倍频 8 桶）排列 tile。
   前向按 pair 数排序；前向顺带输出每个 tile 最后贡献者之前的 pair 数，
   反向按这个精确值排序。XLA `argsort` 排 8,034 个 tile 约需 40 µs，计数排序约 4 µs。
2. **每 block 一个 warp**：前向也改为单 warp block，重 tile 分散到不同 SM。
3. **warp 协作暂存**：每个 lane 预取后 32 个 splat 中的一个到寄存器，
   再经共享内存广播给整个 warp；参数读写都改为两条 128 位访问。
4. **前向每次迭代合成两个 splat**：两个 splat 的指数运算可以重叠，只有 T 递推是顺序的。
   越界的第二个 splat 被整体屏蔽，暂存数据始终是有限值；提前终止最多多看一个
   已被屏蔽的 splat，不影响任何输出。
5. **固定 FMA 形式**：前向与反向共用的二次型改为显式 `fma.rn` / `mul.rn`。
   向量化 load 曾让编译器改变收缩方式，导致约 0.02% 的像素出现 half ULP 差异；
   固定后前向与反向在任意指令调度下舍入一致，并与基线逐位一致。

这些改动不改变任何像素的合成顺序与算术。前向在视图 0、55、150（有 / 无统计）上
与基线逐位一致；fragment 统计与反向梯度只因原子加顺序不同，有 1e-8～1e-6 量级的相对差。

测过但未保留（数据见 JSON）：256 位参数 load（`LDG.E.ENL2.256`，比两条 128 位 load
前向慢 7%、反向慢 1.5%）；每 block 2 个 warp；反向 `red.v2/v4.f32` 向量原子加（无可测收益）；
反向每次迭代两个 splat（分开归约 −1.5%，合并归约使默认无统计路径变慢）；
反向 `min_blocks_per_mp` 占用率提示（无变化）；前向每次迭代四个 splat（比两个再快约 1%，
为控制寄存器压力保留两个）。反向每个 splat 约 370 条 SASS 指令，已接近发射吞吐上限。

## 2. binning：pair 发射

基线发射 kernel 约 215 µs，而做同样椭圆切片计算、只不写结果的计数 kernel 只需 24 µs。
每个线程负责一个 Gaussian，大 Gaussian 让整个 warp 空等：warp 的实际迭代数是
理想值的 **5.6 倍**，写入还按深度序偏移分散到整个 arena。

[sorted_visibility.py](../../src/jaxgs/kernels/sorted_visibility.py) 与
[sorted_binning.py](../../src/jaxgs/kernels/sorted_binning.py) 的改动：

1. **按深度序发射**：第 i 个线程处理深度序第 i 个 Gaussian，相邻 lane 写相邻段；
   同时去掉了把偏移 scatter 回池序的 XLA 操作。
2. **大 Gaussian 由整个 warp 发射**：超过 16 个 pair 时，32 个 lane 并行计算 32 行切片，
   warp 前缀和之后按 pair 平铺，合并写入。
3. **显式舍入的椭圆切片**：计数与发射调用同一组固定舍入的函数，每个 Gaussian
   写出的 pair 恰好等于计数。固定的舍入形式与此前通过 LiteGS 逐位核验的构建相同。

第 3 点修复了一个既有问题：基线的计数与发射 kernel 是同一函数的两次编译，
FMA 收缩方式不同。视图 0 中有一个近退化椭圆被计为 9 个 pair 却只写出 8 个，
留下一个填充槽位（`pair_count` 3,238,819，有效 pair 3,238,818）。
填充键排序后位于末尾，渲染结果不受影响，但计数与写入不一致。

四个视图上排序后的 tile 区间与 Gaussian ID 与基线完全相同；另外十个视图的计数与
发射结果逐位相同（视图 0 仅差上述填充槽位）。LiteGS 原生核验（8×8、8×16、16×16）
的 pair 表仍逐项一致，前向与梯度误差与既有记录相同。

## 3. 投影前向：SH 系数向量化读取

每个 Gaussian 的 SH 系数在内存中连续。[投影前向](../../src/jaxgs/kernels/projection.py)
改为按当前阶数读取系数前缀，用 128 位 load 代替 48 条跨步标量 load。
SH0～3 的全部输出与基线逐位一致。投影反向已达约 1.71 TB/s（显存带宽约 1.79 TB/s），
同样改动没有收益，未采用。

## 4. 验证

- 全量回归 **279 项通过**，Python 行与分支覆盖率均为 100%（1,694 条语句、250 个分支）。
  新增 [test_work_balance.py](../../tests/test_work_balance.py) 的 12 项检查：
  逐 lane 与整 warp 两条发射路径（含超过 32 行的切片）输出逐位相同、每段恰好写满、
  tile 内深度序稳定；tile 顺序是按工作量降序的排列；跨多个暂存批次、并在批次中间
  提前终止的前向与 float32 参考一致，`backward_work` 与最后贡献者一致，
  反向梯度与参考一致。
- `compute-sanitizer`：synccheck 0 错误；racecheck（限定 CuTe kernel）0 hazard；
  memcheck（限定 CuTe kernel）无设备端访存错误，报告的 7 项均为主机端
  `cuGetProcAddress_v2` 返回的 `CUDA_ERROR_INVALID_VALUE`。未限定 kernel 的 racecheck
  在两项测试通过后于工具内部崩溃，未报告 hazard。
- Ruff lint、format check 与 `git diff --check` 通过。

## 5. 计时

固定点数协议与 [独占性能验证](exclusive_performance_20260930.md) 相同：
1,000,064 个渲染点、`_DSC8679.JPG`（822×1237）、SH3、8,000,000 pair arena、
30 次预热后连续计时 300 次更新。基线与本轮按轮交替，各三轮。

| 版本 | Optax ms/步 | CuTe Adam ms/步 |
| --- | --- | --- |
| 基线 | 3.862 / 3.851 / 3.846 | 3.497 / 3.528 / 3.514 |
| 本轮 | 2.980 / 2.986 / 2.980 | 2.648 / 2.652 / 2.656 |

kernel 级中位数（同一视图与输入，每个样本 20 次调用，五轮交替）：

| kernel 组 | 基线 µs | 本轮 µs | 变化 |
| --- | ---: | ---: | ---: |
| 前向（排序 + pack + forward） | 616.4 | 271.8 | −55.9% |
| 前向，收集统计 | 914.1 | 489.6 | −46.4% |
| 反向（排序 + 清零 + backward） | 1,226.5 | 871.8 | −28.9% |
| 反向，收集统计 | 1,357.5 | 997.0 | −26.6% |
| pair 发射（清空 + 发射） | 239.7 | 100.4 | −58.1% |
| 深度排序到发射的全部 kernel | 334.3 | 189.1 | −43.4% |
| 投影前向，SH3 | 145.7 | 132.0 | −9.4% |
| 投影前向，SH0 | 51.3 | 49.8 | −2.9% |

完整训练沿用独占性能验证的协议：`images_4`、`resolution=-1`、`eval=true`、seed 0、
百万容量、8M pair arena，169 个训练视图、177 个 epoch、29,913 次更新；
计时不含预载、预热和 checkpoint 写入，PSNR 为 25 个留出视图的均值。

| 运行 | 训练秒数 | 预热秒数 | 峰值 pair | PSNR |
| --- | ---: | ---: | ---: | ---: |
| 基线，Optax | 89.43 | 10.00 | 3,337,910 | 25.428 dB |
| 本轮，Optax 第 1 次 | 74.48 | 10.55 | 3,312,747 | 25.465 dB |
| 本轮，Optax 第 2 次 | 74.06 | 10.59 | 3,289,080 | 25.444 dB |
| 基线，CuTe Adam | 71.05 | 10.89 | 3,290,238 | 25.498 dB |
| 本轮，CuTe Adam | 55.41 | 11.44 | 3,327,762 | 25.509 dB |

此前两份记录中同协议的基线 PSNR 为 25.426～25.510 dB，本轮各次均在这个范围内。
原子归约顺序与增密轨迹使每次运行的数值略有不同，单次 PSNR 差异不代表质量变化。
新 kernel 的编译使预热增加约 0.6 s。

## 6. 剩余热点

本轮最后一次 profile 采集时 GPU 已有其他训练进程，结果未采用。
前向两 splat 循环之前的独占 profile（每步 GPU kernel 合计 3.04 ms）中，
主要耗时为：raster backward 0.835 ms、Optax SH 参数与 m/v 更新 0.823 ms
（其余参数的 Optax 更新约 0.21 ms）、raster forward 0.277 ms（之后降至约 0.24 ms）、
投影反向 0.193 ms、投影前向 0.142 ms、tile 排序 0.094 ms、fused loss 0.134 ms。

- **Optax 全容量更新**：约 1 ms/步，占固定点数步长的三分之一。它按池容量遍历
  参数与 m/v，而 CuTe Adam 只更新可见 cluster。完整训练中两者相差约 18.9 s
  （74.3 s 对 55.4 s）。Optax 作为默认优化器是既定约束，本轮未改变。
- **raster backward**：每个 splat 约 370 条指令，已接近发射吞吐上限；
  最重 tile 单独约 0.52 ms。若要进一步缩短，需要在重 tile 内切分反向区间
  （先顺序恢复切分点的 T 与颜色，再并行处理各段），这会增加实现复杂度。
- **投影反向** 已接近显存带宽；**fused loss** 与 **排序** 各约 0.1 ms。
