# 可见 cluster 上的 Optax 与每步固定开销

2026-10-01，RTX 5090，驱动 `615.71.09`，JAX `0.11.2`、Flax `0.12.10`、Optax `0.2.8`、
CUTLASS DSL `4.8.0`。基线是 `8becbae`（[负载均衡](kernel_balance_20260930.md) 一轮的提交）。
计时均在 GPU 独占时串行执行，每次运行前确认没有其他 GPU 计算进程；原始数据、命令与
逐 epoch 历史见 [JSON 记录](visible_optax_fixed_costs_20261001.json)。

Optax 仍是默认优化器，且保持可扩展：生产更新直接运行未修改的 `tx.update`，
任何逐行（row-local）的 Optax 变换都可替换进来。CuTe Adam 因可扩展性差不作为默认，
本轮也没有为 Adam 做专门化。默认配置、训练规则、NNX 状态与 checkpoint 格式不变。

## 结论

| 指标 | 基线 `8becbae` | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| 固定百万点更新，Optax，三轮中位数 | 2.979 ms | **2.392 ms** | −19.7% |
| 30k 完整训练，Optax | 74.76 s | **43.72 / 43.67 s** | −41.6% |
| 每步 GPU kernel，池占用 10% | 1,629 µs | **442 µs** | −72.9% |
| 每步 GPU kernel，池占用 100% | 2,741 µs | **1,938 µs** | −29.3% |

留出视图 PSNR：基线 25.458 dB，本轮 25.428 / 25.444 dB，在此前同协议记录的
25.426～25.510 dB 范围内。所有完整训练都得到 993,152 个 Gaussian、8 个编译变体、无 overflow。

与基线相比，投影输出、pair 表（Gaussian ID 与 tile 范围）、前向 RGB / 透射率 /
最后贡献者索引、loss 数值逐位一致（视图 0、55、150，含 / 不含统计）。梯度的相对 L2
差在 2e-5 以内，来源见第 3、4 节；与 LiteGS 的一致性在每一项上都不差于此前记录。

## 1. Optax 只更新可见 cluster

基线的 Optax 更新按池容量遍历全部参数与 m/v：即使只有 10% 的槽位存活，
每步也要约 0.93 ms（三个 XLA fusion）。CuTe Adam 只更新可见 cluster，但它把
Adam 写死在 kernel 里，替换优化器就要重写 kernel。

[visible_optax.py](../../src/jaxgs/kernels/visible_optax.py) 用一个 Pallas kernel，
在可见 cluster 的行块上追踪执行 Optax 变换自身的 `tx.update` 与
`optax.apply_updates`：每个 program 读入 32 行的参数、状态与紧凑梯度（形状补到 2 的幂，
补位以 0 读入、写回时屏蔽），只写回存活的行。其余槽位的参数与状态不变。
变换须是逐行的（槽位的更新只依赖本行的梯度、状态与参数），状态叶需按池槽位排列；
测试中 LiteGS Adam 与稠密更新逐位一致，clip / trace / scale 链与稠密更新的相对差在 1e-6 以内。
sm_120 上 Pallas 只能经 Triton 后端（JAX 0.11 已标记弃用，Mosaic GPU 不支持 sm_120）；
后端不可用时自动回到稠密更新。

池占用 100% 时，profile 的 22 个视图平均 51.6 万个可见 cluster 槽位，该 kernel 每步约
0.565 ms：读写参数、m、v 并读取紧凑梯度共约 0.86 GB，约 1.5 TB/s（显存峰值的 85%）。

## 2. 光栅化：warp 归约与统计

消融实验显示，去掉反向的 warp 归约与原子加，固定视图的反向从 854 µs 降到 538 µs：
每个有贡献的 splat 有 7 次 `redux` 与 10 次 shuffle 的依赖链，再加 10 条单 lane 原子加。

- **转置蝶形归约**（`half2.warp_sums`）：每一步把一半的值交给配对 lane，
  2^k 个和只需 2^k + 4 − k 次 shuffle（8 个和 9 次，原来每个 5 次）。
  几何梯度与平方误差共 8 个 float 和，RG/BA 两个 half2 和；之后每个 lane 持有一个和，
  整个 warp 用一条原子指令写回（lane 各自的地址、步长与系数在循环外算好）。
- **前向统计**：fragment 数与权重和同样走蝶形归约；整个 warp 都没有有效 fragment 的
  splat 跳过归约与原子加（只会加 0）。

| kernel 组（固定视图，三轮交替中位数） | 基线 µs | 本轮 µs |
| --- | ---: | ---: |
| 反向 | 858.4 | 816.7 |
| 反向，收集统计 | 988.0 | 865.0 |
| 前向 | 247.0 | 246.1 |
| 前向，收集统计 | 457.7 | 369.1 |

前向合成的顺序与算术不变，输出逐位一致。float 蝶形和与 LiteGS 的共享指数整数归约
舍入不同；RG/BA 的 half2 求和顺序改为与 LiteGS 相同的 xor 蝶形，颜色梯度与 LiteGS
的相对 L2 差从 4.4e-4 降到 5e-8 量级。

测过但未保留：向量原子加 `red.v2/v4.f32`（反向无收益）；每 block 2 或 4 个 warp
（单 warp block 每 SM 最多 24 个，但多 warp block 反而慢 3～11%）；
前向统计把两个 splat 的计数打包后用向量原子加（比只做跳过慢）。

## 3. L1 + SSIM 合成一个 kernel

原实现是 LiteGS fused-ssim 的两个 kernel：前向写出逐像素 loss 与 3 个偏导数平面
（48 MB），反向再带 halo 读回模糊。新 kernel 每个 block 负责 32×16 输出 tile，
在共享内存里依次完成：读入图像与目标 tile（36×52，零填充）、横向求 x、x²、y、y²、xy
的窗口和、纵向求统计量并算 SSIM、偏导数与 loss、偏导数横向与纵向模糊得到梯度。
统计量在 halo 上重算，loss 图与偏导数不再落到显存。

- 每个线程计算若干相邻输出，共享输入；纵向统计按行流式累加，寄存器从 128 降到 94，
  每 SM 两个 block。
- tile 用 `cp.async`（越界零填充）拷贝；下一通道的 tile 在当前通道最后一次模糊时预取。
- 偏导数中的 7 次 IEEE 除法改为两个近似倒数，与原梯度的相对差约 7e-7。
- 用 `clock64` 逐阶段计时定位：纵向统计（含偏导数）是最重的阶段。

fused loss（含 loss 均值的 XLA 归约）：118.6 → 75.8 µs。各个模糊仍按源实现的
抽头顺序累加。测过但未保留：32×32 tile（512 线程）、持久化 block 跨 tile 预取、
完全展开 tile 读取（寄存器超过 128，每 SM 只剩一个 block）。sm_120 上
`fma.rn.f32x2` 编译成两条标量 FFMA，没有吞吐收益。

## 4. binning：只处理真实存在的数据

基线每步都按容量处理：深度排序 1M 个 Gaussian（XLA 的 `argsort` 用 u64 载荷）、
XLA 前缀和（多个 reduce-window fusion，约 18 µs）、清空并排序整个 8M pair arena、
扫描整个 arena 求 tile 范围。

1. **深度键**：计数 kernel 直接写出可基数排序的深度键（符号翻转，负数取反；
   无 pair 的 Gaussian 排最后），再用 `lax.sort((keys, iota))` 得到 CUB 的
   u32/s32 键值排序。有 pair 的 Gaussian 的深度序不变。
2. **前缀和**：两个 CuTe kernel（向量化读写）按深度序收集计数并求前缀和，5～7 µs。
3. **发射**：多于 16 个 pair 的 Gaussian 先入队，由第二个 kernel 的整个 warp 发射；
   行宽大时整行写出，窄行仍按 32 个 pair 一组二分定位。输出与原来逐位一致。
4. **tile 排序**：[pair_sort.py](../../src/jaxgs/kernels/pair_sort.py) 是只排真实 pair 的
   稳定 LSD 基数排序：每块 digit 直方图、每个 digit 跨块扫描、warp 用 `match.any`
   稳定排名后在共享内存中按序暂存，每个 digit 的段连续写出。8,034 个 tile 需两轮 7 位。
   pair 数从设备内存读取，训练仍完全异步；arena 中 pair 之后的槽位既不清空也不读取。
5. **tile 范围**：只扫描到 pair 数为止。

| 可见性表构建（百万容量，视图 0） | 基线 µs | 本轮 µs |
| --- | ---: | ---: |
| 池占用 10%（322,591 个 pair） | 242.5 | 102.9 |
| 池占用 100%（3,238,582 个 pair） | 336.5 | 223.1 |

## 5. 训练路径少清零

训练投影在不可见 cluster 上只清零可见标记（1 MB，原来清零全部投影字段 49 MB）；
训练反向只清零可见 cluster 的梯度，深度梯度不写（RGB 训练不读），收集统计时
透明度梯度全部清零。计数、打包、发射与投影反向都只读取可见 Gaussian；
通用的 `project_cute_vjp` 与光栅化自动微分接口仍输出完整定义的数组。

## 6. 验证

- 全量回归 **328 项通过**；Python 行与分支覆盖率 100%。新增测试：Optax 可见 cluster
  执行器（与稠密更新逐位一致、其他逐行变换、无 Triton 后端时回到稠密路径）、
  pair 排序（1～3 轮、0～70,001 个 pair、u16/u32 键、越界计数）、计数前缀和、
  cluster 限定的反向与完整反向一致、发射路径用显式填充检查每段恰好写满。
- 与 LiteGS 的直接对比（`benchmarks/packed_parity.py`，8×8 / 8×16 / 16×16 tile）：
  pair 数、Gaussian ID 与 tile 范围精确一致，各项误差均不高于此前记录。
- 与基线的端到端对比：见“结论”。梯度相对差：图像梯度 1.5e-6，光栅化梯度 3e-6～1.3e-5，
  参数梯度 9e-6～1.9e-5，fragment 统计 6e-9。
- `compute-sanitizer`（限定 CuTe kernel）：racecheck 无 hazard，synccheck 0 错误；
  memcheck 无设备端错误（7 项均为主机端 `cuGetProcAddress_v2` 返回值）。

## 7. 计时

固定点数协议与上一轮相同：1,000,064 个渲染点、`_DSC8679.JPG`（822×1237）、SH3、
8,000,000 pair arena、30 次预热后计时 300 次更新。

| 版本 | Optax ms/步 |
| --- | --- |
| 基线 `8becbae` | 2.976 / 2.985 / 2.979；漂移检查 2.978 |
| 本轮（tile 排序之前） | 2.462 / 2.455 / 2.470 |
| 本轮 | 2.392 / 2.387 / 2.392 |

完整训练沿用上一轮协议：`images_4`、`resolution=-1`、`eval=true`、seed 0、百万容量、
8M pair arena，29,913 次更新；计时不含预载、预热和 checkpoint 写入。

| 运行 | 训练秒数 | 预热秒数 | 峰值 pair | PSNR |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 74.76 | 10.53 | 3,317,300 | 25.458 dB |
| 本轮（tile 排序之前） | 45.98 | 12.37 | 3,252,897 | 25.442 dB |
| 本轮第 1 次 | 43.72 | 13.71 | 3,383,642 | 25.428 dB |
| 本轮第 2 次 | 43.67 | 13.77 | 3,306,423 | 25.444 dB |

新增的 Pallas / CuTe kernel 使预热增加约 3.2 s（每个进程一次）。

## 8. 剩余热点

池占用 100% 时（每步 1.94 ms）：raster backward 0.62 ms、Optax 0.57 ms（接近带宽）、
raster forward 0.17 ms、投影反向 0.14 ms 与投影前向 0.11 ms（均接近带宽）、
fused loss 0.08 ms、深度排序约 0.05 ms。池占用 10% 时（每步 0.44 ms）：
raster backward 0.10 ms、loss 0.08 ms、Optax 0.05 ms、深度排序约 0.04 ms。

- 反向约 90% 的 splat 迭代至少有一个像素有贡献，逐像素 half2 计算是主要开销；
  继续缩短需要改变这部分计算，会改变数值。
- 投影反向与 Optax 分开读写参数与梯度；融合两者可省约 20% 的流量，
  但需要把投影反向改写为 Pallas 可追踪的形式。
- 深度排序仍按容量进行；先压缩有 pair 的 Gaussian 再计数感知地排序，
  估计可省 15～25 µs。目标图像的 uint8→float 转换每步约 7 µs。
