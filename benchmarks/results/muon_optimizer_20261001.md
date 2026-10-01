# Muon 优化器（与 LiteGS Adam 相当的学习率）

2026-10-01，RTX 5090，驱动 `615.71.09`，JAX `0.11.2`、Flax `0.12.10`、Optax `0.2.8`、
CUTLASS DSL `4.8.0`。计时均在 GPU 独占时串行执行；原始数据见
[JSON 记录](muon_optimizer_20261001.json)。

`--optimizer muon`（或 `[runtime] optimizer = "muon"`）走与默认 Optax 相同的路径：
同一个按可见 cluster 执行的 Pallas kernel、同一套按槽位的状态与 LiteGS 学习率。
默认优化器仍是 Optax LiteGS Adam。

## 结论

| bicycle 协议 | Optax Adam | Muon |
| --- | ---: | ---: |
| 固定百万点更新 | 2.381 / 2.400 ms | **2.238 / 2.233 / 2.232 / 2.242 ms** |
| 30k 完整训练 | 43.72 / 43.67 / 43.42 s | **41.52 / 41.99 s**（最终代码） |
| 预热（编译） | 13.8 s | 16.9 s |
| 留出视图 PSNR | 25.458 / 25.428 / 25.444 / 25.410 dB，均值 25.435 | 25.519 / 25.494 / 25.438 / 25.461 / 25.475 / 25.451 dB，均值 25.473 |

Muon 的质量与 Adam 相当（均值高约 0.04 dB，约为单次运行波动的两倍）；
每步更快，因为 SH 不再有二阶矩读写。

## 1. 哪些参数用 Muon

Muon 对线性映射的更新做正交化。每个 Gaussian 的 SH 系数是从 SH 基到 RGB 的线性映射，
因此把它的 DC 行（1×3）与高阶系数（[(d+1)² − 1]×3）当作两个矩阵，按 Gaussian 分别正交化
（相当于 `optax.contrib.muon` 以 Gaussian 维为 batch 轴）。位置、尺度、旋转、不透明度是
每个 Gaussian 的向量，沿用 LiteGS Adam，与 `optax.contrib.muon` 对非矩阵参数用 Adam 一致。

不把整个池的 [N, 3] 位置当作一个矩阵：那会把互不相关的 Gaussian 耦合在一起，
更新也不再逐行独立，无法在可见 cluster 上稀疏执行。

细节沿用 `optax.contrib.muon` 的默认值：Nesterov 动量 0.95、Frobenius 预归一化、
5 步五次 Newton-Schulz（系数 3.4445, −4.7750, 2.0315）。偏差校正按槽位自己的更新次数计算
（稀疏更新下每个槽位的步数不同）。不可见与空闲槽位冻结。

状态沿用 Adam 的按槽位布局（m 存动量，SH 的 v 保持为 0），所以增密、槽位复用、
重排、不透明度重置与 checkpoint 都不需要改动。

## 2. “相当的学习率”

`optax.contrib.muon` 的 `consistent_rms` 把每个正交化矩阵缩放 sqrt(max(行, 列)) × r，
使更新 RMS 约为 r，从而沿用 AdamW 的学习率（语言模型中 AdamW 约 0.2）。
这里按 LiteGS Adam 实测的归一化更新 m / sqrt(v) 的 RMS 选 r：

| 情形 | LiteGS Adam 更新 RMS |
| --- | --- |
| 收敛模型重新开始（动量清零后 600 步） | 1.4 → 0.42～0.50，各字段接近 |
| 初始点云训练（3,000 步，无增密） | 0.6～1.0 → 0.22～0.34；SH 新阶激活时短暂升到 0.6～0.7 |

30k 完整训练中对 r 的扫描（单次运行波动约 ±0.03 dB）：

| r | 0.1 | 0.2 | 0.4 | 0.6 | 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PSNR dB | 25.427 | 25.458 / 25.454 | 25.519 / 25.494 / 25.438 / 25.461 / 25.475 / 25.451 | 25.488 | 25.362 |

0.2～0.6 是一个平台，0.1 与 0.8 变差。取 r = 0.4：它等于 Adam 稳态更新 RMS，也在平台中央。

## 3. 消融（r = 0.4）

| 变体 | PSNR | 训练秒数 |
| --- | ---: | ---: |
| SH（DC + 高阶）Muon，其余 Adam（采用） | 25.473（6 次均值） | 41.5～42.0 |
| 所有字段都用逐 Gaussian Muon（向量即归一化动量） | 25.266 | 42.43 |
| 只有高阶 SH 用 Muon，DC 用 Adam | 25.434 | 44.60 |

几何向量上的 Muon（逐 Gaussian 归一化）明显变差；DC 行用 Muon 是收益的来源。

## 4. 性能

Muon 的 Newton-Schulz 在 Pallas（Triton）可见 cluster kernel 中最初每步 1.3 ms
（Adam 0.56 ms）。分步定位：

- 独立测试中同样的计算几乎不花时间；用 SH 的真实布局（每行 16×3，步长 48，
  补到 16×4 并屏蔽第四列）后变慢：Triton 无法向量化 12 字节的行，跨 SH 行的规约
  （每次约 12 µs / 50 万个 Gaussian）占主导。
- 每个 program 只用一个 warp 时，这些规约留在 warp 内：独立测试从 720～900 µs 降到 334 µs。
- 把 Gram 矩阵改用 3×3 小矩阵迭代可减少规约，但对秩亏矩阵误差升到 1e-4，未采用；
  采用逐列形式（与 Optax 相差约 1e-6）。

因此可见 cluster kernel 的 program 形状改为参数（`ProgramShape`）：Adam 保持
32 行 × 4 warp；Muon 用 4 行 × 1 warp、每个 program 处理 8 块。kernel 不再加载与写回
变换原样返回的状态叶（Muon 的 SH 二阶矩）。Newton-Schulz 用 `fori_loop` 而非完全展开，
每个变体的编译从 1.65 s 降到 0.6 s，运行时间不变。

| 优化器 kernel（每步） | 池占用 10% | 池占用 100% |
| --- | ---: | ---: |
| Optax Adam | 53 µs | 564 µs |
| Muon | 75 µs | 476 µs |

## 5. 验证

- 全量回归 344 项通过，Python 行与分支覆盖率 100%（1,928 条语句、282 个分支）。
- [test_muon.py](../../tests/test_muon.py)：逐 Gaussian 正交化与 `optax.contrib.muon` 的
  `orthogonalize_via_newton_schulz`（以 Gaussian 为 batch 轴）相差 < 1e-5，补零的行列不影响结果；
  完整更新与基于 Optax 正交化的参考实现一致（含按槽位偏差校正、SH 阶数 0/1/3、冻结槽位），
  几何字段与 LiteGS Adam 逐位一致；可见 cluster kernel 与稠密路径一致；更新 RMS 符合设定；
  槽位重置。NNX 捐赠训练测试新增 muon；配置与命令行接受 muon。
- 固定点数协议上 Muon 的最终 loss 更高（0.0138 对 0.0129）：该协议从收敛模型出发，
  在单一视图上做 300 次更新；Muon 每个矩阵的步长不随梯度噪声变小，
  而 LiteGS 的 SH 学习率不衰减。完整训练的留出 PSNR 不受影响。
