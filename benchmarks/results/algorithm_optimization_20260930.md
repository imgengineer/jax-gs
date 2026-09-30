# 算法优化与模型命名验证

基线为 `2aeb5c4046fb131ecc0c6e3c512059332558d40b`。默认配置、训练规则与
CuTe 渲染 / Optax 优化器组合保持一致。此次按顺序实现五项优化，并统一模型命名。
完整数值记录见 [JSON](algorithm_optimization_20260930.json)。

后续 GPU 独占测试发现第五项在逐步训练中引入 4 字节 D2H 同步，
生产路径已恢复为完整 arena 的单次排序。当前代码保留前四项优化。
最终耗时、profiler 与质量对照见
[独占性能验证](exclusive_performance_20260930.md)。本文以下保留首次实现的验证记录。

## 五项修改

1. **候选分组**：保留加权 Gumbel 采样排序，将第二次 split/clone 分组排序
   改为稳定 partition。CuTe 每 256 个候选计算一个计数，再扫描这些块计数、写出排列。
   百万容量只扫描 3,907 个块计数；CPU/reference 路径使用固定形状的稳定 scatter。
   分组结果仍为 selected splits、selected clones、未选尾部，各组内部顺序不变。
2. **子 Gaussian 生成**：按设备端 birth budget 选择预编译的容量分支，
   从一个 cluster 开始按四倍增长，返回的模型数组和 Adam 状态仍是完整固定容量。
   默认 partitionable Threefry 的随机前缀保持一致，其他 PRNG 设置回退到完整容量生成。
   读取原始父参数后写入已释放槽位，新生槽位的 Adam 状态仍清零。
3. **SH 梯度带宽**：Optax 路径的 CuTe 投影反向仅输出当前启用的 SH 系数，
   SH0/1/2/3 分别为 1/4/9/16 个。compact 梯度只 gather 这些系数，
   随后补回零梯度尾部。高阶系数已有的 Adam 动量仍衰减，并更新可见参数；
   模型参数和 m/v 的形状保持完整容量。通用 VJP 与 CuTe 优化器仍使用完整 SH 梯度。
4. **对称 conic 原子加**：生产 packed 参数反向将两个相同的非对角梯度合并，
   每次贡献少一次全局 atomic add。投影反向读取两项之和，数学梯度保持一致；
   浮点归约顺序可能不同。通用矩阵 VJP 保留原来的独立非对角梯度约定。
5. **pair 排序范围**：设备端 pair count 选择预编译的静态前缀容量，
   从 65,536 开始按两倍增长，直到完整 arena。前缀中的 padding key 是 sentinel，
   稳定排序保留 Gaussian 深度顺序，输出 ID 数组仍为完整 arena，overflow 检查保留。
   例如 8,000,000 容量中有 3,200,000 个 pair 时，排序范围为 4,194,304。
   Gaussian 深度排序、pair 清零和 emission 仍按完整容量执行。

这些修改保持现有的采样权重、目标点数、split 缩放、pruning、opacity decay、
学习率和 SH 激活规则。参考源码为
[LiteGS densification](https://github.com/MooreThreads/LiteGS/blob/004b95215c90c36cdaf4b354301132b700ac287b/litegs/training/densify.py)、
[LiteGS sparse Adam](https://github.com/MooreThreads/LiteGS/blob/004b95215c90c36cdaf4b354301132b700ac287b/litegs/training/optimizer.py)；
排序使用 [JAX stable sort](https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html)。

## 命名

NNX 模型统一为 `GaussianModel`，固定数组视图由 `GaussianPool` 改为 `GaussianArrays`。
`model.as_arrays()` 共享参数缓冲区，`model.update_from_arrays()` 更新既有 Variables。
相关接口统一为 `create_gaussians`、`seed_gaussians`、`reorder_gaussians`、
`save_gaussians` 和 `load_gaussians`，项目内调用方和文档同步更新。
NPZ 字段和 NNX 参数结构没有改变。

模型命名前后的百万容量 SH0 与 SH3 训练图 StableHLO 哈希完全相同，
编译器参数、输出、alias 和临时显存计划也完全相同。因此该命名调整没有改变训练计算图。

## 正确性验证

- 最终全量回归：**270 项通过**。
- Python 行与分支覆盖率均为 **100%**，1,704 条语句、256 个分支。
  CuTe 编译的 kernel/jit 函数体不计入 Python 覆盖率。
- `compute-sanitizer --tool synccheck`：partition 和 SH compact 梯度的
  **25 项检查通过，0 errors**，包含未满 cluster 和无可见点情况。
- 与基线完整 densification 对照的五组输入涵盖空模型、容量已满、释放槽位再生、
  部分 cluster 和不同 birth budget；模型、m/v、step、birth/prune count 均逐元素一致。
- 稳定分组 oracle、PRNG 前缀与回退、动态 budget 不增加 JIT cache、
  高阶 SH 非零动量、紧凑梯度、对称 conic 参数梯度、pair 前缀排序与 overflow 均有回归检查。
- Ruff lint、format check 和 `git diff --check` 通过。

## 编译资源

固定容量 1,000,000、SH 存储 degree 3、cluster 128、tile 8×16、
pair arena 8,000,000，训练目标图像为 822×1237。以下为 **XLA 编译器显存计划**，
不代表 profiler 测到的峰值显存或迭代时间。

| 训练变体 | 基线临时字节 | 优化后临时字节 | 变化 |
| --- | ---: | ---: | --- |
| SH0，不收集统计 | 293,068,344 | 202,199,608 | 减少 90,868,736 B，约 86.7 MiB / 31% |
| SH3，收集统计 | 293,068,344 | 293,070,904 | 增加 2,560 B |

两种变体的 argument、output、alias 字节都与基线一致，
其中 alias 保持 730,000,004 B。模型与 Adam 缓冲区复用保持一致。
单独百万容量 densification 的临时计划从 3,004,472 B 降至 2,008,160 B，
输出固定容量不变。

## 计时限制

RTX 5090 当前与其他训练进程共享。交替执行的百万容量 densification probe
包含 0、4,992、49,920、249,984 个 birth，计时有明显调度波动；
单次 trace 的优化后 kernel 数从 63 增至 93，记录到的 kernel 总时长下降，
但墙钟时间上升。trace 没有设备到主机的数据拷贝。

这些数据只能说明执行路径和调度开销，不能证明吞吐提升或没有性能回退。
仍需 GPU 独占时，以相同输入、状态、warmup 和迭代数进行完整训练对照；
本次没有重新发布相对 LiteGS 的速度结论，也没有执行完整 30,000 步质量对照。

## 复验命令

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q --cov=jaxgs
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
git diff --check
XLA_PYTHON_CLIENT_PREALLOCATE=false compute-sanitizer --tool synccheck \
  --error-exitcode 86 .venv/bin/python -m pytest -q \
  tests/test_training_protocol.py tests/test_packed.py \
  -k 'cute_partition_preserves or active_sh_prefix'
```
