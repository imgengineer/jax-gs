# Flax NNX 固定状态绑定

2026-09-30；Flax 0.12.10、JAX 0.11.2、Optax 0.2.8。
基线包含上一轮 CuTe RGB 投影反向优化。
原始数据见 [验证记录](flax_partial_validation_20260930.json)。

## 文档依据

Flax 的 [performance guide](https://github.com/google/flax/blob/v0.12.10/docs_nnx/guides/performance.md)
指出，固定模型的 Python 状态遍历会增加调用开销。
当前固定模型已使用 tree mode；本轮采用树模式的
[`nnx.jit_partial`](https://github.com/google/flax/blob/v0.12.10/flax/nnx/transforms/compilation.py)，
预先展开固定状态。该 API 保留 Variable 引用，后续读取最新值并回写更新，
不需要每次调用重新查找这些 Variable。

## 代码改动

- [TrainingState](../../src/jaxgs/training/state.py) 将 GaussianModel、AdamState
  和 fragment statistics 放在同一个 NNX Module。Adam 使用 `nnx.OptState`，
  统计量使用 `nnx.Variable`；对应变量类型见
  [Flax optimizer 源码](https://github.com/google/flax/blob/v0.12.10/flax/nnx/training/optimizer.py)。
- [bind_train_step](../../src/jaxgs/training/step.py) 使用
  `nnx.jit_partial(graph=False, donate_argnums=(0,))`，一次绑定固定状态。
  Array/NNX 两个入口仍共用原有训练计算；独立 `train_step` 接口继续可用。
- [生产循环](../../src/jaxgs/training/trainer.py) 每个 frame 只传入当前视图和
  编译参数；Adam/statistics 在绑定状态内更新。每个 epoch 的 densification、
  opacity reset 与 spatial refinement 通过原 Variable 的 `set_value` 写回。
- 预热使用一份数组副本，在同一绑定函数上编译各个 SH/统计变体；`finally`
  恢复原始数组，再进入计时循环。正常完成和中断路径都验证了原始数组不被删除，
  恢复后的数组对象与缓冲区地址相同。初始 m/v 在预热阶段仍共享零数组，
  预热后分离 v，沿用原有的预热峰值内存安排。
- [固定点数基准](../fixed_model.py) 同步使用固定状态绑定；其报告标记为
  `nnx.jit_partial(graph=False)`。生产报告的 cache size 来自当前绑定函数。

状态树中 Adam、statistics 在 model 前面，以匹配 donation 的缓冲区顺序。
数组内容可以变化，Variable 对象与状态树结构保持固定。

## 验证与性能范围

CPU 微基准使用 128 容量的 Gaussian 参数、Adam 矩和统计量，执行简单数组更新。
预热后各测 5 轮、每轮 1,000 次调用，并在每轮结束等待完成。

| 调用路径 | 每次调用中位耗时 |
| --- | ---: |
| 逐次传 state/statistics/model 的 `nnx.jit` | 57.44 μs |
| 绑定固定状态的 `nnx.jit_partial` | 31.43 μs |

这个 CPU 微基准减少约 45.3%；它包含少量 CPU 数组计算，不能当作 GPU 完整训练
的加速比例。GPU 仍有其他训练任务，本轮未做独占 GPU 耗时对比。

真实 CuTe 检查覆盖 Optax/CuTe、SH0/SH3、统计开关：与 array step 的结果在
`rtol=2e-5, atol=2e-6` 下相同，15 个参数/m/v 缓冲区持续复用，连续步骤的 JIT
cache size 相同。预热后第一次训练调用也复用已经编译的变体。

抽象 GPU 编译对照共有 8 组，包括容量 128 和 1,048,576。
百万容量使用 1237×822 图像、4,194,304 pair arena；两种 NNX 入口的
argument/output/alias/temp memory 完全相同。默认 Optax 大场景的临时空间
均为 307,303,736 B，alias 空间均为 765,460,484 B。

65 个 CuTe 装饰函数与纯 `compute_training_step` 的 AST 均与基线相同。
固定点数基准的 128 点 GPU smoke 检查通过，cache size 为 1，无 overflow。

全量测试 `186 passed`，覆盖 63 次更新的完整训练调度、pruning、opacity reset
与 checkpoint。Python 宿主的 1,617 条语句和 224 个分支覆盖率 100%；CuTe 编译
函数体不计入 Python 覆盖率。Ruff 检查、格式检查、`git diff --check` 通过。
