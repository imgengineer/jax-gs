# Flax / JAX 文档对照整理

日期：2026-09-30。环境：Flax 0.12.10、JAX / jaxlib 0.11.2、Optax 0.2.8。
对照基线是上一轮结构整理后的工作区，包含已完成的屏幕边缘修复。

## 文档依据与代码决策

| 检查项 | 文档依据 | 本次处理 |
| --- | --- | --- |
| NNX 状态管理 | [Flax Tree Mode NNX](https://github.com/google/flax/blob/v0.12.10/docs_nnx/flip/5310-tree-mode-nnx.md) 允许固定树结构中的 Variable 更新，禁止共享 Variable 引用 | 保留 `GaussianModel` 的独立 Variable、`get_value()` / `set_value()` 和显式 `graph=False`；不引入每步 split / merge |
| 编译边界 | [JAX JIT](https://github.com/jax-ml/jax/blob/jax-v0.11.2/docs/jit-compilation.md) 要求纯计算，并建议让外层编译器看到完整计算 | 提取 `compute_training_step`，由 `array_train_step = jax.jit(...)` 和 `nnx.jit` 共用；生产、reference 和基准脚本不再调用 `__wrapped__` |
| 静态参数 | [jax.jit API](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) 将静态参数值纳入编译缓存 | JAX / NNX 共用同一静态参数声明；容量、SH 阶数、统计开关和优化器配置保持静态，步骤、相机数值及占用状态保持动态 |
| Donation | [JAX buffer donation](https://github.com/jax-ml/jax/blob/jax-v0.11.2/docs/buffer_donation.md) 要求交出缓冲区后不再读取旧输入 | 保留 `(state, stats, model, …)` 参数顺序和 `donate_argnums=(0, 1, 2)`；warmup 仅使用独立工作副本，连续接收并复用返回状态 |
| Warmup 与计时 | [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) 要求将预热排除在计时之外，并等待异步计算完成 | 仍执行全部 SH / 统计变体并同步；默认 SH=3 时，全容量工作副本从 8 次复制降至 1 次。densification 仍通过 `.lower(...).compile()` 提前编译 |
| 输入类型 | [JAX type promotion](https://docs.jax.dev/en/latest/101/type_promotion.html) 说明显式 dtype 消除弱类型；类型属于编译签名 | `Camera.from_colmap` 将 `fx` / `fy` / `cx` / `cy` 统一为明确的 `float32` 标量，避免整数、Python 浮点数和 NumPy 标量制造不同签名 |

Flax 的 [performance guide](https://github.com/google/flax/blob/v0.12.10/docs_nnx/guides/performance.md)
主要讨论 graph 模式的 Python 遍历。当前模型只有 8 个 Variable，并已使用 tree 模式。
本轮保留 `nnx.jit`，未引入 `cached_partial` 或改变训练批次组织。

Adam 初始化和 opacity decay 的零矩可能共用缓冲区；在捐赠状态前复制 `v` 的现有步骤继续保留。
仅将两个独立的 `zeros_like` 写入 jitted 函数，并不能保证绕过 XLA 的公共子表达式消除。
默认仍为 CuTe 渲染、Optax 优化器，参数布局、优化器规则和训练调度保持原有实现。

## 验收

- 新增相机类型回归：`int` / `float` / `numpy.float32` / `numpy.float64` 得到相同数值；
  同一 JIT 函数的 trace 次数从基线的 3 次降至 1 次。
- 新增真实 CuTe warmup 回归：预编译后，原始 GaussianPool、AdamState 和统计数组数值、
  缓冲区指针均保持一致，原始输入未被 donation 删除。
- 原有 NNX 回归继续核查参数及矩的 15 个缓冲区指针、已捐赠输入失效和 JIT 缓存稳定。
- 6 组 GPU 编译对照覆盖 Optax / CuTe 优化器、SH 0 / 3、统计开关、独立 Optax 和 densification。
  编译内存用量一致，132 个输出数组的最大绝对差为 `3.73e-9`。
- 独立 Optax 和 densification 的 StableHLO 字节相同。训练模块名随纯计算入口重命名；
  整数内参签名改为 `float32`，对应的类型转换减少。因此训练 StableHLO 不宣称字节相同。
- 65 个 CuTe kernel、launcher 和 DSL 函数的 AST 完全一致，计时 epoch 循环的 AST 完全一致。
- 全量 GPU 测试：167 项通过；Python 宿主覆盖率为 100%（1579 条语句、224 个分支）。
  CuTe 编译到 GPU 的函数体不计入 Python 覆盖率。
- Ruff 检查、格式检查以及 `git diff --check` 通过。

完整测试、覆盖率与编译对照详见
[机器可读验收记录](flax_jax_validation_20260930.json)。
GPU 仍与另一训练进程共享；本轮未测量训练加速比例。
