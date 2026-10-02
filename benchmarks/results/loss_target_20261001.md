# CuTe loss 中直接归一化 uint8 目标图像

2026-10-01，RTX 5090，驱动 615.71.09，JAX 0.11.2、Flax 0.12.10、Optax 0.2.8、
CUTLASS DSL 4.8.0。本轮收益在 loss 阶段和临时显存；整步训练耗时基本持平。
原始样本、编译器内存统计与命令见 [JSON 记录](loss_target_20261001.json)。

训练把预加载的 uint8 RGB 目标直接传给 `fused_loss_and_grad`，在读取共享内存 tile 时
转成 float32 并乘以 `1/255`，省去整张图像的归一化 kernel 和 float32 临时数组。
halo 的越界位置仍填零。CuTe 的字节 tensor 暴露 signless i8，因此在 Python 绑定处
根据 JAX dtype 选择编译变体，读取字节后显式转成 `Uint8`，覆盖 128～255 的像素值。
loss 接口也接受归一化的 float32 目标；训练入口的浮点目标仍按 0～255 像素值归一化。

计时前确认 GPU 无其他计算进程，所有测量串行运行。loss 比较使用 822×1237 RGB 图像，
每条路径预热 30 次，再按交替顺序执行 8 轮、每轮 1,000 次，计时包含主机调度。

| 指标 | 单独归一化 | 合并到 loss |
| --- | ---: | ---: |
| loss 与图像梯度，中位数 | 88.24 µs | 83.83 µs（−5.0%） |
| 编译器报告的临时内存 | 12,210,192 B | 8,208 B |

减少的临时内存为 12,201,984 B（11.6 MiB）。随机图像上 loss 和图像梯度逐位一致。

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cute python \
  benchmarks/loss_target.py --output loss_target.json
```

整步比较沿用固定百万点协议：1,000,000 行 PLY 补齐到 1,000,064 点，`images_4` 的
`_DSC8679.JPG`，SH3，8M pair arena，Optax，预热 30 步、计时 300 步，三轮交替运行。
基线是任务开始时 `e46b03e` 加上已有未提交模块迁移的工作区快照。

| 完整更新 | 基线 | 本轮 |
| --- | ---: | ---: |
| 三轮中位数 | 2.3868 ms | 2.3882 ms |
| 三轮范围 | 2.3847～2.3877 ms | 2.3726～2.3934 ms |

整步差异小于测量波动，不能据此宣称完整训练提速。六次运行均复用一个编译变体，
开启缓冲区 donation，无 overflow。

新增回归覆盖 1×1、小图、边缘 tile、像素 0/128/255、随机像素和相同图像，
直接读取 uint8 与归一化 float32 的 loss/梯度逐位一致。4×4 的诊断渲染与 16×16
的 packed 渲染也比较了 uint8 和 float32 像素值输入的完整训练步。
CuTe loss 的 memcheck、racecheck、synccheck 均通过：0 内存错误、0 竞态、0 同步错误。

本轮同时补齐已有模块迁移：Muon 测试从 `training.muon` 导入其实现，移除 cluster
压缩模块中已迁移的重复清零函数和无用导入，并修正两处格式与 README 的 Muon 链接。

最终全量 GPU 回归 **347 项通过**，Python 行与分支覆盖率均为 **100%**
（1,934 个语句、282 个分支结果），达到项目的 99% 阈值；GPU kernel 的覆盖率排除规则
沿用项目现有配置。Ruff 检查、格式检查与 `git diff --check` 均通过。
