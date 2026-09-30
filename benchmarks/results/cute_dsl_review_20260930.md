# JAX / CuTe DSL 性能审查

2026-09-30，JAX 0.11.2、CUTLASS DSL 4.8.0、RTX 5090。
基线是本轮改动前的工作区，包含此前 Flax/JAX 整理与屏幕边缘修复。
GPU 仍有其他训练任务；本轮核验正确性与编译资源，没有测训练耗时。
具体数据见 [验证记录](cute_dsl_validation_20260930.json)。

## 官方资料与现有实现

参考了 [JAX CuTe 教程](https://docs.jax.dev/en/latest/401/cute-dsl.html)、
NVIDIA 4.8.0 的 [JAX 调用示例](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/dsl_tutorials/jax/cutlass_call_basic.py)
与 [kernel 示例](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/dsl_tutorials/jax/cute_dsl_jax_kernels.py)，
并核对安装版本的 `cutlass.jax` 源码。

| 官方机制 | 当前实现 / 本轮处理 |
| --- | --- |
| `use_static_tensors=True`，固定 shape/stride | 已使用；容量和 SH degree 是静态编译参数 |
| XLA 管理 stream、设备缓冲区与 CUDA Graph | 已使用；`allow_cuda_graph` 默认为 True |
| `cute.autovec_copy` 与对齐访存 | packed 参数的 32 字节加载/存储已使用 |
| 共享内存布局与合并写入 | SH 梯度已有共享转置，coefficient stride 为 129 |
| 编译期分支，裁掉不用的计算 | 本轮增加 RGB 投影反向专用编译分支 |
| CTA 同步必须由同组线程一致执行 | 本轮给 SH 同步/转置增加整个 CTA 一致的工作条件 |
| 缓冲区别名与 donation | 现有 NNX/Optax donation 检查继续通过 |

`warp_redux_sync` 的 Float32 运算支持 min/max；整数支持 add。
因此 packed rasterizer 的 half2 求和不能直接替换成 Float32 redux add。
现有 warp 求和继续沿用原生数值语义，见
[NVIDIA arch API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html)。

## 本轮改动

[生产训练步](../../src/jaxgs/training/step.py) 的损失只依赖 RGB；
深度排序与 tile 分配已 stop_gradient，depth/radius 上游梯度恒为零。
[投影绑定](../../src/jaxgs/kernels/projector.py) 因此显式选择 `rgb_only=True`，
通过 CuTe 的可选 tensor 输入传入 None，移除这两个 FFI 梯度输入。
[反向 kernel](../../src/jaxgs/kernels/projection_backward.py) 用
`cutlass.const_expr` 裁掉对应的梯度检查、读取和 radius 特征值反向计算。
该机制依据 [官方控制流说明](https://github.com/NVIDIA/cutlass/blob/v4.8.0/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst)
和 [4.8.0 JAX primitive](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/jax/primitive.py)。
通用 `project_cute_vjp` 与 compact pullback 的默认接口保留完整 depth/radius 梯度。

投影反向的 launch grid 仍固定。超出可见 cluster 前缀的 CTA 跳过 SH barrier
及转置写入循环；条件仅依赖 block index 和只读 cluster count，整个 CTA 一致。
部分 CTA 内的全部线程仍到达 barrier，写入继续使用原有逐点 mask。
紧凑梯度的未使用尾部仍未定义，优化器只读取有效前缀。

## 编译产物核验

对固定 1,048,576 容量、cluster size 128、16 个 SH coefficient 的投影反向
做抽象 lowering，从正常发布编译的 host object 提取 cubin，再用
`cuobjdump --dump-resource-usage/--dump-sass` 检查；未开启 debug 或 lineinfo。

| compact 投影反向 | 改动前 | RGB 专用版本 |
| --- | ---: | ---: |
| FFI 输入缓冲区 | 17 | 15 |
| SH0 静态 SASS 指令条数 | 3,200 | 3,120 |
| SH3 静态 SASS 指令条数 | 3,504 | 3,424 |
| 寄存器 / thread | 72 | 72 |
| stack / local memory | 0 / 0 B | 0 / 0 B |
| 静态共享内存 | 1,024 B | 1,024 B |

动态 SH 共享缓存布局及 launch block 128 不变。
静态指令条数不是运行时执行次数，也不能换算成训练加速比例。
通用非 compact kernel 的指令条数与资源占用相同；compact 通用版本增加
8 条静态指令来跳过空 CTA 的 SH 工作，仍保持相同资源占用。

小场景完整训练对照包含 Optax/CuTe 两种优化器、SH0/SH3、统计开关，以及独立
Optax 与 densification，共 132 个输出数组，最大绝对差 `3.725290298461914e-9`。
默认 Optax 训练的输入/输出/临时显存计划相同；CuTe 优化器诊断训练的临时空间
减少 1,024 B。独立优化器和 densification 的 StableHLO 完全相同。
65 个 CuTe 装饰函数中，仅投影反向 kernel 与其 launcher 的 AST 改变。

## 测试

新增 14 个 GPU 检查覆盖 SH 0–3、cluster size 2/65/128、容量 257、非整块
前缀、非整 cluster 尾部及空 cluster；RGB pullback 同时对照通用 CuTe
和 JAX reference。已有 depth/radius、零梯度、裁剪颜色及 near-plane 检查通过。

```bash
uv run pytest -q --cov=jaxgs --cov-report=term-missing
compute-sanitizer --tool synccheck --error-exitcode 86 \
  uv run python -m pytest -q tests/test_packed.py -k 'rgb_projection or empty_compact'
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
```

全量 `181 passed`；1,579 条 Python 宿主语句与 224 个分支覆盖率 100%。
CuTe 编译函数体不计入 Python 覆盖率；上述 14 个检查在 synccheck 下报告
0 errors。Ruff 检查通过。

## 后续热点

已有 profile 的优先级仍是 raster backward 的全局梯度累加、Optax 的 SH/m/v
内存读写，以及 pair sorting；见 [此前审查](static_performance_audit_20260930.md)。
这些历史耗时不能当作本轮测量。下一次独占 GPU 应先重测恢复屏幕边缘贡献后的
完整训练，再根据新的 profile 决定 kernel 改动。

4.8.0 的 `cutlass_call(compile_key=...)` 可避免 cache lookup 时生成预编译 IR，
适合继续减少多 SH 变体的预热编译时间；它不会减少已编译训练步的 GPU 工作。
本轮保持现有缓存机制，集中修改 RGB 投影反向的无效工作。
