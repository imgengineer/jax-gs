# jax-gs 项目交接文档

更新日期：2026-07-29

## 1. 一句话状态

本仓库正在以 **pure JAX + Flax NNX** 逐子系统迁移
[`nerfstudio-project/gsplat`](https://github.com/nerfstudio-project/gsplat)。当前兼容基线固定在
upstream `main@2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c`；单进程训练和大部分
backend-independent API 已完成，最新完成的是分布式训练的 owner-local topology plan 与全 rank
preflight——只计划、不提交。下一阶段应提交 device-side topology transaction，而不是提前优化
kernel 性能。

## 2. 仓库与版本状态

- 工作目录：`/home/lzc/Documents/jax-gs`
- 当前分支：`main`
- 远程仓库：`https://github.com/imgengineer/jax-gs.git`
- 已推送代码基线：`40c57e9`
  (`feat(training): plan distributed topology without committing`)
- 前一提交：`6e8227d` (`docs: add project handoff guide`)、
  `f74efbd` (`feat(training): add distributed densification stats`)
- upstream `main` 在 2026-07-29 再次确认仍为 `2b902ff`

检查状态：

```bash
git status --short --branch
git log -6 --oneline --decorate
git ls-remote https://github.com/nerfstudio-project/gsplat.git refs/heads/main
```

如果 GitHub 直连不稳定，使用用户指定的本地代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
git fetch origin main
git push origin main
```

认证已经配置在当前环境中；不要把 token、credential helper 内容或其他密钥写入仓库。

## 3. 项目目标与工程原则

目标是尽可能复现 current-main gsplat 的 API、数值语义和训练流程，同时遵守以下边界：

1. 项目源码完全脱离 gsplat CUDA extension、FFI、CuTile 和其他专用 kernel DSL。
2. 投影、intersection、排序、compositing 和反向传播使用普通 JAX primitives，由 XLA 在
   CPU/CUDA 上执行。
3. 当前优先级是正确性、API parity 和代码可读性；暂不宣称性能等价。
4. 对 PyTorch mutable/autograd 语义无法逐字复制时，提供显式、可微、可测试的 JAX 调用面，
   并在兼容文档中记录差异。
5. JAX 动态 shape 用固定物理 capacity、padding、valid count 和 overflow metadata 表达；
   不要为追求表面 API 一致而引入不可 JIT 的运行时形状。
6. 只做当前阶段需要的最小改动。新行为先写失败测试，再实现并跑定向及全量回归。

## 4. 权威文档与代码入口

- [README.md](README.md)：用户入口、环境、CLI、当前能力和最新验收结论。
- [COMPATIBILITY.md](COMPATIBILITY.md)：API/行为兼容矩阵和明确边界。
- [CURRENT_MAIN_MIGRATION.md](CURRENT_MAIN_MIGRATION.md)：pinned-tree inventory、阶段记录和
  上游语义说明。
- [src/jax_gs/rasterization.py](src/jax_gs/rasterization.py)：根级 3DGS rasterization 路由。
- [src/jax_gs/two_dgs.py](src/jax_gs/two_dgs.py)：2DGS 路径。
- [src/jax_gs/model.py](src/jax_gs/model.py)：Flax NNX `GaussianModel`。
- [src/jax_gs/strategy/__init__.py](src/jax_gs/strategy/__init__.py)：Default/MCMC 策略、统计和
  fixed-slot topology 操作。
- [src/jax_gs/optimizers/](src/jax_gs/optimizers/)：dense、row-selective 和 SelectiveAdam 语义。
- [src/jax_gs/training/__init__.py](src/jax_gs/training/__init__.py)：统一训练 step、host trainer、
  scene normalization 和 distributed step factory。
- [src/jax_gs/distributed.py](src/jax_gs/distributed.py)：named-axis collectives 和兼容型分布式
  renderer。
- [tests/test_training_distributed.py](tests/test_training_distributed.py)：最新分布式训练契约。
- [scripts/test_safe.sh](scripts/test_safe.sh)：fresh-process CPU/GPU 安全验收入口。

## 5. 当前已完成的主要能力

详细清单以兼容矩阵为准，下面只列继续开发时最重要的状态。

### 5.1 渲染与微分

- 3DGS dense/static-packed rendering、projection、tile intersection、sorting 和 compositing。
- RGB、depth、expected depth、hit-distance 及组合 render modes。
- pinhole、ortho、fisheye、FTheta/UT、Eval3D 及结构化 LiDAR 相关路径。
- 2DGS projection/rasterization、normal/distortion/depth 辅助输出。
- sparse tile layout、visibility/contributor 查询和 overflow metadata。
- 普通 signed gradient 由 JAX autodiff 得到；true AbsGrad 由独立零值 probe 在逐像素贡献处取
  绝对值，不能用 `abs(signed_gradient)` 近似。

### 5.2 模型、策略与优化器

- Flax NNX `GaussianModel`，逻辑 capacity 与物理 bucket 分离。
- DefaultStrategy 和 MCMCStrategy 的 fixed-slot duplicate/split/prune/relocate/reset。
- Scene/Dynamic sidecar 在 resize、compact 和 topology transaction 中保持行对齐。
- current-main global-batch Adam 超参数缩放、means scene-scale、sparse row selection 和
  uncorrected SelectiveAdam。
- overflow 时 model、optimizer、strategy stats、camera/appearance state 和 MCMC noise 原子跳过。

### 5.3 单进程训练

- 3DGS/2DGS、Default/MCMC、full-image/patch、packed sparse、visible Adam。
- pose optimization、pose noise 和 appearance module 的独立 optimizer/checkpoint。
- current-main COLMAP world normalization；trainer checkpoint v6 保存精确变换和 scene scale。
- checkpoint/resume、定期 eval、export 及 CLI 已接入单进程 host training。
- `train()` 仍明确拒绝 `jax.process_count() != 1`，避免伪装成完整多进程训练。

### 5.4 分布式训练当前切片

`make_distributed_train_step()` 已支持绑定 `nnx.pmap`/named axis 的设备训练：

- 每 rank 持有等物理 capacity 的 Gaussian/Adam shard 和本地 camera batch。
- dense、SH、pinhole 3DGS。
- Gaussian photometric gradient 等于各 rank local mean-loss 梯度之和，不做 `pmean`。
- global visibility 在全局 Gaussian 坐标中归约后切回 owner shard。
- optimizer batch/world/scene-scale/config 契约和 rank step/SH-degree 一致性检查。
- 当前或 sticky overflow、以及 state mismatch，会使所有 rank 的 model、optimizer 和 strategy stats
  原子 no-op；safety diagnostics 仍会更新。
- owner-local signed densification statistics 已包含所有 rank 的 camera contribution。
- refinement schedule 已被接受：每 rank 生成 owner-local duplicate/split/prune plan，标量摘要全局
  归约成 `refine_*` metrics，任一 rank 的计划 capacity overflow 使全体原子 no-op；但**不提交**
  任何拓扑变更。

## 6. 分布式 statistics 与 topology plan 的关键语义

这是最容易被后续“简化”破坏的部分。

### 6.1 densification statistics

设：

- `W = world_size`
- `C = rank-local camera batch size`
- `L = local Gaussian capacity`
- `G = W * L`

当前数据流是：

```text
screen_probe             [C, G, 2]  每个 camera rank 独立
screen_grad              [C, G, 2]
projection_radii         [C, G, 2]
projection_valid         [C, G]
global_active_mask       [G]

build_densification_stats(...)
  -> local grad/count/max-radii [G]

grad_sum:   psum
count:      psum
max_radii:  pmax
  -> global stats [G]

dynamic_slice(axis_index * L, L)
  -> owner-local stats [L]
```

必须保持以下顺序：

1. 每个可见 `(camera, Gaussian)` 的 signed `[dx, dy]` 先按 width/height 和 **local `C`**
   归一化。
2. 对该二维向量取 L2 norm。
3. 再跨 rank 聚合标量 `sum/sum/max`。
4. 最后按 owner slice。

禁止以下两种写法：

- `norm(psum(signed_gradient))`：不同相机方向相反的梯度会错误相消。
- 各 rank 先 owner-slice 再 `psum`：不同 owner 的相同局部 slot 会被错误相加。

分布式 renderer 的私有 `_means2d_offset` 兼容两种静态形状：

- `[C, L, 2]`：旧的 rank-local probe，会沿 Gaussian 轴 gather。
- `[C, G, 2]`：训练使用的 already-global、camera-rank-local probe，不再 gather。

Pinned upstream 明确拒绝 `distributed=True && absgrad=True`，所以本阶段继续拒绝 distributed
AbsGrad；这不是用 signed gradient 的绝对值可以补上的缺口。

### 6.2 owner-local topology plan

数据流是：

```text
plan_refine(owner-local model[L] / StrategyState[L], scene_scale, step)
  -> planned_new_count / pruned_count / required_capacity / capacity_overflow

psum(planned_new_count)          -> refine_planned_new_count
psum(pruned_count)               -> refine_planned_pruned_count
pmax(required_capacity)          -> refine_required_capacity（每 shard 需要的容量）
pmax(capacity_overflow)          -> 任一 rank 溢出
& refine_scheduled               -> strategy_capacity_overflow -> has_overflow
```

必须保持的约束：

1. 计划在 owner-local `[L]` 上做，只有标量摘要跨 rank；`[L]` 决策数组永远不做 collective。
2. plan 使用 train step 的 `scene_scale`（已与 optimizer 校验过的那个），不使用
   `StrategyState.scene_scale`；后者是 per-rank 设备状态，会让不同 rank 用不同阈值打分。
3. `refine_scheduled` 与 `reset_scheduled` 是 `optimizer.step` 的确定性函数，而 step 已有
   pmin/pmax 一致性检查，所以不需要再为 schedule 增加 collective。
4. plan 必须在 optimizer update 之前算（它要 gate `has_overflow`），而 upstream 的
   `step_post_backward` 是在 optimizer step 之后决策的。因此当前 plan 是**preflight 估计**，
   不是 Slice B 提交时会重算的那份 event 列表。
5. `max_new_per_refine` 是每 shard 的上界，`W` 个 rank 的全局上界是 `W * max_new_per_refine`。
   这是 fixed-capacity 的取舍，不是 upstream 语义。
6. host 看到 `refine_capacity_overflow=True` 后必须扩容所有 shard 再重放该 step；否则 step 被
   原子跳过、`optimizer.step` 不前进，同一步会无限重复。

## 7. 当前明确未完成的内容

不要在 README 或提交说明中声称完整分布式/性能 parity。当前仍缺少：

1. 分布式 device-side duplicate/split/prune/reset topology transaction 的**提交**（plan 与
   preflight 已完成，见 6.2）。
2. 分布式 checkpoint/resume、shard manifest、world-size 校验与重分片。
3. host camera-data sharding、multi-process/multi-host trainer orchestration、distributed eval。
4. distributed pose/appearance、packed/sparse/visible Adam、UT/Eval3D、2DGS。
5. Gaussian leading-batch distributed renderer。
6. 大规模显存/吞吐优化；当前 compatibility renderer 会在每 rank gather/replicate global
   Gaussian scene，内存约有 `W` 倍开销。
7. 完整、安全结束的 current-slice GPU acceptance。

## 8. 建议的下一阶段

建议继续按最小可验收切片推进 distributed topology，而不是一次实现完整 host trainer。

### Slice A：owner-local topology plan 与全局 preflight（已完成）

实现见 `_make_train_step()` 中 `distributed_plan_strategy` 分支，语义见 6.2，验收见
`tests/test_training_distributed.py` 的 8 个 plan 测试（owner-only duplicate、同父
duplicate+split、跨 rank prune 求和、inactive padding、未提交的 opacity reset、scene-scale
来源、单 rank overflow 原子性，以及 named `nnx.vmap` 与双虚拟 CPU `nnx.pmap` 一致）。

### Slice B：提交 topology transaction（下一步）

1. owner-local apply duplicate/split/prune/reset。
2. 同步 Gaussian 参数、Adam moments、strategy statistics 和 Scene/Dynamic sidecar。
3. 所有 collectives 必须在每个 rank 上以相同静态顺序执行；不要把 collective 首次放入
   data-dependent `nnx.cond` 分支。
4. overflow replay 前后 schedule/optimizer step 必须保持对齐。
5. 先决定并写清楚一个语义问题：preflight 在 optimizer update 之前算，而 upstream 的 refine 决策
   在 optimizer step 之后。提交时是
   (a) 按 upstream 用 post-update 状态重算 events，再依赖 `_default_refine` 内部的 overflow
   自保护；还是 (b) 严格提交 preflight 过的那批 events。(a) 更贴 upstream，(b) 更容易证明
   原子性。不要在没做决定的情况下混用两者。

### Slice C：分布式 checkpoint

1. 将每 rank model、optimizer、StrategyState、TrainingSafetyState 作为不可分割 shard 保存。
2. manifest 记录 world size、rank、local/global capacity、slot layout 和 config fingerprint。
3. 先支持 same-world-size exact resume；不支持的 world-size 变化应明确拒绝。
4. 后续再实现 model/optimizer/stats 三者一起重分片。

## 9. 环境和常用命令

当前项目使用 `uv` 管理环境。新增依赖使用 `uv add`，不要直接修改环境或使用裸 `pip`：

```bash
uv sync --all-groups
uv add <package>
uv run python -c "import jax; print(jax.devices())"
```

当前机器记录：RTX 5090、JAX 0.11.0、Flax 0.12.8、Grain 0.2.18。

定向测试：

```bash
uv run pytest -q tests/test_distributed.py tests/test_training_distributed.py
uv run pytest -q tests/test_strategy_current_main.py tests/test_training.py
```

完整安全 CPU 验收：

```bash
JAX_PLATFORMS=cpu scripts/test_safe.sh
```

2026-07-29 最新结果：

- 常规：`847 passed, 1 skipped, 38 deselected`
- fresh-process resource-heavy：`19+5+9+3+2=38 passed`
- CPU 总通过数：`885`
- 唯一 skip：本机没有可选 Mip-NeRF360 stump 数据集
- 4 条 warning：既有 Orbax restore sharding 提示

GPU 安全脚本此前两次分别通过 21 和 32 个独立 CUDA case，随后因瞬态
`libuv-worker`/kernel-journal D-state 被 preflight 按设计停止。不要设置
`ALLOW_D_STATE_GPU_TESTS=1` 绕过保护，除非用户明确接受风险。本轮 distributed statistics
没有修改 CUDA 专用路径。

## 10. 开发与验证注意事项

- 编辑文件使用小而明确的 patch，保留用户已有改动；不要清理无关代码。
- 搜索优先使用 `rg` / `rg --files`。
- 每个行为变化先有失败测试，再实现，再跑 focused tests。
- 涉及 collective 时，至少有 named `vmap` 数值测试和真实双虚拟 CPU `pmap` 测试。
- `StrategyState` 是 owner-local `[L]`，不要直接对状态数组做 `psum/pmean`。
- `dynamic_slice_in_dim` 的 slice size 必须保持静态；动态量只应是 owner start。
- 不要在 `nnx.cond` 分支里首次创建/rebind `nnx.Variable`。
- stats collective 可以在 overflow 判断前执行，但 mutation 必须只在统一 apply 分支提交。
- `refine_*`/`reset_scheduled` metrics 只在 distributed + DefaultStrategy 下出现；单进程 metrics
  字典不含这些 key，host 侧读取前要按 key 存在性判断。
- checkpoint 格式已有 v6；不要因为 owner-local stats 已可保存就宣称 distributed checkpoint
  完成。rank 0 单独保存只包含 owner-0 shard。
- 技术结论优先对照 pinned upstream 的 production code 和官方测试，不依据旧 release 或二手
  文档猜测。
- 修改 API/边界/验收结果时同步更新 README、COMPATIBILITY 和
  CURRENT_MAIN_MIGRATION，避免三份文档互相矛盾。

## 11. 提交前检查表

```bash
git diff --check
git status --short --branch
uv run pytest -q <changed-subsystem-tests>
JAX_PLATFORMS=cpu scripts/test_safe.sh  # 高风险或阶段性切片
git diff --stat
```

提交时显式暂存本次文件，不要用破坏性 Git 命令覆盖其他人的 worktree。推送前先
`git fetch origin main` 并确认本地基线未落后。完成后核对：

```bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/main
```
