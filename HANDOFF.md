# jax-gs 项目交接文档

更新日期：2026-07-29

## 1. 一句话状态

本仓库正在以 **pure JAX + Flax NNX** 逐子系统迁移
[`nerfstudio-project/gsplat`](https://github.com/nerfstudio-project/gsplat)。当前兼容基线固定在
upstream `main@2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c`；单进程训练和大部分
backend-independent API 已完成，最新完成的是分布式 owner-local topology transaction 与不可分割的
shard checkpoint。下一阶段应做 host 编排（shard 扩容 + overflow 重放、数据分片、eval），而不是
提前优化 kernel 性能。

## 2. 仓库与版本状态

- 工作目录：`/home/lzc/Documents/jax-gs`
- 当前分支：`main`
- 远程仓库：`https://github.com/imgengineer/jax-gs.git`
- 已推送代码基线：`a09114a` (`feat(checkpoints): persist distributed shard sets`)
- 前一提交：`e8ea88b`
  (`feat(training): commit distributed topology transactions`)、
  `40c57e9` (`feat(training): plan distributed topology without committing`)
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
- refinement schedule 已被接受：update 前每 rank 生成 owner-local duplicate/split/prune plan，标量
  摘要全局归约成 `refine_*` metrics，任一 rank 的计划 capacity overflow 使全体原子 no-op。
- update 后每个 owner 用普通 `DefaultStrategy.refine` 提交自己的 duplicate/split/prune 和
  scheduled opacity reset；shard 物理容量不变，step 内不做 bucket 扩容，也没有 Scene/Dynamic
  sidecar。
- `save_distributed_checkpoint`/`restore_distributed_checkpoint`/
  `load_distributed_checkpoint_manifest` 把 stacked model/optimizer/StrategyState/
  TrainingSafetyState 作为不可分割 shard 集存取，只支持同 world-size、同 shard capacity 的精确
  resume。

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

### 6.2 owner-local topology plan 与 commit

一个 step 内的顺序是：

```text
[update 前]
plan_refine(owner-local model[L] / StrategyState[L], scene_scale, step)
  -> planned_new_count / pruned_count / required_capacity / capacity_overflow

psum(planned_new_count)          -> refine_planned_new_count
psum(pruned_count)               -> refine_planned_pruned_count
pmax(required_capacity)          -> refine_required_capacity（每 shard 需要的容量）
pmax(capacity_overflow)          -> 任一 rank 溢出
& refine_scheduled               -> strategy_capacity_overflow -> has_overflow

[apply 分支内，无 collective]
optimizer.update -> 统计累加
  -> cond(refine_scheduled)  DefaultStrategy.refine（owner-local，用 post-update 状态重算 events）
  -> cond(reset_scheduled & ~commit_overflow)  reset_opacities

[两个分支汇合后]
psum(new_count) / psum(pruned_count) / pmax(commit_overflow) / pmax(opacity_reset)
  -> refine_new_count / refine_pruned_count / refine_commit_overflow / opacity_reset
```

必须保持的约束：

1. 计划与提交都在 owner-local `[L]` 上做，只有标量摘要跨 rank；`[L]` 决策数组永远不做
   collective。
2. plan 与 commit 都使用 train step 的 `scene_scale`（已与 optimizer 校验过的那个），不使用
   `StrategyState.scene_scale`；后者是 per-rank 设备状态，会让不同 rank 用不同阈值打分。
3. `refine_scheduled` 与 `reset_scheduled` 是 `optimizer.step` 的确定性函数，而 step 已有
   pmin/pmax 一致性检查，所以不需要再为 schedule 增加 collective。
4. plan 在 optimizer update 之前算（它要 gate `has_overflow`），commit 在 update 与统计累加之后
   按 upstream 顺序**重算** events。两者可以合理地不同，这是刻意的；不要为了让二者一致而把
   commit 改成重放 plan，也不要把 plan 挪到 update 之后（那就无法 gate 整步）。
   `tests/test_training_distributed.py::test_commit_recomputes_from_post_update_statistics`
   钉住了这个差异。
5. commit 溢出（preflight 通过但重算后放不下）由 `_default_refine` 自己保护：该 owner 不改任何
   行、保留统计、跳过 opacity reset，其它 owner 照常提交。这与 upstream 每 rank 独立跑 strategy
   一致；不要把它升级成全局原子 no-op（那需要在 cond 内做 collective）。
6. `max_new_per_refine` 是每 shard 的上界，`W` 个 rank 的全局上界是 `W * max_new_per_refine`。
   这是 fixed-capacity 的取舍，不是 upstream 语义。
7. host 看到 `refine_capacity_overflow=True` 后必须扩容所有 shard 再重放该 step；否则 step 被
   原子跳过、`optimizer.step` 不前进，同一步会无限重复。`refine_commit_overflow=True` 则相反：
   该 step 已经消费，host 只需在下一次 refine 前扩容。
8. 所有 collective 必须留在 cond 外：plan 归约在 cond 前，commit 计数归约在两个分支汇合后。

## 7. 当前明确未完成的内容

不要在 README 或提交说明中声称完整分布式/性能 parity。当前仍缺少：

1. 分布式 host 编排：shard bucket 扩容 + overflow 重放、camera-data sharding、
   multi-process/multi-host trainer、distributed eval。设备侧 topology transaction 已完成
   （见 6.2），但没有任何 host 会消费 `refine_*` metrics 或调用 shard checkpoint API。
2. world-size 变更的重分片；当前 restore 明确拒绝。分布式 checkpoint 也不保存 scene
   transform/scene scale，resume 方必须自己带这两个值（optimizer 与 train step 都需要）。
3. 分布式 Scene/Dynamic sidecar lineage；device 提交会丢弃 `_slot_copy_*` transaction。
4. distributed pose/appearance、packed/sparse/visible Adam、UT/Eval3D、2DGS。
5. Gaussian leading-batch distributed renderer。
6. 大规模显存/吞吐优化；当前 compatibility renderer 会在每 rank gather/replicate global
   Gaussian scene，内存约有 `W` 倍开销。
7. 完整、安全结束的 current-slice GPU acceptance。

## 8. 建议的下一阶段

建议继续按最小可验收切片推进 distributed topology，而不是一次实现完整 host trainer。

### Slice A/B：owner-local topology plan、preflight 与 commit（已完成）

实现见 `_make_train_step()` 中 `distributed_plan_strategy` 分支与 `apply_update` 末尾的
`topology_commit`，语义见 6.2。Slice B 的语义选择已定：commit 用 post-update 状态**重算**
events（贴 upstream），preflight 保留为 host 扩容信号。验收见
`tests/test_training_distributed.py` 的 12 个 topology 测试。

### Slice C：分布式 checkpoint（已完成）

实现见 `src/jax_gs/checkpoints.py` 的 `save_distributed_checkpoint`、
`restore_distributed_checkpoint` 和 `load_distributed_checkpoint_manifest`，验收见
`tests/test_training_distributed.py` 末尾的 8 个 checkpoint 测试。重分片仍未实现：restore 在
world-size 或 shard capacity 不一致时明确拒绝，需要 model/optimizer/stats 三者一起重分片才能放开。

### Slice D：host shard 扩容与重放（下一步）

设备侧已经会在 `refine_capacity_overflow=True` 时原子跳过整步，但没有 host 消费这个信号。

1. 读取 `refine_required_capacity`，对所有 rank 同时 `resize_training_state` 到同一 bucket，
   重编译 pmap step，再重放被跳过的那一步（参照单进程 `synchronize_pending_steps` 的
   intersection-overflow 重放）。
2. 所有 shard 的 `L` 必须保持相同静态值；不要让某个 rank 单独扩容。
3. 到达 `max_capacity` 且仍然溢出时必须显式失败或明确停止 refine，否则同一步会无限重复。
4. `refine_commit_overflow=True` 不需要重放，只需在下一次 refine 前扩容。
5. 扩容会改变 shard checkpoint 的 `local_capacity`；重放前后保存的 checkpoint 不能互相 resume，
   这一点要在 host 里显式处理，不要指望 restore 帮你迁移。

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

- 常规：`858 passed, 1 skipped, 38 deselected`
- fresh-process resource-heavy：`19+5+9+3+2=38 passed`
- CPU 总通过数：`896`
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
- 单进程 checkpoint 与分布式 shard checkpoint 是两种 artifact：manifest 的 `kind` 字段区分，
  两个 restore 入口会互相拒绝。不要用 `save_checkpoint` 保存 stacked shard（`model.capacity`
  会读成 world size）。
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
