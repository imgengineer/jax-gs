# jax-gs 分布式 checkpoint 推理交接

更新日期：2026-08-11。兼容目标仍固定为 gsplat `main@2b902ff`。

## 提交基线

Phase 2 已提交并推送到 `main`：

- `47b3590 feat(training): wire local-device distributed loop`
- `48e5706 feat(training): checkpoint distributed camera poses`

这两个提交接通了一个 JAX process 管理多个本地设备的训练 host loop：相机 batch 分片、Gaussian owner shard、`P('rank')` placement、容量 overflow 扩容/重放、mapped eval、distributed checkpoint/exact resume，以及 replicated camera-pose/pmean、固定 pose noise 和 canonical pose checkpoint。Phase 2 的颜色路径是 SH；不包含 distributed appearance。

Phase 3 已实现并完成 CPU 验收。本次提交的改动集中在：

- `src/jax_gs/checkpoints.py`
- `src/jax_gs/training/_distributed_loop.py`
- `src/jax_gs/training/_loop.py`
- `src/jax_gs/training/_memory.py`
- `src/jax_gs/training/_step.py`
- `src/jax_gs/training/appearance.py`
- `tests/test_training.py`
- `tests/test_training_distributed.py`

根目录未跟踪的 `CLAUDE.md` 不属于本阶段，除非用户另行确认，不要随 Phase 3 暂存或提交。

Phase 3 已以 `498166d feat(training): support distributed appearance` 提交并推送到 `main`。Phase 4 在此基础上接通 distributed shard checkpoint 的通用单设备 `render/export` 入口；仍须排除根目录用户文件 `CLAUDE.md`。

## Phase 3 已实现设计

目标只是在 Phase 2 的标准 pinhole 3DGS distributed trainer 中加入 `app_opt`，不扩大其余 renderer 组合。

数据流采用 JAX gather-before-MLP 扩展：

1. 每个 rank 接收自己的相机 batch；`dataset_index` 必须是训练 split-local 的 `int32 [B_local]`，pose/noise 后的 camera-to-world 仍是 rank-local。
2. 沿 named rank axis 对 activated Gaussian arrays 做 differentiable `all_gather`，得到 rank 顺序的全局 means、features、colors、geometry 和 active mask。
3. 每个 rank 用自己的相机中心与全局 means 计算 `[B_local, N_global, 3]` view directions，再由 replicated Appearance module 生成该 rank 相机条件下的 `[B_local, N_global, 3]` direct RGB。
4. geometry 与逐视角颜色已经全局化，因此调用本地 rasterizer 时传 `distributed=False`；不要再次 gather 逐视角颜色，也不要把其他 rank 相机预计算的颜色错配到本 rank。
5. `all_gather` 的 transpose 把 Gaussian `features`/`colors` 梯度求和散回 owner shard；Appearance module/optimizer 保持 replica，并对梯度做 rank `pmean`。

这一路径是为 pure JAX 分布式语义增加的工程扩展；上游 current-main 没有可直接照搬的 distributed appearance orchestrator。当前优先级是清晰与正确性，gather 会在每个 rank 物化全局场景，因此不作吞吐、显存或可扩展性承诺。

## 状态、resize 与 checkpoint

- Gaussian `features`/`colors` 是逐行 owner state，随 duplicate/split/relocate、capacity resize、compaction 和 `allow_reshard=True` 一起迁移。
- Appearance embedding/MLP 及其 optimizer 是相机相关的 replicated state，不随 Gaussian bucket resize。
- 保存 distributed appearance checkpoint 前必须逐叶验证 replicas 一致；文件中只保存一份 canonical module/optimizer。
- manifest 记录 appearance component、有序训练 image names、camera count、feature dimension 和静态 optimizer contract。恢复时先校验这些契约，再把 canonical state 广播到目标 ranks；Gaussian reshard 失败不得留下部分 appearance restore。
- 训练 eval 应传 `embed_ids=None`，即零 camera embedding；方向仍来自真实 held-out camera，并使用 `config.model.sh_degree`。只有 export bake 使用零方向。

这里的 canonical appearance checkpoint 是“replicated 训练状态只落一份”。它与 generic appearance export 的 canonical degree-0 bake 是两件不同的事。

## Phase 4：distributed checkpoint 推理物化

- `load_distributed_inference_checkpoint()` 是独立的 inference-only public primitive；它不伪造 world=1 training resume，也不读取 Gaussian/Appearance optimizer、strategy、safety 或 pose。
- Orbax partial target 只读取 stacked model、host step 和可选 canonical `appearance.module`。Appearance 先恢复到临时 holder，Gaussian manifest 校验与压紧全部成功后才更新调用方模块，失败路径不会留下部分 restore。
- 全局行顺序固定为 `(rank, local slot)`。loader 用 payload 的真实 `active_mask` stable-select 所有 Gaussian leaves，manifest 的 `active_counts`/`active_prefix` 只用于反校验，不能拿 counts 猜 active prefix。
- 输出是只读 active-prefix 单模型：非空时 `capacity=max_capacity=active_count`；全空时构造一个 finite neutral inactive slot，绝不复制可能含 NaN/Inf 的废弃 owner slot。active 总数超过 1000 万时在物化前明确拒绝。
- SH checkpoint 原样保留 `sh0/sh_rest`；appearance checkpoint 合并 owner-local `features/colors` 并恢复唯一 canonical module。render 继续使用 held-out 实际方向、零 embedding 和完整配置 degree；export 才使用零方向 canonical bake 为 degree-0 SH。
- `_load_training_objects()` 自动区分 generic 与 distributed checkpoint，并把 distributed 推理 config 的逻辑 capacity 调整为合并后的实际物理长度。现有 render/export 后半段无需两套实现。
- distributed render 必须含 v6 scene metadata；缺失时坐标约定不可判断，因此明确拒绝，不再静默套用 generic legacy mean/max-extent transform。export 保持训练坐标输出。
- CLI render 的 70% 显存预检复用统一 evaluation estimator，除 raster workspace 外也计入 appearance MLP forward workspace。

当前仍是“单进程、单设备物化完整 local-device shard set”：不依赖当前设备数，但需要本进程读到全部 shard，并会短暂同时持有 stacked source 与 compact model。它不是 multi-host checkpoint reader、mapped CLI renderer，也不是训练态 reshard。

## 明确边界

- 仅支持单进程、多本地设备；multi-process/multi-host 仍须在创建输出前拒绝。
- Phase 3 不新增 2DGS、UT/eval3d、非 pinhole、`sparse_grad` 或 AbsGrad distributed 支持。
- `jax-gs train --distributed --app-opt` 已在上述单进程多本地设备范围内接通。
- `jax-gs render`/`jax-gs export` 已能自动加载 generic checkpoint 或完整 distributed shard set；distributed 路径受 1000 万 active 单模型上限和物化峰值内存约束。
- `jax-gs train --distributed --resume` 目前是 same-world exact resume；跨 world/capacity 的 `restore_distributed_checkpoint(..., allow_reshard=True)` 仍是 Python API，没有 CLI 开关。
- 不要把内部 already-gathered appearance 颜色路径描述成根级公共 rasterization API 的通用布局支持。

## Phase 3 验收结果

- `tests/test_training_distributed.py`：73 passed，覆盖 global Gaussian gather、rank-local camera/image-id、Appearance DDP `pmean`、optimizer step mismatch 原子 no-op、canonical checkpoint、2→4 broadcast/reshard、失败 restore 原子性、真实 reference rasterizer、mapped eval，以及双虚拟 CPU app-only/pose-only/noise-only smoke 和 pose+app exact resume。
- 训练、appearance、pose、capacity、distributed renderer 与 CLI 定向回归：158 passed；4 条 warning 是既有 Orbax generic checkpoint sharding 提示。
- `./scripts/test_safe.sh`：退出码 0；唯一 skip 是本机缺少可选 Mip-NeRF360 stump 数据集。
- `git diff --check` 与 `compileall` 通过。

## Phase 4 验收结果

- `tests/test_training_distributed.py`：76 passed；新增覆盖非前缀 active holes、所有 Gaussian leaves 的 rank-major 压紧、config/manifest 拒绝、finite empty slot、canonical Appearance module-only partial restore 与失败原子性。
- `tests/test_cli.py`：21 passed；真实 SH/Appearance distributed Orbax checkpoint 均经过 render 命令 seam 和 PLY export 回读，generic loader/render/export 回归保持通过。
- 两文件同进程合跑：97 passed；`tests/test_capacity.py tests/test_storage.py`：61 passed、4 条既有 Orbax generic sharding warning；`tests/test_training.py` 的 checkpoint/restore/memory/preflight/evaluation/appearance 定向选择：18 passed；`tests/test_api.py`：5 passed。以上共 181 个互不重复的当前 worktree 测试通过。
- Appearance 的 holes + 非平凡相机 reference rasterizer 独立对拍中，mapped distributed 与 checkpoint-merged local 输出 `max_abs=0.0`。
- `git diff --check` 与 `compileall` 通过。Phase 4 未重跑完整 CPU safe script，完整脚本最近一次证据仍是 Phase 3；不要把两者混写。

## 后续迁移顺序

1. 再设计 multi-process/multi-host host ownership、per-process checkpoint、数据加载与一致 preflight；现有 stacked host state helper 不能直接宣称支持它。
2. 让 CLI render 默认消费 checkpoint 中动态增长后的 intersection/candidate high-water mark，再由显式 CLI 参数覆盖；当前仍使用保存的 TrainConfig 值，容量不足时会报告 overflow，最终输出应使用 `--strict-overflow` 或显式覆盖容量。
3. 继续迁移 distributed UT/eval3d、非 pinhole、`sparse_grad` 和 AbsGrad；每项按 current-main 实际组合单独对齐，不用过时的“上游统一拒绝”归因。
4. 有可用 GPU 时按串行安全脚本补跑本阶段 GPU 回归；当前验收仅承诺 CPU 与双虚拟 CPU `pmap`。

提交或继续开发时仍应检查实际暂存清单，不能带入根目录用户文件 `CLAUDE.md`。
