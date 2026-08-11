# jax-gs 分布式 appearance 交接

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

## 明确边界与 CLI 遗留

- 仅支持单进程、多本地设备；multi-process/multi-host 仍须在创建输出前拒绝。
- Phase 3 不新增 2DGS、UT/eval3d、非 pinhole、`sparse_grad` 或 AbsGrad distributed 支持。
- `jax-gs train --distributed --app-opt` 已在上述单进程多本地设备范围内接通。
- generic `jax-gs render`/`jax-gs export` 使用 generic checkpoint restore，不能直接加载 distributed shard set。host loop 内 mapped eval 不代表通用 distributed render/export CLI 已完成。
- `jax-gs train --distributed --resume` 目前是 same-world exact resume；跨 world/capacity 的 `restore_distributed_checkpoint(..., allow_reshard=True)` 仍是 Python API，没有 CLI 开关。
- 不要把内部 already-gathered appearance 颜色路径描述成根级公共 rasterization API 的通用布局支持。

## Phase 3 验收结果

- `tests/test_training_distributed.py`：73 passed，覆盖 global Gaussian gather、rank-local camera/image-id、Appearance DDP `pmean`、optimizer step mismatch 原子 no-op、canonical checkpoint、2→4 broadcast/reshard、失败 restore 原子性、真实 reference rasterizer、mapped eval，以及双虚拟 CPU app-only/pose-only/noise-only smoke 和 pose+app exact resume。
- 训练、appearance、pose、capacity、distributed renderer 与 CLI 定向回归：158 passed；4 条 warning 是既有 Orbax generic checkpoint sharding 提示。
- `./scripts/test_safe.sh`：退出码 0；唯一 skip 是本机缺少可选 Mip-NeRF360 stump 数据集。
- `git diff --check` 与 `compileall` 通过。

## 后续迁移顺序

1. 为 distributed shard checkpoint 补通 generic `jax-gs render`/`jax-gs export` loader；不要把 host-loop mapped eval 当成该能力。
2. 再设计 multi-process/multi-host host ownership、per-process checkpoint、数据加载与一致 preflight；现有 stacked host state helper 不能直接宣称支持它。
3. 继续迁移 distributed UT/eval3d、非 pinhole、`sparse_grad` 和 AbsGrad；每项按 current-main 实际组合单独对齐，不用过时的“上游统一拒绝”归因。
4. 有可用 GPU 时按串行安全脚本补跑本阶段 GPU 回归；当前验收仅承诺 CPU 与双虚拟 CPU `pmap`。

提交或继续开发时仍应检查实际暂存清单，不能带入根目录用户文件 `CLAUDE.md`。
