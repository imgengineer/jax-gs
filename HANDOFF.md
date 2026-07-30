# jax-gs 项目交接文档

更新日期：2026-07-29

## 1. 一句话状态

本仓库正在以 **pure JAX + Flax NNX** 逐子系统迁移
[`nerfstudio-project/gsplat`](https://github.com/nerfstudio-project/gsplat)。当前兼容基线固定在
upstream `main@2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c`；单进程训练和大部分
backend-independent API 已完成；2026-07-29 的结构审计确认**库面已对上游 HEAD 完整**（见 §2）。
最新完成的是 compositing 的两级 rematerialization 与 tile 批加宽（RTX 5090 实测前向 2.6×、
反向 4.6×、峰值显存 12.3×），GPU 验收也首次完整跑通。剩下的是这次性能改动的 GPU 复跑、
进一步吞吐优化，以及 host 编排（驱动扩容/重放循环、相机数据分片、distributed eval）。

## 2. 仓库与版本状态

- 工作目录：`/home/lzc/Documents/jax-gs`
- 当前分支：`main`
- 远程仓库：`https://github.com/imgengineer/jax-gs.git`
- 已推送代码基线：`26f9b35` (`perf(rasterization): gate the chunk loop by the busiest tile`)
- 前置：`7bc7291` (`perf(sparse): rematerialize the per-pixel compositor`)
- 前置：`45715a4`
  (`perf(rasterization): rematerialize chunks and widen the tile batch`)
- 前一提交：`fad8ea0` (`perf(rasterization): rematerialize per-tile compositing in reverse mode`)、
  `25272ce` (`feat(training): wire packed, visible_adam, and MCMC into sharded training`)、
  `2ffb0a8` (`feat(training): support distributed camera-pose optimization`)
- upstream `main` 在 2026-07-29 再次确认仍为 `2b902ff`
- 同日做过一次结构审计（脚本可重写，未入库）：上游全部非 CUDA 模块路径都能在同名 `jax_gs`
  路径解析，134 个共有 callable 的参数名一致，唯一差异是已记录的 `SelectiveAdam` 调用面适配；
  根级 65 个公开符号全部齐备。库面对上游 HEAD 已完整，剩下的是 trainer 编排与上表第 4/5/6 项。

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
- 反向对 compositing 做两级 rematerialization：`jax.checkpoint` 同时包住 tile 函数（dense、
  reference、2DGS、两条 eval3d）和 dense 路径 tile 内部的 chunk 循环。后者让反向显存不再随
  intersection 容量增长，从而能把 `tile_batch_size` 默认值从 4 提到 64 换取 GPU 并行度
  （实测前向 2.6×、反向 4.6×、峰值显存 12.3×）。`prevent_cse=False` 试过并被否决：反向从
  2,685 ms 变成 4,104 ms。不要删掉它：删掉会让反向临时显存重新随 tile 数
  增长，实测 200k 高斯 640×360 会从 5.16 GiB 回到 1,169.6 GiB 并直接 OOM。
  `tests/test_rasterization_jax.py::test_backward_memory_does_not_grow_with_the_tile_count`
  是护栏。sparse 的 per-pixel compositor 也已按同样方式 remat（护栏在
  `test_sparse_backward_does_not_store_every_pixel`）；但它的 `_sparse_sample_weights`
  仍对每个像素展开整个 intersection 容量，那是 Slice E 的 sparse 对应物，未做。
  `visibility.py` 的三处 pixel map 形状相同，但属查询面、未确认会被求导，按不做投机改动处理。
- 与 upstream `step_post_backward` 的提前返回一致：`refine_stop` 之后统计累加、refine 和
  opacity reset 全部停止；判定集中在 `DefaultStrategy.should_refine`/`should_reset`。

### 5.3 单进程训练

- 3DGS/2DGS、Default/MCMC、full-image/patch、packed sparse、visible Adam。
- pose optimization、pose noise 和 appearance module 的独立 optimizer/checkpoint。
- current-main COLMAP world normalization；trainer checkpoint v6 保存精确变换和 scene scale。
- checkpoint/resume、定期 eval、export 及 CLI 已接入单进程 host training。
- `train()` 仍明确拒绝 `jax.process_count() != 1`，避免伪装成完整多进程训练。

### 5.4 分布式训练当前切片

`make_distributed_train_step()` 已支持绑定 `nnx.pmap`/named axis 的设备训练：

- 每 rank 持有等物理 capacity 的 Gaussian/Adam shard 和本地 camera batch。
- SH、pinhole 3DGS；dense 与 packed 投影、`visible_adam`、Default 与 MCMC 策略都已接入。
- camera-pose 优化与 pose noise（pose 模块复制、梯度 `pmean`，对齐上游 DDP）。
- packed 的 metadata 按 gathered scene 的全局 active mask 解包，再切回 owner；MCMC 每 shard 用
  自己的 cap 与 5% 出生预算（同上游每 rank 独立），但 scheduled capacity overflow 必须跨 rank
  归约，否则某个 shard 单独跳过会让 `optimizer.step` 分叉。
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
- `resize_distributed_training_state()` 统一扩容所有 shard；被 preflight overflow 跳过的 step
  扩容后可原样重放。

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
   pmin/pmax 一致性检查，所以不需要再为 schedule 增加 collective。两者都以
   `refine_stop` 为上界，和 upstream `step_post_backward` 的提前返回一致，也和
   `DefaultStrategy.should_refine`/`should_reset`、单进程 host 循环保持同一套判定。
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

1. 分布式 host 编排：camera-data sharding、multi-process/multi-host trainer、distributed eval，
   以及驱动扩容/重放的循环。设备侧 topology transaction、shard checkpoint 和扩容原语都已完成，
   但没有任何 host 会消费 `refine_*` metrics 或调用这些 API。对应上游的
   `examples/simple_trainer.py`，不是 `gsplat` 包 API。
2. world-size 变更的重分片；当前 restore 明确拒绝。分布式 checkpoint 也不保存 scene
   transform/scene scale，resume 方必须自己带这两个值（optimizer 与 train step 都需要）。
3. 分布式 Scene/Dynamic sidecar lineage；device 提交会丢弃 `_slot_copy_*` transaction。
4. 吞吐优化与其余显存项：compositing 之外的阶段实测只占前向 0.44 ms（projection 0.12、
   SH 0.02、intersect+sort 0.30），compositing 占 99.95%。已定位到具体根因与方案，见 §8
   Slice E；那是当前最大的一笔（约 900× 空转）。分布式 compatibility renderer 的 W× 场景复制
   与 sparse per-pixel compositor 的反向 remat 仍未做。分布式 compatibility renderer 仍在每 rank gather/replicate 全局
   Gaussian scene（实测 L=50k/rank、640×360、SH3：每 rank temp 24/51/94 MiB 对应 world
   1/2/4，flops 1.19/1.79/2.99e8），上游用的是 gather cameras + all-to-all projection；
   sparse per-pixel compositor 的反向也还没做 remat。GPU 上的 wall-clock 优化未开始。
5. 完整、安全结束的 current-slice GPU acceptance。2026-07-29 尝试时 GPU 上有用户的其它
   进程（`transientMamba`，约 2.5 GiB），`scripts/test_safe.sh` 的 preflight 会直接拒绝；
   不要杀别人的进程，也不要设 `ALLOW_D_STATE_GPU_TESTS=1` 绕过。GPU 空闲后再跑
   `RUN_GPU_TESTS=1 RUN_RESOURCE_HEAVY_GPU_TESTS=1 scripts/test_safe.sh`。

**不要再把下列项当成缺口**（2026-07-29 对照上游 `gsplat/cuda/csrc/Rendering.cpp` 的
`distributed` 校验确认）：upstream 自己就拒绝 `distributed=True` 与 Gaussian batch 维、
`sparse_grad`、AbsGrad、UT、eval3d、return_normals、custom rays、非 pinhole、rolling shutter、
camera distortion、LiDAR 系数、以及 per-view `[C, N, D]` colors（即 appearance）；
`rasterization_2dgs` 根本没有 `distributed` 参数。这些拒绝是 parity。

## 8. 建议的下一阶段

建议继续按最小可验收切片推进 distributed topology，而不是一次实现完整 host trainer。

### Slice A/B：owner-local topology plan、preflight 与 commit（已完成）

实现见 `_make_train_step()` 中 `distributed_plan_strategy` 分支与 `apply_update` 末尾的
`topology_commit`，语义见 6.2。Slice B 的语义选择已定：commit 用 post-update 状态**重算**
events（贴 upstream），preflight 保留为 host 扩容信号。验收见
`tests/test_training_distributed.py` 的 12 个 topology 测试。

### Slice E：动态候选遍历的 compositor（已完成一半）

**已做（无需 custom_vjp）**：chunk 循环现在由"最忙 tile 的占用"门控。该标量在 tile map 之外
算好，所以谓词在 tile 批的 vmap 内仍是标量、`lax.cond` 能真正短路，标准反向自动微分照常工作。
被跳过的 chunk 原本就被完全掩码，因此结果不变（loss 逐位相同、梯度约 1e-9 相对差）。前向在
容量 4,096/16,384/65,536 上分别快 6.5×/8.9×/13.7×；护栏是
`tests/test_rasterization_jax.py::test_forward_time_does_not_track_the_intersection_capacity`。

**GPU 实测**：20 万高斯 640×360 上前向 197.8→37.9 ms、`value_and_grad` 1,613.8→1,002.9 ms。

**仍欠的（这才需要 custom_vjp）——收益已量化**：静态 `fori_loop` 的**步数**没变，门控只让每步
变便宜。GPU 上每步固定开销约 90 µs，容量 65,536 配 512 的块 = 128 步 ≈ **11.8 ms 地板**：
用一个只有 202 个交集、两种容量都装得下的场景实测，有门控 0.87 ms（2,048）vs 11.8 ms（65,536），
无门控是 1.05 vs 24.2 ms。也就是说门控砍掉一半多，剩下的一半是纯粹的循环步数开销，**只有动态
trip count 能消**——这正是下面方案要做的事，收益就是这个地板。

**注意**：我为这个性质写过一个 wall-clock 护栏，又删了。能在 CPU 上判别的比值阈值落在 GPU 的
循环开销地板之内，结果是 GPU 验收误报失败而并无回归。这类性能性质请用 `benchmarks/` 手工复核，
不要写成单测。

### Slice E 续：per-tile 动态遍历（**建议不做**，2026-07-30 已量化否决）

2026-07-30 在 RTX 5090、20 万高斯、640×360、bucket 65,536 上做了两组测量，结论是这个切片
不值得做：

1. **加大块（=减少循环步数）反而更慢**，说明真实场景下循环开销地板不是主导：
   `(K, tile_batch)` = (512,64) 前向 37.6 ms / 反向 999 ms；(2048,64) 24.4 / 1419；
   (8192,64) 46.7 / 1724；(512,256) **18.7** / 1353；(512,16) 117.9 / 1707。
   当前默认 (512, 64) 就是训练场景的最优点（训练由反向主导）。
2. 因此动态 trip count 能省的只是那个地板：小场景占绝对多数，但 20 万高斯的训练步里
   前向只占 37.6/999≈3.8%，地板又只是其中一部分，**整体收益约 1–3%**。为此重写核心
   compositor 并手写反向，风险收益不成立。

**另一个被证伪的猜测**：我曾以为门控到位后内层 `jax.checkpoint(composite_chunk)` 可以去掉。
实测去掉后直接 OOM（要 85 GiB）。两级 remat 都必须保留；反向/前向 26.6× 是"双重重算换显存"
的必然代价，不是浪费。

**纯渲染场景的顺手优化**（不影响训练）：前向在 `tile_batch_size=256` 下是 18.7 ms，比默认的
64（37.6 ms）快约 2×，代价是显存。CLI render / eval 这类无梯度路径可以用 `--tile-batch-size`
自行调大，不需要改默认值。

**如果还要继续压吞吐**，下一个真正的大头不在这里，而是 compositor 的并行结构本身
（上游是逐像素线程 + 共享内存分块走 tile 列表），那在 JAX/XLA 里没有直接对应物，属于另一个
量级的工作，应先确认值不值得。

### Slice E 续（原方案 A 的设计，仅作存档）

**动机**：见上，全局门控已把 95.0/379.4/1501.6 ms 降到 14.7/42.5/109.6 ms；剩下的是批内
最忙 tile 与其它 tile 的差距。compositing 仍占前向的绝大部分（其余阶段实测合计 0.44 ms）。

**不要用 `cost_analysis` 判断这类改动**：它对循环体只计一次、不乘 trip count，会完全掩盖这个
问题。用 wall-clock。

**上游语义**：CUDA kernel 按 tile 的交集区间做动态 `while` 遍历，所有像素透射率耗尽即提前终止，
**没有每 tile 上限、从不截断**。所以方案是复刻它，而不是加一个截断上限。

**实现方案**：

1. 只改 dense 路径（`low_level.rasterize_to_pixels` 的非 packed 分支）。reference、2DGS、
   eval3d 以及 absgrad probe 分支先保留现有静态实现作为回退，缩小爆炸半径。
2. 把 `composite_chunk` 重构成显式函数
   `chunk(carry, flat_means, flat_conics, flat_colors, flat_opacities, chunk_index) -> carry`，
   不再靠闭包捕获可微数组（`custom_vjp` 的可微输入必须是显式实参）。
3. 给"单 tile compositing"套 `jax.custom_vjp`：
   - forward：`lax.while_loop`，trip count = `ceil(count / max_gaussians_per_tile)`，
     并在所有像素 `transmittance <= threshold` 时提前退出；每步把 carry 写进静态缓冲
     （`ceil(capacity / K)` 槽 × `[P, C+2]`，256 像素 3 通道时约 655 KB/tile，
     tile batch 64 约 42 MB，可接受）。residual 只存这个缓冲和输入数组。
   - backward：按逆序在执行过的步上循环，对每步用 `jax.vjp(chunk, 存下的carry, ...)` 取梯度并
     scatter-add 回全局数组。**不要手推 compositing 的导数**——让 JAX 对 chunk body 求导，
     absgrad probe 的 `custom_vjp` 也会被自然包含。
4. `vmap` 交互：tile batch 下 `while_loop` 的 trip count 变成该批的最大值（JAX 语义），
   所以上界从"全局容量"降到"批内最大 tile 占用"，仍是主要收益；不要为此放弃 tile 批。
5. 完成后 `jax.checkpoint(composite_chunk)` 可以去掉（动态循环已自带重算结构），但
   `jax.checkpoint(render_tile)` 要保留——两个显存护栏会告诉你。

**验收**：

- 与现有静态实现逐点比对 loss 与全部梯度（容差参照 COMPATIBILITY 已声明的重结合级别）。
- 现有两个护栏：反向显存不随 tile 数、不随 intersection 容量增长。
- 新增护栏：前向 wall-clock 不再随 `max_intersections` 线性增长（现在是 15.8×/16×桶）。
- 完整 CPU 验收 + GPU 验收（GPU 需设备空闲；不要杀别人的进程）。

### Slice C：分布式 checkpoint（已完成）

实现见 `src/jax_gs/checkpoints.py` 的 `save_distributed_checkpoint`、
`restore_distributed_checkpoint` 和 `load_distributed_checkpoint_manifest`，验收见
`tests/test_training_distributed.py` 末尾的 8 个 checkpoint 测试。重分片仍未实现：restore 在
world-size 或 shard capacity 不一致时明确拒绝，需要 model/optimizer/stats 三者一起重分片才能放开。

### Slice D：host shard 扩容与重放（原语已完成，host 循环待做）

已完成：`jax_gs.capacity.resize_distributed_training_state()` 把 stacked
model/optimizer/StrategyState 的每个 shard 按单进程规则扩容后重新 stack，所有 rank 保持同一
physical capacity；`tests/test_training_distributed.py::
test_growing_all_shards_lets_the_skipped_step_replay` 已经跑通「溢出跳过 → 扩容 → 原样重放并
提交」整条链路。

待做的 host 循环：

1. 读取 `refine_required_capacity` 选 bucket（参照单进程
   `bounded_required` / `config.model.bucket_capacity`），调用上面的原语，重编译 pmap step，
   再重放被跳过的那一步（参照 `synchronize_pending_steps` 的 intersection-overflow 重放）。
2. 所有 shard 的 `L` 必须保持相同静态值；不要让某个 rank 单独扩容。
3. 到达 `max_capacity` 且仍然溢出时必须显式失败或明确停止 refine，否则同一步会无限重复。
4. `refine_commit_overflow=True` 不需要重放，只需在下一次 refine 前扩容。
5. 扩容会改变 shard checkpoint 的 `local_capacity`；重放前后保存的 checkpoint 不能互相 resume，
   这一点要在 host 里显式处理，不要指望 restore 帮你迁移。
6. 显存预算检查还没有分布式版本：`_check_bucket_transition_memory_budget` 只算单进程，世界级
   转换大约需要 `W` 倍，且原语过程中会短暂多占一份世界拷贝。

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

- 常规：`874 passed, 1 skipped, 38 deselected`
- fresh-process resource-heavy：`19+5+9+3+2=38 passed`
- CPU 总通过数：`912`
- 2026-07-30 完整 GPU 验收：CPU 段 912 + GPU 段 122 个独立 CUDA case = `1,034` 全通过
- 唯一 skip：本机没有可选 Mip-NeRF360 stump 数据集
- 4 条 warning：既有 Orbax restore sharding 提示

2026-07-29 GPU 验收首次完整跑通：`RUN_GPU_TESTS=1 RUN_RESOURCE_HEAVY_GPU_TESTS=1
scripts/test_safe.sh` 的 CPU 段 66 组 910 项 + GPU 段 121 个独立 CUDA case，合计 1,031 项通过、
零失败。该次覆盖的是 per-tile remat 状态。其后 chunk-level remat、`tile_batch_size` 默认值变更、
sparse remat 与 chunk gate 已由 2026-07-30 的完整 GPU 验收覆盖（1,034 项全通过）。历史记录：曾有一次尝试**待做**：2026-07-30 尝试过一次，CPU 段 66 组 912 项全绿跑完，但
GPU 段启动瞬间被 preflight 拒绝——用户的 transientMamba 训练在 CPU 段中途重新上了 GPU。
重跑时先确认 `nvidia-smi --query-compute-apps` 为空。不要设置
`ALLOW_D_STATE_GPU_TESTS=1` 绕过保护，也不要杀别人的 GPU 进程。

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
