# jax-gs 性能与 Pallas 交接

更新日期：2026-08-12。兼容目标仍固定为 gsplat `main@2b902ff`。

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

根目录未跟踪的 `CLAUDE.md` 是用户文件，不属于仓库改动；除非用户另行确认，后续提交仍须显式排除。

Phase 3–8 均已提交并推送到 `origin/main`：

- `498166d feat(training): support distributed appearance`
- `1e325fc feat(checkpoints): load distributed inference snapshots`
- `1ec52d8 perf(rasterizer): add experimental Pallas forward`
- `14f9a60 perf(rasterizer): add Pallas compositor backward`
- `b76d243 perf(rasterizer): add Pallas AccuTile counting`
- `19cd7c1 perf(rasterizer): add Pallas AccuTile emission`

Phase 8 的实现基线是已推送的 `19cd7c1`；该阶段已完成 AccuTile pair emission 的增量替换和 Flax HiJAX 隔离评估。

Phase 8 不改 projection、geometry state、prefix/searchsorted、sort、offsets、默认 pure-JAX 路径或 checkpoint schema。`_pallas_intersections.py` 现在负责 AccuTile count 与固定输出槽 pair-emission scan；`_pallas.py` 仍只把原有四次 Gaussian scatter 合并为一次宽 scatter。

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

## Phase 5/6：Pure JAX 审计与 Pallas 前向/反向

性能审计先围绕当前 87% 训练步占比的 compositor 做 Pure JAX 改写。动态 `lax.switch` 静态 loop 变体、复用 `cumprod` 尾元素以及 `lax.associative_scan` 都没有在稳态生产 shape 上得到可靠收益：前两项完整 renderer 持平或更慢，associative scan 在 K=512 时 forward 约慢 27%、backward 约慢 8%，且改变结合顺序；这些试验均已撤回，默认 Pure JAX 数学没有改动。

当前落地的是实验性 `RasterizationConfig.compositor_backend="pallas"`：

- projection、visible packing、AccuTile/AABB intersection、排序及 overflow 都继续走 JAX；Pallas 只替换最后的逐 tile alpha compositor。
- 一个 Mosaic GPU program 拥有一个 tile，按该 tile 的真实候选数执行动态 `fori_loop`。像素寄存器按 128-lane `WG_STRIDED` 布局，RGB/深度通道使用独立连续 `[pixel]` 累加器；tile 边缘使用 finite mask，输出再恢复普通 HWC。
- forward 额外保存每 tile/pixel 的最终 transmittance。custom VJP 在 tile 内逆序恢复 `T_before=T_after/(1-alpha)`，用后缀 weight cotangent 计算 alpha 链梯度，并输出无竞争的 per-intersection means/conics/colors/opacity 梯度；普通 JAX 再按 Gaussian id scatter-add，避免在 Pallas 中使用全局 atomic。
- 默认 backend 仍是 `"jax"`。Pallas 要求 JAX 0.11 的 Mosaic GPU、NVIDIA Hopper 或更新架构及 float32 输入；当前支持单设备 3DGS reverse-mode 训练，不支持 distributed training、AbsGrad、2DGS、Eval3D、reference backend、JVP 或高阶梯度。CPU interpreter 只用于数值测试。
- `max_candidates_per_tile` 仍是被检查的静态承诺，并按 `max_gaussians_per_tile` 取整；Pallas 遇到更忙 tile 同样设置 `tile_overflow`，不会静默截断。`tile_batch_size` 只属于 Pure JAX 静态分组，Pallas 不使用它。
- CLI 可在 `jax-gs render` 与 `jax-gs train` 上用 `--compositor-backend pallas` 显式启用；benchmark 的 `--backward` 也可比较两种 compositor。训练 factory 对上述不支持组合在建图前明确拒绝。
- `compositor_backend` 不进入 v6 training checkpoint fingerprint：它是可在 resume 时切换的执行后端，且旧 checkpoint 在该字段出现前已生成；跨 backend 只承诺容差一致，要求逐位续训时必须保持 backend、设备与 JAX/XLA 版本相同。其它 config/模型/optimizer 契约保持不变。

RTX 5090、JAX 0.11.0、`K=512`、tile batch 64 的质量等价完整 renderer 基准如下。候选上限恰好覆盖实际最忙 tile；10k 报告 30 次 hot mean，200k 报告 20 次，均无 overflow：

| 场景 | intersections / 最忙 tile | JAX forward | Pallas forward | JAX value+grad | Pallas value+grad | 反向加速 |
|---|---:|---:|---:|---:|---:|---:|
| 10,000，640×360 | 61,970 / 116 | 3.403 ms | 1.051 ms | 10.722 ms | 3.408 ms | 3.15× |
| 200,000，640×360 | 1,237,085 / 1,846 | 13.036 ms | 3.668 ms | 43.802 ms | 11.559 ms | 3.79× |

两组 benchmark loss 逐位相同。10k 的 JAX/Pallas `peak_bytes_in_use` 约 263/74 MiB；200k 约 298/297 MiB。低层 JIT 对拍覆盖五组参数、重复 Gaussian scatter、clamp、padding 与 overflow；高层两相机对拍覆盖 3D geometry 与 viewmat 链路，原生 NNX train-step 的模型、Adam 和策略状态也在容差内一致。

## Phase 7：真实训练 profile 与增量 intersection kernel

这一步没有先猜 kernel。profile 使用 Mip-NeRF360 garden 的 138,766 个真实 COLMAP 点、实际相机与图像；训练采用 256×256 patch、262,144 物理 bucket，另用 `test_garden.npz` 在 640×360 下稳定复测完整 renderer。结果显示 Pallas compositor 已不是训练首要热点：四组 Gaussian gradient scatter 合计约 1.84 ms/步，AccuTile count 的 JAX loop 约 0.14 ms/步。

落地改动保持两个小边界：

1. compositor VJP 仍先在 Mosaic 中生成无竞争的 per-intersection cotangent，但把 means、conics、colors、opacity 拼成 `[slot, 6+C]`，只做一次 JAX scatter，再拆回四组 Gaussian 梯度。真实训练 trace 中该部分约 1.84→1.12 ms/步。
2. `intersection_backend="pallas"` 只把已经准备好的 AccuTile state 交给每 128 Gaussian 一个 Mosaic program 的 count scan。pair emission、prefix、tuple sort 与 offsets 仍是 JAX；AABB 路径不使用这个 kernel。真实训练 trace 中 count 约 0.14→0.007 ms/步。

640×360 的实际 garden 输入产生 352,091 个 intersection，最忙 tile 1,992，无 overflow。四组交错 500 次 hot-forward 中，Pallas count 相对 JAX count 的完整 renderer mean 改善 8–11%；两组 200 次 `value_and_grad` 改善 2–4%，loss 相同。12 步真实 NNX/Adam 训练完整通过。当前 JAX 0.11 Mosaic GPU 无法为动态向量 Gaussian id 降低所需 atomic reduction，因此没有用更复杂的 kernel 强行取代最后一个 JAX scatter。

默认仍为 JAX；Pallas intersection 要求单设备 Hopper 或更新 NVIDIA GPU，distributed factory 会提前拒绝。`compositor_backend` 仍不进入 checkpoint fingerprint；`intersection_backend` 历史上已属于训练 config，为兼容既有 v6 checkpoint 保持原契约。

## Phase 8：Pallas AccuTile pair emission 与 HiJAX 探针

Phase 7 的 GPU trace 显示 count 已退出热点，剩余 intersection 端主要成本是 pure-JAX pair-emission outer scan。Phase 8 保留原有 count prefix，让 JAX 用 `searchsorted` 为每个固定输出槽解析 owner Gaussian；随后每 128 个输出槽由一个 Mosaic program 顺序走查各自 owner 的椭圆 outer 轴并直接写 Gaussian/tile id。geometry state、prefix/searchsorted、tuple sort 和 offsets 都没有迁移，AABB 与默认 JAX 路径也没有变化。

同一 RTX 5090 进程中交错比较 Phase 7 的“Pallas count + JAX emit”和 Phase 8 的“Pallas count + Pallas emit”，输入为 garden 138,766 点、640×360、352,091 intersections、最忙 tile 1,992、无 overflow：

- 500 次 hot-forward mean：1.294 → 1.058 ms，1.22×。
- 200 次完整 `value_and_grad` mean：3.828 → 3.457 ms，1.11×。
- 两路 render、alpha、排序后的 intersection ids、计数、loss 逐位相同；参数梯度最大绝对差 `8.44e-7`。

低层测试用 129 个 Gaussian 及 193/1,345 个输出槽同时覆盖 overflow、无效 padding 和非整块尾部，CPU interpreter 与原生 Mosaic 都和 pure JAX ids 精确一致。Pallas intersection 仍只支持单设备 Hopper 或更新 NVIDIA GPU，distributed factory 继续提前拒绝。

Flax 0.12.8 的实验性 HiJAX 只做了隔离探针，没有保留半成品入口。仓库当前 JAX 0.11.0 下，官方最小 Linear+Adam 结构在 `nnx.with_vars`/`nnx.vars_as(..., mutable=False)` 处就因 traced Variable 元数据失败；跳过 immutable view 后，`jax.grad` 又明确报告 HiJAX `get_variable` 没有 JVP。由于基础训练用例尚不能完成一步，而本项目还需要 NNX optimizer alias、条件分支和桶内拓扑 mutation，当前继续使用已验证的 `nnx.jit`，不复制第二套 Optax 状态机。

## 明确边界

- 仅支持单进程、多本地设备；multi-process/multi-host 仍须在创建输出前拒绝。
- Pallas training 是独立的单设备 3DGS opt-in 路径，不属于 local-device distributed trainer；默认和所有 CPU 运行仍使用 pure JAX compositor 与 intersection。
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

## Phase 5 验收结果

- CPU：config/Pallas interpreter/CLI/storage 95 passed；`tests/test_rasterization_jax.py` 33 passed、1 个 native Pallas case 按平台跳过；`tests/test_training.py tests/test_training_distributed.py` 168 passed；2DGS Pallas 拒绝项单独通过。
- GPU：tile-size 16 / RGB 的 native Mosaic compile 与高层 JAX 对拍通过；额外手工覆盖两相机、RGB+D、10,000 与 200,000 高斯。200,000 组的完整数值与 overflow 结果见上表。
- `git diff --check` 与 `compileall` 通过。Phase 5 尚未重跑完整 CPU/GPU safe script，不要把定向结果写成全套验收。

## Phase 6 验收结果

- CPU：`tests/test_pallas_compositor.py tests/test_rasterization_jax.py tests/test_training.py tests/test_cli.py tests/test_config.py tests/test_storage.py tests/test_two_dgs.py` 共 229 passed、5 个 native-only skip、20 deselected；`tests/test_training_distributed.py` 76 passed。
- GPU：低层 native/interpret、候选 overflow 与重复 Gaussian 梯度对拍通过；高层 Pallas renderer 定向组 10 passed、1 个 CPU-only skip（含两相机 native reverse）；NNX Pallas train-step 的 interpret/native 两路 2 passed。
- benchmark 使用不溢出的 500,000/2,000,000 intersection capacity 和恰好覆盖最忙 tile 的 candidate bound；10k/200k 的完整 `value_and_grad` 分别加速 3.15×/3.79×，loss 逐位相同。详细配置和数值见上表与 README。
- `git diff --check`、`compileall` 通过。Phase 6 未重跑完整 `scripts/test_safe.sh`，不得把这些定向结果写成完整 safe-script 验收。

## Phase 7 验收结果

- CPU 相关整文件回归：AccuTile/intersection/config/storage/rasterization/training 共 222 passed、4 个 native-only skip；CLI 21 passed。Pallas 定向选择另为 22 passed、6 个 native-only skip。
- RTX 5090：AccuTile count、compositor、两相机高层梯度与 NNX train-step 的 Pallas 定向选择 27 passed、1 个 CPU-only skip；actual garden 的 JAX/Pallas count 输出与 overflow metadata 精确一致。
- `./scripts/test_safe.sh` 完整 CPU 串行回归退出码 0；6 个 native Pallas case 与本机缺失的 stump 数据集按预期跳过，4 条 warning 是既有 Orbax generic checkpoint sharding 提示。
- `git diff --check` 与 `compileall` 通过。Phase 7 未运行完整 GPU safe script，不得把隔离的 RTX 5090 定向结果写成全套 GPU 验收。

## Phase 8 验收结果

- CPU 相关整文件回归：AccuTile/intersection/config/storage/rasterization/training/CLI 共 243 passed、4 个 native-only skip；加强后的 overflow/padding scan 定向用例随后再次通过。
- RTX 5090：AccuTile/Pallas/rasterization/training 定向选择 21 passed；真实 garden 的 Pallas emit 与 Phase 7 JAX emit 做了上述同进程交错 A/B，并额外确认完整 JAX 与完整 Pallas 的 render/alpha/排序 intersection metadata 精确相同。
- `compileall`、`git diff --check` 与提交前暂存清单检查通过；Phase 8 提交只包含实现、接线、测试和文档，没有带入 `CLAUDE.md`。
- Phase 8 未重跑完整 CPU/GPU safe script；最近一次完整 CPU safe-script 证据仍是 Phase 7。上述 243/21 是 Phase 8 的相关 CPU/GPU 定向验收，不应描述成完整 GPU 回归。

## 后续迁移顺序

1. 若继续性能线，Phase 8 后优先重新量化约 1.12 ms 的宽 Gaussian scatter、JAX `searchsorted` 与 optimizer；tuple sort 和 pair emission 已不再是首要热点。Mosaic 当前不能表达动态向量-id atomic，因此不要把宽 scatter 原样搬回 kernel。只有测到稳定瓶颈后才增加下一枚 Pallas kernel。distributed/AbsGrad/2DGS/Eval3D 必须各自做数值与 overflow 门禁，不能直接放开当前 guard。
2. 再设计 multi-process/multi-host host ownership、per-process checkpoint、数据加载与一致 preflight；现有 stacked host state helper 不能直接宣称支持它。
3. 让 CLI render 默认消费 checkpoint 中动态增长后的 intersection/candidate high-water mark，再由显式 CLI 参数覆盖；当前仍使用保存的 TrainConfig 值，容量不足时会报告 overflow，最终输出应使用 `--strict-overflow` 或显式覆盖容量。
4. 继续迁移 distributed UT/eval3d、非 pinhole、`sparse_grad` 和 AbsGrad；每项按 current-main 实际组合单独对齐，不用过时的“上游统一拒绝”归因。
5. Phase 7 的完整 CPU safe script 已通过；GPU 仍按 case 隔离通过定向门禁，若扩大 Pallas surface 再运行完整 GPU safe script。
6. HiJAX 只在 Flax/JAX 的官方最小 value-and-grad/update 模式能在锁定版本上稳定通过后再评估；在此之前不要为它复制训练循环或放宽 topology/alias 语义。

提交或继续开发时仍应检查实际暂存清单，不能带入根目录用户文件 `CLAUDE.md`。
