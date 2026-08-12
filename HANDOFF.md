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

Phase 9–11 当前尚未提交，worktree 只改动：

- `src/jax_gs/_pallas.py`
- `src/jax_gs/_pallas_intersections.py`
- `src/jax_gs/intersections.py`
- `tests/test_pallas_compositor.py`
- `README.md`
- `HANDOFF.md`

它不改 projection、geometry state、count prefix、tuple sort、默认 pure-JAX/AABB 路径、checkpoint schema 或 Pallas 支持边界。Pallas AccuTile emit 的固定输出槽 owner 从逐槽 `searchsorted` 改为 run-start marker + associative prefix-max；仅 Pallas AccuTile 的 direct-sort offsets 从 `searchsorted` 改为 tile histogram + prefix。Pallas compositor backward 现在直接输出 packed slot cotangent，并让 Gaussian scatter 用越界 sentinel 丢弃无效 padding；没有新增 Mosaic atomic kernel，optimizer 也没有复制第二套状态机。

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
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
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

## Phase 9：Pallas AccuTile owner/offset 整数展开

Phase 8 留下的两处 `searchsorted` 已在真实 garden shape 上重新量化。固定输出槽 owner 反查约 0.167 ms，等价的 run-start marker + `associative_scan(max)` 约 0.080 ms；排序后的 920 个 tile offsets 反查约 0.162 ms，等价的 `bincount` + prefix 约 0.110 ms。两项都是整数、stop-gradient 元数据变换，不增加新 Pallas kernel。

实现边界保持窄：

1. owner marker 在每个 Gaussian 的 retained-run 起点写 `gaussian_id + 1`；零 count Gaussian 的相同起点用 scatter-max 选择最大 id，等价于 `searchsorted(..., side="right")`。只标记 `starts < valid_count`，overflow 截断后的无效起点不会污染前缀。
2. `associative_scan(jnp.maximum)` 把 owner 沿固定输出容量传播；`local = rank - starts[owner]` 保持原 Gaussian-major run offset。capacity 为 0、capacity 小于/大于 Gaussian 数、empty valid prefix、padding 和 overflow 都与 JAX emit 对拍。
3. direct-sort padding 的 sentinel key 是 `tile_count`；Pallas AccuTile 路径用长度 `tile_count + 1` 的 histogram 接住 sentinel，再只对真实 tile bins 做 prefix。JAX AccuTile、AABB、UT/畸变/2DGS fallback 和小容量 lexsort 路径仍使用原有 offsets 实现。

RTX 5090 / JAX 0.11.0、garden 138,766 active、640×360、352,091 intersections、最忙 tile 1,992、无 overflow：3 次 pristine Phase 8 与 4 次 Phase 9 fresh-process、每次 200 次热运行的完整 renderer mean 为 forward 1.082→1.028 ms（约 5.0%），`value_and_grad` 3.455→3.450 ms（约 0.1%，视为噪声内）。render、alpha、排序后的 packed intersection ids、offsets、counts 和 overflow metadata 逐位相同；loss 逐位相同，五组参数梯度最大绝对差 `4.34e-7`。

单独的 262,144 行 dense Adam + quaternion normalize 同步调用约 0.90 ms，但 Perfetto 中 GPU kernels 合计约 0.36 ms，其余主要来自把 optimizer 独立拿出 `nnx.jit` 后的 NNX graph flatten/unflatten host 开销。真实训练把 loss/backward/update 放在同一 compiled step，因此这个 microbenchmark 不支持复制 Optax 状态或新增 optimizer kernel。

## Phase 10/11：丢弃 padding scatter 与 packed backward 输出

Phase 10 的关键观察是：旧宽 scatter 对整个固定 intersection capacity 先做 `[slot, 6+C]` 零掩码，再把所有 padding 零行 scatter 到 clipped Gaussian 0。garden 640×360 的 524,288 槽只有 352,091 有效，隔离 scatter 因无效零 atomic 被放大到约 0.46 ms。现在无效行改指向 `gaussian_count` 越界 sentinel，并用 `.add(..., mode="drop")` 丢弃；有效非负 id 仍先 clip，保留原 malformed-input 语义。隔离 shape 约 0.46→0.064 ms。

Phase 11 让 Mosaic backward 直接写单个 `[slot, 6+C]` 输出，替代 means/conics/colors/opacity 四个独立缓冲与 kernel 后 concatenate。这个变化单独约 0.7% 完整反向收益，但减少了一次全容量拼接且让 VJP 数据流与最终宽 scatter 一致，因此和 Phase 10 一起保留。

RTX 5090 / JAX 0.11.0 的完整结果：

- garden 640×360、352,091/524,288 intersections：四次 fresh-process mean `value_and_grad` 3.450→3.038 ms，1.14×、约 11.9%；forward 未改，均值约有 1% 时钟波动。
- garden 256×256、207,218/524,288 intersections：三次 fresh-process mean `value_and_grad` 5.268→4.718 ms，1.12×、约 10.4%；forward 1.466→1.411 ms，但只视作同进程公共路径/时钟波动，不归因给纯反向改动。
- 相对 Phase 9，loss 逐位相同；五组参数梯度 relative L2 均 ≤`5.4e-8`，最大绝对差 `1.03e-6`。原生 RGB 与四通道 RGB+D 低层对拍均在既有 float32 容差内。
- 真实 Mip-NeRF360 garden 的 12 步与 110 步 NNX/Adam 训练均完成、零 tile/intersection overflow，并生成最终 checkpoint。长跑 loss/PSNR 轨迹只出现预期的 float32 累积差异。

## Tokamax Mosaic 配置探针（未保留）

参考 OpenXLA Tokamax 隔离筛选了 `approx_math=True`、`unsafe_no_auto_barriers=True` 和 2–12 KiB `reduction_scratch_bytes`。backward-only `unsafe_no_auto_barriers=True` 的普通四次 fresh-process A/B 确实把完整 `value_and_grad` 从 3.033 降到 2.784 ms（约 8.2%），render/alpha/loss/metadata 逐位相同，五组梯度也只出现 float32 归约级差异；12 步真实训练 smoke 同样通过。

但 JAX 0.11 对该 `unsafe_*` 选项列出的充分安全条件在这里无法被证明：`jnp.sum` 会使用编译器内部 reduction scratch，标量 ref assignment 又按 swap 语义 lowering。NVIDIA Compute Sanitizer 的控制实验也不具区分力：flag 开/关时 `synccheck` 都是 0 error，`racecheck` 都报告同样 24 个 replicated write/write hazards。该 flag 因此按保守原则完全撤回，不能把普通数值对拍当作安全证明。安全的单行写回替代也被当前 native lowering 卡住：`stack` 未实现、splat `concatenate` 不支持、WG-strided 短行要求至少 128 元素；为 7–10 个 float 引入 SMEM/TMA 不值得。`approx_math=True` 单独使完整反向约慢 2%，reduction scratch 从默认 2 KiB 扫到 4/6/8/12 KiB 也无额外稳定收益，均未保留。Tokamax 的 warp-specialized SMEM pipeline 主要服务可分块矩阵运算；这里 candidate 之间有严格反向 transmittance recurrence，不能直接并行流水。`pl.loop` 在锁定 JAX 版本中也只是 `lax.fori_loop` 包装，不构成独立优化。

当前宽 scatter 已约 0.04 ms，不再是热点；保留自动 barrier 后，首要 renderer 热点仍是约 1.77 ms 的 Pallas backward kernel 本体。共享 reciprocal/改写 transmittance 代数只改善约 0.16%，已撤回。Mosaic 仍不能表达动态向量-id atomic，但现在没有理由为已退出热点的 scatter 强行增加 atomic kernel。

## Pallas Projection 探针（未保留）

曾按“单设备、dense 3DGS、单相机 float32 pinhole、quats/scales”边界实现独立 Mosaic projection 原型，并用 custom VJP 在反向重算权威 pure-JAX projection，以保留 means/quats/scales/opacities、viewmat 与 K 的梯度。默认 JAX、AABB、UT/Eval3D、2DGS、畸变和其它相机路径均未放宽。

当前 JAX 0.11 Mosaic Lane lowering 没有 `sqrt` 与 `ceil`，原型只能用精确 `x * rsqrt(x)` 正平方根和非负整数截断补一实现。更关键的是，Lane 的逐标量乘加允许 contraction：若把 camera transform、quaternion factor 和 covariance projection 全搬入 Mosaic，少量半径/valid 边界会相对现有 JAX `einsum` 漂移。为了保持离散几何契约，最终安全原型让现有 JAX 按原阈值选择 covariance/factor route，并预计算 means2d、depth 与原始 2D covariance；Pallas 只完成 eps2d、conic、opacity extent、radius 与 valid 后处理。

该 hybrid 在 interpreter/native 的构造对拍中覆盖 Gaussian 数 0/1/127/128/129、classic/antialiased、active mask 和非整块尾部：radii、means2d、depth、valid 精确一致，conic/compensation 保持 float32 容差；Gaussian 与相机输入 custom-VJP 梯度 relative-L2 <`2e-5`、最大绝对差 <`5e-5`。隔离原型测试为 15 passed。

但 RTX 5090 / JAX 0.11.0、实际 garden 138,766 Gaussian、640×360、radius clip 3、20 次 warmup、每个 fresh process 200 次同步 hot iteration 的三组交错 projection-only A/B 没有收益：

- JAX forward median：0.1328 / 0.1330 / 0.1307 ms；Pallas：0.1412 / 0.1401 / 0.1298 ms。三组中位数约 0.1328→0.1401 ms，慢 5.5%。
- JAX `value_and_grad` median：0.3046 / 0.3236 / 0.3180 ms；Pallas：0.3553 / 0.3437 / 0.3532 ms。三组中位数约 0.3180→0.3532 ms，慢 11.0%。

这些数字只覆盖 projection stage，不外推到完整 renderer 或训练。由于安全版本必须先在 JAX 完成最重的 projection 代数，再承担额外 Mosaic launch/custom-VJP 重算，结果符合预期。`projection_backend`、CLI、renderer dispatch、测试和 `_pallas_projection.py` 已全部撤回；production worktree 没有保留任何 projection 改动。除非后续 Mosaic 能在不改变离散几何契约的前提下直接降低权威 projection 数学，或新的 profile 证明形状/瓶颈已变化，否则不要重复这个 hybrid。

## Pallas compositor backward v2 探针（后续因 correctness 修正保留）

按 gsplat CUDA 的 `last_ids` 思路做了 go/no-go 原型：forward 为每个 pixel 保存最后一个 accepted candidate；backward 以 tile 内最大 endpoint 缩短 reverse loop，并对每个 pixel 屏蔽 endpoint 之后的候选。真实 garden 离线重放显示理论上可跳过 88,951 / 352,091 candidate-slot（25.3%），383 / 920 个 tile 至少缩短一个候选。

实现过程中确认了两个 Mosaic 约束。第一，Pallas 未写 output 不是可靠零值；因此缩短 loop 后必须精确屏蔽每个 tile 实际写过的 prefix。第二，原生 Lane `fori_loop` 的 int endpoint carry 需要与 WG-strided fragment layout 完全一致；原型改用 float carry 再写回 int32 才能通过 lowering。补齐 written-prefix mask 后，1/3/4 通道、candidate bound 1/5 的 interpreter/native 梯度矩阵与大容量 empty-tail 回归均通过。

但 fresh-process 结果明确未达到门槛。和原 benchmark loss 一致时：

- 预先保存的 retained baseline：forward median `1.011 ms`，完整 `value_and_grad` median `3.032 ms`。
- 紧邻原型前按同一 official benchmark 命令重跑的 retained baseline：`1.017 / 3.042 ms`。
- endpoint 原型：forward median `1.051 ms`，完整 `value_and_grad` median `2.670 ms`；loss 同为 `0.32643967866897583`，352,091 intersections、无 overflow。相对紧邻 baseline，forward 回归约 3.3%，完整 `value_and_grad` 改善约 12.2%。

endpoint 原型只把完整 renderer 降到约 `2.67 ms`，对应反向净段仍约 `1.62 ms`，因此作为纯性能路线仍未达到 `~0.8 ms` go/no-go 目标。后来在 CUDA FFI 的 garden 梯度门禁中发现，旧 packed-output Pallas 实现会把未写固定容量槽的未初始化值 scatter 回 Gaussian；这个 correctness 问题使 endpoint/written-prefix 方案从“性能 no-go”变成必须保留的修复。修正版大场景梯度重新与 pure JAX 对齐。

同一轮原生 atomic 探针进一步确认当前 JAX 0.11 Mosaic 不能作为 fallback：Lane/长度一 slice 仍触发 `_atomic_store`/discharge 三字段对四字段错误；Warpgroup 路径改报 `memref.subview` offset 不是 value sequence；`plgpu.kernel` 也不接受 `input_output_aliases`，不能用预零 alias buffer 规避未写 output。不要通过 patch 私有 JAX internals 把这些探针带入 production。

结论更新：Pallas backward v2 作为追求 gsplat 级性能的路线仍是 **no-go**，但 endpoint + written-prefix mask 作为 native 大容量 backward correctness 修复保留。进一步性能由 CUDA/XLA FFI 承担；不再继续 Pallas shared-memory/warp-level micro-tuning。

## CUDA/XLA FFI compositor 与 Pallas backward 修正

参考 gsplat `RasterizeToPixels3DGSSerialBatchFwd/Bwd` 新增显式 `RasterizationConfig.compositor_backend="cuda_ffi"`。默认 `jax` 与现有 Pallas/intersection 路径不变；只有显式选择时才通过 JAX 0.11 typed FFI 加载 CUDA target。实现边界是单设备 float32 3DGS、tile size 16、channels 1/2/3/4/8/16/32，不支持 reference backend、2DGS、Eval3D、AbsGrad、distributed、JVP 或高阶梯度。

CUDA 实现使用每 tile 一个 16×16 CTA、shared-memory Gaussian batches、per-pixel last contributing slot、reverse shared batches、warp reduction 与直接 per-Gaussian atomicAdd。它保留本项目而非盲拷 gsplat 的数值契约：`MAX_ALPHA=0.999`、自定义 alpha/transmittance threshold、clamp equality 的 0.5 VJP、fixed-capacity `valid_count`/padding、candidate-bound 取整和 overflow。背景混合留在普通 JAX，因此 background gradient 不经过 atomic kernel。反向 typed-FFI outputs 在 launch 前用 supplied stream 上的 `cudaMemsetAsync` 清零。

运行时 loader 默认按 CUDA source、JAX version、nvcc version 和 compute capability 哈希到 `~/.cache/jax-gs/cuda-ffi`，用 POSIX file lock 避免并发构建；`JAX_GS_CUDA_FFI_LIBRARY` 可提供预编译 `.so`，`JAX_GS_CUDA_FFI_CACHE_DIR` 和 `JAX_GS_NVCC` 可覆盖缓存/toolchain。普通 import 不触发 CUDA 编译。`uv build` 已成功生成 sdist/wheel，两者都包含 `_cuda_ffi.py` 与 `cuda_ffi/compositor.cu`；从 wheel 安装到隔离环境后，随包 source 可被首次运行编译并完成原生 forward。

本轮同时发现并修复了此前遗漏的 Pallas 大场景 backward correctness bug：packed-output kernel 缩短 reverse loop 后，未写的固定容量 output slot 在 native Mosaic 中不是可靠零值，却仍被外部 scatter。小构造测试恰好没有暴露。修正版 forward 保存每 pixel last contributor/accepted transmittance；backward 只扫描到 tile endpoint，并在 scatter 前只保留各 tile 实际写过的 prefix。garden low-level mean-loss 的 means/conics/color/opacity gradient relative-L2 均约 `1e-7` 量级；完整参数 gradient norms 与 pure JAX/CUDA FFI 对齐。原 focused compositor suite 继续通过，并新增大 empty-tail gradient zero 回归。

RTX 5090、garden 138,766 active、640×360、352,091 intersections、无 overflow，official benchmark 每次 30 warmup + 200 hot、三次 fresh process：

- 最终 Pallas median-of-runs：forward `1.054 ms`，完整 `value_and_grad` `2.689 ms`。
- 最终 CUDA FFI median-of-runs：forward `0.864 ms`，完整 `value_and_grad` `1.594 ms`；满足预设 `≤2.0 ms` go/no-go。
- 相对修正版 Pallas，CUDA FFI 完整参数梯度 relative-L2：means `3.1e-6`、quats `5.3e-5`、scales `2.8e-5`、opacities `6.0e-7`、colors `2.0e-6`；loss 精确一致。render max abs `5.26e-4`、alpha max abs `9.54e-7`，主要来自 CUDA fast exp/累加顺序。
- 100 次 atomic repeat probe 的最大变化：means `1.15e-7`、quats `1.78e-7`、scales/SH chain `4.74e-6`、opacities `1.16e-10`、colors `5.82e-11`。这是 explicit backend 的预期 float32 reduction-order nondeterminism，默认 JAX/Pallas 不受影响。
- 隔离 projected-input compositor：Pallas forward/value-and-grad `0.482 / 2.354 ms`，CUDA FFI `0.289 / 0.799 ms`。
- 一步 native NNX/Adam smoke 的 loss、overflow、step 与 JAX 一致，model/optimizer leaf 最大差约 `3e-11`。
- 真实 Mip-NeRF360 garden、138,766 active、262,144 bucket、256×256 patch 的 CUDA FFI 12 步和 110 步 NNX/Adam 均完成并生成 checkpoint，全程 tile/intersection overflow 为零、显存约 `0.19/23.55 GiB`。12 步 step 1/10 的打印 loss 与 Pallas 一致；110 步因 fast-exp/atomic 累积出现预期轨迹差异，最终 loss `0.167782`（Pallas retained run `0.169162`），不能据此宣称端到端训练数值逐步一致。
- 空缓存首次 native build + tiny forward 约 4 秒，随后复用约 1.48 MiB cache library；普通 import 即使设置无效 prebuilt path 也不加载 `_cuda_ffi`。
- Compute Sanitizer 对 channels 1/2/3/4/8/16/32、partial tile、negative/repeated IDs 的 forward+backward：memcheck 0 error、racecheck 0 hazard、synccheck 0 error。Nsight Systems garden 单次记录为 forward kernel约 `0.282 ms`、backward kernel 约 `0.513 ms`，12 次 `cudaMemsetAsync` 合计约 `0.070 ms`；Nsight Compute 因主机 `ERR_NVGPUCTRPERM` 未能采集硬件 counter。

和 gsplat 旧 target 的 640×360 `0.377 / 1.039 ms` 相比，当前完整 jax-gs CUDA FFI renderer 约为 forward `2.29×` 慢、完整 `value_and_grad` `1.53×` 慢；反向差距已显著缩小，前向剩余主要是仍在 JAX 的 projection/intersection/sort 与 CUDA compositor 自身约 `0.29 ms`。

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

## Phase 9–11 验收结果

- Phase 9：`tests/test_accutile_intersections.py tests/test_intersections.py tests/test_rasterization_jax.py` 为 53 passed；额外 Pallas/intersection/training 定向选择为 52 passed、1 个 CPU-only rejection skip、108 deselected。
- Phase 10/11：Pallas compositor/rasterization/training 最终定向组 28 passed、1 个 CPU-only rejection skip、123 deselected；扩展后的 1/3/4 通道低层 compositor 整文件为 14 passed、1 skip，native 与 interpreter 都覆盖 packed slicing 和重复 Gaussian scatter-add。
- Tokamax compiler 配置探针：`unsafe_no_auto_barriers` 的 native backward `synccheck` 为 0 error，`racecheck` 在 flag 开/关时都报同样 24 个 replicated-store hazards，无法证明候选安全；候选已按保守原则撤回，不属于保留实现的验收结果。
- owner 展开的额外 interpreter 对拍覆盖 capacity 0/1/17/97/193/1,345（同时小于与大于 129 Gaussian）以及 empty valid prefix；全部和 pure-JAX emit ids 精确一致。
- 真实 640×360 garden 的 render/alpha/packed intersection ids/offsets/counts/overflow 全部逐位对拍；loss 逐位相同，五组梯度最大绝对差见 Phase 9 节。
- Mip-NeRF360 garden 138,766 active、262,144 物理 bucket、256×256 patch 的 12 步原生 Pallas NNX/Adam 训练完成；step 1/10 均为零 tile/intersection overflow，并生成 step 12 checkpoint。
- 独立 fresh-context reviewer 未发现 blocker/high/medium correctness 问题；只指出新增分支的 direct unit-test 覆盖可更显式，当前已由上述构造对拍、真实 garden packed metadata 对拍和整文件回归覆盖。
- `git diff --check`、edited-file `py_compile` 和 `compileall` 通过。Primary Python LSP 对编辑文件 clean；辅助 Pyright 实例无法解析项目 `.venv` 中的 JAX/NumPy/Pytest imports，且不会热重载虚拟环境配置，属于工具环境问题。Phase 9–11 未运行完整 CPU/GPU safe script；最近一次完整 CPU safe-script 证据仍是 Phase 7。

## 后续迁移顺序

1. 下一条性能线优先优化 CUDA FFI forward 和共同 JAX 前端，而不是继续 Pallas micro-tuning。先用 Nsight 拆分 `~0.864 ms` 完整 forward：isolated CUDA compositor 约 `0.289 ms`，其余主要是 projection/intersection/sort/launch。只有 profile 证明 sort/intersection 占主导时再迁移 CUB sort 或 fused intersection；保持 checkpoint schema、projection/geometry/count-prefix/tuple-sort 的默认契约和现有 support guards。`unsafe_no_auto_barriers`、`approx_math`、扩大 reduction scratch、共享 reciprocal、简单代数改写、把 `pl.loop` 当作新 lowering，以及“JAX 预计算几何 + Pallas projection 后处理”的 hybrid 均已证伪，不要重做。distributed/AbsGrad/2DGS/Eval3D 必须各自做数值、atomic nondeterminism 与 overflow 门禁，不能直接放开当前 guard。
2. 再设计 multi-process/multi-host host ownership、per-process checkpoint、数据加载与一致 preflight；现有 stacked host state helper 不能直接宣称支持它。
3. 让 CLI render 默认消费 checkpoint 中动态增长后的 intersection/candidate high-water mark，再由显式 CLI 参数覆盖；当前仍使用保存的 TrainConfig 值，容量不足时会报告 overflow，最终输出应使用 `--strict-overflow` 或显式覆盖容量。
4. 继续迁移 distributed UT/eval3d、非 pinhole、`sparse_grad` 和 AbsGrad；每项按 current-main 实际组合单独对齐，不用过时的“上游统一拒绝”归因。
5. Phase 7 的完整 CPU safe script 已通过；GPU 仍按 case 隔离通过定向门禁，若扩大 Pallas surface 再运行完整 GPU safe script。
6. HiJAX 只在 Flax/JAX 的官方最小 value-and-grad/update 模式能在锁定版本上稳定通过后再评估；在此之前不要为它复制训练循环或放宽 topology/alias 语义。

提交或继续开发时仍应检查实际暂存清单，不能带入根目录用户文件 `CLAUDE.md`。
