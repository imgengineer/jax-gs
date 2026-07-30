# jax-gs

基于 pure JAX、Flax NNX 和 Grain 的可微 Gaussian Splatting 实现。当前兼容目标已升级并固定到 gsplat [`main@2b902ff`](https://github.com/nerfstudio-project/gsplat/tree/2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c)（2026-07-24），并按渲染、稀疏可见性、传感器、场景/动态、损失/推理、训练集成等子系统分阶段迁移；已实现能力、明确边界和最终验收状态见 [`CURRENT_MAIN_MIGRATION.md`](CURRENT_MAIN_MIGRATION.md) 与 [`COMPATIBILITY.md`](COMPATIBILITY.md)。投影、可见项压缩、tile intersection、排序、前向 compositing 和反向传播均由普通 JAX primitives 表达，同一数值路径可由 XLA 在 CPU 或 CUDA GPU 上执行，不依赖项目自带的原生扩展、FFI 或专用 GPU kernel DSL。

核心设计是分桶的固定 shape 参数池：`ModelConfig.capacity` 默认是 1,000,000 个高斯的逻辑上限，可配置到硬上限 10,000,000；`GaussianModel.capacity` 则是当前实际分配的物理 bucket。物理 bucket 默认从 65,536 开始，按 2 倍增长到逻辑上限。在同一个 bucket 内，增密、分裂、重定位和裁剪只更新槽内容与 `active_mask`，active Gaussian 数量变化不会触发 JIT；跨 bucket 时参数、Adam moment 和策略状态一起扩容，相关 JIT 函数为新的 bucket shape 各编译一次，随后继续复用。全状态 active-prefix compaction 只在 checkpoint 前执行，不再在每次 refine 后搬运完整模型与 Adam。图像尺寸、tile 大小、候选上限、相机模型和输出通道仍是静态编译维度。

## 已实现

- Flax NNX `GaussianModel`，默认逻辑上限 100 万、可配置至 1000 万、最小物理 bucket 65,536；颜色参数严格二选一：常规模式保存 degree-0–4 `sh0`/`sh_rest`，appearance 模式保存每高斯 32 维 `features` 与 3 维 base color logits `colors`。`ModelConfig.initial_scale`（current-main `init_scale`）默认 1.0，点云 active scale 使用最多 3 个最近邻距离的 RMS × 该倍率，不足 4 点时使用全部可用邻居。
- 参数分组 Optax Adam、inactive 槽梯度屏蔽；所有 Gaussian 参数组按 current-main 的有效 batch `B=batch_size*world_size` 使用 `lr*sqrt(B)`、`eps/sqrt(B)` 和线性缩放 beta，means 学习率再乘训练坐标中的 `scene_scale`。训练器会把本地 batch 与归一化场景 extent 接入 optimizer，resume 无条件校验 batch size；`B>10` 会像上游 PyTorch Adam 一样因无效负 beta1 而提前拒绝。appearance 的 Gaussian `features`/`colors` 各自使用 `sh0_lr` 参数组。`sparse_grad` 使用带 bias correction 的 row-selective Optax Adam，`visible_adam` 使用 current-main SelectiveAdam 的 uncorrected moments。两者都保留 dense 参数/moment 存储、只更新当步可见行，并让全局 optimizer/schedule counter 前进；refine/MCMC 通过固定长度 indexed scatter 只清新分配或重定位槽的 moment，opacity reset 只清 opacity moment。
- Default 与 MCMC 桶内固定槽策略，包含增密、split/clone、prune、relocate、opacity reset、checkpoint 前 active-prefix compaction、同步扩容和容量溢出报告；duplicate/split/relocate/birth、resize 与 compaction 会同步复制或重排当前启用的 SH 或 appearance 颜色表示。Default 支持同一父高斯在 current-main 条件下同时 duplicate 与 split，并从原始父槽快照执行两个事件；默认每次 refine 最多新增 8192。与 upstream 一致，`refine_stop` 之后统计累加、refine 和 opacity reset 全部停止。
- 纯 JAX 3DGS EWA 投影、固定容量 visible packing、tile-intersection 构建、排序和多通道 forward/backward alpha compositing。
- 普通 EWA 3DGS 默认执行 pure-JAX opacity-aware SNUGBOX + AccuTile 精确 count/emit；缺少 conic/opacity、UT/畸变/2DGS 或显式选择 `aabb` 时使用保守 AABB 候选。
- 固定容量 intersection 使用 JAX tuple sort；padding 使用 sentinel，运行时 `valid_count` 限定有效前缀并生成每个 tile 的 offsets。
- dense projection 后使用固定容量 visible packing；只有保留的可见项进入 SH、intersection、排序和 compositor，超出容量会显式报告 `visible_overflow`。
- 常规颜色训练使用 split SH，避免百万槽每步先物化完整 `[N,K,3]` 拼接；appearance 模式直接消费分开的 `features`/`colors`；NNX train/refine/compaction 启用 buffer donation。
- pinhole projection 直接计算非零 Jacobian/covariance 分量，factor 路径避免物化完整 camera covariance；intersection capacity 从 65,536 起使用更快且更省显存的直接 tuple sort，小 buffer 保留 `lexsort`。
- 普通 Gaussian、feature、opacity 和 background 梯度直接由可读的 JAX compositor/autodiff 路径产生；true AbsGrad 仅在零值 screen-space probe 的广播边界使用窄 custom VJP，对每个 Gaussian×pixel 的局部 cotangent 先逐分量取绝对值再跨像素/相机累加。
- compositing 的反向做两级 rematerialization，并按设备调 tile 批：tile 的 chunk 循环与 tile map 都被 checkpoint，反向只保留"一批 tile 的一个 chunk"的 `[max_gaussians_per_tile, pixel]` 中间量，因此反向显存不再随 intersection 容量或 tile 数增长；这让更大的 `tile_batch_size` 变得可行，默认值从 4 提到 64（4 会把 920 个 tile 串成 230 步，现代 GPU 基本空转）。RTX 5090、20 万高斯、640×360、SH degree 3、65,536 intersection 桶实测：前向 915.6→353.3 ms（2.6×）、`value_and_grad` 13,409.9→2,929.3 ms（4.6×）、峰值显存 5.524→0.448 GiB（12.3×）。loss 到小数点后 10 位不变，梯度绝大多数元素逐位不变，1.2 万个元素中数百个病态项最大相对差 1e-4，属本文档既有的重结合容差范畴。
- 反向传播对 per-tile compositing 做 rematerialization：dense 3DGS、reference、2DGS 与两条 eval3d 路径都在反向按 tile 批重算，而不是把每个 tile 的 `[max_gaussians_per_tile, pixel]` 中间量同时保留。改前反向临时显存随 tile 数增长并远超前向：20 万高斯、640×360、SH degree 3、训练默认 65,536 intersection 桶下，XLA 编译期显存分析报告反向 1,169.6 GiB（前向仅 30 MiB），实跑会因申请数十 TB 而失败；改后同一场景为 5.16 GiB（227×），更大的桶保持同样比例。reference 降 35×、2DGS 降 26×、eval3d 降 50×，tile 数翻四倍后反向显存不再变化。代价是 `value_and_grad` flops 实测 +21%；dense 与 reference 梯度逐位不变，2DGS 在一个 float32 ulp 内（相对 6.2e-8）。上游 backward kernel 同样重算逐 (Gaussian, pixel) 响应，所以这是靠近上游的显存行为。
- 训练 overflow 使用设备端 sticky 状态：触发当步及后续 Gaussian/pose/appearance optimizer 与 strategy 更新立即 no-op，主机只在日志、refine、reset、checkpoint、eval 和 final 边界同步检查；scheduled MCMC refine 会先做容量计划，overflow 时 Gaussian/pose/appearance optimizer step、model、pose/appearance state、strategy stats、refine 和 position noise 整体不提交，扩容后的 replay 保持这些状态同步。
- RGB、D、ED、RGB+D、RGB+ED，以及 current-main 的 d、Ed、RGB-d、RGB-Ed hit-distance 模式；支持 classic/antialiased、background、任意前导 batch、多相机、pinhole/ortho/fisheye、直接 covariance。
- 2DGS dense/static-packed 投影与渲染，包括 leading-batch packed metadata、normals、surface normals、distortion、median/expected depth；`info["gradient_2dgs"]` 保持 current-main 的 forward-zero 语义，显式 JAX probe 在反向中产生 ray-transform 定义的 signed densify VJP，独立 AbsGrad probe 在逐像素取绝对值后累加。统一训练器支持 `model_type="2dgs"`、packed sparse training 及 normal/distortion 正则。
- 3DGUT 七 sigma 点 UT 投影、FTheta/OpenCV distortion、rolling shutter 和固定分块；`with_eval3d` 用 UT 构建候选，并以纯 JAX 按世界射线到 Gaussian 的距离精确计算响应，已支持自定义 rays、hit distance、累积 normals、extra signals/SH extra signals 和 eval3d debug outputs。
- COLMAP binary 完整解析与 Grain 数据管线，训练流按 `shuffle → repeat → batch` 组织，连续 epoch 会覆盖尾部图像，小场景不会因 `drop_remainder` 停滞，resume fast-forward 也保持确定性；已覆盖本机 Mip-NeRF 360 数据。
- COLMAP trainer 默认使用 current-main 的相机 up 对齐、focus 中位数居中、camera-distance 中位数缩放、点云 PCA 主轴对齐与 upside-down 修正；`normalize_world_space=False` 保留 identity 坐标，`global_scale` 只缩放 scene-size 相关训练参数。归一化是训练前的确定性 host NumPy 预处理，渲染和反向仍走 pure JAX。
- 默认 full-image 训练和显式静态 square-patch 训练、L1+SSIM、PSNR、SH degree schedule、评估渲染；非方形 full-image 的独立 `height`/`width` 会贯穿 renderer、densification stats、intersection capacity 和训练/扩桶 memory preflight。
- Orbax checkpoint format v6 在 v5 的 Gaussian 颜色模式、camera-pose/appearance 模块及独立 optimizer state 基础上，增加可选的 scene component。统一 trainer 生成的 checkpoint 会保存训练时的 4×4 world-to-training transform 与最终 `scene_scale`；resume 和 CLI render 直接复用该矩阵，不重新运行 PCA。v1–v5 以及 generic v6 checkpoint 仍可读取；任何缺少 scene component 的 checkpoint 都保持旧版 camera-center mean/max-extent 坐标。分布式训练另有 `save_distributed_checkpoint`/`restore_distributed_checkpoint`：把 stacked `[world, ...]` 的 model、optimizer、StrategyState、TrainingSafetyState 作为不可分割 shard 集写入，manifest 记录 world size、每 shard 与全局 capacity、每 shard slot layout 和 config fingerprint；只支持同 world-size、同 shard capacity 的精确 resume，重分片与跨类型 restore 会明确报错。兼容 gsplat 的 PLY 读写、`.splat`、Supersplat compressed PLY 字节导出与有损 PNG/SH-codebook compression；`GaussianModel` 另保留无损 PNG array transport。
- 静态 padding 的低层 intersection/indices API；运行时长度通过 `valid_count` 与 `overflow` 表达。
- current-main sparse tile layout/intersection/pixel compositing，以及 dense/sparse contributor count、all-ID/weight、top-contributor 查询；低层 dense 与 sparse pixel API 都支持显式零 probe 的 true AbsGrad；默认构造完整静态容量，显式压低容量时通过 `required_count`/`overflow` 报告截断。
- current-main `geometry.functional`：独立 `xyzw` 四元数、SE(3) 变换/矩阵互转、packed pose-track 插值、单/双 pose trajectory 与 frame transform，均为纯 JAX 并覆盖 JIT/梯度。
- current-main `scene`：NNX `GaussianScene` 组件/sidecar/拓扑事务、训练状态 resize/active-prefix compact 同步，以及 no-grad `GaussianInferenceScene` 的 planar means、FP16 QSO、RGB/SH0–3 packing 与生命周期管理。
- current-main `stage`：按 scene id 注册 `GaussianScene` 与 renderer，并以 `render_fn(splats=..., **kwargs)` 原样分发调用和返回值。
- current-main `contrib.dynamic`：NNX `DeformNetwork`、六平面多分辨率 `HexPlaneField`、纯 JAX plane/time regularization，以及与固定物理 capacity 对齐并随 duplicate/split/resize/compact 保持父子 lineage 的 `DynamicStrategy` mask。
- current-main losses/regularizers：逐元素 photometric/standard loss、NCHW SSIM、depth/LiDAR loss、Gaussian fused-loss NNX facade、color correction、targeted TV 与 occlusion mask 工具。
- current-main experimental inference：packed scene 的纯 JAX stateless RGB/alpha 渲染、32B/16B SH codec 数值路径，以及可复用的 NNX float16 RGBT renderer/lifecycle。
- current-main 初始化/训练结构：多帧 depth unprojection、分块 KNN scale 初始化、点云 active scale 的 3-NN RMS × `initial_scale`、`TwoStageScheduler`，以及 `training`、`strategy`、`optimizers` 包级导入路径；2DGS preset 使用 near/far 0.2/200、prune opacity 0.05 和 `gradient_2dgs` key，MCMC 3DGS preset 使用 initial opacity/scale 0.5/0.1 及 0.01 opacity/scale regularization。
- 3DGS 训练的 `opacity_reg`/`scale_reg` 从 raw opacity logits/log-scales 计算 sigmoid/exp 后只对 active rows 求均值，支持 JSON、`--opacity-reg`/`--scale-reg` 和同名 loss metrics；2DGS 对任一非零权重在配置边界拒绝。
- Flax NNX camera-pose 与 appearance 训练已端到端接入单进程统一 3DGS/2DGS trainer。`jax_gs.training.pose.CameraOptModule` 用每图平移 + 6D rotation embedding 右乘 camera-to-world local delta，并保留 upstream 的随机构造与显式 `zero_init()` 训练起点。`jax_gs.training.appearance.AppearanceOptModule` 拼接训练 image embedding、32 维 Gaussian feature 与方向 SH basis，输出零初始化的 `[C,N,3]` color-logit correction；训练以 split-local image id 选择 embedding，direct RGB 为 `sigmoid(colors + correction)`，两类模块都有独立 optimizer、原子 overflow replay 和精确 checkpoint/resume。
- current-main strategy 公共 hooks 与固定槽操作：显式 signed `<key>_gradient` 或 AbsGrad `<key>_absgrad` 统计、duplicate/split/remove/reset/relocate/sample-add、同父 duplicate+split，以及 overflow 时模型、optimizer、统计、Scene/Dynamic sidecar 和 MCMC noise 的原子跳过；MCMC 训练支持 UT/Eval3D，并在这些模式下不构建其不消费的 screen densification stats。
- current-main 工程辅助面：传感器 tensor helpers/完整 dispatch 表、JAX trace/profile capture 与 forward/gradient replay、由外部 JAX 进程启动器驱动的 distributed CLI，以及根级 `rasterization(distributed=True)` 的 single-rank 与 named-axis multi-shard 路由。
- `jax_gs.training.make_distributed_train_step()` 提供 Gaussian-sharded 设备训练：绑定 `nnx.pmap`/named axis 后，每个 rank 持有等物理容量的 Gaussian shard 与本地相机 batch，pure-JAX differentiable gather 令 Gaussian 光度梯度等于各 rank 局部 mean-loss 梯度之和。当前切片限定 SH、pinhole 3DGS，支持 dense/packed 投影、`visible_adam`、Default 与 MCMC 策略，以及 camera-pose 优化与 pose noise（pose 模块按 upstream 的 DDP 语义复制并对梯度跨 rank 求平均，Gaussian 仍是 owner-scatter 求和）；它校验 optimizer 的 batch/world/scene-scale/config 契约，在全局 Gaussian 索引上归约 visibility，并按每个 camera/Gaussian 的 signed screen gradient 先取范数、再以 `sum/sum/max` 聚合 densification statistics，最后切回 owner shard。当前/历史 overflow 及 rank step/SH-degree 不一致会对 optimizer/model/statistics 执行全 rank 原子 no-op。
- 分布式 refinement 在 update 前 plan、update 后 commit：每个 rank 用 train step 的 `scene_scale` 在自己的 owner-local 行上生成 duplicate/split/prune 计划，仅把标量摘要以 `psum/psum/pmax` 归约成全局 `refine_scheduled`、`reset_scheduled`、`refine_planned_new_count`、`refine_planned_pruned_count`、`refine_required_capacity`、`refine_capacity_overflow` metrics；任一 rank 计划 capacity overflow 会让所有 rank 原子跳过本步，便于 host 扩容后重放。update 之后每个 owner 按 current-main 的 post-optimizer 顺序用普通 `DefaultStrategy.refine` 提交自己的 duplicate/split/prune 与 opacity reset，并用后验状态重算 events（因此与 preflight 计划可以合理地不同），提交结果通过 `refine_new_count`、`refine_pruned_count`、`refine_commit_overflow`、`opacity_reset` 汇报。shard 物理容量在 step 内不变；`jax_gs.capacity.resize_distributed_training_state()` 提供把所有 shard 统一扩容到同一 capacity 的原语，被跳过的那一步扩容后可原样重放，但选择目标 bucket、预算显存和驱动重放循环仍属 host。

## 当前验收与明确边界

- 2026-07-29 compositing 性能切片的 forced-CPU 验收为常规 `873 passed, 1 skipped, 38 deselected`，随后五个 fresh-process resource-heavy 分组通过 `19+5+9+3+2=38` 项；合计 911 项通过。唯一 skip 是本机缺少可选的 Mip-NeRF360 stump 数据集，4 条 warning 为既有 Orbax sharding 恢复提示。2026-07-29 首次完整跑通 GPU 安全脚本（`RUN_GPU_TESTS=1 RUN_RESOURCE_HEAVY_GPU_TESTS=1`）：CPU 段 66 个 fresh-process 分组 910 项，GPU 段 121 个独立 CUDA case，合计 1,031 项通过、零失败、无 preflight 中止。该次验收覆盖的是 per-tile rematerialization 状态；其后的 chunk-level rematerialization 与 `tile_batch_size` 默认值变更已有完整 CPU 验收，GPU 复跑待设备空闲。
- JAX 数组不能承载 PyTorch backward 后写入的 mutable `.absgrad`。低层 dense/sparse pixel API 和高层 3DGS/2DGS reference/intersections 路径改用独立的显式零值 probe，probe 对 forward 严格无影响；3D 使用 projected-means VJP，2DGS 使用其独立的 ray-transform densify VJP。训练器和 strategy hook 已接入该路径。3D eval3d 与 distributed AbsGrad 不在当前支持组合内。
- `sparse_grad=True` 要求 unbatched `packed=True`；3DGS 还拒绝 distributed、显式 UT、eval3d，以及隐式走 UT 的 `camera_model="ftheta"`。renderer 参数梯度、模型与 Adam moment 仍是 fixed-bucket dense arrays；`sparse_grad`/SparseAdam-compatible 路径使用 bias-corrected Adam，`visible_adam` 使用 uncorrected SelectiveAdam，二者都不宣称 COO/稀疏内存收益。JAX 兼容类以 `update(model, grads, visible_mask)` 和只读 `.step` counter 取代 PyTorch/source 的 autograd 后 `step(visibility)` 调用面。
- 2DGS 支持 packed sparse training，但 `backend="reference"` 的 packed 训练缺少跨 projection/intersection metadata，因此在配置边界明确拒绝；3DGS 的 `visible_adam` 不要求 packed，2DGS 不支持 `visible_adam`。
- MCMC 3DGS 训练支持 UT 和 Eval3D，因为它不依赖 Default 的 screen densification stats；Default+Eval3D 仍明确拒绝。Default full-image 训练已实现，且保留显式 patch；full-image 模式要求训练图像共享同一 `(height, width)`，混合分辨率数据需指定 patch。Camera pose 与 appearance 的 config/CLI、3DGS/2DGS train-step、独立 optimizer、overflow replay 与 checkpoint/resume 已接入统一单进程训练流程；分布式切片尚未包含 pose/appearance、设备内 bucket 扩容、host 数据分片或 checkpoint/eval orchestration。
- 根级 distributed renderer 支持 single-rank 和绑定 named axis 的等长 padded multi-shard；`make_distributed_train_step()` 已在 named `nnx.vmap` 与双虚拟 CPU `nnx.pmap` 上验证，但项目仍不内置多进程 launcher，也不支持 distributed Gaussian leading batch 或完整 host 训练编排。其 `loss/l1/ssim/psnr/active_count/visible_count` metrics 保持 rank-local，overflow/intersection/refinement diagnostics 为 global；owner-local signed densification statistics 已与所有 rank 的相机贡献同步并驱动 owner-local duplicate/split/prune/reset 提交；shard 集也可整体 checkpoint/resume，但设备内 bucket 扩容、world-size 变更重分片与 host 训练编排仍未实现。distributed 下对 Gaussian leading batch、appearance per-view colors、UT/eval3d、非 pinhole、`sparse_grad`、AbsGrad 的拒绝与上游自身的 `distributed=True` 校验一致，不是缺口。多进程/多主机仍由 `jax.distributed`、MPI、Slurm 等在入口外启动；在数据、拓扑与 checkpoint 同步接入前，统一 `train()` 对 `jax.process_count() != 1` 明确拒绝。
- 当前迁移以 API、数值语义和工程可读性为优先目标；appearance 与其余训练路径尚未做专用 kernel、分布式 state sharding 或正式吞吐/显存对标，因此不宣称与 upstream gsplat 性能等价。GPU 最终验收也仍待安全串行重跑。

## 环境

```bash
uv sync --all-groups
uv run python -c "import jax; print(jax.devices())"
```

`pyproject.toml` 使用 `jax[cuda13]`、Flax NNX、Grain 和 Optax，并通过 setuptools 构建纯 Python wheel。`jax[cuda13]` 只提供 JAX/XLA 的 CUDA 运行环境；项目源码不编译或调用自带的 CMake/CUDA 扩展及第三方 kernel DSL。本机当前环境为 RTX 5090、JAX 0.11.0、Flax 0.12.8、Grain 0.2.18。

## Rasterizer backend

`RasterizationConfig.backend` 支持以下模式：

| backend | 行为 |
|---|---|
| `auto`（默认） | 使用 pure-JAX fixed-capacity sorted-intersections 路径 |
| `jax` / `intersections` | 与 `auto` 相同；保留显式名称便于配置迁移和对照 |
| `reference` | 调试/对照路径；每个 tile 扫描完整 Gaussian 物理 bucket 后执行 `top_k`，通常最慢 |

旧配置中的 `backend="cutile"`、`intersection_backend="cutile"` 或 `sort_backend="cutile"` 会发出弃用警告并归一化为 `"jax"`；更早的 `cuda_ffi`/`pallas` 配置也会在加载时迁移到 JAX。CLI 不再提供旧 backend 选项。这个兼容入口只负责读取已有配置，不会加载旧依赖或选择另一条运行时路径。

`RasterizationConfig.intersection_backend` 的 `auto` 与 `jax` 都使用 pure JAX。`intersection_mode="auto"` 在普通 EWA 3DGS 上使用 opacity-aware AccuTile，`aabb` 强制矩形候选，`accutile` 强制精确椭圆路径；UT、非线性投影和 2DGS 自动使用 AABB，因为其近似 conic 不保证是保守边界。这里的 **AccuTile 是 gsplat/SpeedySplat 的保守椭圆与 tile 相交算法，不是 NVIDIA cuTile 编程模型**；本项目的 AccuTile count/emit 由 pure JAX 实现。

`RasterizationConfig.sort_backend` 的 `auto` 与 `jax` 都使用 JAX tuple sort。排序只消费 `valid_count` 指定的前缀，padding 使用 sentinel，并为每个 tile 生成 offsets。固定 capacity 仍是 JIT shape，改变它会触发新编译。

可通过 metadata 检查固定容量结果是否发生截断：

```python
render, alpha, info = rasterization(...)
print(info["intersection_overflow"], info["tile_overflow"])
print(info["visible_count"], info["visible_capacity"], info["visible_overflow"])
```

## 数据检查

```bash
uv run jax-gs inspect-data /home/lzc/datasets/stump \
  --image-dir images_8
```

加载器直接读取：

```text
scene/
├── images_8/
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

内参按图片实际宽高分别缩放，COLMAP `qvec/tvec` 按 world-to-camera 约定解析。无需 `points3D.ply`。

## 训练

先生成配置：

```bash
uv run jax-gs init-config \
  --data /home/lzc/datasets/stump \
  --output stump.json
```

默认配置使用 100 万逻辑上限、65,536 最小物理 bucket、`initial_scale=1.0`、`backend="auto"` 和 full-image 训练。训练选择能覆盖初始点数的最小 bucket；refine 前再按精确的计划新增数判断是否需要跨桶。独立估算 full-image 内存时需提供实际尺寸，例如：

```bash
uv run jax-gs estimate-memory --config stump.json \
  --image-height 480 --image-width 640
uv run jax-gs estimate-memory --config stump.json --active-target 1000000 \
  --image-height 480 --image-width 640
uv run jax-gs train --config stump.json
```

训练命令会从 COLMAP scene 自动取得真实 `height`/`width`；上面的 480×640 只演示 `estimate-memory` 的显式参数，请替换为数据集尺寸。full-image 支持非方形图像，但训练 split 中的图像必须共享同一尺寸。需要随机 square patch 或处理混合分辨率数据时显式传 `--patch-size N`。

COLMAP 训练默认启用 current-main world normalization。可用 `--no-normalize-world-space` 保留输入坐标，`--global-scale X` 只调整 means 学习率与 densification 策略所消费的 scene scale；resume 要求这两个值与 checkpoint config 一致。Trainer-generated v6 checkpoint 会复用保存的矩阵与尺度，不重新依赖点云 PCA。

`--capacity` 的允许范围是 1–10,000,000。需要百万级增密时可通过 JSON 或 CLI 调整 `--max-new-per-refine`、`--refine-every` 和 `--refine-stop`；默认每次 refine 上限已提高到 8192。

3DGUT 投影训练可添加 `--with-ut`；MCMC 策略可使用 `--with-eval3d`，该选项会在 renderer 内启用 UT，并跳过 MCMC 不需要的 screen densification stats。Default 策略的 Eval3D 训练尚不支持。FTheta 相机需通过 Python API 提供标定多项式。

不加载已有 config/checkpoint 时，`--model-type 2dgs` 选择 current-main 2DGS profile：near/far plane 为 0.2/200、`prune_opacity=0.05`、densification key 为 `gradient_2dgs`；`--strategy mcmc` 选择 3DGS MCMC profile，把 current-main `init_opa=0.5`/`init_scale=0.1` 映射为 `initial_opacity`/`initial_scale`，并设置 `opacity_reg=scale_reg=0.01`。3DGS 可用 `--opacity-reg`、`--scale-reg` 覆盖权重，metrics 中分别报告加权的 `opacity_reg_loss` 和 `scale_reg_loss`；2DGS 配置会拒绝非零值。加载 config/checkpoint 后，CLI 只应用显式 override，不会隐式重置整套 profile。

固定容量 packed/可见行训练可使用 `--packed --sparse-grad`，或在 3DGS 中单独使用 `--visible-adam`。前者要求 unbatched packed renderer，使用 bias-corrected Adam；后者匹配 current-main uncorrected SelectiveAdam。两种模式都保留 dense 参数和 Adam state，只改变可见行的更新语义。`sparse_grad` 不能与 UT/Eval3D/FTheta 组合。2DGS 支持 `--packed --sparse-grad`，但不支持 `--visible-adam`，且 packed 训练不能使用 `backend="reference"`。

Camera pose 优化可通过 `--pose-opt` 启用，并用 `--pose-opt-lr`、`--pose-opt-reg` 和 `--pose-noise` 配置；后三者默认分别为 `1e-5`、`1e-6` 和 `0.0`。统一 3DGS/2DGS train step 按训练 image id 应用局部 pose delta；`pose_noise` 先施加固定、停止梯度的扰动，再学习 pose adjustment。独立 pose Adam 使用随 batch size 缩放并指数衰减的学习率和 coupled L2，Orbax checkpoint 保存模块与 optimizer state，并通过相机数量和 image-name 顺序校验 resume 映射。为避免重建指数 LR 时间轴时发生隐式跳升，pose resume 还要求总 `steps` 与 checkpoint 配置一致。

Appearance 优化可通过 `--app-opt` 启用，并用 `--app-embed-dim`、`--app-opt-lr` 和 `--app-opt-reg` 配置；默认分别为 16、`1e-3` 和 `1e-6`。启用后 Gaussian 不再保存 SH 参数，而是保存固定 32 维 `features` 与 base `colors` logits；统一 3DGS/2DGS train step 以当前训练 split 的 image id 选择 embedding，并把 MLP correction 与 base logits 相加后 sigmoid 为 direct RGB，因此传给 rasterizer 的 `sh_degree=None`。Appearance embedding Adam 的学习率为 `app_opt_lr * sqrt(batch_size) * 10` 并使用 coupled weight decay，color head Adam 使用 `app_opt_lr * sqrt(batch_size)` 且无 decay；二者与 Gaussian/pose 更新共享原子 overflow/replay。resume 会校验 appearance 开关、embedding 维度、学习率、正则、SH degree、batch size，以及 manifest 中的相机数量和 image-name 顺序。

快速冒烟可缩小逻辑上限和步数；`--capacity` 表示逻辑最大值，必须不少于初始 COLMAP 点数，并不表示总是立即预分配该数量：

```bash
uv run jax-gs train \
  --data /home/lzc/datasets/stump \
  --image-dir images_8 \
  --capacity 40000 \
  --patch-size 16 \
  --num-workers 4 \
  --steps 1 \
  --output outputs/stump-smoke
```

本机真实数据跨桶冒烟结果：读取 32,049 个 stump 稀疏点，以 32,768 物理 bucket 启动，refine 前安全扩容到 65,536，连续完成两个 Grain → 随机 patch → GPU 前向/反向 → Adam 步骤并保存/恢复 Orbax checkpoint；最终 active 40,241，tile/intersection overflow 均为 0。

## 渲染与导出

```bash
uv run jax-gs render outputs/default/checkpoints/step_00005000 \
  --output render.png \
  --strict-overflow

uv run jax-gs export outputs/default/checkpoints/step_00005000 model.ply
uv run jax-gs export outputs/default/checkpoints/step_00005000 model.splat
```

Appearance checkpoint 的 held-out evaluation、CLI render 和其他无训练 image id 的推理使用零 camera embedding，并在评估时使用配置的完整 SH direction-basis degree。CLI export 会先按 current-main 的 canonical 约定用零 embedding、零方向和完整配置 degree 计算 RGB，再烘焙为 degree-0 `sh0`（`sh_rest` 为空）后交给常规 PLY/`.splat` exporter；原始 `features`/`colors` 不会被直接当作 SH 导出。这里的 canonical bake 只定义可移植的单一外观，不保留逐训练图像的 embedding 变化。

点云初始化会把 RGB 限制到稳定的开区间后再求 color logit，避免精确 0/1 产生无穷值。这是有意的数值健壮性差异：upstream 当前直接使用 `torch.logit(rgb)`，边界颜色可能得到 `±inf`。

需要定位 pure-JAX fixed-capacity 路径与完整 bucket 扫描之间的差异时，可显式添加 `--rasterizer-backend jax`、`intersections` 或 `reference`；普通使用建议保留 `auto`。全分辨率 render 会先估算工作区，`--strict-overflow` 会在写出图像前拒绝 intersection 或 tile 候选截断。

Python API：

```python
from jax_gs import GaussianModel, ModelConfig, rasterization

model = GaussianModel.empty(ModelConfig())
assert model.capacity == 65_536       # 当前物理存储 bucket
assert model.max_capacity == 1_000_000  # 逻辑上限
params = model.activated()
render, alpha, info = rasterization(
    params["means"],
    params["quats"],
    params["scales"],
    params["opacities"],
    params["sh_coeffs"],
    viewmats,
    Ks,
    width,
    height,
    active_mask=params["active_mask"],
    sh_degree=3,
)
```

## OOM 防护

包会在用户没有显式设置时，于首次导入 JAX 前设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，并用 `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1` 限制 GPU 编译并行度。后者会让首次 JIT 略慢，但避免 CUDA 13.3 `ptxas` 多 worker 同时编译大型反向图时的宿主内存尖峰和已观测 SIGSEGV；热运行吞吐不受影响。JAX 0.11 已移除旧的 LLVM module parallelism flag，因此不再注入该 flag。用户显式提供的环境变量优先。benchmark 脚本会强制关闭预分配。训练开始前按实际启动的物理 bucket 与训练 `(height, width)` 估算固定 shape 工作集；非方形 tile grid、intersection capacity 和 raster workspace 都使用独立 H/W。跨桶扩容前还会计入旧、新训练状态短暂并存的峰值并再次预检。估算超过设备内存限制的 70% 时拒绝分配，运行中超过 85% 时停止。全分辨率 CLI render 同样执行 70% 工作区预检。

未显式设置 `JAX_COMPILATION_CACHE_DIR` 时，包会启用私有的持久化编译缓存 `${XDG_CACHE_HOME:-~/.cache}/jax-gs/jax-compilation-cache-v1`，目录权限为 `0700`。这让相同设备、XLA flags 和静态 shape 的后续进程复用 executable，减少重复 `ptxas` 编译；显式 cache 路径或 `JAX_ENABLE_COMPILATION_CACHE=false` 均优先。缓存内容等价于受信任的可执行代码，不应放在其他用户可写的共享目录。当前安全测试脚本串行运行；若主动并发启动多个独立冷编译进程，建议为每个任务指定不同 cache 目录。

Grain/KD-tree worker 默认 4 个，也可通过 `--num-workers` 配置为更大的正整数。可用 worker 数按当前进程的 CPU affinity 计算（不支持 affinity 的平台退回 `os.cpu_count()`）；配置值超过它时会发出 oversubscription warning，但不会拒绝运行。Grain 预取队列固定为 8，避免更高并发自动放大图像预取内存。运行时不设置或截断 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS`，这些库使用各自默认值或用户环境配置。XLA/PTX codegen 默认保持单 worker，用户在导入前显式提供的 `XLA_FLAGS` 仍然优先。

当物理 bucket 为 1,000,000、degree-3 SH 时，模型、两组 Adam moment、mask 和策略持久状态约 0.7 GiB；本机百万 active 的完整 NNX/Adam 受控训练步峰值约 1.13 GiB，配置的保守峰值估计约 2.04 GiB。物理 10,000,000、degree-3 的完整训练保守估计约 15.63 GiB；degree-4、大尺寸 full-image 和大规模 bucket transition 会显著增加风险。默认最小 bucket 为 65,536，实际启动值只需覆盖初始点数；`jax-gs estimate-memory --active-target N --image-height H --image-width W` 会按目标 active 数选择物理 bucket，并按 full-image 的实际非方形尺寸给出估算；patch 配置则直接使用 `patch_size`。估算会按相机 batch 数放大 dense projection/UT 工作区，训练内全分辨率评估也会把当前常驻模型与 Adam 显存计入 70% 预检。

`active_mask` 的作用是让增密、裁剪和重定位在当前物理 bucket 内不改变参数 shape，从而复用同一个 JIT executable；它不会让 bucket 内的参数、dense projection 或 optimizer state 稀疏执行。projection 后的 visible packing 会缩小 SH/intersection/compositor 工作量；active-prefix compaction 则延后到 checkpoint 前，避免每次 refine 全量搬运状态。训练不再每步 `device_get` overflow 标量；设备端 sticky 标记会先冻结所有更新，最迟在下一个控制边界（常规日志间隔最多 10 步）由主机抛错，且任何 refine/reset/checkpoint/eval 都发生在检查之后。

建议：

- 不要同时启动多个 JAX GPU 测试或训练进程。
- 若在导入 `jax_gs` 前先导入了 JAX，请在 shell 中预先设置 `XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"`。
- 先运行 `jax-gs estimate-memory`。
- 冒烟测试使用小 patch、较小逻辑上限/最小 bucket 和较小 `max_gaussians_per_tile`。
- `info["intersection_overflow"]` 表示全局固定 intersection buffer 被截断，结果不能用于训练或最终评估；提高 `max_intersections`。
- `info["visible_overflow"]` 表示投影后的可见数已经超过同一固定容量；它会并入 `intersection_overflow`，不会静默用于训练。
- `max_gaussians_per_tile` 只控制 pure-JAX compositor 的临时分块；超过它仅设置 `candidate_limit_exceeded` 诊断，不会截断。`tile_overflow` 只表示输入 intersection buffer 声称的 tile 候选超过其实际物理容量，训练会拒绝这种不完整结果。
- 训练检测到会截断梯度的 intersection/tile overflow 时会终止。最终渲染建议加 `--strict-overflow`。
- 全图评估按 camera/tile 串行批处理，优先控制显存而非峰值吞吐。
- 直接从千万点 COLMAP 点云初始化会同时构建主机 KD-tree 和参数 staging arrays；更安全的常规路径是从较小初始点云按 bucket 增长到目标规模。
- benchmark 默认不运行 backward；显式 backward 还受小分辨率、physical capacity、K 和 tile batch 安全门限保护，除非用户主动传入 `--allow-unsafe`。

## 性能状态

当前阶段优先保证 current-main API、数值语义、训练流程和代码可读性，尚未发布 pure-JAX 路径与 upstream gsplat 的可复现性能结论。早期专用后端的测量不能代表当前实现，已从项目说明中移除。benchmark 仍可用于本机回归，但结果必须注明设备、JAX/XLA 版本、静态 capacity、图像尺寸、warmup 和是否包含梯度；这些数字不应外推为端到端训练吞吐。

## 测试

为避免单个 pytest 进程累积大量 CPU/GPU executable，建议使用脚本按测试文件启动独立进程并严格串行运行：

```bash
./scripts/test_safe.sh
# 额外串行 GPU 冒烟：
RUN_GPU_TESTS=1 ./scripts/test_safe.sh
```

GPU 模式会先取得单用户独占锁，并在开始前及每个用例之间检查残留的 `nvcc/cicc/cudafe++/ptxas/ninja`、`sd-rmrf`、NVIDIA compute process 和 D-state 线程；任一存在都会拒绝继续。只有用户明确接受内核 D-state 风险时才可临时设置 `ALLOW_D_STATE_GPU_TESTS=1`，编译器或 NVIDIA compute 残留仍不可绕过。CUDA 用例按 test case 使用独立进程，并以 180 秒为默认超时；超时会终止整个普通后代进程组，测试失败或脚本退出时仍会执行 postflight。collection 失败或没有收集到用例也会直接失败，不会静默跳过。显式 resource-heavy 分组使用 `-o addopts="-ra" -m resource_heavy` 覆盖项目默认的 `addopts`，避免其中的 `not resource_heavy` 将目标用例误 deselect。构建并行度在该安全脚本中固定为 1；测试脚本默认把主机数学库线程设为 4，但同样尊重显式环境变量，训练数据 worker 则服从配置。

`RUN_GPU_TESTS=1 ./scripts/test_safe.sh` 在 GPU 上执行与 CPU 相同的 pure-JAX
数值路径。不要把整个 CUDA 测试集合放进同一个 pytest 进程；本机已观察到
这种做法会累积编译器状态并最终触发 `pxla`/`ptxas` 崩溃。

可复现性能测试使用：

```bash
uv run python benchmarks/benchmark_rasterization.py \
  --backend auto \
  --capacity 1000000 \
  --active 10000 \
  --resolution 640x360 \
  --k 512 \
  --tile-batch 4
```

接近 gsplat v1.5.3 garden benchmark 的可复现命令：

```bash
uv run python benchmarks/benchmark_rasterization.py \
  --npz /path/to/gsplat/assets/test_garden.npz \
  --capacity 10000 --active 10000 \
  --resolution 640x360 --k 512 --tile-batch 4 \
  --radius-clip 3 --gsplat-v153-garden-profile \
  --hot-iters 100 --backward --backward-iters 100 --allow-unsafe
```

脚本输出 cold/hot timing、warmup 次数、intersection/tile overflow 和 peak memory。这里的 `--capacity` 始终是 benchmark 的物理数组长度。默认只测试 forward；不要在桌面 GPU 上直接为 physical capacity 100 万启用 backward，除非已经独立核对工作区和设备余量。

需要定位 kernel 热点时，可在 warmup 后额外采集短 trace；forward 和 backward 会写入不同子目录，可直接用 Perfetto/TensorBoard profile viewer 打开：

```bash
uv run python benchmarks/benchmark_rasterization.py \
  --capacity 65536 --active 8192 --resolution 640x360 \
  --k 512 --max-intersections 262144 \
  --profile-dir /tmp/jax-gs-profile --profile-iters 3
```

CPU 用例可以手工运行，例如：

```bash
JAX_PLATFORMS=cpu uv run pytest -q \
  tests/test_math.py tests/test_cameras.py tests/test_spherical_harmonics.py \
  tests/test_losses.py tests/test_colmap.py tests/test_dataset.py
```

GPU 用例不要用上面的手工合并方式；统一使用安全脚本，让每个 test case
进入独立进程并在前后执行资源预检。

## 与 gsplat 的静态 shape 差异

gsplat CUDA 的 `packed` projection、tile intersections 和 pixel hits 使用运行时 `nnz` 长度。JAX 中这与稳定 JIT shape 冲突，因此本项目采用：

- 高层 `rasterization()` 在每个物理 bucket 内保持固定 shape 与兼容返回语义。
- 低层动态结果使用 padded arrays、`valid_count`、mask 与 `overflow`。
- 仅在 JIT 外提供裁剪为动态长度的兼容辅助函数。
- 3D 高层 `packed=True` 在 intersections、reference 和 eval3d 路径都返回固定 `P=B*C*N` 的全局有效前缀、IDs、`indptr` 和重映射后的实际 compositor intersection metadata；2DGS 也返回跨 leading batch 的全局 packed 前缀与 metadata。
- `sparse_grad` 的 renderer 参数梯度与 optimizer storage 仍为 dense fixed-bucket arrays，并报告 `sparse_grad_is_dense=True`；其 row-selective Adam 保留 bias correction，`visible_adam` 则使用 uncorrected moments。3D signed densification 使用 projected-means probe；2DGS 的 forward-zero `gradient_2dgs` 通过 ray-transform densify probe 获得独立 VJP。两者的 true AbsGrad 都由独立零 probe 累加逐像素绝对 cotangent，不用 `abs(signed_gradient)` 近似。
- active 数量在 bucket 内变化不 retrace；跨 bucket 时训练状态同步扩容，相关 JIT 函数为新的静态 shape 各编译一次。
- 所有 compositor 模式共享 pure-JAX 数值实现；不同设备、XLA 版本和 upstream CUDA reduction order 可能产生小量浮点差异，因此只承诺容差内一致而非逐位一致。

当前目标固定到 gsplat `main` 的
`2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c`（2026-07-24）。该提交后端无关的
公开 Python 模块和导入路径已经完成清单审计；可观察数值能力按本文列出的边界
继续分阶段验收。已覆盖的子系统包括 HiGS experimental inference、AV 相关
camera/LiDAR/stage、PPISP loss/regularization，以及本轮训练/分布式集成切片。
PyTorch autograd
包装类、根级 `cuda` extension wrapper、原生 CUDA backend loader/build 模块和上游
测试辅助 `_helper.py` 不在 pure-JAX 包中伪造；对应数值能力由直接可微的 JAX 函数
提供。精确边界和最终验收见
[`CURRENT_MAIN_MIGRATION.md`](CURRENT_MAIN_MIGRATION.md) 与
[`COMPATIBILITY.md`](COMPATIBILITY.md)。

## JAX 项目参考

- [`jaxsplat`](https://github.com/yklcs/jaxsplat) 通过修改后的 gsplat CUDA kernels 与 JAX FFI 避开运行时动态 shape。
- [`splax`](https://github.com/amacati/splax) 使用 Warp FFI 实现数据依赖 intersection/sort，并采用 opacity-aware 紧致 tile、32 位排序键、持久 scratch、early-exit 和反向重算。当前项目只参考其与 gsplat 的 SNUGBOX/AccuTile 数值公式，以 fixed-shape pure JAX 实现 count/emit、排序和 autodiff compositor。
