# CPU + GPU 混合焦散加速：可验证实施方案

> 版本：v3（可证伪实施收敛版）
>
> 状态：规划文档，尚未实施
>
> 适用仓库：`E:\Caustica_Blender_GPL`
>
> 目标功能：在不拖慢渲染的前提下，评估并逐步实现 CPU + GPU 共同加速表面焦散和体积焦散

---

## 1. 结论先行

CPU + GPU 混合加速只能作为有条件方案，不能默认开启，也不能把光子数量平均分给 CPU 和 GPU。

修正后的路线是：

```text
先确定 CPU 是哪一种角色
  A. CPU 已经是 Cycles 渲染设备
  B. GPU-only 会话外挂一个只做 Gather 的 CPU work
        ↓
阶段 B0 固定选择 B
        ↓
仅体积焦散 F12
        ↓
固定小方块、固定 5% 份额、2 线程
        ↓
整帧墙钟至少提升 3% 到 5%
        ↓
才允许进入调度器阶段
```

关键结论：

1. 当前项目的 GPU 光子追踪速度很高，CPU 追踪通常是短板。
2. 焦散真正随分辨率、采样数、体积查询增长的部分是收集和平滑，不是固定数量的光子追踪。
3. 现有 Cycles 多设备框架已经能创建多个 `PathTraceWork`，CPU 和 GPU 都能执行光子收集。
4. 当前 `photon_pick_trace_device()` 已明确在 CPU + GPU 混合设备中优先选择 GPU 追踪光子。
5. 单 GPU、非体积焦散的 device-resident 路径只向主机回传少量计数表；让 CPU 参与 Gather 会切断该路径，并新增完整光子图 D2H。
6. 体积焦散当前已经需要主机侧 PhotonGrid 和光束 BVH，D2H 属于基线成本；CPU + GPU 收集在这里相对更有可行性。
7. 视口每代更新频繁，D2H、线程唤醒和同步延迟占比高，默认不启用 CPU Gather。
8. F12、高分辨率、厚体积介质是唯一值得优先验证的场景。
9. “CPU 已是 Cycles 渲染设备”和“GPU-only 会话外挂 CPU Gather”是两条不同路径，必须分开验证。
10. 第一阶段复用现有 tile 划分，不另建平行任务图。

修正后的目标：

- 表面焦散单 GPU 默认保持 device-resident GPU-only。
- 体积焦散 F12 先验证 CPU 分区收集。
- 已有多设备 D2H 场景可以继续评估 CPU 收益。
- 视口默认禁用 CPU Gather。
- CPU 只接收预计能在 GPU 完成前做完的工作。
- 第一版使用固定份额，不引入吞吐 EMA 和复杂撤销。

不能承诺“1 + 1 > 1”。是否净加速必须通过固定场景实测决定。

正确比较对象是整帧墙钟：

```text
T_hybrid =
    max(T_gpu_only * (1 - s), T_cpu_only_full_frame * s)
    + d2h_extra
    + h2d_extra
    + sync
    + power_loss

allow =
    T_hybrid < T_gpu_only
```

其中：

- `s` 是分配给 CPU 的像素比例。
- `T_cpu_only_full_frame` 是 CPU 独自完成整帧收集的墙钟时间。
- `d2h_extra` 是 CPU 参与新增的光子图回读。
- `h2d_extra` 是 CPU 收集结果回传 GPU 的成本。
- `power_loss` 是 CPU 负载导致 GPU 降频后的额外时间。

CPU 的“工作时长”不能直接当作帧时间节省。

---

## 2. 目标与非目标

### 2.1 目标

1. 建立可复现的 GPU-only 与 CPU + GPU 对照基准。
2. 单独测量每代 PhotonGrid 的 D2H 字节数和耗时。
3. 验证 CPU 对体积焦散、有限光束查询的 F12 渲染是否有净收益。
4. 验证已有多设备场景中 CPU 参与 Gather 是否有净收益。
5. 在 CPU 落后时自动收回任务，避免拖慢整帧。
6. 后续动态调度阶段再记录 GPU 频率、功耗和利用率。
7. 保持 CPU-only、GPU-only、CUDA、OptiX 现有路径稳定。
8. 保持混合渲染结果与 GPU-only 在统计误差范围内一致。
9. 所有调度逻辑保持可关闭，默认行为不能因混合调度改变。

### 2.2 非目标

第一阶段不处理以下内容：

- 不把光子追踪平均拆给 CPU 和 GPU。
- 不在表面焦散单 GPU 的 device-resident 路径上默认启用 CPU Gather。
- 不在视口渲染中默认启用 CPU Gather。
- 不接受新增 D2H 成本高于 CPU 节省时间的情况。
- 不引入 StarPU、SYCL、Kokkos 等大型运行时。
- 不重写 Cycles 的通用多设备系统。
- 不改变 SPPM、光子光束的物理定义。
- 不把 CPU 当作无条件启用的加速设备。
- 不在没有实测数据前默认开启混合收集。
- 不在阶段 A/B0 接入 NVML、ROCm SMI 或跨平台功耗采样。
- 阶段 B0 只验证当前 CUDA/OptiX 主后端；Metal、HIP、OneAPI 不纳入首轮验收。
- 不把“其他后端编译通过”当作混合功能已验证。

---

## 3. 可核验资料

以下资料已核验到 DOI、Crossref/OpenAlex 元数据或公开源码仓库。论文用于证明方案方向，源码用于确认现实实现路径。

部分 DOI 落地页会对自动请求返回 `403`，但 DOI 本身及 Crossref/OpenAlex 元数据均已核对。

### 3.1 焦散与光子方法

| 资料 | 可核验地址 | 对本方案的用途 |
| --- | --- | --- |
| Stochastic Progressive Photon Mapping | https://doi.org/10.1145/1618452.1618487 | 当前 SPPM 收集、半径收缩和渐进统计的基础 |
| Photon Mapping on Programmable Graphics Hardware | https://doi.org/10.1145/1198555.1198797 | 证明光子追踪、分箱和查询适合映射到 GPU |
| Parallel Progressive Photon Mapping on GPUs | https://doi.org/10.1145/1899950.1900004 | 证明渐进光子映射可在 GPU 上并行实现 |
| Distributed Out-of-Core Stochastic Progressive Photon Mapping | https://doi.org/10.1111/cgf.12340 | 证明 SPPM 可以跨多资源分配追踪与视线计算，并降低资源空置 |
| Progressive Photon Beams | https://doi.org/10.1145/2024156.2024215 | 体积光束和渐进式光束估计的依据 |
| A Comprehensive Theory of Volumetric Radiance Estimation Using Photon Points and Beams | https://doi.org/10.1145/1899404.1899409 | 光子点、光子光束在体积估计中的统一理论 |
| Real-Time Volume Caustics with Adaptive Beam Tracing | https://doi.org/10.1145/1944745.1944753 | 有限体积光束、自适应光束构造的参考 |

### 3.2 混合 CPU + GPU 渲染与调度

| 资料 | 可核验地址 | 对本方案的用途 |
| --- | --- | --- |
| Combinatorial Bidirectional Path-Tracing for Efficient Hybrid CPU/GPU Rendering | https://doi.org/10.1111/j.1467-8659.2011.01863.x | 将渲染算法拆成适合 CPU、GPU 并行执行的阶段，并使用双缓冲和异步执行 |
| Out-of-Core Data Management for Path Tracing on Hybrid Resources | https://doi.org/10.1111/j.1467-8659.2009.01378.x | CPU 和 GPU 同时参与路径追踪时，需要独立的调度和数据管理层 |
| Hybrid CPU-GPU Scheduling and Execution of Tree Traversals | https://doi.org/10.1145/2925426.2926261 | CPU/GPU 混合遍历任务的任务划分和调度参考 |
| A Survey of CPU-GPU Heterogeneous Computing Techniques | https://doi.org/10.1145/2788396 | 异构计算中的任务划分、负载均衡和数据传输问题总览 |

论文和外部项目只支持以下原则：

- 异构资源不能简单平均分吞吐型任务。
- CPU/GPU 需要任务划分、负载均衡和数据传输管理。

论文不能证明本项目的 SPPM Gather 一定变快。性能结论只接受固定场景实测。

### 3.3 官方与公开源码

| 源码 | 可核验地址 | 对本方案的用途 |
| --- | --- | --- |
| Blender Cycles MultiDevice | https://projects.blender.org/blender/blender/src/branch/main/intern/cycles/device/multi/device.cpp | 官方多设备包装、CPU/GPU 子设备、内存搬运和多设备渲染基础；镜像：https://github.com/blender/blender/blob/main/intern/cycles/device/multi/device.cpp |
| PBRT-v4 | https://github.com/mmp/pbrt-v4 | CPU、GPU、Wavefront 后端分目录实现；用于参考后端边界，不代表其支持同时混合渲染 |
| Mitsuba 3 | https://github.com/mitsuba-renderer/mitsuba3 | 可重定向渲染器；CPU、CUDA、OptiX 等后端分离的参考 |
| phosphorus mk2 | https://github.com/jkrueger/phosphorus_mk2 | 一个小型公开 hybrid CPU/GPU 物理渲染器，用于参考工程组织，不作为性能结论来源 |

### 3.4 本项目当前源码事实

以下结论来自当前源码，不是论文推测：

| 文件 | 当前事实 |
| --- | --- |
| `intern/cycles/integrator/photon_map.cpp` | `photon_pick_trace_device()` 在多设备中优先选择 GPU 追踪光子 |
| `intern/cycles/integrator/photon_map.cpp` | 多设备时使用单张 GPU 追踪，完成后的光子图复制到全部渲染设备 |
| `intern/cycles/integrator/path_trace.cpp` | `photon_gather_after_work()` 对全部 `path_trace_works_` 并行调用 `photon_gather()` |
| `intern/cycles/integrator/path_trace_work_cpu.cpp` | CPU 已有独立的 SPPM 收集和平滑实现 |
| `intern/cycles/integrator/path_trace_work_gpu.cpp` | GPU 已有独立的 SPPM 收集和平滑实现 |
| `intern/cycles/device/multi/device.cpp` | MultiDevice 支持 CPU 与 GPU 子设备、设备间内存映射和独立渲染工作 |
| `intern/cycles/session/session.cpp` | 当前已有光子批次、额外样本、视口调度和 F12 避让逻辑 |

### 3.5 D2H 与 device-resident 的硬约束

当前源码存在以下硬约束：

```c
const bool device_resident =
    device_bin && !d.multi_device && !d.scene.volume_caustics;
```

含义：

1. 单 GPU、非体积焦散可以保持 device-resident。
2. 只要进入多设备，device-resident 失效。
3. 体积焦散即使只启用一张 GPU，也会关闭 device-resident。
4. 使用 CPU + GPU 时，渲染设备数量大于 1，必然进入多设备路径。

多设备路径会执行 `scatter_device_to_host()`，当前每代回读：

```text
photon_sorted_pos          float4
photon_sorted_beam_start   float4
photon_sorted_flux         float4
photon_sorted_beam_sigma   float4
```

每代基础 D2H 估算：

```text
每沉积字节数 = 4 * 16 = 64 字节
10,000,000 沉积 ≈ 640,000,000 字节 ≈ 610 MiB
除沉积数组外，还有 cell_start、计数器、同步和主机端 BVH 构建成本
```

主机内存保护：

```text
host_required =
    photon_grid_bytes
    + volume_beam_bvh_bytes
    + cpu_gather_staging_bytes
    + safety_headroom
```

阶段 B0 在 `host_required` 超过可用物理内存的 `25%` 时默认禁用混合模式。该阈值是工程保护线，不是硬件理论极限。

工程结论：

| 场景 | D2H 是否新增 | CPU Gather 判断 |
| --- | --- | --- |
| 单 GPU 表面焦散 | 是，新增完整多设备回读 | 默认禁用 |
| 单 GPU 体积焦散 | 否，体积路径已经回读 | 可优先验证 |
| 多 GPU 表面焦散 | 否，多设备路径已经回读 | 可在已有回读成本上评估 |
| 多 GPU 体积焦散 | 否，已经回读 | 可优先验证 |
| 视口 | 每代频率高 | 默认禁用 |
| F12 大分辨率 | 每代可摊销 | 优先验证 |

后续可以评估压缩传输、仅传必要数组、异步双缓冲和区域裁剪，但第一版方案不能假定这些优化已经存在。

---

## 4. 为什么不能直接平分光子

当前实现倾向让 GPU 追踪光子，原因是光子追踪是吞吐型任务：

- 光子之间独立。
- 每条光子路径较短。
- GPU 的并行吞吐远高于 CPU。
- CPU 与 GPU 同步一次的成本，可能高于 CPU 多追少量光子节省的时间。

如果按固定 50:50 分光子：

1. GPU 很快完成自己的一半。
2. CPU 仍在追踪剩余光子。
3. GPU 空闲等待 CPU。
4. 整批完成时间由 CPU 决定。

因此，CPU 参与光子追踪只有在满足以下条件时才值得考虑：

- CPU 光子追踪实现与 GPU 使用同一随机种子和路径规则。
- CPU 只承担按实测吞吐计算出的很小比例。
- CPU 批次与 GPU 批次可以独立合并。
- CPU 的额外同步和内存传输成本低于节省时间。

当前阶段建议完全不采用 CPU 追踪，保留 GPU 追踪作为唯一主路径。

---

## 5. 推荐架构

### 5.1 总体流程

```text
场景同步
  ↓
GPU 光子追踪
  ↓
GPU 设备端分箱 / 前缀和
  ↓
判断 PhotonGrid 是否已经需要主机侧副本
  ├─ 是：进入混合收集候选路径
  └─ 否：保持 device-resident GPU-only
  ↓
阶段 B0：固定收集分区
  ├─ GPU 收集其余区域
  └─ CPU 收集固定 5% 方块
阶段 C：动态收集调度器
  ↓
CPU 结果 H2D，统一同步
  ↓
GPU 或 CPU 单一设备执行显示平滑
  ↓
去噪 / 合成 / 输出
```

### 5.2 阶段职责

| 阶段 | 主设备 | CPU 是否参与 | 原因 |
| --- | --- | --- | --- |
| 光子追踪 | GPU | 否 | 吞吐型任务，GPU 优势最大 |
| 光子分箱 | GPU | 否 | 当前已有设备端分箱，避免回读 |
| D2H 判断 | 主机 | 否 | 决定 device-resident 是否已经失效 |
| 焦散收集 | GPU + CPU | 条件启用 | 只在主机侧副本本来就需要或净收益可验证时启用 |
| 显示平滑 | 优先 GPU | 第二阶段评估 | 需要处理跨区域边界 |
| 去噪 | 保持现状 | 不改变 | 避免引入新的同步路径 |

### 5.3 场景优先级

| 场景 | 第一阶段策略 |
| --- | --- |
| 单 GPU 表面焦散 | 保持 GPU-only，不启用 CPU Gather |
| 单 GPU 体积焦散 F12 | 优先验证 CPU Gather |
| 多 GPU 表面焦散 | 在已有 D2H 路径上验证 |
| 多 GPU 体积焦散 F12 | 优先验证 |
| 任何视口渲染 | 默认保持 GPU-only |

### 5.4 可选的两种 CPU 参与方式

#### 方案 A：CPU 只做收集

特点：

- GPU 追踪全部光子。
- CPU 和 GPU 分担 SPPM 收集和光束查询。
- 风险最低。
- 与当前代码结构最接近。

建议作为第一个验证目标，但只允许在体积焦散 F12 或已有多设备 D2H 场景启用。

#### 方案 B：CPU 做收集和部分追踪

特点：

- GPU 追踪大部分光子。
- CPU 追踪小比例光子。
- 需要动态吞吐反馈和独立批次合并。
- 同步和数据传输风险更高。

只有方案 A 实测有效、D2H 成本可接受且仍受追踪时间限制时，才考虑方案 B。

---

## 6. 混合收集设计

### 6.1 任务粒度

阶段 B0 只记录固定 tile 范围，不建立完整任务图：

```text
HybridTile:
    x, y, width, height
    generation
    consume
```

第一批粒度的建议：

- GPU：大块 tile。
- CPU：小方块 tile，初始考虑 `64x64` 或 `128x128`。
- 不使用窄横条；光子格查询对二维局部性敏感，横条会增加缓存和哈希访问成本。
- 初始 CPU 块大小应小于 GPU 块，避免 CPU 长时间占住大块。

### 6.2 收益上限

设：

```text
F = GPU-only Gather 墙钟
C = CPU-only 完成同等工作量的墙钟
k = C / F

T_hybrid(s) =
    max((1 - s) * F, s * C)
    + d2h_extra
    + h2d_extra
    + sync_extra
    + power_loss
```

不考虑额外成本时，最优份额：

```text
s_opt ≈ 1 / (1 + k)
```

理论上的 Gather 段最多缩短：

| GPU/CPU 倍速 `k` | 最优 CPU 份额 | Gather 最多缩短 |
| --- | --- | --- |
| 5x | 17% | 约 17% |
| 10x | 9% | 约 9% |
| 20x | 5% | 约 5% |

这只是 Gather 段上限。如果 Gather+Smooth 只占整帧 30%，整帧最多只能快约 3%。

阶段 A 的停止条件：

- Gather+Smooth 小于整帧时间的 `25%`。
- 实测 `k > 8`。
- 体积焦散 D2H 已经很重，而 Gather 仍不是主要瓶颈。
- 新增 D2H 或 H2D 在任何固定场景中超过预计 GPU Gather 节省。

阶段 B0 不使用吞吐 EMA。固定测试：

- CPU 份额固定为 `5%`。
- CPU 使用 `2` 个线程。
- 使用固定小方块 tile。
- 固定采样数。
- 平滑仍在 GPU。
- CPU 结果必须显式 H2D 回 GPU。

### 6.3 动态调度顺序

本节只在阶段 C 使用，阶段 B0 不实现。

1. GPU 优先拿大块。
2. CPU 只拿小方块。
3. CPU 任务进入现有 TBB task arena，避免与外层 `parallel_for` 嵌套过订阅。
4. GPU 任务通过现有 CUDA/OptiX 队列执行。
5. 第一代 CPU 份额为 `0`，没有吞吐 EMA 时不猜。
6. 每一块完成后更新吞吐 EMA。
7. CPU 块只有在预计能在 GPU 剩余时间的 `50%` 内完成时才派发。
8. CPU 超时后，未开始的块立即归还 GPU。
9. 总同步点只等待已分配任务，不为未开始的 CPU 任务等待。

### 6.4 防止 CPU 拖尾

SPPM 下一代需要等待当前代全部像素的半径和统计完成。一个慢 CPU tile 仍可能阻塞整个代际。

阶段 B0 的保护：

- 第一代不启用 CPU。
- 固定 `5%` 份额。
- 只使用小方块。
- 超时后将未开始块交回 GPU。
- 不实现动态撤销。

阶段 C 的保护：

- CPU 份额上限。
- 最小 GPU 占空比。
- CPU 预计完成时间检查。
- 未开始任务可由 GPU 接管。
- 连续两代整帧墙钟没有收益后关闭混合模式。
- 用户强制关闭开关。

### 6.5 防止功耗墙与降频

阶段 A/B0 不接入 NVML、ROCm SMI 或跨平台功耗采样。

阶段 B0 只使用两个 CPU 线程，初始线程数规则：

```text
threads = max(1, min(2, hardware_concurrency / 4))
```

阶段 C 再评估：

- 每代记录 CPU 占用、GPU SM 频率、显存频率和 GPU 利用率。
- 如果 GPU 降频损失超过 CPU 节省，立即停止 CPU 参与。
- 连续两代 `T_hybrid > T_gpu_only` 时自动关闭。

跨平台功耗守卫不是第一阶段前置条件。先用整帧墙钟回退控制风险。

---

## 7. 平滑与边界

SPPM 收集可以按像素区域独立执行，但显示平滑会读取邻近像素。

第一版建议：

1. CPU 和 GPU 只并行执行原始收集。
2. CPU 写出的每像素 tau、radius、N、M、flux 必须 H2D 回 GPU。
3. 收集完成后统一同步。
4. 显示平滑继续在单一设备执行，优先 GPU。
5. 确认收集加速有效后，再做 CPU/GPU 分区平滑。

如果后续并行平滑，需要：

- 每个区域增加 halo。
- 统计和 tau 使用统一 generation。
- 边界像素只由一个设备最终写入。
- 不能多个设备同时写同一像素。
- 验收使用逐像素 `|A-B|` 热图和 tile 边界条带检查，不能只看整图平均误差。
- 第一阶段使用固定采样；自适应采样下 tile 工作量差异大，静态份额没有参考价值。

---

## 8. 需要修改的模块

### 8.1 先区分两种 CPU 角色

#### 角色 A：CPU 已经是 Cycles 渲染设备

现状：

- 已经存在 CPU `PathTraceWork`。
- `photon_gather_after_work()` 已经对全部 work 调用 `photon_gather()`。
- 多设备路径中 CPU 当前已经在参与收集。

处理：

- 不新增 `GatherTask`。
- 先测量现有多设备路径中 CPU Gather 是净收益还是拖尾。
- 如果当前路径已拖慢，停止该方向。

#### 角色 B：GPU-only 会话外挂 CPU Gather

现状：

- CPU 不参与路径追踪。
- 只接收少量像素的 Gather 工作。
- 这是本方案真正的新功能。

阶段 B0 固定选择角色 B。

### 8.2 阶段 B0 的最小改动

不新建完整任务图，先复用现有 tile 划分：

- 为现有 `PathTraceWork` 增加指定区域 Gather 的能力。
- GPU-only 会话中的 CPU gather work 只处理固定数量的方块 tile。
- CPU work 不参与路径追踪、sample 计数和 buffer 归属。
- CPU 结果 H2D 回 GPU。
- 平滑仍在 GPU。

暂不新增 `photon_gather_scheduler.*`。只有 B0 整帧实测有效后，才评估独立调度器。

### 8.3 可能的接入点

| 文件 | 阶段 B0 内容 |
| --- | --- |
| `intern/cycles/integrator/path_trace.cpp` | 在现有 Gather 调用旁边增加固定 B0 分支 |
| `intern/cycles/integrator/path_trace_work_cpu.cpp` | 支持固定小方块收集 |
| `intern/cycles/integrator/path_trace_work_gpu.cpp` | 保持 GPU 收集路径并接收 CPU H2D 结果 |
| `intern/cycles/integrator/photon_map.cpp` | 暴露只读 PhotonGrid 和 generation |
| `intern/cycles/session/session.cpp` | 仅读取 F12 模式开关 |

不修改 `device/multi/device.cpp` 的通用多设备逻辑。

### 8.4 阶段 B0 调试开关

只增加最小开关：

```text
CYCLESPLUS_PHOTON_HYBRID_GATHER=0/1
CYCLESPLUS_PHOTON_HYBRID_TILE=64/128
CYCLESPLUS_PHOTON_HYBRID_SHARE=0.05
CYCLESPLUS_PHOTON_HYBRID_LOG=0/1
```

默认值：

- `HYBRID_GATHER=0`
- `TILE=128`
- `SHARE=0.05`
- `LOG=0`

阶段 B0 固定使用 `2` 个线程，不提供线程数环境变量。

---

## 9. 分阶段实施

### 阶段 A：基准测量

只增加统计，不启用 CPU 参与。

记录：

- 光子追踪时间。
- GPU 分箱时间。
- GPU 收集时间。
- GPU 平滑时间。
- 每代 D2H 字节数。
- `photon.scatter_and_readback` 时间。
- `photon.deposit_readback_bytes` 数值。
- H2D 预估字节数。
- 每帧总时间。

退出条件：

- GPU-only 的 3 次运行中位数稳定。
- Gather+Smooth 占整帧时间的比例已明确。
- CPU/GPU 实测倍速 `k` 已明确。
- D2H、H2D 和同步成本已明确。
- 建立平直玻璃、透镜、体积雾、相机前玻璃四组固定结果。

否决条件：

- Gather+Smooth 小于整帧时间 `25%`。
- 实测 `k > 8`。
- 单 GPU 表面焦散需要新增 D2H，且预计无法摊销。
- 体积焦散 D2H 很重，而 Gather 仍不是主要瓶颈。

### 阶段 B0：静态一刀验证

实现：

- 只对体积焦散 F12 开启。
- 视口保持关闭。
- 只验证当前 CUDA/OptiX 主后端。
- 只使用固定采样场景；自适应采样场景暂不参与静态份额验收。
- CPU 不参与路径追踪。
- 第一代 CPU 份额为 `0`。
- 固定 CPU 份额为 `5%`。
- CPU 只做 `128x128` 小方块收集，使用 `2` 个线程。
- GPU 做剩余收集。
- 平滑保持单一设备。
- CPU 结果 H2D 回 GPU。
- 不实现 EMA、动态撤销、功耗守卫和多个调度开关。

退出条件：

- 同一场景、固定采样下，CPU + GPU 整帧墙钟稳定提升 `3%` 到 `5%`。
- 没有任何一个测试场景整帧回退超过 `1%`。
- 逐像素 `|A-B|` 热图和 tile 边界条带检查可接受。
- CPU 结果 H2D 后的图像无接缝、黑块或亮度跳变。

只有 B0 通过，才允许进入阶段 C。

### 阶段 C：动态调度

实现：

- 复用现有 tile 划分，不另建平行任务图。
- 第一代 CPU 份额为 `0`。
- 吞吐 EMA。
- 动态 CPU 份额。
- CPU 块预计只能使用 GPU 剩余时间的 `50%`。
- 超时未开始的块归还 GPU。
- 自动关闭。
- 可选功耗和频率守卫。

退出条件：

- CPU 慢时不增加总帧时间。
- GPU 快时 CPU 能自动收回份额。
- 连续运行无死锁、无显存错误、无渲染停顿。

### 阶段 D：体积焦散重点优化

重点检查：

- 有限光束 BVH。
- 体积查询。
- 非均匀介质步进。
- 透明玻璃进入体积的相机路径。

此阶段才评估 CPU 并行平滑和更细粒度光束查询。

### 阶段 E：视口验证

只有阶段 C 和阶段 D 均通过后才进入。

进入条件：

- 小批次 D2H 已被异步重叠或区域裁剪消除。
- CPU 线程唤醒延迟低于视口单帧预算。
- GPU 频率未因 CPU 负载明显下降。

否则视口永久保持 GPU-only。

---

## 10. 验收指标

### 10.1 必须满足

1. GPU-only 和 CPU + GPU 使用相同场景、相同采样、相同光子数。
2. CPU + GPU 输出与 GPU-only 无结构性差异。
3. CPU 参与后不出现固定接缝、黑块或亮度跳变。
4. CPU 落后时总帧时间不增加。
5. 设备丢失、CUDA 错误、OptiX 错误、显存不足时必须自动回退 GPU-only。
6. 单 GPU 表面焦散的 device-resident 路径默认保持不变。
7. 视口默认不使用 CPU Gather。
8. 连续两代 `T_hybrid > T_gpu_only` 时自动关闭混合模式。

### 10.2 建议的自动判定阈值

- 单 GPU 表面焦散：默认不启用；只有显式实验开关可以覆盖。
- 阶段 B0：固定采样、同一体积焦散 F12 场景，整帧墙钟稳定提升 `3%` 到 `5%`。
- 阶段 B0：其他测试场景不得回退超过 `1%`。
- 阶段 C：只有 B0 通过后才启用动态份额和 EMA。
- 新增 D2H、H2D、同步和功耗成本全部计入 `T_hybrid`。
- 使用墙钟公式比较，不使用 CPU 工作时长作为“节省值”。
- 如果收集和平滑合计占帧时间很低，默认关闭混合模式。

这些阈值是本项目的工程验收线，不是论文给出的性能保证。

---

## 11. 风险与规避

| 风险 | 原因 | 规避 |
| --- | --- | --- |
| CPU 拖慢 | CPU 单像素处理速度低 | CPU 小份额、预计完成时间检查、自动收回 |
| 同步开销吃掉收益 | 每次收集都要等待 CPU 和 GPU | 使用粗粒度任务、异步队列、避免逐像素同步 |
| 新增 D2H 吃掉收益 | CPU 参与让单 GPU 表面焦散失去 device-resident | 单 GPU 表面焦散默认禁用；先测 D2H |
| PCIe/内存带宽争用 | 光子图、缓冲区和纹理同时搬运 | 只复制必要数组、固定 generation、减少回读 |
| 视口延迟增加 | 小批次、频繁 generation、线程唤醒 | 视口默认禁用 CPU Gather |
| 功耗墙和 GPU 降频 | CPU 满载抢功耗和散热预算 | 限制到 2 到 4 线程；记录频率；自动关闭 |
| 平滑边界错误 | 邻域读取跨设备区域 | 第一阶段只做单设备平滑 |
| 结果不一致 | CPU/GPU 浮点差异 | 统一公式和容差；固定场景对照 |
| 驱动问题 | OptiX/CUDA 队列与 CPU 内存竞争 | 保持现有队列边界；失败自动回退 GPU-only |
| 维护复杂 | 调度逻辑侵入官方文件 | B0 只做固定分支；只有 B0 通过才新增独立调度文件 |

---

## 12. 不采用或暂不采用的方案

### 12.1 不采用 CPU + GPU 平均追光子

原因：

- CPU 会形成长尾。
- 光子追踪是 GPU 强项。
- 同步成本不稳定。

### 12.2 第一阶段不引入 StarPU

原因：

- Cycles 已有 TBB 和 CUDA/OptiX 队列。
- 引入新运行时会影响 Blender 构建、打包和许可证。
- StarPU 适合作为调度模型参考，不是当前最小改动。

### 12.3 不用简单加权平均合并独立图像

原因：

- 不同设备的噪声相关性不同。
- 简单平均会改变能量和收敛速度。
- 混合调度应分配互斥像素，不能重复计算同一像素。

---

## 13. 最终建议

推荐按以下顺序执行：

1. 先做阶段 A 测量，不做功能改动。
2. 如果 Gather+Smooth 小于整帧 `25%` 或 `k > 8`，停止。
3. 只对体积焦散 F12 做阶段 B0：固定 `5%`、固定方块、`2` 线程、无调度器。
4. B0 只有整帧稳定提升 `3%` 到 `5%`，才允许进入阶段 C。
5. 单 GPU 表面焦散、视口和平均追光子仍保持关闭。
6. 任何阶段都保留 GPU-only 回退。

论文只提供异构调度原则，不为本项目的 SPPM Gather 性能背书。是否保留这一功能，只依据固定场景的整帧墙钟、H2D/D2H、逐像素差异和 tile 边界检查。
