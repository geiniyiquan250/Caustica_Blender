# 体积焦散丁达尔效应（Volumetric Caustics / Tyndall Effect）实施方案

> 编写基线：Blender 5.2.0 LTS，CyclesPlus 焦散源码快照，2026-09-02
>
> 文档类型：研究与工程落地方案
>
> 目标：让玻璃、水面或透明介质形成的焦散，在参与介质中产生连续的空间光束、光柱和丁达尔亮带。

## 1. 结论先行

当前焦散模块已经有光子追踪、渐进式 SPPM、CPU/CUDA/OptiX 路径和体积吸收衰减。光子在体积内部还没有独立的散射事件和体积光子记录，因此只能得到表面焦散，无法稳定形成雾中的空间光束。

推荐采用“两张光子图”方案：

1. 保留现有表面光子图，继续负责地面、水底、墙面的焦散。
2. 新增体积光子图，记录光子在雾中的散射位置、入射方向、能量和介质信息。
3. 相机路径在体积散射采样点查询体积光子图，将结果写入体积直接光、体积间接光和独立焦散通道。

第一阶段只做均匀体积、单次散射、Henyey-Greenstein 相位函数和 CPU/CUDA/OptiX 结果一致。异质体积、多次散射和 Vulkan 后端放到后续阶段。

## 2. 当前基线与缺口

### 2.1 已有能力

- `intern/cycles/integrator/photon_map.cpp` 已负责场景提取、材质分类、光源预算、光子批次和渐进发布。
- `intern/cycles/kernel/integrator/photon_trace.h` 已支持光子在真实 BVH 中行走，包含反射、折射、粗糙反射、透明穿透、清漆和体积吸收衰减。
- `intern/cycles/kernel/integrator/photon_lookup.h` 已提供空间哈希网格和表面光子收集。
- `intern/cycles/kernel/film/photon_passes.h` 已提供每像素 SPPM 状态、半径收缩、平滑、灯光组分离和焦散 pass。
- `shade_volume.h` 已具备体积距离采样、直接散射、间接散射、相位函数和体积透射流程。

### 2.2 关键缺口

- 光子追踪遇到一段体积时只更新吸收后的功率，没有按自由程采样体积散射点。
- 现有 photon deposit 只有位置、法线和灯光组，没有体积散射方向、介质 ID 和相位函数信息。
- 现有 gather 以表面位置和法线为中心，无法在体积采样点工作。
- 现有 `PASS_CAUSTICS` 是表面焦散语义，体积焦散需要独立的体积 pass 和重复计数隔离。
- 方案文档中的 Vulkan 路线目前没有对应的 Cycles Vulkan 光子后端，不能作为第一阶段实现依赖。

## 3. 物理目标与边界

### 3.1 目标现象

典型场景是太阳或聚光灯穿过水面、玻璃或透明物体进入雾体。雾中的散射点沿着折射后的光路形成连续亮带，地面仍然保留表面焦散。相机移动时，亮带的位置跟随光路和体积密度变化。

### 3.2 第一阶段支持范围

- 均匀或低频近似均匀的参与介质。
- 单次体积散射。
- 各向同性和 Henyey-Greenstein 相位函数。
- 表面焦散链之后进入体积的光子。
- 点光源、聚光灯、面光源、太阳和世界光源。
- 独立体积焦散 pass，以及与 Combined 的一致合成。

### 3.3 暂缓范围

- 多次体积散射的完整光子链。
- 高密度烟雾的细粒度异质采样。
- 体积自发光和复杂体积节点混合的精确光子材质评估。
- 体积光子图与表面光子图的统一可变半径。
- Vulkan RT/Compute 后端。

## 4. 核心估计器

### 4.1 光子在体积段中的散射采样

设光子从 `x0` 行进到下一个表面或离开体积的位置 `x1`，段长为 `L`。均匀介质中：

```text
sigma_t = sigma_a + sigma_s
u       = uniform(0, 1)
t       = -log(max(1-u, epsilon)) / sigma_t
```

当 `t < L` 时，在 `x = x0 + t * direction` 产生体积散射事件；否则光子继续按当前表面逻辑行走。光子功率按 `exp(-sigma_t * t)` 衰减，散射事件使用 `sigma_s` 和相位函数进行权重校正。

均匀介质使用指数自由程采样时，采样 PDF 与消光系数可在事件权重中抵消，减少第一版的方差。介质密度为零时直接跳过体积采样，保持当前表面结果。

### 4.2 体积光子记录

建议新增结构：

```cpp
struct VolumePhotonDeposit {
  float3 position;       // 体积散射位置
  float3 incoming_dir;   // 光子到达散射点的方向
  float3 flux;           // 已包含表面链路和体积透射的能量
  int volume_shader;     // 介质材质槽
  int lightgroup;        // 灯光组，-1 表示未分组
  float sigma_s;         // 散射系数或归一化散射权重
  float anisotropy;      // Henyey-Greenstein g
};
```

实际布局应按 GPU 对齐规则压缩为 16 字节组，避免为每条光子路径增加过大的设备内存。第一版可以将 `volume_shader`、`lightgroup` 和相位参数放入额外数组，确认内存占用后再合并。

### 4.3 相机路径上的体积收集

相机路径已经由 `shade_volume.h` 得到体积散射点 `P`、相机出射方向 `wo` 和介质相位函数。新增 `volume_photon_grid_gather()`，在 `P` 周围查询体积光子：

```text
L_caustic(P, wo) =
    sum_i [flux_i * phase(volume_i, incoming_i, wo) * K(P - position_i)]
    / (volume_photon_count * kernel_normalization)
```

其中：

- `K` 使用三维 Epanechnikov 或紧支撑核。
- 光子入射方向与相机出射方向通过同一个介质相位函数计算。
- 体积散射点到相机的透射率由现有体积积分器继续负责，避免重复计算。
- 体积光子和表面光子使用独立半径、独立统计量和独立成熟度。

第一版建议沿用现有空间哈希代码，新增 `volume_photon_grid.h` 和 `volume_photon_lookup.h`，不要把表面光子和体积光子混进同一条查询路径。两类记录的距离尺度、法线条件和相位函数条件不同，强行共用会增加偏差排查难度。

## 5. 光子追踪流程改造

### 5.1 光子发射与表面链

保持当前发射、灯光预算、caster target、Fresnel、反射和折射逻辑。光子只有在已经经过至少一个镜面事件后，才允许写入“体积焦散”记录，这样普通直接照亮雾体的能量仍由 Cycles 原有体积直接光负责。

### 5.2 体积段插入点

在 `photon_trace_single()` 的每次 BVH 相交前，维护当前介质状态：

```text
segment_start = current_origin
segment_end   = next_surface_or_miss
medium        = current_volume
```

当 `medium.sigma_t > 0` 且光子处于镜面焦散链时，先在该段采样一次体积散射：

1. 计算自由程 `t`。
2. 更新到达散射点的 Beer-Lambert 透射率。
3. 按 `sigma_s / sigma_t` 计算散射反照率权重。
4. 写入体积光子记录。
5. 第一阶段结束该光子的焦散链，避免单次散射光子继续产生未经验证的多次散射偏差。

当采样未命中散射点时，按现有逻辑处理表面相交、体积进入/离开和表面 deposit。体积吸收状态继续由当前 `vol_sigma` 维护。

### 5.3 介质取得方式

第一阶段在主机提取阶段为每个体积 shader 建立紧凑的 `PhotonVolumeMaterial`：

```cpp
struct PhotonVolumeMaterial {
  float3 sigma_a;
  float3 sigma_s;
  float anisotropy;
  int phase_type;
  int accurate;
};
```

均匀的 `Volume Scatter`、`Volume Absorption`、`Principled Volume` 常量输入可以进入 fast path。贴图驱动密度、体积混合、体积噪波和复杂节点先进入 accurate/unsupported 报告，避免静默生成错误丁达尔效果。

## 6. 与现有路径追踪的分工

### 6.1 普通体积光

普通直接光和普通间接光继续走 Cycles 原有体积积分器。体积光子只补充“经过玻璃、水面、镜面链路后到达雾中的焦散光”。

### 6.2 重复计数控制

新增体积专用路径标记，例如：

```text
PATH_RAY_PHOTON_VOLUME_CAUSTIC_CHAIN
```

光子图已经覆盖的镜面链体积贡献，在路径追踪对应的直接光或发光连接处跳过一次；普通穿过雾体的路径、体积本身的环境光和多次散射保持原有行为。控制范围必须限定在贡献写入位置，不能关闭整个体积 closure。

### 6.3 Pass 设计

建议新增：

- `PASS_VOLUME_CAUSTICS`：合并后的体积焦散。
- `PASS_VOLUME_PHOTON_STATE`：内部统计，可选调试使用。
- `PASS_VOLUME_CAUSTICS_<LightGroup>`：按灯光组拆分的体积焦散。

表面 `PASS_CAUSTICS` 保持兼容。用户打开独立焦散 pass 时，同时显示表面和体积两个结果，合成器可以分别调节地面亮斑和雾中光柱。

## 7. 数据结构与调度改动

### 7.1 PhotonMapData

在 `PhotonMapData` 中增加两套双缓冲存储：

```text
volume_pos
volume_dir
volume_flux
volume_cell_start
volume_grid_radius
volume_photon_num
```

GPU 模式沿用当前 device-resident deposit、计数、prefix cursor、scatter 流程。体积网格和表面网格分开上传，避免一张哈希表同时承受两种记录的容量峰值。

### 7.2 SPPM 统计

表面和体积分别维护 `tau`、`N`、`r2`、`gather_weight`。体积半径初值建议按相机像素尺寸和雾体深度估算，并设置独立下限。体积光束比表面焦散更容易形成细长结构，不能直接复用表面半径。

### 7.3 视口和最终渲染

沿用现有 `restart()`、`advance()`、`photon_gather_after_work()` 调度。每次新体积光子代发布后：

1. 绑定表面和体积两张光子图。
2. 执行相机路径。
3. 执行表面焦散 gather。
4. 执行体积焦散 gather。
5. 执行各自的显示平滑和自适应采样更新。

视口与最终渲染使用同一套物理参数、批次权重和半径规则。视口可以缩小首批光子数量，但不能替换成单独的体积后处理假象。

## 8. 用户参数设计

建议把参数放在现有 Photon Caustics 区域下：

| 参数 | 建议默认值 | 作用与联动 |
| --- | ---: | --- |
| Volume Caustics | 关闭 | 需要先打开 Photon Caustics；开启后才创建体积光子图。 |
| Volume Photon Share | 25% | 从总光子预算中分给体积光子。场景没有有效体积时自动回收给表面光子。 |
| Volume Detail | 4 | 控制体积查询半径。提高后光束边缘更清楚，稀疏体积需要同步提高光子数。 |
| Max Volume Scatter Events | 1 | 第一阶段固定为 1；后续允许 2 或 3，但每增加一次都会明显增加方差和时间。 |
| Volume Phase | HG | 支持 Isotropic 和 Henyey-Greenstein；`g > 0` 让前向光束更明显，`g < 0` 增强反向散射。 |
| Volume Caustics Intensity | 1.0 | 只缩放体积焦散显示结果，不改变光子传播和统计。 |

参数组合规则：

- `Photon Caustics` 关闭时，所有体积焦散参数不生效。
- `Volume Caustics` 关闭时，体积仍可参与普通 Cycles 体积渲染。
- `Max Volume Scatter Events = 1` 时，结果只表达单次体积散射，适合清晰丁达尔光束。
- 当体积没有散射系数，只有吸收系数时，不生成体积焦散；表面焦散和体积透射仍按当前逻辑工作。

## 9. 分阶段实施

### P0：离线 CPU 验证

目标是先证明估计器和能量归一化正确：

1. 新增均匀体积材质表和 CPU 体积光子记录。
2. 在单个玻璃平面加雾盒场景中生成体积光子图。
3. 用固定种子比较：无体积、普通体积光、体积焦散三组结果。
4. 验证光子数翻倍后噪声下降，平均能量不会随半径和 Detail 无故改变。

验收条件：光束位置随玻璃折射方向移动，光束强度随 `sigma_s` 增加而增加，随 `sigma_a` 增加而衰减；关闭体积焦散后回到普通体积结果。

### P1：接入现有 Film Gather

1. 新增体积网格查询。
2. 在 `shade_volume.h` 的散射点确定后执行体积 gather。
3. 新增体积焦散 pass 和灯光组 pass。
4. 为体积统计加入独立 SPPM 半径和成熟度。
5. 先关闭显示平滑，只验证原始估计器。

验收条件：体积焦散能量能在 Combined 和独立 pass 之间一致分解，灯光组相加等于总体积焦散。

### P2：CUDA/OptiX 与视口渐进

1. 增加 GPU 体积 deposit、计数、分桶和 scatter kernel。
2. 保持 CPU/GPU 相同的随机种子派生和光子布局。
3. 接入现有视口异步批次与后台渲染保护。
4. 对 GPU 设备增加容量溢出重试和取消安全检查。

验收条件：CPU、CUDA、OptiX 的低光子数量结果统计一致；长时间视口渲染不持续增长显存；场景更新时不会使用旧体积网格。

### P3：异质体积与质量扩展

1. 对低频密度场采用 delta tracking 或 ratio tracking。
2. 将体积属性采样接入 `ATTR_STD_VOLUME_DENSITY` 和颜色属性。
3. 支持有限次数多次散射，并单独评估路径权重和 Russian roulette。
4. 研究体积光子差分和各向异性椭圆核，解决长距离光束被圆形核抹平的问题。

这一阶段完成前，不建议默认开启复杂烟雾场景的体积焦散。

## 10. 测试场景与指标

### 10.1 必备场景

- 单个玻璃板 + 均匀雾盒 + 太阳光。
- 水面 + 水下雾 + 地面接收面。
- 聚光灯 + 玻璃棱镜 + 窄体积通道。
- 彩色玻璃 + 有吸收的雾体。
- 多灯光组 + 体积焦散 pass。
- 摄像机移动和材质热更新场景。

### 10.2 必测指标

- 体积光子命中率和每批有效散射数。
- 光子平均功率、散射点空间分布和目标体积覆盖率。
- 体积焦散的均值、方差、收敛速度和半径变化。
- `Volume Caustics` 与普通体积路径的重复计数比例。
- CPU/GPU 图像差异、显存峰值、批次发布延迟和取消耗时。

### 10.3 物理回归

- `sigma_s = 0` 时体积焦散严格为零。
- `sigma_a = 0` 时只剩散射造成的角度变化，不出现额外颜色吸收。
- 相位函数积分归一化后，改变 `g` 只改变角度分布，不改变总能量。
- 光源强度翻倍时体积焦散平均能量翻倍。
- 光子数量翻倍时平均结果稳定，噪声下降。
- 关闭 Photon Caustics 时现有表面和普通体积结果保持不变。

## 11. 主要风险与处理顺序

1. **重复计数**：先实现独立 pass，再接入 Combined；确认能量分解后再启用路径分区。
2. **体积半径过大导致光束变粗**：体积使用独立半径，后续再引入方向差分核。
3. **稀疏体积白点**：保留现有批次权重、稀疏 gather 保护和成熟度门控，避免直接套用表面 clamp。
4. **复杂节点材质误判**：fast path 只接收已确认的常量体积；其余材料显示诊断报告并回退普通体积路径。
5. **GPU 内存峰值**：体积光子图使用单独容量上限、分批上传和溢出重试，先保证取消和场景更新安全。
6. **多次散射方差过高**：第一版单次散射完成后停止光子焦散链，用普通路径追踪承担剩余多次散射。
7. **Vulkan 范围膨胀**：Vulkan 作为独立后端项目，等 CPU/CUDA/OptiX 估计器稳定后再设计跨后端接口。

## 12. 推荐的最小落地提交

第一份工程提交只包含以下内容：

- `PhotonVolumeMaterial` 和均匀体积材质提取。
- `VolumePhotonDeposit`、体积光子双缓冲和三维哈希表。
- CPU `photon_trace_single()` 的单次自由程采样。
- CPU 体积 gather 和 `PASS_VOLUME_CAUSTICS`。
- 一个独立回归场景和能量/收敛日志。

先完成这条最小链路，再移植 GPU。只要 P0/P1 的能量和收敛检查通过，后续 CUDA/OptiX 与异质体积才有可比较的基准。

## 13. 最终判断

这条路线可以复用当前 CyclesPlus 的光子调度、SPPM 统计、设备分桶和灯光组机制，新增部分集中在体积段采样、体积光子记录、体积 gather 和 pass 分工。第一阶段的关键验收点是“雾中亮带由真实散射点产生，关闭后普通体积结果不变，能量不随 Detail 无故漂移”。

