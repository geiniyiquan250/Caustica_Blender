# Caustica Guided Adaptive SPPM + Vulkan 完整优化方案（WYSIWYG 单一光子配置版）

> 目标版本：Caustica 1.2 → 2.0
>
> 基线：Caustica 1.1（Blender / Cycles 源码分支中的 GPU SPPM 焦散求解器）
>
> 文档日期：2026-09-01
>
> 文档用途：技术评审、开发拆解、性能验收、术语统一

---

## 0. 文档结论

### 0.1 最终建议

Caustica 1.1 **不应该推翻现有 SPPM，也不应该马上切换到完整 BDPT / VCM**。

当前最优升级路线应当是：

**Guided + Adaptive + Anisotropic SPPM**

即：

1. **GPU Photon Guiding**：让更多光子走“真正有用”的方向，而不是只提高总光子数；
2. **Camera-aware Contribution Feedback**：只有真正进入当前相机图像并降低误差的光子，才反向训练下一代发射分布；
3. **Defensive Sampling + Exact PDF**：避免 Guiding 因极低概率方向产生超高权重白点，同时保持能量正确；
4. **Adaptive Radius + ESS**：局部光子不足时不要机械缩小 SPPM 半径；
5. **Photon Differential / Anisotropic Kernel**：让每个光子的密度核顺着焦散结构拉伸，减少模糊、漏光和细线焦散的白点；
6. **True GGX Transmission + Medium Stack**：排除“看起来像 SPPM 白点、实际来自材质/介质近似”的问题；
7. **Auto Photon Culling**：只在大场景确实有收益时启用；
8. **Viewport / Final 强制 WYSIWYG**：预览与最终渲染必须使用同一个 Guided Adaptive SPPM 实例语义、同一套 Photon Count / Batch / 材质 / 光源 / PDF / Guiding / Radius / Kernel 规则；Caustica 内部不再划分“预览光子预算”和“最终光子预算”，总运行次数只由 Blender 已有的 Viewport Samples 与 Render Samples 分别控制；
9. **ReSTIR FG+ 降为实验性加速模块**：只有在能够作为“同一目标估计器”的候选生成/重采样模块，并通过同预算一致性与能量回归测试后，才允许进入默认 Viewport；否则只保留 Experimental，不作为所见即所得链路；
10. **BDPT / VCM 放到 2.0**：作为 SPPM 无法高效解决的困难路径补充，而不是现在替换现有系统。

核心思想从：

> “每秒追更多 photons”

升级为：

> **“每秒产生更多 Useful Photons，并让每个 Useful Photon 的统计权重更稳定。”**

### 0.2 WYSIWYG 是硬性架构约束

Caustica 后续所有优化必须满足一个核心产品原则：

> **Viewport 与 Final Render 调用同一套 Photon Solver 配置。Caustica 不再额外区分预览预算和渲染预算；二者运行多少次，只由 Blender 已有的 Viewport Samples 与 Render Samples 控制。**

因此默认生产链路必须保证：

```text
Viewport Samples = B
≈
Render Samples = B
```

这里的“≈”指在相同场景、相同分辨率、相同 Caustica 参数和相同样本数下的统计近似：噪声分布和随机序列可以不同，但**期望值、焦散位置、焦散形状、能量、材质响应、光源贡献和收敛方向必须一致**。

必须相同的内容：

- Integrator / Path Classification；
- Photon emission 与完整 PDF；
- Light / Caster / Directional Guiding；
- Defensive Mixture；
- BSDF / GGX Transmission / Medium Stack；
- Measurement Point 定义；
- SPPM `N / τ / R` 更新公式；
- Adaptive Radius / ESS 判定；
- Photon Differential / Anisotropic Kernel；
- Kernel normalization；
- Light Group / AOV 能量归属；
- Camera-aware contribution credit；
- 所有会进入物理估计器的参数。

只允许由 Blender 宿主天然不同的内容：

- Viewport Samples 与 Render Samples 的用户设定值；
- 刷新频率；
- 结果输出方式；
- 仅显示层的 denoise / smoothing；
- 取消、后台渲染与 GPU stream 生命周期等宿主调度细节。

以下内容即使在宿主调度层也不得分叉：

- Photon Count 的含义与数值；
- 每个 sample/work 对应的 photon batch 规则；
- batch index 到随机种子、发射数和 Guide 更新的映射；
- 任何进入物理估计器的停止、成熟度或自适应判定。

显示层结果**不得反向训练 Guiding**。最终渲染也不得在开始时切换到另一套 radius policy、另一套 photon estimator 或另一套材质近似。

理想行为应类似 Cycles：

```text
Viewport Samples 较低
= 同一真实结果 + 更多噪声 / 更粗糙的统计估计

Render Samples 较高
= 同一真实结果 + 更低噪声 / 更高精度
```

而不能是：

```text
Preview A 算法
↓ F12
Final B 算法
```

这条约束优先级高于 ReSTIR、Viewport 专用 estimator 或任何“看起来更快”的近似方案。

### 0.3 现有参数冻结是硬性产品约束

Guided / Adaptive / Anisotropic SPPM 必须在现有 Caustica 参数背后自动工作，不得修改或替换现有参数的名称、位置、默认值、范围和用户语义。

当前冻结参数为：

| 参数 | 范围/枚举 | 默认值 | 约束 |
|---|---|---|---|
| Photon Caustics Solver | Off / On | Off | 保持现有总开关 |
| Photon Count | 1–1000，单位仍为每渐进代的百万光子 | 100 | Viewport/Final 共用同一个值 |
| Detail | 0.1–20.0 | 10.0 | 保持现有锐度/半径控制语义 |
| Intensity | 0–100，soft max 5 | 1.0 | 1.0 继续表示物理强度 |
| Casters | All Materials / Selected Materials Only | All Materials | 保持现有材质选择行为 |

禁止新增会与现有参数重叠或替代它们的生产级设置：

- Mode / Fast / Balanced / High Quality；
- Quality 总滑块；
- Preview/Final Photon Count；
- Preview/Final Photon Budget；
- Preview/Final Radius Policy；
- 用户可调的 Guide Strength、Guide Resolution、ESS threshold、EMA alpha、Beta schedule。

新算法参数必须使用经过回归验证的内部 Auto 策略；诊断与消融优先通过 Debug AOV、日志、环境变量或开发者构建提供，不改变普通用户现有参数面板。

---

## 1. 依据与边界

本文把信息分成三类：

- **[现状]**：来自现有 Caustica 1.1 源码分析报告；
- **[研究依据]**：来自公开论文、项目页或公开实现；
- **[工程建议]**：针对 Caustica 当前架构做出的设计方案，不代表论文已经直接验证这些具体参数。

### 1.1 主要研究依据

1. Hachisuka & Jensen, **Stochastic Progressive Photon Mapping**, SIGGRAPH Asia 2009。
2. Grittmann et al., **Efficient Caustic Rendering with Lightweight Photon Mapping**, EGSR 2018。
3. Kern, Brüll, Grosch, **Accelerating Photon Mapping for Hardware-Based Ray Tracing**, JCGT 2023。
4. Kern, Brüll, Grosch, **ReSTIR FG: Real-Time Reservoir Resampled Photon Final Gathering**, EGSR 2024；公开实现：`TU-Clausthal-Rendering/ReSTIR-FG`。
5. Lin et al., **Hypothesis Testing for Progressive Kernel Estimation and VCM Framework**, 2025。
6. De Barro, Bugeja, Spina, **ARPΔ: Accelerated Ray-Tracing Photon Differentials for Real-Time Global Illumination**, 2025。
7. Kern, Brüll, Kastning, Grosch, **Guided ReSTIR FG+ Photon Resampling for Large Scenes and Many Lights**, EGSR 2026；公开实现：`TU-Clausthal-Rendering/Guided-ReSTIR-FG-Plus`。
8. Benoist, Litalien, Gruson, **Neural Progressive Photon Mapping**, Eurographics 2026；公开实现：NPPM。

---

# 第一部分：Caustica 1.1 现在到底是什么

## 2. 当前系统基线

### 2.1 当前架构

[现状]

Caustica 不是 Blender Python 插件，而是直接修改 Cycles 内核的 Blender 源码构建。

当前有三层：

```text
Blender / Cycles Host
        │
        ├── Scene Extract
        ├── Material Classification
        ├── Light Evaluation
        ├── Pilot Photon Budget
        └── Caster Target Guiding
                 │
                 ↓
        GPU Photon Trace
        OptiX / CUDA
                 │
                 ├── scene_intersect
                 ├── Fresnel
                 ├── GGX reflection
                 ├── refraction
                 ├── absorption
                 └── photon deposit
                 │
                 ↓
        GPU Photon Grid / Binning
                 │
                 ↓
        SPPM Film Gather
                 │
                 ├── N
                 ├── radius²
                 ├── tau
                 └── preview smoothing
```

### 2.2 当前已有的重要优势

[现状]

Caustica 1.1 已经有：

- OptiX 独立 `PIP_PHOTON` pipeline；
- 独立 CUDA stream，与正常 Cycles 渲染并发；
- 一个线程对应一个 photon；
- photon 直接调用 Cycles `scene_intersect`；
- GPU device-resident photon binning；
- Pilot 按灯分配 photon budget；
- Caster 包围球 Target Guiding；
- EMA 目标权重；
- exploration 下限；
- PCG32 确定性随机数；
- 每像素 SPPM `N / radius² / tau`；
- Photon Caustics AOV；
- Light Group Caustics；
- 与普通 Path Tracing 相加共存；
- Accurate Shader fallback。

因此，**Caustica 当前真正缺少的不是 GPU Photon Tracer，而是更聪明的 Sampling、Density Estimation 和反馈体系。**

---

## 3. 当前 SPPM 的主要问题

### 3.1 问题 A：光子“发对物体”但没有“发对方向”

当前 Target Guiding 已经知道：

```text
哪盏灯 → 哪个 caster
```

例如：

```text
Sun
 ├── Glass A  20%
 └── Glass B  80%
```

但进入 Glass B 以后，目前并没有充分学习：

```text
Glass B 的哪个入射区域 / 哪个方向
最容易最终形成相机看到的焦散？
```

因此仍然会有大量 photon：

```text
Light
  ├── useful
  ├── useless
  ├── useless
  ├── useless
  └── useful
```

### 解决方向

**GPU Directional Photon Guiding**。

---

### 3.2 问题 B：Target Yield 不等于 Camera Utility

当前一个 caster 产生很多 deposit，不代表这些 deposit 对当前相机有价值。

例如：

```text
Caster A
→ 产生 10000 photons
→ 9000 落在摄像机完全看不到的位置

Caster B
→ 产生 3000 photons
→ 2500 真正进入当前画面焦散
```

只看 photon yield：

```text
A > B
```

真正图像贡献：

```text
B >> A
```

### 解决方向

**Camera-aware Contribution Feedback**。

---

### 3.3 问题 C：光子稀疏导致局部高权重白点

SPPM 局部 gather 半径中如果只有极少光子：

```text
radius
  │
  ├── photon
  └── photon
```

一个 photon 的能量对最终估计影响就很大。

典型结果：

- 白点；
- 彩色斑点；
- 某一代突然亮；
- 高 Detail 初始阶段颗粒很重。

### 解决方向

- Directional Guiding；
- Adaptive Radius；
- ESS；
- 防止低 PDF 高权重；
- Anisotropic Kernel。

---

### 3.4 问题 D：固定/机械半径策略存在 Bias-Variance 冲突

当前 SPPM 使用渐进缩半径，同时设置：

```text
r >= r0 / 4
```

优点：

- Viewport 更稳定；
- 防止 GPU photons 太多时半径快速缩成亚像素；
- 避免焦散越渲染越淡。

缺点：

- Final Render 永久保留一个最小 density kernel；
- 细焦散可能被固定半径模糊；
- 不同区域 photon density 差异巨大，但 radius policy 相似；
- 稀疏区域半径太小会产生白点；
- 密集区域半径太大又会糊。

### 解决方向

**Viewport 与 Final 共用同一 Adaptive Radius / Confidence-based Shrink；Viewport 只是更早停止。**

---

### 3.5 问题 E：圆形/各向同性 Kernel 不适合真实焦散

真实焦散经常呈：

```text
────────────
```

或者：

```text
///////
```

而不是：

```text
○
```

当前近似圆形的 Epanechnikov gather 会导致：

- 细线变粗；
- 高曲率焦散被糊；
- 阴影/焦散边界泄漏；
- 为了保持锐利只能继续缩半径；
- 半径缩小后又变稀疏并产生白点。

### 解决方向

**Photon Differential + Anisotropic / Elliptical Kernel**。

---

### 3.6 问题 F：部分“白点”根本不是 SPPM Density 问题

当前还有物理近似：

- Rough Refraction 使用近似方向抖动，而不是完整 GGX Transmission；
- Medium 是单槽状态，不是真正嵌套介质栈；
- 复杂节点可能进入 fast classification 或 accurate fallback；
- 部分纹理材质存在分类边界。

因此以下场景的异常亮点：

```text
rough glass
liquid inside glass
nested dielectric
high IOR
normal map glass
```

可能来自错误的路径权重/折射分布，而不是 photon gather。

### 解决方向

- True GGX Transmission；
- Medium Stack；
- 更强 shader validation；
- Photon Debug AOV。

---

# 第二部分：目标系统

## 4. Caustica 下一代完整架构

目标名称建议：

# Guided Adaptive SPPM

完整数据流：

```text
                           CAMERA SIDE
                               │
                        Camera Path Trace
                               │
                     SPPM Measurement Point
                               │
              ┌────────────────┴────────────────┐
              │                                 │
      Receiver Visibility               PT Variance / Need
              │                                 │
              └────────────────┬────────────────┘
                               ↓
                      Camera Utility Map
                               │
                               ↓
                        Feedback Statistics
                               │
                               └─────────────────────┐
                                                     │
                                                     ↓
LIGHT SIDE                                   GPU PHOTON GUIDE
   │                                                 │
Pilot photons                                  Light Guide
   │                                                 ↓
Light Budget                                   Caster Guide
   │                                                 ↓
Caster Selection                             Direction Guide
   │                                                 ↓
   └────────────────────────────────────→ Defensive Mixture
                                                     │
                                              Exact PDF
                                                     │
                                                     ↓
                                               Photon Emit
                                                     │
                                               OptiX Trace
                                                     │
                                       Physical Specular Chain
                                                     │
                                              Photon Deposit
                                                     │
                                      Optional Receiver Culling
                                                     │
                                               GPU Binning
                                                     │
                 ┌───────────────────────────────────┴──────────────┐
                 │                                                  │
         Adaptive Radius                               Photon Differential
                 │                                                  │
                 │                                      Anisotropic Kernel
                 │                                                  │
                 └──────────────────────┬───────────────────────────┘
                                        ↓
                                   SPPM Gather
                                        │
                                N / R² / Tau Update
                                        │
                     contribution / variance / ESS
                                        │
                                        └────────────→ Next Photon Guide
```

---

# 第三部分：P0 —— Guided SPPM

## 5. P0-1：Photon Statistics / 诊断层

### 5.1 为什么必须先做

当前只有“看结果好不好”，缺少对 photon 效率的完整定量诊断。

如果没有统计层，很容易把：

- 发射问题；
- 材质问题；
- PDF 问题；
- radius 问题；
- kernel 问题；

全部误认为“SPPM 白点”。

### 5.2 新增统计

建议每批记录：

```cpp
struct PhotonStatistics {
    uint64 emitted;
    uint64 hit_caster;
    uint64 completed_caustic_chain;
    uint64 deposited;
    uint64 gathered;

    float mean_weight;
    float max_weight;
    float weight_variance;

    float mean_photons_per_hitpoint;
    float empty_gather_ratio;
    float useful_photon_ratio;

    float max_over_median_weight;
    float effective_sample_size;
};
```

### 5.3 新指标：Useful Photon Ratio

定义：

```text
Useful Photon Ratio
=
真正参与当前图像 gather 的 photons
───────────────────────────
总 emitted photons
```

这是未来最重要的性能指标之一。

### 5.4 和现在不同

当前：

```text
重点：Photons/s
```

新增：

```text
Photons/s
+
Useful Photons/s
+
Gather Efficiency
+
Weight Stability
```

### 5.5 解决什么

- 判断 Guiding 是否真的有效；
- 判断白点是否来自低样本密度；
- 判断是否有超高权重 photon；
- 判断 Radius 是否缩太快；
- 判断某灯/caster 是否浪费大量预算。

---

## 6. P0-2：Light Guiding 2.0

### 6.1 当前

Caustica 已经使用 Pilot Photon 测每盏灯实际焦散产出，并按比例分配预算。

### 6.2 新方案

Light 权重不再只来自：

```text
Pilot Caustic Yield
```

而逐步改成：

```text
Light Utility
=
Pilot Yield
+
Previous Gathered Contribution
```

更进一步在 1.3 后：

```text
Light Utility
=
Gathered Photon Contribution
×
PT Need
```

### 6.3 和现在不同

当前回答：

> 哪盏灯“比较能产生焦散”？

新方案回答：

> 哪盏灯“比较能产生当前相机真正需要的焦散”？

### 6.4 解决什么

- 多灯场景 photon budget 浪费；
- HDRI / Virtual Sun 中大量无关方向；
- 摄像机移动后旧的光源重要度失配。

---

## 7. P0-3：Caster Guiding 2.0

### 7.1 当前

每个焦散 caster 有包围球目标；通过 `target_yield` + EMA 更新权重，并保留 exploration 下限。

### 7.2 新方案

Caster 权重从：

```text
Target Deposit Yield
```

升级成：

```text
Target Gather Utility
```

即：

```text
Photon Deposit
       ↓
是否真的被当前 measurement points 收集？
       ↓
实际贡献多少？
       ↓
再训练 caster 权重
```

### 7.3 和现在不同

当前：

> “打中了 caster 后产生了多少焦散 photon。”

新方案：

> “这些 photon 最后有多少真的影响当前画面。”

### 7.4 解决什么

- 摄像机背后、遮挡区域浪费 photon；
- 大 caster 只有局部区域真正形成可见焦散；
- 多 caster 时权重被“高产但无用”对象占用。

---

## 8. P0-4：GPU Directional Photon Guiding

这是整个 1.2 中最重要的新功能。

### 8.1 研究依据

2026 Guided ReSTIR FG+ 提出了 GPU histogram-based photon guiding：

- 在 GPU 上记录 photon contribution histogram；
- 分层给 Light 与发射方向重新分配 photon；
- 公开结果展示了 large scene / many lights 中 photon density 的明显改善；
- 论文示例使用 64×64 direction guiding map。

### 8.2 Caustica 应该怎么做

Caustica 已有 Caster，因此建议扩展成三级概率：

\[
P(L,T,\omega)=P(L)P(T|L)P(\omega|L,T)
\]

其中：

- `L` = Light；
- `T` = Caster Target；
- `ω` = 发射方向/目标参数。

### 8.3 Direction Map

建议：

```text
Light
 └── Caster
      └── Direction Histogram
```

第一版工程建议：

```text
Low budget    → 不建 map
Medium budget → 16×16 / 32×32
High budget   → 64×64
```

### 8.4 注意：不同灯型的 UV 含义不同

不要强行把所有灯都变成世界球面方向。

建议：

#### Sun / Directional / World Virtual Sun

```text
u,v = source angular / target cone parameter
```

#### Point / Spot

```text
u,v = caster aim disk / cone coordinates
```

#### Area Light

第一版只 Guiding：

```text
目标方向 / caster aim UV
```

保留光源表面 origin sampling 使用原来的正确分布。

未来再研究：

```text
light_surface_uv × direction_uv
```

因为这已经接近 4D 分布，第一版代价太大。

### 8.5 和现在不同

当前：

```text
Light → Caster
```

新方案：

```text
Light → Caster → Useful Direction
```

### 8.6 解决什么

主要解决：

- photon 稀疏；
- 大量 photon 命中 caster 后仍走向无效区域；
- 小焦散区域难收敛；
- 同样 photon count 下白点多；
- 需要持续增加 photon count 才能变干净。

---

## 9. P0-5：Camera-aware Contribution Feedback

### 9.1 目标

Guiding 的训练信号不能只看“photon 有多亮”，而要看：

> 这个 photon 对当前图像有多有用。

### 9.2 第一阶段 1.2

建议训练值：

```text
utility = gathered photon contribution
```

暂时不要立即乘 Path Tracing variance，先把系统稳定。

### 9.3 第二阶段 1.3/1.4

逐步升级：

\[
Utility = PhotonContribution \times PTNeed
\]

其中 `PTNeed` 可以来自：

- PT variance；
- adaptive sampling residual；
- photon/PT residual；
- caustics confidence。

### 9.4 与 Lightweight Photon Mapping 的关系

Lightweight Photon Mapping 的核心思想不是简单“相机可见度”，而是：

> 把 photon computation 集中到相对于 Path Tracing 真正能降低当前图像方差的光路。

Caustica 天然适合这个思想，因为它本身就是：

```text
Cycles Path Tracing
+
Photon Caustics
```

### 9.5 和现在不同

当前：

```text
photon 成功 deposit
→ 算成功
```

新方案：

```text
photon deposit
→ 被当前 measurement point 收集
→ 对图像有实际贡献
→ 才作为强训练信号
```

### 9.6 解决什么

- 相机视野之外 photon 浪费；
- PT 本来就能很好采样的路径还在浪费 photons；
- large scene 中 photon density 不足。

---

## 10. P0-6：Robust Guiding Training

### 10.1 为什么不能直接使用 Raw Contribution

假设：

```text
999 photons = 1~10
1 photon     = 100000
```

如果直接：

```text
histogram += contribution
```

下一代 Guide 会被单个异常 photon 支配。

形成：

```text
Firefly
  ↓
Guide 追逐 Firefly
  ↓
更多集中采样
  ↓
更极端权重
```

### 10.2 正确做法

**Estimator 保持真实 contribution。**

但 Guiding 的训练统计可以使用 robust transform，例如：

```text
sqrt(luminance)
```

或：

```text
min(L, rolling_percentile_limit)
```

注意：

这只改变 proposal distribution 的训练方式，不直接修改最终 estimator 的 photon contribution。

### 10.3 解决什么

- Guide 被单个白点污染；
- 一次错误样本影响后面很多批；
- Guiding feedback 正反馈失控。

---

## 11. P0-7：Defensive Sampling

这是 Guiding 方案中必须正确实现的部分。

### 11.1 错误方案

```text
P = Pguide
```

如果某方向 Guide 给的概率接近 0，却偶然被采中：

```text
weight ∝ 1 / PDF
```

就可能产生超高权重 photon。

### 11.2 正确方案

使用 mixture：

\[
p(\omega)
=
\beta p_{guide}(\omega)
+
(1-\beta)p_{base}(\omega)
\]

推荐策略：

```text
Guide 未成熟：β = 0

随后：
0.2
0.4
0.6
0.8

成熟后建议最高约：0.85~0.90
```

始终保留 Base Distribution。

### 11.3 Base Distribution 是什么

就是当前 Caustica 已经正确工作的：

- light sampling；
- target/caster sampling；
- physically valid source emission distribution。

### 11.4 和现在不同

当前 Target Guiding 已经有 exploration floor。

新方案把“探索下限”进一步扩展到：

```text
Directional Guiding PDF
```

### 11.5 解决什么

- Guiding 自己制造白点；
- 新出现焦散永远采不到；
- 灯/物体移动后旧 Guide 锁死；
- 动画不稳定。

---

## 12. P0-8：Exact PDF / 能量正确性

### 12.1 总发射 PDF

简化表示：

\[
p_{emit}
=
p(L)
\times p(T|L)
\times
[(1-\beta)p_{base}(\omega|L,T)+\beta p_{guide}(\omega|L,T)]
\]

实际实现还要包含光源位置/面积采样等源分布项。

### 12.2 Photon Flux

必须根据**实际使用的总 PDF**计算初始 photon power/flux。

否则：

```text
Guide Strength ↑
→ 焦散整体亮度改变
```

这说明 estimator 已经错误。

### 12.3 验收规则

Guide OFF / ON：

- equal-time 图像可以不同；
- 收敛后的平均能量必须趋向一致；
- Guide Strength 不应该成为“焦散亮度”滑块。

### 12.4 解决什么

- Guiding bias；
- 焦散整体变亮/变暗；
- 不同参数产生能量漂移。

---

## 13. P0-9：Guide Confidence 与 ESS

### 13.1 ESS

ESS = Effective Sample Size，有效样本数。

常用定义：

\[
ESS = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
\]

直觉：

```text
10 个权重差不多的 photon
→ ESS 接近 10

10 个 photon 但 1 个占 99% 权重
→ ESS 接近 1
```

### 13.2 用 ESS 控制 β

```text
ESS 低
→ Guide 不可靠
→ β ↓

ESS 高
+ histogram 多批稳定
→ β ↑
```

推荐：

```text
β = max_beta × confidence
```

### 13.3 和现在不同

当前 EMA 的权重更新主要来自 target yield。

未来：

```text
EMA
+
ESS
+
History Stability
+
Sample Count
```

共同决定 Guiding 强度。

### 13.4 解决什么

- 少量 photon 误导整张 guide map；
- 首批训练不稳定；
- 极端场景 Guide 过度自信。

---

# 第四部分：P1 —— Adaptive SPPM

## 14. 当前 SPPM 半径逻辑

标准 SPPM 的核心思想是：

随着每一代 photon 到来：

- 有效 photon 数增长；
- radius 渐进缩小；
- tau 重标定；
- estimator 逐步降低 density estimation bias。

典型形式：

\[
N' = N + \alpha M
\]

\[
R' = R\sqrt{\frac{N+\alpha M}{N+M}}
\]

\[
\tau' = (\tau+\Phi)\frac{R'^2}{R^2}
\]

其中：

- `M`：本代 radius 中的 photon 数；
- `Φ`：本代 photon flux 总贡献；
- `α`：收缩控制参数；
- `N`：累积有效 photon 数；
- `R`：gather radius；
- `τ`：累积通量统计。

Caustica 当前还叠加了 batch weighting、clamp、radius floor 等工程修正，因此实际实现不能机械替换成教科书公式。

---

## 15. P1-1：统一 Radius Policy —— Preview 是 Final 的早期收敛状态

### 15.1 原则

Viewport 和 Final **不得使用两套 radius policy，也不得使用两套 photon budget/schedule**。二者必须使用同一 Photon Count 解释、同一 batch-index 规则、同一 SPPM recurrence、同一 adaptive confidence gate、同一 ESS 条件和同一 kernel normalization。

总运行次数的区别已经由 Blender 自带设置表达：

```text
Viewport
→ 运行 Viewport Samples 指定的次数
→ R 自然仍较大
→ 噪声更高

Final
→ 运行 Render Samples 指定的次数
→ R 随统计证据继续缩小
→ 噪声更低、焦散更锐
```

如果 Viewport Samples 与 Render Samples 设置成相同数值，则 Caustica 的 photon batch 数、每批 Photon Count 和物理状态演化也必须相同；不能再叠加 Viewport/Final multiplier、专用起始批次或隐藏质量档。

### 15.2 取消“Preview 专用永久 Radius Floor”

当前 `r >= r0 / 4` 可以继续作为兼容/调试基线，但新的默认生产路径不应把 Viewport 和 Final 分成：

```text
Viewport = 有 floor 的稳定估计器
Final    = 无 floor 的渐进估计器
```

正确方式是引入统一的 adaptive lower-bound / confidence gate：

```text
R_next = SPPM_shrink(R, N, M)

if ESS / photon density / confidence 不足：
    暂缓 shrink
else:
    正常 shrink
```

这个规则在 Viewport 和 Final 中完全相同。Viewport 之所以看起来更“粗”，只是因为统计尚未成熟，而不是因为使用了另一套 estimator。

### 15.3 同预算一致性验收

必须增加一个硬测试：

```text
Viewport:  64 camera spp + X photons
Final:     64 camera spp + X photons
```

在关闭仅显示层 denoise 后，两者应满足：

- 平均焦散能量一致；
- 焦散几何位置一致；
- radius / ESS 分布一致或在并行调度容差内；
- Light Group / Caustics AOV 一致；
- 不允许出现 Preview 有焦散、Final 换算法后焦散形状变化。

### 15.4 解决什么

- 真正做到所见即所得；
- 避免 Preview 调好灯光后 F12 焦散形状/亮度改变；
- Final 不再被固定 kernel bias 限制；
- Viewport 不需要靠另一个 estimator“伪稳定”；
- 后续所有 Guiding / Vulkan backend 都只维护一套物理逻辑。

---

## 16. P1-2：Adaptive Radius

### 16.1 为什么不要立即完全替换 SPPM recurrence

Hypothesis-testing PPM 等方法很值得研究，但当前 Caustica 已经有大量工程 heuristics。

直接换公式风险：

- 很难定位亮度变化来自哪里；
- 与 batch weighting 冲突；
- 和 adaptive sampling maturity 冲突；
- 会同时改变 bias 和 variance。

### 16.2 第一版建议

保留现有 SPPM 主更新公式，增加一个：

```text
Shrink Confidence
```

输入：

- 本代 `M`；
- empty gather ratio；
- ESS；
- local flux variance；
- history stability。

逻辑：

```text
M 太少 / ESS 低
→ 不要激进缩小

M 充足 / ESS 稳定
→ 正常 shrink

极高 photon density
→ 允许更积极减小 radius
```

### 16.3 第二版

再正式 A/B：

- Current SPPM；
- Adaptive PPM；
- Hypothesis-testing radius。

### 16.4 解决什么

- 稀疏区半径缩太快产生白点；
- 密集区半径长期过大产生 blur；
- 不同局部 photon density 使用同样节奏。

---

# 第五部分：P1 —— Photon Differential 与 Anisotropic Kernel

## 17. Photon Differential

### 17.1 是什么

普通 Photon 只知道：

```text
位置
方向
能量
```

Photon Differential 还追踪：

> 邻近的一小束光线经过同一路径后会如何展开、压缩、旋转。

因此到 receiver 时，不再认为一个 photon 是：

```text
○
```

而是一个有方向的 footprint：

```text
━━━━━━
```

或：

```text
/////
```

### 17.2 研究依据

Photon Differential Splatting、NVIDIA AAPS、2025 ARPΔ 都说明：

- photon footprint 可以根据路径传播；
- elliptical / anisotropic footprint 能在较低 photon count 下恢复更锐利焦散；
- 特别适合硬件光追和 caustics。

### 17.3 Caustica 的实现原则

不要永久存完整：

```text
dPdu
dPdv
dDdu
dDdv
```

因为会增加：

- Photon payload；
- register pressure；
- VRAM；
- bandwidth；
- 降低 occupancy。

建议：

Trace 阶段维护 compact differential；

Deposit 时压缩成：

```text
major axis
minor axis
orientation
```

或者：

```text
2D covariance
```

### 17.4 解决什么

- 细线焦散糊掉；
- 焦散边界泄漏；
- 曲面焦散必须用很小圆形 radius 才能保持锐利；
- 小圆形 radius 导致 photon 稀疏白点。

---

## 18. Anisotropic Kernel

### 18.1 当前

近似：

\[
d^2=x^2+y^2
\]

也就是圆形距离。

### 18.2 新方案

使用 covariance：

\[
d^2=x^T\Sigma^{-1}x
\]

其中 `Σ` 描述椭圆：

- 方向；
- 长轴；
- 短轴。

### 18.3 必须加的保护

Photon Differential 在极端 grazing / refraction 情况可能得到超长 footprint。

所以要有：

```text
max eccentricity
max footprint
min footprint
fallback to isotropic SPPM
```

### 18.4 和现在不同

当前：

```text
所有 photon 基本共享局部圆形 density 逻辑
```

未来：

```text
每个 photon 自带局部形状信息
```

### 18.5 解决什么

主要解决：

- Blur；
- Light leaking；
- Thin caustics；
- 曲面焦散；
- 通过减小圆形 radius 获得锐利结果时导致的高 variance。

---

# 第六部分：P1 —— 光学正确性升级

## 19. True GGX Transmission

### 19.1 当前问题

当前粗糙折射属于近似方向扰动，不是真正完整的 GGX transmission distribution。

### 19.2 新方案

统一成：

```text
GGX VNDF Reflection
GGX VNDF Refraction / Transmission
```

并正确处理：

- half-vector；
- eta；
- Jacobian；
- transmission PDF；
- adjoint transport correction；
- Fresnel split。

### 19.3 和现在不同

当前：

> rough glass 的方向大致“抖开”。

未来：

> rough glass photon 按和 Cycles BSDF 一致的微表面分布采样。

### 19.4 解决什么

- Rough Glass 焦散形状错误；
- 局部亮点不是统计噪声而是错误 PDF；
- Path Tracing 与 Photon 路径能量不一致。

---

## 20. Medium Stack

### 20.1 当前

只有单一当前 medium 状态。

### 20.2 问题

```text
Air
 ↓
Glass
 ↓
Water
 ↓
Glass
 ↓
Air
```

单槽状态很难正确恢复嵌套关系。

### 20.3 新方案

Photon path 维护轻量介质栈：

```cpp
MediumStack {
    object_id;
    ior;
    absorption;
};
```

考虑 GPU 成本，第一版建议：

```text
fixed depth 4~8
```

而不是动态容器。

### 20.4 解决什么

- 玻璃杯 + 水；
- 彩色液体；
- 多层透明壳；
- 嵌套 dielectric 中吸收颜色丢失。

---

# 第七部分：P2 —— Photon Culling

## 21. Photon Culling

### 21.1 是什么

先从 camera measurement points 建一个 receiver relevance mask。

Photon deposit 时：

```text
这个 cell 附近不存在可能 gather 我的 measurement point
→ 不存 photon
```

### 21.2 研究依据

2023 Hardware RT Photon Mapping 的 Photon Culling：

- 大场景中可以减少无用 photon 存储和后续 acceleration/gather 工作；
- 论文测试中大场景有明显收益；
- 但 Caustic Water / Caustic Glass 这种小型解析光场景可能反而变慢。

### 21.3 因此 Caustica 必须 Auto

建议 Pilot 评估：

```text
预计 rejected photons 太少
→ OFF

大部分 photons 都落在无关 receiver cells
→ ON
```

### 21.4 和现在不同

当前：

```text
符合 deposit 条件
→ 存
```

未来：

```text
符合 deposit 条件
+
receiver relevance
→ 才存
```

### 21.5 解决什么

- 大场景显存浪费；
- Binning 工作量；
- 不可见区域 photon 存储。

### 21.6 不解决什么

它不是主要的白点算法。

所以优先级低于 Guiding / Radius / Kernel。

---

# 第八部分：P2 —— GPU Camera Path Guiding

## 22. GPU Path Guiding

### 22.1 与 Photon Guiding 完全不是一回事

Photon Guiding：

```text
Light → Camera
```

帮助：

- caustics；
- specular-diffuse 路径；
- light path efficiency。

Camera Path Guiding：

```text
Camera → Scene → Light
```

帮助：

- 小窗户室内；
- HDRI difficult lighting；
- diffuse multi-bounce；
- 普通 GI noise。

### 22.2 为什么不先做

当前核心痛点是：

> SPPM 焦散白点和 photon efficiency。

普通 Camera Path Guiding 不直接解决这一点。

### 22.3 最终关系

未来可以：

```text
Camera Guiding
      ↓
   Receiver
      ↑
Photon Guiding
```

形成双端高效采样，但不需要立即进入 1.2。

---

# 第九部分：P3 —— ReSTIR FG+

## 23. ReSTIR FG

### 23.1 是什么

ReSTIR = Reservoir-based Spatiotemporal Importance Resampling。

核心思想：

不用每个像素保留大量 samples，而维护一个 Reservoir，代表很多候选样本。

然后在：

- 时间；
- 空间；

之间复用样本。

### 23.2 ReSTIR FG

ReSTIR FG 把：

```text
Photon Final Gathering
+
Reservoir Resampling
```

结合起来，可以实时显示 caustics 和 GI。

---

## 24. Guided ReSTIR FG+

2026 版本增加：

- GPU photon guiding；
- large scenes / many lights 支持；
- backprojection；
- random replay；
- 更好的直接可见焦散锐度；
- 更好的时序一致性。

### 24.1 Caustica 应该怎么使用

**ReSTIR FG+ 不再作为默认 Viewport 的另一套 estimator。**

默认生产链路必须是：

```text
Viewport
→ Guided Adaptive SPPM

Final Render
→ Guided Adaptive SPPM
```

Caustica 参数和 batch 规则完全相同；若两者表现出总工作量差异，只能来自 Blender 已有的 Viewport Samples 与 Render Samples 设置。

ReSTIR FG+ 只允许以两种方式进入：

1. **Experimental Preview**：明确标记为非 WYSIWYG 实验模式；
2. **Estimator-neutral acceleration**：只作为 candidate generation / reservoir proposal / temporal reuse 机制，并通过完整 PDF 重加权，证明不会改变 Guided Adaptive SPPM 的目标估计与同预算结果。

### 24.2 为什么不能默认“Viewport ReSTIR / Final SPPM”

如果默认：

```text
Viewport = ReSTIR FG+
Final    = SPPM
```

用户调灯、材质和玻璃时看到的焦散可能在 F12 后改变：

- 局部亮度不同；
- 高频细节不同；
- temporal history 带来的稳定结构在 Final 不存在；
- reservoir selection 与 SPPM density estimate 的误差形态不同。

这违反 Caustica 的核心产品目标：**预览就是最终渲染的低采样版本。**

### 24.3 推荐研究方向

如果未来希望使用 ReSTIR 技术提高 Viewport 速度，优先研究：

```text
Same Guided SPPM target
        │
ReSTIR candidate / reservoir reuse
        │
Exact reweighting / validity
        ↓
Same photon contribution estimator
```

也就是说 ReSTIR 只改变“怎样更聪明地找到候选 photon / light sample”，不改变最终被估计的物理量。

### 24.4 进入默认链路的验收条件

ReSTIR 加速只有同时通过以下测试才能从 Experimental 升级为默认：

- ReSTIR ON/OFF 同预算平均能量差 < 1%；
- 高频焦散结构无系统性偏移；
- 静态场景长时间收敛到同一参考；
- history reset 后没有 ghost caustics；
- 动画不出现 reservoir history 引起的亮度漂移；
- Guide training 不读取 display filtered / denoised value。

在这些条件满足前，**不要为了更“稳”的预览牺牲所见即所得。**

---

# 第十部分：P3 —— Neural Progressive Photon Mapping

## 25. NPPM

### 25.1 是什么

2026 Neural Progressive Photon Mapping 不直接让神经网络随意预测每个 photon 权重。

它学习：

> 根据局部 photon statistics 预测一个可解析、可归一化的 density kernel 参数。

目的：

- 更锐利；
- 降低 bias；
- 兼容 progressive photon mapping。

### 25.2 为什么不作为默认

额外复杂度：

- 神经推理；
- 模型部署；
- GPU 架构差异；
- 调试复杂；
- 推理成本；
- 和 Cycles device backend 集成成本。

### 25.3 建议定位

未来：

```text
Photon Kernel
  Classic
  Adaptive
  Anisotropic
  Neural HQ
```

### 25.4 解决什么

主要是：

- low photon count density reconstruction；
- 复杂局部焦散 kernel 形状；
- bias-quality tradeoff。

但不是当前第一优先级。

---

# 第十一部分：P4 —— BDPT / VCM

## 26. Lightweight BDPT

未来先做：

```text
PT + NEE + Light Tracing
```

而不是立即 Full BDPT。

适合补：

- 某些 photon mapping 收敛慢但 light tracing 能直接到 camera 的路径；
- 复杂镜面路径。

---

## 27. Full BDPT

需要统一：

```text
PathVertex
PDF forward
PDF reverse
BSDF
eta
delta flag
geometry term
visibility
MIS
```

工程量显著大于 Guided SPPM。

---

## 28. VCM

VCM = Vertex Connection and Merging。

可以理解为：

```text
BDPT Vertex Connection
+
Photon Mapping Vertex Merging
```

Caustica 已经拥有 photon infrastructure，因此未来很适合走 VCM。

但 1.1 当前最直接的问题不是“缺少全局通用 integrator”，而是：

> photon 已经追得很快，但有效率和 density estimator 仍然可以明显提升。

因此 VCM 应该进入 2.0，而不是 1.2。

---

# 第十二部分：白点问题完整分类

## 29. 白点不是一种问题

| 白点类型 | 根因 | 现有 Caustica | 新方案 |
|---|---|---|---|
| A. Sparse Photon | radius 内 photon 太少 | clamp / floor | Directional Guiding + Adaptive Radius |
| B. High Weight | 极低 PDF 样本权重巨大 | 部分 clamp | Defensive Mixture + Exact PDF + ESS |
| C. Guide Feedback Outlier | guide 被异常 photon 训练 | 无完整机制 | Robust Training Statistic |
| D. Kernel Mismatch | 圆形 kernel 不符合焦散形状 | plane/normal reject | Photon Differential + Ellipse Kernel |
| E. Radius Too Small | 收缩过快 | r0/4 floor | Adaptive Shrink |
| F. Radius Too Large | blur / leak | boundary heuristic | Anisotropic Kernel + Final shrink |
| G. Measurement Jitter | hitpoint 抖动 | tau average | confidence / geometry validation |
| H. Rough Refraction Error | BSDF 分布错误 | cone jitter | True GGX Transmission |
| I. Nested Medium Error | medium 状态错误 | single state | Medium Stack |
| J. GPU Timing Error | buffer race / stale state | 已做大量修复 | regression + debug counters |

### 29.1 非常重要

不要使用：

```text
亮度超过阈值
→ 一律 clamp
```

因为真实焦散本身就可能非常亮。

正确判断 Firefly 更应该看：

- 是否只出现一代；
- max / median weight；
- ESS；
- 邻域稳定性；
- PDF；
- 是否随更多 photons 稳定。

---

# 第十三部分：和 Caustica 1.1 的完整差异

## 30. 总对比表

| 模块 | Caustica 1.1 | 新方案 | 解决问题 |
|---|---|---|---|
| Light Budget | Pilot yield | Pilot + Gather Utility | 多灯预算更准确 |
| Caster Guide | deposit/yield EMA | actual gathered utility | 不给不可见焦散浪费 photons |
| Direction Guide | 无 | GPU histogram | 提高有效 photon density |
| Guide Feedback | target yield | camera-aware contribution | 相机相关优化 |
| Guide Robustness | exploration floor | mixture + ESS + robust history | 防止 guide 自己制造白点 |
| PDF | target probability correction | full hierarchical mixture PDF | 保持能量正确 |
| Radius | SPPM + r0/4 floor | viewport floor / final adaptive | 减 blur 和 sparse noise |
| Kernel | isotropic Epanechnikov | anisotropic elliptical | 锐利焦散、少漏光 |
| Photon Shape | point photon | compact differential footprint | 适应路径聚焦/拉伸 |
| Rough Refraction | approximate jitter | GGX transmission | 物理正确 rough glass |
| Medium | single state | stack | glass + liquid |
| Culling | 无或不作为核心 | auto receiver culling | 大场景省算力/VRAM |
| Viewport reuse | progressive batch | optional ReSTIR FG+ | 更快实时预览 |
| Final estimator | SPPM | Guided Adaptive SPPM | 维持渐进求解 |
| Debug | 环境变量为主 | Debug AOV + statistics | 更易定位问题 |
| Performance KPI | raw photons/s | useful photons/s + error/time | 衡量真实效率 |

---

# 第十四部分：GPU 数据结构设计

## 31. 新增 Guiding 数据结构

建议新增：

```text
intern/cycles/kernel/integrator/
    photon_guiding.h
    photon_guiding_types.h
```

### 31.1 Key

```cpp
struct PhotonGuideKey {
    uint light_id;
    uint target_id;
};
```

### 31.2 Header

```cpp
struct PhotonGuideHeader {
    uint offset;
    ushort resolution;
    ushort flags;

    uint sample_count;

    float confidence;
    float beta;
};
```

### 31.3 Histogram Statistics

```cpp
struct PhotonGuideCell {
    uint contribution_sum;
    uint contribution_sq_sum;
    uint sample_count;
};
```

为了 GPU 原子效率，建议 contribution histogram 优先用量化整数累积，然后在 reduce 阶段恢复 scale。

---

## 32. Photon 数据增加什么

不建议第一版把所有信息永久塞进最终 photon array。

### 必须

```text
light id
caster/target id
guide cell id
```

### 可能需要

```text
emission pdf / compact reconstruction data
```

如果 PDF 只在 trace 阶段用于 flux 初始化，deposit 后可以不永久存。

### Anisotropic 版本

Deposit 后压缩：

```text
major radius
minor radius
orientation
```

或者 covariance。

---

# 第十五部分：新增 GPU Kernel

## 33. 推荐 Kernel

当前已经有：

```text
photon_trace
photon_bin_count
photon_bin_cursor_init
photon_bin_scatter
film_photon_gather
film_photon_smooth
```

新增：

```text
photon_statistics_clear
photon_statistics_reduce

photon_guide_clear
photon_guide_accumulate
photon_guide_reduce
photon_guide_history_update
photon_guide_budget
photon_guide_build_mip

photon_receiver_mask_build

film_photon_variance
film_photon_debug_aov
```

### 33.1 推荐顺序

```text
Gather
 ↓
Guide Accumulate
 ↓
Reduce
 ↓
History / EMA
 ↓
Confidence / ESS
 ↓
Budget
 ↓
Mip / Prefix
 ↓
Next Photon Trace
```

---

# 第十六部分：GPU 性能原则

## 34. 不允许 Guiding 把 Photon Tracer 拖死

Caustica 当前最大资产是高速 Photon Trace。

因此 Guiding 必须遵守：

### 34.1 禁止重型树结构作为第一版

第一版不建议直接搬：

- CPU SD-tree；
- OpenPGL 数据结构；
- 大量动态分配；
- 每 photon 高复杂度 KNN。

### 34.2 推荐

- Fixed / pooled histogram；
- integer atomic；
- block-level reduction；
- compact mip/prefix；
- 只有 active Light × Caster 才分配 Direction Map；
- 小 budget 不启用 direction map。

### 34.3 Guiding 性能目标

这些是**工程验收目标，不是论文保证值**：

```text
Guiding GPU overhead
目标 < 10~15% raw photon trace cost
```

如果 raw photons/s 降低 10%，但 Useful Photons/s 提高数倍，则仍然是巨大净收益。

---

# 第十七部分：统一调度、WYSIWYG 与确定性

## 35. Viewport —— 同一 Photon Solver，由 Viewport Samples 控制次数

Viewport 默认必须运行与 Final 完全相同的：

```text
Pilot / Guiding
→ Photon Trace
→ Bin
→ Guided Adaptive SPPM Gather
→ N / τ / R Update
```

Viewport 不得拥有额外的 photon budget、Photon Count multiplier、专用 batch growth 或物理质量档。它运行多少次只读取 Blender 的 Viewport Samples。

Viewport 只可以在以下宿主层面不同：

- 更频繁刷新；
- 允许仅显示层 denoise/smoothing；
- 交互取消、结果呈现和 GPU stream 生命周期管理。

Viewport 的 history 默认应理解为**同一 SPPM / Guide 物理累计状态**，而不是另一套 temporal estimator。只要场景没有发生需要 invalidate 的变化，就可以继续累积。

ReSTIR temporal reuse 仍为 Experimental，除非证明 estimator-neutral。

---

## 36. Final Still Image —— 同一 Photon Solver，由 Render Samples 控制次数

F12 可以重新从确定性初始状态启动：

```text
Pilot
→ Guide warm-up
→ Progressive guided batches
```

也可以未来支持在分辨率/场景状态完全兼容时 warm-start，但**不得因为进入 Final Render 而切换 estimator、Photon Count、batch 规则、Guiding 或 radius policy**。F12 的总运行次数读取 Blender Render Samples，不再读取任何 Caustica 专用 Final Budget。

所有随机种子与 batch schedule 必须可重复。

关键不变量：

```text
Viewport Samples = B
≈
Render Samples = B
```

当 Render Samples 高于 Viewport Samples 时，Final 的提升来自 Blender 要求它运行更多次，而不是 Caustica 切换算法或使用另一套光子预算。

---

## 37. Final Animation

默认不要直接：

```text
Frame N guide
→ Frame N+1 guide
```

建议：

```text
每帧 deterministic reset
→ deterministic pilot
→ deterministic guide evolution
```

未来可以增加显式：

```text
Temporal Guide Cache [Experimental]
```

但不能默认依赖。

### 原因

Caustica 已经有过按实测速度动态批量导致动画亮度变化的历史问题，因此确定性仍然应是最高优先级。

---

# 第十八部分：UI 方案

## 38. 普通用户 UI —— 保持现状

普通用户面板不改名、不换位置、不增加模式或质量滑块，继续保持现有结构：

```text
Photon Caustics Solver
Photon Count
Detail
Intensity
Casters
```

Guided / Adaptive / Anisotropic SPPM 在这套参数背后自动运行。现有 `.blend` 文件中的参数值必须继续兼容，加载旧文件不得发生数值重映射或静默改默认值。

---

## 39. Advanced —— 不新增普通用户参数

不新增普通用户可调的 Advanced 光子算法面板。以下内容全部使用内部 Auto 策略：

- Photon Guiding；
- Guide Strength / Beta；
- Guide Resolution；
- Adaptive Radius；
- Anisotropic Kernel；
- Photon Culling；
- EMA alpha；
- ESS threshold；
- exploration epsilon；
- histogram quantization；
- history confidence。

开发与验证阶段需要的开关只允许出现在 Debug AOV、日志、环境变量或显式开发者构建中；它们不是成品参数，也不得写入普通用户 `.blend` 成为新的长期兼容负担。

---

# 第十九部分：Debug AOV

## 40. 必须新增

```text
Photon Density
Useful Photon Density
Photon Weight
Max Photon Weight
ESS
Gather Radius
Guide Confidence
Guide Direction
Guide Beta
Kernel Eccentricity
Caster ID
Light ID
Receiver Utility
```

### 最重要四张

```text
Useful Photon Density
Max Photon Weight
ESS
Gather Radius
```

看到白点时可以快速判断：

```text
Density 低
→ Sampling / Radius

Weight 极高
→ PDF / Guiding

ESS 极低
→ Guide / Outlier

Radius 极小
→ Shrink policy
```

---

# 第二十部分：源码文件级修改

## 41. `photon_map.cpp/.h`

新增：

- Guiding Manager；
- Light/Caster/Direction budget；
- Histogram resource pool；
- Guide history；
- ESS / confidence；
- Auto culling decision；
- Unified WYSIWYG Viewport/Final schedule。

---

## 42. `photon_trace_types.h`

新增：

- guide key；
- guide UV / cell；
- sampling mode；
- compact photon differential state；
- optional medium stack state。

---

## 43. `photon_trace.h`

修改：

- hierarchical guided emission；
- defensive mixture sampling；
- exact PDF correction；
- true GGX transmission；
- medium stack；
- photon differential propagation；
- statistics counters。

---

## 44. `photon_grid.h`

新增：

- optional receiver relevance mask；
- anisotropic footprint bounds；
- debug density counters。

---

## 45. `photon_lookup.h`

新增：

- ESS；
- adaptive shrink information；
- elliptical distance；
- footprint fallback；
- anisotropy clamp。

---

## 46. `photon_passes.h`

新增：

- per-hitpoint contribution；
- second moment；
- variance；
- useful photon counter；
- guiding feedback；
- debug pass。

---

## 47. `shade_surface.h`

新增：

- receiver utility；
- PT need / variance hook；
- camera-visible measurement metadata。

---

## 48. `path_trace.cpp`

新增：

- Unified WYSIWYG Viewport/Final schedule；
- guide warm-up；
- unified adaptive radius lifecycle；
- guide reset / lifecycle；
- deterministic final schedule。

---

## 49. OptiX / Device

新增：

- guide/stat kernels；
- device buffers；
- stream dependency；
- event/barrier；
- debug synchronization mode。

必须继续遵循当前 Caustica 的原则：

> Photon Pipeline 的 launch params 与普通 render pipeline 隔离。

---

# 第二十一部分：测试场景

## 50. 必须固定 Regression Scenes

### Scene A：Glass Sphere + Sun

测试：

- 基础 specular-diffuse caustic；
- energy；
- directional guide。

### Scene B：Glass + Large Area Light

测试：

- area light guide；
- low-PDF outlier；
- defensive mixture。

### Scene C：Water Caustics

测试：

- 高频 caustics；
- anisotropic footprint；
- radius。

### Scene D：Rough Glass

测试：

- GGX transmission；
- white speckle 分类。

### Scene E：Glass + Liquid

测试：

- Medium Stack；
- absorption；
- nested eta。

### Scene F：HDRI Product

测试：

- Virtual Suns；
- multi-light guiding；
- 产品渲染常见场景。

### Scene G：Bistro / Interior

测试：

- large scene；
- photon culling；
- camera-aware utility。

### Scene H：Moving Light / Camera

测试：

- Guide adaptation；
- history reset；
- animation flicker。

### Scene I：No-Caustics Control

必须保证：

> 没有明显焦散的普通 Cycles 场景，开启 Auto 后不能有明显额外开销。

---

# 第二十二部分：性能指标

## 51. 不再只看 Photons/s

必须同时记录：

```text
Raw Photons/s
Useful Photons/s
Deposited Photons/s
Gathered Photons/s

Trace ms
Binning ms
Guiding ms
Gather ms
Culling ms

VRAM

Empty Gather %
Mean photons / hitpoint
ESS
Max / Median weight

MRSE / MSE
99% / 99.9% error
Time-to-quality
```

---

## 52. 主 KPI

建议主 KPI：

# Time-to-Quality

即：

> 达到同样误差/同样视觉质量，需要多少时间。

而不是：

> 一秒能发多少 photons。

---

## 53. 工程验收目标

以下均为**目标值，需要实际 A/B 验证，不是研究论文对 Caustica 的保证**。

### 53.1 Guiding Overhead

目标：

```text
< 10~15%
```

### 53.2 Useful Photon Ratio

典型焦散场景目标：

```text
至少明显高于 1.1
```

理想目标：

```text
2× 以上
```

但不应在没有实测前作为发布承诺。

### 53.3 Firefly / White Speckle

Equal-time：

- high-weight outlier 数明显下降；
- Max/Median Photon Weight 降低；
- ESS 提高。

### 53.4 Energy

Guide ON/OFF 最终均值应接近，并随 sample 增加进一步收敛。

### 53.5 No-caustics Scene

Auto 模式下额外开销应尽可能接近 0；建议目标控制在几个百分点以内。

---

# 第二十三部分：开发阶段

## 54. Caustica 1.2 —— Guided SPPM

### 必须完成

```text
Photon Statistics
Light Guide 2.0
Caster Guide 2.0
Directional Guide
Camera-aware Gather Feedback
Robust Guide Training
Defensive Mixture
Exact PDF
Guide ESS / Confidence
Debug AOV
```

### 1.2 不做

- Full BDPT；
- VCM；
- Neural PPM；
- Full GPU Camera Path Guiding；
- 默认使用“Viewport ReSTIR / Final SPPM”双估计器架构。

---

## 55. Caustica 1.3 —— Adaptive / Physical SPPM

新增：

```text
Unified Viewport/Final Radius Policy
Adaptive Radius
ESS-based Shrink
True GGX Transmission
Medium Stack
Photon Differential Prototype
```

---

## 56. Caustica 1.4 —— Anisotropic / Smart GPU

新增：

```text
Production Anisotropic Kernel
Auto Photon Culling
PT Need Feedback
Lightweight Photon Mapping-style Utility
GPU Camera Path Guiding Prototype
```

---

## 57. Caustica 1.5 —— WYSIWYG Preview Acceleration

默认仍然使用与 Final 完全相同的 Guided Adaptive SPPM，只优化不改变物理 batch 语义的宿主调度与候选生成；不得新增 Viewport/Final 两套 Photon Count 或预算。

研究项：

```text
Estimator-neutral Temporal Reuse
Estimator-neutral Spatial Reuse
ReSTIR candidate generation [Experimental]
Backprojection [Experimental]
Random Replay [Experimental]
```

硬约束：

```text
任何 Preview 加速
必须保持 Same Target Estimator
否则只能放在 Experimental 模式
```

---

## 58. Caustica 1.6 —— HQ Experimental

新增：

```text
Neural Progressive Photon Kernel
```

作为可选 HQ 模式。

---

## 59. Caustica 2.0 —— Hybrid Light Transport

路线：

```text
Light Tracing
       ↓
Lightweight BDPT
       ↓
Full BDPT
       ↓
Vertex Connection
       +
SPPM Vertex Merging
       ↓
VCM
```

最终可能形成：

```text
Integrator
  Path Tracing
  Path + Guided Caustics
  Lightweight BDPT
  BDPT
  VCM
```

---

# 第二十四部分：风险矩阵

## 60. 主要风险

| 风险 | 严重度 | 原因 | 防护 |
|---|---:|---|---|
| Guiding Bias | 极高 | PDF 修正错误 | Guide ON/OFF energy regression |
| Guiding Firefly | 高 | pGuide 极小 | Defensive mixture + ESS |
| GPU Atomic Contention | 高 | histogram 热点 | integer atomic + block reduce |
| Register Pressure | 高 | differential/medium state 太大 | compact state / fixed stack |
| VRAM 增长 | 高 | per-light-target maps | pooled active maps |
| Animation Flicker | 极高 | 历史/动态 batch | deterministic final schedule |
| Radius Brightness Drift | 高 | adaptive shrink 与 tau 不匹配 | analytical regression |
| Anisotropic Leak | 中高 | ellipse 过大/法向错误 | eccentricity clamp + fallback |
| Rough Glass Energy Error | 高 | transmission PDF/Jacobian | compare against Cycles reference |
| Culling False Negative | 极高 | 把有贡献 photon 丢掉 | conservative mask only |
| OptiX Race | 极高 | 多 stream / shared state | private params + explicit events |

---

# 第二十五部分：哪些东西暂时不建议做

## 61. 不建议：继续加大量 Clamp

原因：

- 隐藏真实 sampling 问题；
- 吃掉真实高能焦散；
- Detail 改变亮度；
- 难以保证 estimator 性质。

Clamp 应该最后用于极端保护，而不是主要降噪器。

---

## 62. 不建议：第一版直接 OpenPGL GPU 化

原因：

- 数据结构太重；
- Caustica photon tracer 当前强项是吞吐；
- histogram guiding 更符合当前架构。

---

## 63. 不建议：第一版所有 Light×Caster 都 64×64

原因：

- large light count 会爆 VRAM；
- 很多组合根本没有足够 photon 训练；
- 低 budget map 只会产生噪声分布。

必须 active allocation。

---

## 64. 不建议：默认跨动画帧继承 Guide

原因：

- Temporal bias / flicker 风险；
- 场景变化后 proposal lag；
- 不利于 deterministic render。

---

## 65. 不建议：现在直接 Full VCM

原因：

- 工程规模大；
- 不能直接解决当前白点最根本的 photon efficiency 问题；
- 当前 SPPM infrastructure 已经非常强，先把它利用率提升更划算。

---

# 第二十六部分：名词解释

## 66. 基础渲染名词

### Rendering Equation / 渲染方程

描述一个点向某方向发出的 radiance，由自身 emission 和来自其它方向的入射光经过 BSDF 反射/透射组成。

所有 Path Tracing、Photon Mapping、BDPT、VCM 本质上都在数值求解它。

### Monte Carlo / 蒙特卡洛积分

用随机样本估计积分。

样本越多通常方差越小，但 sampling distribution 是否合理会决定收敛速度。

### Bias / 偏差

估计值的平均结果与真值存在系统性差异。

Photon density estimation 在有限 radius 下通常带 bias，但 progressive shrink 可以让 bias 渐进下降。

### Variance / 方差

随机估计结果每次波动有多大。

图像上常表现为噪点、白点、闪烁。

### Consistent Estimator / 一致估计器

随着样本趋向无穷，估计结果趋向正确答案。

### Radiance / 辐亮度

渲染器中最常用的光能描述之一，带空间与方向意义。

### Flux / 光通量/能量包

Photon Mapping 中 photon 携带的能量权重通常称 flux/power。

---

## 67. Path Tracing 相关

### PT / Path Tracing

从 Camera 发射路径，在场景中随机反弹，估计完整光传输。

### NEE / Next Event Estimation

在路径顶点直接采样光源，而不是等待随机方向刚好撞到灯。

### MIS / Multiple Importance Sampling

把多种 sampling strategy 按概率权重组合，降低方差。

### BSDF

Bidirectional Scattering Distribution Function。

描述光线在材质表面如何反射和透射。

### BRDF

BSDF 中只描述反射部分。

### BTDF

BSDF 中描述透射/折射部分。

### Delta BSDF

理想镜面、理想折射等集中在单一方向的分布。

### Throughput

一条 Path 到当前顶点为止累计的能量权重。

---

## 68. Photon Mapping 相关

### Photon Mapping / 光子映射

从 Light 端发射 photon，将光子存储在表面，再从 Camera 端进行 density estimation。

特别擅长焦散。

### Photon / 光子

这里不是物理量子模拟，而是 Monte Carlo 中代表一束能量的 sample/particle。

### Photon Deposit

Photon 在合格 receiver 表面存入 photon map。

### Photon Gather

从 measurement point 周围收集 photons，估计局部 radiance。

### Density Estimation / 密度估计

通过一定 radius/kernel 内 photon 数量和能量估计局部光照。

### PPM / Progressive Photon Mapping

让 radius 随 iteration 渐进缩小，不需要无限保存 photons。

### SPPM / Stochastic Progressive Photon Mapping

SPPM 是 PPM 的 stochastic 扩展，可以正确处理像素域 stochastic effects，例如 DOF、glossy eye paths 等。

Caustica 当前核心 estimator 就是 SPPM。

### Measurement Point / Hitpoint

Camera path 上用于接收 photon density estimate 的点。

### N

SPPM 中累计的有效 photon 统计量。

### Radius / R

SPPM density estimation 的局部搜索半径。

### Tau / τ

SPPM 中累积并经过 radius rescale 的 photon flux 统计量。

### Kernel

规定 radius 内 photon 对最终估计贡献权重的函数。

### Epanechnikov Kernel

一种常见 compact kernel，中间权重大、边缘逐渐降低。

---

## 69. 焦散相关

### Caustics / 焦散

光经过镜面反射或折射后聚集在 diffuse receiver 上形成的高频亮纹。

### Caster

在 Caustica 中指可以形成焦散 specular chain 的物体，例如玻璃、金属、水面。

### Receiver

接收焦散 photon 的表面，通常是 diffuse/rough surface。

### Specular-Diffuse-Specular Path

含镜面与漫反射连接的困难光传输路径类别之一。

### Light Transport Chain

从 Light 到 Camera 的完整光线路径序列。

---

## 70. Guiding 相关

### Path Guiding

学习“哪些方向更可能有高贡献”，然后用学习到的 distribution 辅助下一批路径采样。

### Photon Guiding

专门引导从 Light 端发射的 photons。

### Light Guiding

学习应该给哪些 Light 更多 photon budget。

### Caster Guiding / Target Guiding

学习 photons 应该更优先投向哪些焦散 caster。

### Directional Guiding

在选定 Light/Caster 后，再学习哪个发射方向或目标 UV 更有用。

### Guiding Map

存储 learned distribution 的数据结构，例如 2D histogram。

### Histogram

把连续方向空间分成离散 bins，累计 contribution。

### EMA / Exponential Moving Average

指数移动平均。

新数据占一部分，旧历史保留一部分，使 guide 不会每批剧烈变化。

### Exploration

故意保留一定概率继续采样当前 Guide 认为“不重要”的区域，避免漏掉新路径。

### Exploitation

利用已学习 Guide，把更多样本给当前认为重要的区域。

### Defensive Sampling

用 Base Distribution 与 Learned Distribution 混合，防止 proposal 变得过窄或概率接近 0。

### Proposal Distribution

Monte Carlo 中用于生成样本的概率分布。

### Base Distribution

没有 Guiding 时原本的正确采样分布。

### Guide Distribution

Guiding 学习得到的分布。

### PDF

Probability Density Function。

样本被采到的概率密度。Monte Carlo 权重必须正确考虑 PDF。

### Exact PDF Correction

当 sampling distribution 改变后，按实际最终 PDF 重新计算权重，保持能量与 estimator 正确。

### Utility / Usefulness

本文中表示 photon 对当前图像是否真正有价值的训练指标。

### Camera-aware

说明一个策略会考虑当前摄像机看到什么，而不是只考虑场景全局能量。

---

## 71. 统计与白点相关

### Firefly

Monte Carlo 中少量极高权重 sample 产生的异常亮点。

### Outlier

统计上远离主要样本分布的异常样本。

### ESS / Effective Sample Size

衡量“虽然有很多样本，但真正有效的独立权重样本有多少”。

### Max/Median Weight

最大 photon 权重与中位数权重的比值。

比值极高通常意味着 sample weight distribution 不稳定。

### Second Moment

二阶矩，用于估计 variance。

### Variance Estimate

根据样本历史估算当前局部随机误差大小。

### Empty Gather Ratio

measurement points 中完全没有或几乎没有 photon 的比例。

### Useful Photon

本文定义：最终真正参与当前 Camera image photon gather 的 photon。

---

## 72. Kernel / Differential 相关

### Isotropic Kernel

所有方向尺度相同，例如圆形。

### Anisotropic Kernel

不同方向尺度不同，例如椭圆。

### Photon Differential

追踪 photon 周围微小方向扰动，估算一束邻近 photon 经过路径后如何展开。

### Footprint

一个 photon 在 receiver 上代表的局部面积/形状。

### Elliptical Kernel

椭圆 density kernel。

### Covariance Matrix

用于描述二维椭圆尺度和方向的一种数学表示。

### Eccentricity

椭圆长轴/短轴差异程度。

### Light Leaking

density kernel 跨越几何/阴影/焦散边界，把能量错误扩散到不该亮的位置。

---

## 73. 光学名词

### Fresnel

光在介质界面反射和透射比例随入射角、IOR 改变的规律。

### IOR

Index of Refraction，折射率。

### Eta / η

光传输代码中常用来表示折射率或两侧折射率比例。

### GGX

常见微表面法线分布，用于粗糙金属/玻璃。

### VNDF

Visible Normal Distribution Function。

只采样从当前方向可见的微表面法线，提高 GGX sampling 质量。

### GGX Transmission

使用 GGX 微表面模型正确采样 rough dielectric transmission/refraction。

### Medium

光线当前所处介质，例如 air、glass、water。

### Medium Stack

维护嵌套介质进入/退出关系的栈。

### Beer-Lambert

光在吸收介质中随传播距离指数衰减的规律。

### Adjoint / Eta² Correction

从 Light 端与 Camera 端传播时，由于折射 measure 变化需要正确处理的能量/PDF 修正。

---

## 74. GPU / OptiX 名词

### GPU Kernel

运行在大量 GPU threads 上的函数。

### OptiX

NVIDIA 基于 RTX hardware 的 ray tracing API/framework。

### RT Core

RTX GPU 中专门加速 BVH traversal 和 ray/triangle intersection 的硬件。

### CUDA Stream

GPU 工作队列。多个 stream 在条件允许时可以并发执行。

### Pipeline

OptiX 中一组 raygen/miss/hit program 和状态组成的执行管线。

### Launch Params

GPU pipeline 启动时使用的参数缓冲。

### BVH

Bounding Volume Hierarchy，用来加速 ray-scene intersection。

### scene_intersect

Cycles 内部真实场景求交入口。

### Occupancy

GPU 同时保持活跃 warps 的能力，受 registers/shared memory 等限制。

### Register Pressure

单 thread 使用过多寄存器导致 occupancy 降低。

### Atomic Operation

多个 GPU threads 安全更新同一内存位置的原子操作。

### Atomic Contention

大量 threads 同时更新同一个 atomic 地址造成排队和性能下降。

### Prefix Sum / Scan

GPU 并行算法，用累计计数生成连续 offset；常用于 binning。

### Device-resident

数据长期保留在 GPU VRAM，不来回传 CPU。

### PCIe

CPU 与独立 GPU 之间常见的数据传输总线。

---

## 75. ReSTIR 名词

### ReSTIR

Reservoir-based Spatiotemporal Importance Resampling。

通过 Reservoir 对大量候选 sample 做压缩和重采样。

### Reservoir

用少量状态代表大量候选样本及其统计权重的数据结构。

### Temporal Reuse

复用前一帧样本。

### Spatial Reuse

复用邻近像素样本。

### Spatiotemporal Reuse

同时做空间和时间复用。

### RIS

Resampled Importance Sampling。

从一组候选样本根据目标权重重新选择代表样本。

### Final Gathering

从 camera-side surface 再采样/收集间接光信息的步骤。

### Backprojection

把已有路径/焦散信息重新投影到当前像素或当前帧，用于提高直接可见焦散的复用与稳定性。

### Random Replay

保存随机决策/随机种子并重放路径，常用于复用难以直接存储全部路径状态的 specular transport。

---

## 76. 双向算法名词

### Light Tracing

从 Light 发射路径，并尝试连接/投影到 Camera。

### BDPT

Bidirectional Path Tracing。

同时从 Camera 与 Light 构建 subpath，并尝试不同顶点连接策略。

### Path Vertex

Camera/Light subpath 上的一个顶点，通常需要存：位置、法线、throughput、PDF、BSDF 状态等。

### Forward PDF

从前一个 vertex 生成当前方向的概率密度。

### Reverse PDF

从反方向看，对应 path strategy 的概率密度。

### VCM

Vertex Connection and Merging。

把 BDPT 的 vertex connection 与 photon mapping 的 vertex merging 统一到 MIS 框架。

### Vertex Connection

显式连接 camera path vertex 和 light path vertex。

### Vertex Merging

把附近 light vertices/photon 与 camera vertex 做 density estimation 合并。

---

# 第二十七部分：最终推荐

## 77. 如果只允许做三个核心变化

### 第一：Camera-aware GPU Photon Guiding

包括：

```text
Light
→ Caster
→ Direction
→ Gather Contribution Feedback
→ Defensive Mixture
→ Exact PDF
```

这是最直接提升 Useful Photon Density 的方案。

### 第二：Adaptive Radius + ESS

解决：

- 稀疏区白点；
- 机械 radius shrink；
- Viewport / Final 不允许目标估计器冲突。

### 第三：Photon Differential + Anisotropic Kernel

解决：

- 细焦散；
- 边缘漏光；
- blur；
- 圆形 kernel 为保持锐利而被迫缩得太小导致的 variance。

---

## 78. 最终一句话

Caustica 1.1 已经解决了：

> **“怎样高速追踪大量物理可用的焦散 photons。”**

下一代应该解决：

> **“怎样只追真正有用的 photons，并用更符合真实焦散形状的 estimator 重建它们。”**

因此正确的技术主线不是继续堆光子数，而是：

# Photon Efficiency → Statistical Stability → Adaptive Density → Physical Accuracy → Real-time Reuse → Hybrid Integrator

对应版本：

```text
1.1  Fast GPU SPPM
 ↓
1.2  Guided SPPM
 ↓
1.3  Adaptive + Physical SPPM
 ↓
1.4  Anisotropic + Smart GPU
 ↓
1.5  WYSIWYG Preview Acceleration
 ↓
2.0  BDPT / VCM Hybrid
```

---

# 附录 A：推荐开发任务清单

## A.1 1.2 第一批提交

- [ ] 添加 photon statistics buffer
- [ ] 添加 Useful Photon Ratio
- [ ] 添加 photon weight / ESS 统计
- [ ] 给每个 photon batch 建立 Light/Caster feedback
- [ ] Direction Histogram 16×16 prototype
- [ ] Base + Guide mixture sampling
- [ ] 完整 PDF regression test
- [ ] Guide ON/OFF energy test
- [ ] Guide Debug AOV

## A.2 1.2 第二批提交

- [ ] Direction Histogram 32/64 adaptive allocation
- [ ] Robust training contribution
- [ ] Guide confidence
- [ ] ESS-driven beta
- [ ] pooled active Light×Caster map allocator
- [ ] Integer atomic / block reduction optimization
- [ ] camera gather contribution feedback

## A.3 1.3

- [ ] Unified Viewport/Final radius policy
- [ ] Adaptive shrink gate
- [ ] True GGX transmission
- [ ] fixed-size Medium Stack
- [ ] Photon Differential prototype

## A.4 1.4

- [ ] Production anisotropic gather
- [ ] Auto Photon Culling
- [ ] PT Need feedback
- [ ] GPU Camera Path Guiding prototype

## A.5 1.5+

- [ ] ReSTIR candidate/reuse prototype [Experimental]
- [ ] estimator-neutral temporal/spatial reuse validation
- [ ] neural kernel experiment
- [ ] PathVertex common structure
- [ ] Light Tracing
- [ ] BDPT
- [ ] VCM

---

# 附录 B：A/B 测试必须保留的基线

任何优化都至少保留以下模式：

```text
Baseline 0: Cycles PT
Baseline 1: Caustica 1.1 current SPPM heuristics
Test 1: Guiding only
Test 2: Guiding + Adaptive Radius
Test 3: Guiding + Anisotropic Kernel
Test 4: Full proposed
```

所有对比必须尽量同时报告：

```text
equal photons
equal time
equal visual quality
```

只报其中一种会产生误导。

---


# 专章：Vulkan 应该放在哪里

## 1. 结论先行

Vulkan **应该加入 Caustica 的完整技术路线**，但它不是“减少 SPPM 白点”的算法模块，而是一个新的 **GPU Device Backend / Hardware Ray Tracing Backend**。

最合理的定位不是：

```text
OptiX → 全部替换成 Vulkan
```

而是：

```text
                         Caustica Light Transport
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
             NVIDIA OptiX Backend       Vulkan RT Backend
                    │                           │
            RTX / CUDA / OptiX          NVIDIA / AMD / future
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                         Guided Adaptive SPPM
                                  │
                      future BDPT / VCM / ReSTIR
```

因此 Vulkan 是 **平台与架构扩展路线**，而 Guided / Adaptive / Anisotropic SPPM 是 **采样与重建质量路线**。两者应并行推进，但优先级不同。

---

## 2. 为什么现在 Caustica 不能简单“切 Vulkan”

Caustica 1.1 当前真正快的核心是：

```text
Cycles Scene
    ↓
Cycles BVH / scene_intersect
    ↓
独立 OptiX photon pipeline
    ↓
CUDA stream 并发 photon trace
    ↓
GPU binning
    ↓
SPPM gather
```

当前 photon tracer 并不是一个独立、后端无关的渲染器，而是深度寄生在 Cycles GPU 内核和 OptiX device implementation 上。

如果把 OptiX 直接改成 Vulkan，需要重新解决至少以下能力：

1. Vulkan instance / physical device / logical device 管理；
2. `VK_KHR_acceleration_structure`；
3. `VK_KHR_ray_tracing_pipeline` 或 Ray Query；
4. BLAS / TLAS 构建与更新；
5. Shader Binding Table；
6. SPIR-V shader 编译；
7. Descriptor / resource table；
8. Buffer / image / texture binding；
9. synchronization2 / timeline semaphore / barriers；
10. Vulkan command buffer 与 Blender/Cycles 调度协调；
11. photon buffers 与 Cycles film buffer 的数据共享；
12. 材质、纹理、实例、motion blur、light linking 的语义映射。

因此 Vulkan Backend 本质是一个 **设备后端工程**，不是把几行 `scene_intersect()` 换成 Vulkan API。

---

## 3. Blender 的 Vulkan 与 Cycles Vulkan 是两件事

这一点必须特别说明。

Blender 自身目前已经有 Vulkan Graphics Backend，用于 UI、Viewport、EEVEE 等 GPU 图形层；但这并不意味着 Cycles 已经拥有 Vulkan Ray Tracing backend。

当前 Cycles 的生产 GPU 路线主要仍是：

```text
NVIDIA  → CUDA / OptiX
AMD     → HIP / HIP RT
Intel   → oneAPI / Embree GPU
Apple   → Metal
```

因此：

```text
Blender Vulkan Backend
≠
Cycles Vulkan Path Tracing Backend
```

也就是说，不能因为 Blender 可以用 Vulkan 显示界面，就直接让 Caustica photon tracer 调用 Blender 的 Vulkan backend 完成光追。

---

## 4. Lumen 对 Vulkan 路线的真正价值

`yuphin/Lumen` 的重要性不只是它有 SPPM / BDPT / VCM，而是它已经证明：

```text
Vulkan RT
+
GPU Path Tracing
+
SPPM
+
BDPT
+
VCM
+
ReSTIR
```

可以在同一个轻量 GPU 框架里运行。

Lumen 当前公开实现包括：

- Path Tracing + MIS；
- BDPT；
- SPPM；
- VCM；
- PSSMLT；
- VCMMLT；
- ReSTIR；
- ReSTIR GI；
- ReSTIR PT / GRIS；
- DDGI；
- Vulkan Render Graph；
- SPIR-V reflection；
- Vulkan ray tracing hardware acceleration。

它的硬件目标包括 Turing+ 与 RDNA2 级 GPU。

因此 Lumen 最适合成为：

> **Caustica Vulkan Backend 与未来多 Integrator 的参考实现 / donor architecture。**

而不是把整个 Lumen scene/material system 原样搬进 Blender。

---

# 5. 推荐的 Vulkan 三阶段路线

## Stage V0：先做 Device Abstraction，不改实际后端

这是最重要的一步。

当前不要先写 Vulkan，而要把 Caustica 里面 OptiX 专属逻辑抽象出来。

现在可能类似：

```text
PhotonMap
   ↓
OptiXDevice
   ↓
PIP_PHOTON
   ↓
photon_trace
```

改成：

```text
PhotonMap
   ↓
PhotonDeviceBackend
   │
   ├── PhotonOptiXBackend
   └── PhotonVulkanBackend   [future]
```

建议接口：

```cpp
class PhotonDeviceBackend {
public:
    virtual bool supported() const = 0;

    virtual void upload_scene(const PhotonSceneData &) = 0;
    virtual void upload_guiding(const PhotonGuideData &) = 0;

    virtual void trace(const PhotonTraceParams &) = 0;
    virtual void bin(const PhotonBinParams &) = 0;
    virtual void gather(const PhotonGatherParams &) = 0;

    virtual PhotonStatistics statistics() = 0;
};
```

这一阶段实际仍只运行：

```text
PhotonOptiXBackend
```

但以后 Vulkan 不需要重写 PhotonMap 的 host orchestration。

### 解决什么

- 防止未来所有 Guiding / SPPM 功能继续和 OptiX 硬耦合；
- 为 AMD/跨平台铺路；
- 为未来 BDPT / VCM integrator 统一 device interface；
- 降低 Vulkan 分支长期维护成本。

### 与现在不同

**现在：** Photon subsystem 知道 OptiX pipeline、CUDA stream 等细节。

**改后：** Photon subsystem 只知道“发射 / 分桶 / gather”；设备细节隐藏到 backend。

---

# 6. Stage V1：Vulkan Photon Backend Only

第一版 Vulkan **不要移植整个 Cycles**。

只移植：

```text
Photon Emission
Photon Trace
Photon Deposit
Photon Binning
Photon Guiding
```

相机 Path Tracing 仍然使用原始 Cycles backend。

即：

```text
                         Camera Paths
                             │
                        Cycles Device
                  OptiX / HIP / oneAPI / Metal
                             │
                             ↓
                          Film


                         Light Paths
                             │
                   Caustica Photon Backend
                    ┌────────┴────────┐
                  OptiX            Vulkan RT
                    │                 │
                    └────────┬────────┘
                             ↓
                         Photon Grid
                             ↓
                         SPPM Gather
```

这是最现实的 Vulkan 第一步。

---

# 7. Vulkan Photon Backend 的最大难点：场景与材质

Caustica 当前有一个非常重要的优势：

```text
photon_trace
↓
Cycles scene_intersect
```

因此实例、motion blur、BVH 等全部天然与 Cycles 一致。

Vulkan Photon Backend 如果使用自己的 TLAS/BLAS，就会产生：

```text
Cycles Scene
     ↓
     ├── Cycles BVH
     └── Vulkan BLAS/TLAS
```

即 **双份场景加速结构**。

这会带来：

- VRAM 增加；
- scene update 成本；
- animation synchronization；
- instance transform 同步；
- motion blur 数据同步；
- displacement / subdivision 一致性；
- generated geometry 一致性。

所以 Vulkan Backend 必须明确为：

> 用 Caustica `extract_scene()` 得到的 **紧凑 Photon Scene** 建立 Vulkan RT scene，不能重新做 Blender scene importer。

---

# 8. 推荐 Photon Scene 数据层

增加：

```text
PhotonScene
│
├── GeometryTable
├── InstanceTable
├── MaterialTable
├── TextureTable
├── LightTable
├── CasterTable
├── MediumTable
└── LightLinkTable
```

Host 端一次生成：

```text
Cycles Scene
     ↓
PhotonSceneBuilder
     ↓
Canonical PhotonScene
     │
     ├── OptiX upload
     └── Vulkan upload
```

这是 Vulkan 能否长期维护的关键。

---

# 9. Fast Material 与 Accurate Material 必须重新设计

这里是 Vulkan 路线最大的技术风险。

Caustica 现在有：

```text
Fast classified material
        │
        └── 不确定 → Accurate Cycles shader fallback
```

也就是说，复杂材质可以直接在 photon hit 上跑真实 Cycles surface shader。

但是 Vulkan shader 无法直接调用 CUDA/OptiX 里的 Cycles shader evaluator。

因此第一版 Vulkan 只能支持：

```text
Diffuse
Glass
GGX Reflection
GGX Transmission
Metal
Coat
Transparent
Emission
Volume Absorption
```

再加有限纹理：

```text
Base Color
Roughness
Metallic
Transmission
IOR
Normal
Alpha
```

复杂 shader：

```text
Vulkan Unsupported
↓
自动回退 OptiX Photon Backend
```

而不是静默近似。

### 与现在不同

**OptiX：** 可直接使用 Cycles accurate fallback。

**Vulkan V1：** 只能运行 canonical PhotonMaterial。

### 解决方案

长期再做：

```text
Cycles Shader IR
      ↓
Photon Shader IR
      ↓
     ├─ CUDA / OptiX
     └─ SPIR-V / Vulkan
```

这才有可能实现跨后端一致的 Accurate Mode。

---

# 10. Vulkan 应该使用 Ray Pipeline 还是 Ray Query

建议分功能。

## Photon Trace

优先：

```text
VK_KHR_ray_tracing_pipeline
```

原因：

- 一个 photon 对应一个 raygen invocation；
- 与当前 `__raygen__kernel_optix_photon_trace` 思路接近；
- 支持 hit / miss shader；
- 方便递归状态由 loop 自己维护；
- 与 Lumen 架构更接近。

## Culling / visibility / connection ray

可考虑：

```text
Ray Query
```

例如未来：

```text
BDPT visibility
VCM connection
shadow / camera connection
```

用 compute shader + ray query 可能更灵活。

所以最终可能是混合：

```text
Photon Walk        → RT Pipeline
BDPT Connections   → Ray Query
Spatial Pass       → Compute
Guiding            → Compute
Binning            → Compute
Gather             → Compute
```

---

# 11. Vulkan 下 GPU Guiding 反而非常合适

前面的 Guided SPPM 路线几乎天然适合 Vulkan Compute。

例如：

```text
photon_guide_clear          Compute
photon_guide_accumulate     Compute / atomics
photon_guide_reduce         Compute
photon_guide_mipmap         Compute
photon_guide_budget         Compute
```

Directional Guiding：

```text
64×64 histogram
↓
GPU mip pyramid
↓
quadtree traversal sampling
```

完全可以使用 Vulkan storage buffer / storage image 实现。

因此：

> Vulkan 不会改变 Guided SPPM 算法设计，只改变 kernel 调度和资源绑定方式。

---

# 12. Vulkan 下的 Photon Binning

当前 Caustica：

```text
photon_bin_count
↓
CPU prefix
↓
photon_bin_scatter
```

Vulkan 版不建议把 prefix sum 回 CPU。

应该进一步升级为：

```text
Photon Deposit
     ↓
GPU Count
     ↓
GPU Parallel Prefix Scan
     ↓
GPU Scatter
```

从而：

```text
0 PCIe synchronization
```

这甚至可能反过来优化 OptiX 版本。

建议最终把 GPU scan 写成后端无关算法：

```text
DeviceScan
├── CUDA implementation
└── Vulkan compute implementation
```

### 解决什么

- 去掉当前 cell prefix host round-trip；
- 降低 batch latency；
- 更适合大量小 photon generations；
- 为实时 viewport ReSTIR photon reuse 减少同步。

---

# 13. Vulkan 的真正性能目标不是比 OptiX 快

第一版 Vulkan 不应该设目标：

```text
Vulkan > OptiX
```

这非常不现实。

NVIDIA 上 OptiX 对 RTX hardware、driver 和 BVH 的集成非常成熟。

正确目标是：

```text
NVIDIA:
OptiX = fastest / reference
Vulkan = portable

AMD:
Vulkan RT = native Caustica path

future:
other Vulkan RT GPUs
```

工程验收建议：

```text
NVIDIA Vulkan photon throughput
>= OptiX 的 70~85%
```

第一阶段已经可以接受。

AMD 的目标则是：

```text
从“不支持”
→
“可生产使用”
```

这本身就是巨大提升。

以上百分比是工程目标，不是已有实测结论。

---

# 14. Vulkan 会不会让 SPPM 白点减少？

**不会直接减少。**

同样算法、同样随机数、同样 PDF：

```text
OptiX SPPM
≈
Vulkan SPPM
```

白点来自：

- photon density；
- emission PDF；
- high-weight outlier；
- radius；
- kernel；
- BSDF；
- medium；
- numerical robustness。

不是来自 OptiX 还是 Vulkan。

真正减少白点的是：

```text
Directional Guiding
Camera-aware feedback
Defensive PDF
Adaptive Radius
Anisotropic Kernel
GGX transmission
Medium Stack
```

Vulkan 只是让这些算法可以运行在更多 GPU 上。

---

# 15. Vulkan 能不能让速度更快？

分情况。

## NVIDIA

不应预期 Vulkan RT 自动比 OptiX 快。

更可能：

```text
OptiX >= Vulkan
```

尤其是复杂 Cycles scene、motion blur、curve 等情况下。

## AMD

现在 Caustica GPU photon backend 基本没有对应高性能实现，因此 Vulkan 可以从：

```text
不可用
```

提升到：

```text
hardware RT photon mapping
```

收益巨大。

## 算法调度层

Vulkan Render Graph / compute pipeline 的优势可能体现在：

- guiding；
- GPU scan；
- reservoir reuse；
- spatial hash；
- BDPT connections；
- VCM merge；
- ReSTIR passes。

因此长期 Vulkan 对 **复杂多 pass integrator** 的结构价值可能比单纯 Photon Trace 更大。

---

# 16. 为什么 Vulkan 对未来 VCM / ReSTIR 特别重要

SPPM 当前主要是：

```text
Trace
→ Bin
→ Gather
```

而未来：

```text
BDPT
VCM
ReSTIR DI
ReSTIR GI
ReSTIR PT
ReSTIR FG
```

会需要大量：

```text
Compute Pass
Ray Query
Buffer Compaction
Prefix Scan
Reservoir Update
Temporal Pass
Spatial Pass
Connection Visibility
```

这与 Vulkan 的 Render Graph / explicit synchronization 模型非常匹配。

Lumen 的意义也正是在这里：它已经把许多不同 integrator 组织在同一个 Vulkan RT + Render Graph 环境中。

因此长期架构可以变成：

```text
Caustica Light Transport Graph
│
├── PT
├── Photon Trace
├── Guiding
├── SPPM Merge
├── BDPT Connection
├── VCM Merge
├── ReSTIR Temporal
└── ReSTIR Spatial
         │
         ↓
Device Graph Backend
├── CUDA / OptiX
└── Vulkan
```

---

# 17. Vulkan 与 HIP RT 的关系

这是一个现实问题。

如果目标只是：

> “让 Caustica 支持 AMD”

那么还有两条路线：

```text
A. 直接扩展现有 Cycles HIP RT backend
B. 新写 Vulkan RT backend
```

### HIP RT 优点

- 已经属于 Cycles 正式 GPU 体系；
- scene / shader / kernel infrastructure 更容易复用；
- 不需要维护第二份 scene acceleration structure；
- 与 Cycles feature parity 更自然。

### Vulkan 优点

- 跨厂商；
- 可借鉴 Lumen；
- 对未来独立 BDPT/VCM/ReSTIR graph 更自由；
- 不依赖 CUDA/HIP 编程生态；
- 一套 API 可以覆盖 NVIDIA + AMD。

因此如果目标是 **近期让 Caustica 支持 AMD**：

> 优先研究 HIP RT photon kernel 是否能复用当前 `ccl_gpu_kernel_call`。

如果目标是 **长期打造自己的跨平台高级 light transport backend**：

> Vulkan 更有战略价值。

---

# 18. 审核后的 Vulkan 优先级

最终建议：

| 优先级 | 项目 | 说明 |
|---|---|---|
| P0 | Guided SPPM | 直接解决当前焦散质量/效率 |
| P0 | Adaptive / ESS | 直接解决白点与 radius |
| P1 | GGX / Medium | 保证物理正确 |
| **P1** | **Device Abstraction** | 为 Vulkan/HIP RT 做准备 |
| P2 | Photon Differential | 提高 kernel reconstruction |
| **P2** | **Vulkan Photon Prototype** | 独立 photon trace backend |
| P2 | HIP RT Photon Backend 调研 | 快速 AMD 支持路线 |
| P3 | Vulkan GPU Binning/Scan | 全 GPU pipeline |
| P3 | Vulkan Viewport ReSTIR FG | 实时多 pass |
| P4 | Vulkan BDPT/VCM | 2.0 架构 |

所以 Vulkan **不是不要做**，而是不能把它放在 Guided SPPM 前面。

---

# 19. 推荐版本规划：加入 Vulkan 后

## Caustica 1.2 — Guided SPPM

```text
Photon Statistics
Light / Caster / Direction Guiding
Camera-aware Feedback
Defensive Sampling
Exact PDF
```

Backend：

```text
OptiX only
```

---

## Caustica 1.3 — Adaptive & Physical

```text
ESS
Adaptive Radius
True GGX Transmission
Medium Stack
Anisotropic kernel prototype
```

同时开始：

```text
PhotonDeviceBackend abstraction
```

---

## Caustica 1.4 — Portable Photon Core

```text
Canonical PhotonScene
Canonical PhotonMaterial
Device-independent guide/bin/gather interfaces
```

Backend：

```text
OptiX reference
HIP RT research
Vulkan prototype
```

---

## Caustica 1.5 — Vulkan Photon

目标：

```text
Vulkan RT photon emission
Vulkan RT photon walk
Vulkan compute guiding
Vulkan compute binning
Vulkan compute prefix scan
```

功能先限制：

```text
Fast PhotonMaterial subset
```

复杂材质自动报告 unsupported/fallback。

---

## Caustica 1.6 — Cross-platform Guided SPPM

```text
NVIDIA → OptiX / Vulkan
AMD    → Vulkan / HIP RT
```

同一：

```text
PhotonScene
Guiding
SPPM estimator
Statistics
AOV
```

进行 cross-backend consistency test。

---

## Caustica 2.0 — Light Transport Graph

```text
PT
+
Light Tracing
+
BDPT
+
SPPM
+
VCM
+
ReSTIR
```

Device：

```text
OptiX
Vulkan RT
```

这时候 Vulkan 才真正成为 Caustica 的核心架构之一。

---

# 20. Vulkan 新增名词解释

### Vulkan
Khronos 制定的低层跨平台图形与计算 API。与 OpenGL 相比，它把资源、同步、内存和执行调度更明确地交给应用控制。

### Vulkan RT / Vulkan Ray Tracing
Vulkan 的硬件光线追踪扩展体系，主要包括 acceleration structure、ray tracing pipeline、ray query 等能力。

### SPIR-V
Vulkan shader 使用的中间表示。CUDA kernel 不能直接作为 Vulkan shader 使用，需要把对应算法编译/改写为 SPIR-V 可执行 shader。

### BLAS
Bottom Level Acceleration Structure。通常对应单个 mesh/geometry 的光追加速结构。

### TLAS
Top Level Acceleration Structure。组织多个实例及其 transform，并引用 BLAS。

### SBT
Shader Binding Table。Vulkan/RTX 风格 ray tracing pipeline 用于把几何实例与 hit shader 等程序绑定起来的数据表。

### Raygen Shader
光追 pipeline 的发射入口。Caustica 的“一线程一个 photon”很适合映射为一个 raygen invocation 对应一个 photon。

### Closest Hit Shader
射线命中最近表面后执行的 shader。可以读取材质、法线、UV 等命中数据。

### Miss Shader
射线离开场景没有命中几何时执行。

### Ray Query
允许 compute/fragment 等普通 shader 主动执行光线求交，而无需完整 RT pipeline。适合 visibility ray、连接测试等。

### Descriptor Set
Vulkan shader 访问 buffer、texture、sampler 等资源的绑定机制。

### Push Constant
非常小、更新频繁的 shader 参数通道，适合 batch id、photon generation parameters 等。

### Command Buffer
记录 GPU 命令的对象，例如 trace rays、dispatch compute、barrier 等，然后提交 GPU queue 执行。

### Queue
Vulkan GPU 工作提交队列。可以有 graphics / compute / transfer 等不同能力队列。

### Synchronization2
Vulkan 较新的显式同步 API，用于描述不同 pass 之间的资源依赖和 pipeline barrier。

### Timeline Semaphore
带递增计数值的 GPU 同步对象，适合异步 batch、render graph、多 queue 协调。

### Render Graph
把渲染/计算过程描述为多个 pass 和资源依赖，由系统自动安排 barrier、资源生命周期与顺序。未来 ReSTIR/VCM 很适合这种结构。

### Device Abstraction
把算法与 CUDA/OptiX/Vulkan/HIP 等具体 API 分离的接口层。

### Canonical PhotonScene
本文建议新增的统一光子场景描述。OptiX、Vulkan 等 backend 都从同一数据源构建设备端 scene，避免每个 backend 自己解释 Blender 场景。

### Backend Parity
不同设备 backend 在相同 seed、相同参数下应获得统计上一致的结果，而不是完全逐 bit 一致。

---

# 21. Vulkan 路线最终结论

Vulkan 必须纳入长期方案，但它解决的是：

```text
跨平台
AMD 支持
未来高级 integrator 架构
GPU 多 pass 调度
```

它**不直接解决**：

```text
SPPM 白点
半径过早收缩
high-weight photon
焦散 blur
GGX transmission 错误
medium nesting 错误
```

这些仍然由 Guided Adaptive SPPM 解决。

所以正确顺序是：

```text
先把算法做对
        ↓
Guided Adaptive SPPM
        ↓
把设备边界抽象干净
        ↓
OptiX / HIP RT / Vulkan
        ↓
再扩展 BDPT / VCM / ReSTIR
```

而不是：

```text
先 Vulkan 重构
↓
然后再解决现有 SPPM 问题
```

后者风险更高，而且短期几乎不会改善当前实际渲染质量。


# 附录 C：参考文献

1. Hachisuka, T.; Jensen, H. W. **Stochastic Progressive Photon Mapping**. ACM Transactions on Graphics, 2009. DOI: 10.1145/1618452.1618487.
2. Grittmann, P.; Pérard-Gayot, A.; Slusallek, P.; Křivánek, J. **Efficient Caustic Rendering with Lightweight Photon Mapping**. Computer Graphics Forum / EGSR, 2018.
3. Kern, R.; Brüll, F.; Grosch, T. **Accelerating Photon Mapping for Hardware-Based Ray Tracing**. Journal of Computer Graphics Techniques, 2023.
4. Kern, R.; Brüll, F.; Grosch, T. **ReSTIR FG: Real-Time Reservoir Resampled Photon Final Gathering**. EGSR, 2024. Public implementation: `TU-Clausthal-Rendering/ReSTIR-FG`.
5. Lin, Z.; Hu, C.; Jia, J.; Li, S. **Hypothesis Testing for Progressive Kernel Estimation and VCM Framework**. 2025, arXiv:2504.04411.
6. De Barro, A.; Bugeja, K.; Spina, S. **ARPΔ: Accelerated Ray-Tracing Photon Differentials for Real-Time Global Illumination with Combined Specular and Diffuse Solutions**. The Visual Computer, 2025. DOI: 10.1007/s00371-025-03898-6.
7. Kern, R.; Brüll, F.; Kastning, J.; Grosch, T. **Guided ReSTIR FG+ Photon Resampling for Large Scenes and Many Lights**. EGSR 2026. DOI: 10.2312/SR.20261001. Public implementation: `TU-Clausthal-Rendering/Guided-ReSTIR-FG-Plus`.
8. Benoist, J.; Litalien, J.; Gruson, A. **Neural Progressive Photon Mapping**. Computer Graphics Forum / Eurographics 2026. DOI: 10.1111/cgf.70408.
9. Frisvad, J. R.; Schjøth, L.; Erleben, K.; Sporring, J. **Photon Differential Splatting for Rendering Caustics**. Computer Graphics Forum, 2014.
10. NVIDIA. **Generating Ray-Traced Caustic Effects in Unreal Engine 4, Part 1** — Adaptive Anisotropic Photon Scattering / Photon Differentials.
11. Yuphin. **Lumen: A Vulkan Raytracing Framework for Various Bidirectional Path Tracing Techniques**. GitHub, accessed 2026-08-31.
12. Blender Developer Documentation. **Cycles GPU Backends / Kernel Scheduling / Render Scheduling**. Accessed 2026-08-31.
13. Blender Developer Documentation. **Vulkan Backend** — Blender graphics/viewport backend documentation. Accessed 2026-08-31.

---

# 附录 D：Caustica 1.1 源码基线说明

本文关于 Caustica 1.1 当前实现的描述基于用户提供的《Caustica 1.1 技术分析报告》，其关键基线包括：

- 独立 OptiX photon pipeline；
- device-resident photon binning；
- pilot light budget；
- caster target guiding；
- PCG32 deterministic RNG；
- SPPM N/r²/tau；
- current heuristic mask；
- accurate shader fallback；
- current radius floor；
- rough transmission 与 medium 的现有限制。

后续真正进入代码实现前，仍建议以对应 Caustica GPL 源码版本做逐函数验证，因为静态分析报告描述的是当前版本行为，而实际开发分支可能继续变化。

---

# 附录 E：最终审核补全——V2 中遗漏或需要加强的关键项

> 本附录不是推翻前述路线，而是补齐“真正做成生产级 Blender/Cycles 渲染分支”时不可缺少的工程闭环。
> 其中 [必须] 表示不解决就可能导致后端不可用、结果不一致或性能严重倒退；[建议] 表示可以后续迭代。

## E.1 [必须] Cross-Backend Interop：OptiX/CUDA 与 Vulkan 如何共存

### 当前遗漏

V2 提出“Camera Path 继续 Cycles，Photon Path 先做 Vulkan Backend”，但没有完整解决：

```text
Cycles Camera Path
    ↓
CUDA / OptiX device data

Vulkan Photon Path
    ↓
Vulkan buffers / Vulkan AS
```

两套 API 默认不能直接共享：

- Cycles 内部 BVH / OptiX GAS/IAS 不能直接作为 Vulkan TLAS/BLAS 使用；
- CUDA device pointer 不能默认当 Vulkan buffer 使用；
- Vulkan AS 也不能被 OptiX `scene_intersect` 直接遍历；
- 如果各自复制 geometry / texture / material，会显著增加 VRAM；
- 如果每批 photon 都经过 CPU/PCIe 搬运，可能完全抵消 GPU Photon Mapping 的优势。

### 正确方案

Stage V0 必须增加一层明确的 **Interop / Ownership Policy**：

```text
Canonical Scene Data
       │
       ├── Cycles/OptiX representation
       │
       └── Vulkan representation
```

允许两种实现路线：

**路线 A：双份 AS，静态共享原始几何数据**

- Geometry/instance/material canonical buffers 为唯一源；
- OptiX 和 Vulkan 各自构建 AS；
- Photon results 只在代结束时以 compact buffer / film contribution 交换；
- 优点：简单、稳定；
- 缺点：VRAM 更高，动态场景更新两遍。

**路线 B：External Memory / External Semaphore Interop**

- 能共享的线性 buffer 使用 external memory；
- CUDA/Vulkan 使用 external semaphore 或 timeline 语义同步；
- AS 仍通常各自构建；
- 优点：减少线性数据重复；
- 缺点：跨厂商、跨驱动复杂，必须 capability-gated。

### 验收

必须记录：

- duplicated geometry VRAM；
- duplicated texture/material VRAM；
- AS VRAM；
- interop copy MB/frame；
- interop synchronization ms；
- photon result transfer ms。

若 Vulkan Photon Backend 导致典型产品场景总 VRAM > OptiX 版 1.5×，必须重新评估默认启用策略。

---

## E.2 [必须] Acceleration Structure 生命周期：Build / Refit / Compaction / Update

V2 只解释 BLAS/TLAS 名词，但生产级 Blender 场景必须明确 AS 生命周期。

需要覆盖：

```text
Static Mesh
Transform-only animation
Deforming Mesh
Subdivision / Displacement result
Instancing
Geometry add/remove
Material-only change
```

建议策略：

| 变化 | Vulkan Photon AS 策略 |
|---|---|
| 相机变化 | 不重建 |
| 灯光变化 | 不重建 geometry AS |
| instance transform | TLAS update/refit |
| mesh deformation | BLAS update 或 rebuild |
| topology change | rebuild |
| 静态完成后 | compaction |

注意：Vulkan KHR RT 的 acceleration structure build/update 是显式资源管理，不能假设像 Cycles `scene_intersect` 一样自动继承全部更新逻辑。

---

## E.3 [必须] Motion Blur 是跨平台 Vulkan 路线中的特殊问题

当前 Caustica 复用 Cycles `scene_intersect`，因此可以天然继承 Cycles 的运动模糊语义。

Vulkan KHR Ray Tracing 的核心 AS 并不等价于完整的跨厂商 motion-blur ray tracing。现有 motion-instance 支持仍存在厂商扩展路径，因此 Vulkan V1 不应承诺与 Cycles 完全等价的 motion blur。

建议 Capability Matrix：

```text
VulkanPhotonFeature.motion_blur
VulkanPhotonFeature.curves
VulkanPhotonFeature.procedural_geometry
VulkanPhotonFeature.volumes
```

不支持时必须：

- UI 明确报告；
- 自动回退 OptiX/HIP/CPU Photon Path；
- 禁止静默用静态 geometry 代替动画 geometry。

---

## E.4 [必须] Geometry Coverage：不能只讨论 Mesh

当前 Caustica 已知 Photon Path 主要针对三角形图元，头发/粒子等存在限制。

Vulkan 版必须明确支持矩阵：

- Triangles；
- Instances；
- Curves / Hair；
- Point Cloud；
- Procedural/AABB primitives；
- Subdivision/Displacement 的最终几何；
- Volume boundary；
- Alpha cutout / transparent traversal。

KHR acceleration structures 原生主要处理 triangles 与 AABBs；非三角几何需要自定义 intersection 或预转换。

第一版建议：

```text
Required: triangles + instances + alpha cutout
Optional: procedural AABB
Fallback: hair / curves / special primitives
```

---

## E.5 [必须] Material ABI：不是“列出几个 BSDF”就够了

V2 已指出 Vulkan 无法直接调用 Cycles shader evaluator，但还需要更明确的 ABI。

建议建立：

```cpp
PhotonClosureSet {
    Closure closures[MAX_PHOTON_CLOSURES];
    uint count;
    uint flags;
}

Closure {
    type;
    weight;
    roughness;
    eta;
    anisotropy;
    normal;
    coat;
    transmission;
}
```

Canonical Material Compiler 负责：

```text
Cycles Shader Graph
     ↓
Photon-compatible closure IR
     ↓
OptiX Photon Evaluator
Vulkan Photon Evaluator
HIP Photon Evaluator
```

真正目标不是让 Vulkan 复制 Cycles 所有节点，而是让所有 Photon Backend 对同一个 **Photon Closure IR** 给出一致结果。

必须验证：

- eval() 一致；
- sample() 一致；
- PDF 一致；
- delta/non-delta 分类一致；
- eta/adjoint 语义一致；
- normal mapping 语义一致。

---

## E.6 [必须] Texture / Node-driven Parameters

Caustica 1.1 已存在纹理驱动 IOR、roughness、metallic、transmission 等难以快速分类的问题。

Guided SPPM + Vulkan 后，如果仍然只使用常量近似，会让 Guiding 更快地收敛到“错误的光子分布”。

因此需要三档：

```text
FAST
  常量/均值材质

TEXTURED
  支持关键 Image Texture + UV + mapping

ACCURATE
  回退真实 Cycles closure evaluation 或 backend-specific full evaluator
```

至少应该优先支持纹理驱动：

- Roughness；
- Transmission；
- IOR；
- Normal/Bump；
- Base Color；
- Metallic；
- Alpha。

---

## E.7 [必须] Light Sampling Coverage / Light Tree

V2 对 Light Guiding 讲得较多，但缺少“所有 Blender 光源类型如何进入统一发射 PDF”。

需要完整覆盖：

- Point；
- Spot；
- Area rectangle/disk/ellipse；
- Sun；
- Mesh emission；
- HDRI / World；
- IES；
- Light Linking；
- Light Groups。

建议建立统一：

```text
P(light)
× P(emission position | light)
× P(emission direction | light, position)
× P(target | light)
× P(guide direction | light,target)
```

所有因素必须能还原最终 emission PDF，否则 Guided SPPM 会出现亮度随 Guide 改变的问题。

大规模 many-light 场景未来应考虑接入 Cycles Light Tree / hierarchical light sampling，而不是无限扩大 flat pilot table。

---

## E.8 [必须] Guiding 的“信用分配”要定义清楚

Camera-aware Guiding 不能只写“使用 gathered contribution”，必须规定哪个 photon 的哪个事件获得 credit。

建议唯一 credit key：

```text
(light_id,
 target_id,
 emission_bin,
 path_class,
 receiver_cluster)
```

credit 来源：

```text
actual photon contribution to film
× optional variance need
```

必须防止：

- 一个 photon 被多个像素 gather 后重复训练导致 guide 过度放大；
- 高 exposure / tone mapping 影响训练；
- denoised image 反向影响物理采样；
- Light Group/AOV 分支重复 credit。

训练必须使用 **linear scene-referred radiance**，不能使用显示变换后的像素。

---

## E.9 [必须] Adaptive / Anisotropic Kernel 的归一化与一致性证明

这是 V2 中最大的“理论完整性”遗漏之一。

当 kernel 从圆形变成椭圆，或者 radius 由统计自适应变化时，不能只换距离函数。

必须保证：

```text
∫ K(x; Σ) dx = 1
```

以及椭圆变换的 Jacobian 正确进入 density normalization。

如果：

```text
Σ → 很扁
```

必须设置 eigenvalue floor / max eccentricity，否则会产生无限高 density estimate。

Final 模式要明确：

- radius schedule 满足一致性条件；
- adaptive decision 不能把 radius 永久冻结；
- guide 不改变 estimator target；
- anisotropic kernel 面积进入 flux normalization。

否则“更锐利”可能来自隐藏 bias。

---

## E.10 [必须] Numerical Robustness / HDR Accumulation

Guiding + SPPM 会增加大量：

- atomic accumulation；
- second moment；
- histogram；
- PDF ratio；
- covariance。

必须防止：

- uint histogram overflow；
- float atomic 精度损失；
- NaN / Inf propagation；
- half differential underflow；
- very small PDF；
- huge HDR emissive lights。

建议：

```text
Guide training value → robust/log/sqrt transform
Estimator flux → full precision
Histogram → saturating or rescaled integer
Statistics reduction → FP32 minimum, selective FP64 CPU reference
PDF floor → only through defensive mixture, not arbitrary estimator clamp
```

必须增加 NaN/Inf Debug Counter。

---

## E.11 [必须] Russian Roulette / Bounce Budget 与 Guiding 的关系

当前 Photon Trace 有固定 bounce 上限。

未来加入 Guiding 后必须明确：

- RR probability 基于 throughput，而不是 guide confidence；
- guide 只改变 proposal，不改变 path survival physics；
- transmission eta scaling 后再计算合理 RR；
- Light Path 的最大深度与 Camera Path 的 path partition 规则一致。

否则会出现 guide on/off 能量差。

---

## E.12 [必须] Multi-GPU 方案

当前 Caustica 多 GPU 已是弱项；Vulkan 并不会自动解决。

需要明确多 GPU Photon 策略：

### 推荐第一版

```text
每 GPU 独立 trace photon subset
每 GPU 独立 bin/gather partial
CPU/主 GPU 合并 tau/N/statistics
```

### 后续

- replicated guide；
- per-batch histogram reduction；
- peer-to-peer 可用时走 P2P；
- 无 P2P 时避免传完整 photon array，只传 compact statistics / film partials。

Guiding training 必须保证 GPU 数量变化不导致明显亮度差或动画闪烁。

---

## E.13 [必须] Denoiser / Adaptive Sampling / AOV 的边界

Caustica 已与 Cycles combined pass、自适应采样、Caustics AOV 共存。

新系统必须规定：

```text
Physical SPPM statistics
        │
        ├── Guide training
        ├── Caustics AOV
        └── Combined

Display smoothing / denoiser
        ↓
只能影响显示
```

禁止：

```text
Denoised/filtered result → Guide Training
```

否则 denoiser bias 会进入 sampling feedback loop。

同时自适应采样不能只根据普通 PT variance 退休仍在等待 photon 收敛的像素。

---

## E.14 [必须] Unified Physical History Invalidation

SPPM 累积、Guiding EMA 以及任何可选 temporal reuse 都属于物理历史状态，必须使用统一的 history invalidation 规则。

至少以下变化要 reset 或 partial invalidate：

- camera teleport / projection change；
- light transform/intensity；
- caster transform；
- material roughness/IOR/transmission；
- topology；
- world HDRI；
- exposure 不应 reset physical history；
- color management 不应改变 physical history。

否则会出现 ghost caustics / stale guide。

---

## E.15 [必须] Preview / Final Parity Test

这是新增的生产级硬门槛。任何新优化在进入默认链路前，都必须比较 Viewport 与 Final 的**同预算结果**。

### 测试方式

固定：

- 相同分辨率；
- 相同场景状态；
- 相同 camera spp；
- 相同 emitted photon count；
- 相同 Guiding 配置；
- 关闭 display-only denoise/smoothing。

比较：

```text
Viewport capture @ budget B
vs
Final render @ budget B
```

### 必须满足

1. Combined linear energy 近似一致；
2. Caustics AOV 近似一致；
3. Light Group photon energy 近似一致；
4. 焦散峰值位置没有系统偏移；
5. `N / τ / R / ESS` 的统计分布近似一致；
6. Guiding direction histogram 不因运行模式改变目标；
7. 不允许 Viewport-only clamp / kernel / radius floor；
8. 不允许 Final-only material path / BSDF 近似；
9. 差别应该主要表现为随机噪声，而不是结构或亮度变化。

建议把这一测试做成 CI regression，并为每个典型焦散场景保留 reference statistics。

---

## E.16 [建议] Shader Execution Reordering / Wavefront Coherence

2026 Vulkan 已出现跨厂商方向的 ray-tracing invocation reorder 扩展能力，但它不能作为基础依赖。

Photon Path 在多 bounce 后方向高度发散，未来可以把：

```text
Trace
→ hit classification
→ reorder by material / path state
→ shade
```

作为 NVIDIA/支持设备的高级优化。

必须是 Optional Fast Path：

```text
SER/Invocation Reorder available → ON
否则 → 正常 pipeline
```

不能让核心正确性依赖它。

---

## E.16 [建议] Spectral / Dispersion 路线

当前方案整体仍是 RGB Photon Transport。

对于：

- prism；
- crystal；
- diamond；
- chromatic caustics；
- wavelength-dependent IOR；

RGB 分通道近似不能完整表达真实波长相关折射。

长期可增加：

```text
Hero Wavelength Photon
or
Wavelength Packet Photon
```

Photon 数据增加 wavelength/lambda，并使用 `eta(lambda)`。

建议定位：

```text
Caustica 2.x Experimental Spectral Caustics
```

不是 1.x 主线。

---

## E.17 [建议] Volume Scattering / SSS Caustic Chain

当前 Caustica 主要支持 volume absorption，不完整支持 scattering/SSS 产生焦散链。

长期需要区分：

- surface photon map；
- volume photon / beam / path representation；
- SSS transport。

这属于新的 estimator，不建议塞进 1.2–1.4，但“完整路线图”必须记录为已知缺口。

---

## E.18 [必须] Backend Parity Test 不够，需要 Transport Correctness Test

除 OptiX vs Vulkan 图片对比外，必须增加单元/物理测试：

### BSDF

- white furnace；
- reciprocity/adjoint consistency；
- sample/eval/pdf matching；
- TIR；
- eta² transport；
- GGX reflection/transmission normalization。

### Photon estimator

- constant density field；
- kernel integral = 1；
- isotropic vs anisotropic equal-energy；
- guide ON/OFF asymptotic energy；
- radius schedule convergence。

### Backend

- same RNG seed；
- same scene；
- same emitted photon count；
- emitted energy parity；
- deposited energy parity；
- gathered energy parity。

图片“看起来一样”不能作为最终正确性标准。

---

## E.19 [必须] Vulkan Capability / Extension Matrix

Vulkan Photon Backend 不能只判断“支持 Vulkan 1.3”。

至少检测：

```text
VK_KHR_acceleration_structure
VK_KHR_ray_tracing_pipeline
VK_KHR_ray_query               (optional path)
VK_KHR_buffer_device_address
VK_KHR_deferred_host_operations
Synchronization2
Timeline Semaphore
Descriptor Indexing / bindless strategy
Subgroup capabilities
Optional invocation reorder
Optional vendor motion blur
```

并建立：

```cpp
PhotonBackendCapabilities
```

根据硬件选择：

```text
RT Pipeline
Ray Query
Compute fallback
Feature disable
Backend fallback
```

---

## E.20 [必须] Licensing / Third-party Boundary

Lumen 是很好的算法与 Vulkan 工程参考，但“参考设计”和“直接复制代码”要区分。

Caustica/Blender/Cycles 属于 GPL 体系；未来引入：

- Lumen；
- ReSTIR FG+；
- Falcor-derived code；
- NPPM；
- third-party Vulkan helpers；

都应在真正合并代码前做许可证兼容与 NOTICE 审核。

技术方案中建议保留：

```text
Algorithm Reference
Implementation Reference
Code Reuse Allowed?
Required Attribution
License
```

字段，避免后期工程完成后才处理授权问题。

---

# 附录 F：审核后的最终优先级（补全版）

## F.1 Caustica 1.2 — Sampling Correctness First

必须：

1. Photon Statistics；
2. Light/Caster/Directional Guiding；
3. Camera-aware credit assignment；
4. Defensive Mixture；
5. Exact PDF；
6. Numerical safeguards；
7. Guide/Estimator separation；
8. Debug AOV。

## F.2 Caustica 1.3 — Estimator + Optics

必须：

1. Unified Viewport/Final radius law；
2. Adaptive Radius；
3. Kernel normalization tests；
4. True GGX Transmission；
5. Medium Stack；
6. Texture-driven critical optical parameters。

## F.3 Caustica 1.4 — Anisotropic + Portable Core

必须：

1. Photon Differential；
2. Anisotropic Kernel；
3. Photon Closure IR；
4. Canonical PhotonScene；
5. Device Abstraction；
6. AS lifecycle model；
7. Backend capability matrix。

## F.4 Caustica 1.5 — Vulkan Photon Backend

必须：

1. Vulkan BLAS/TLAS；
2. scene synchronization；
3. backend-neutral material evaluation；
4. Vulkan photon trace；
5. Vulkan GPU bin/scan/gather；
6. interop/memory ownership；
7. Backend parity + transport correctness tests。

## F.5 Caustica 1.6 — Cross-platform / Multi-GPU / WYSIWYG Reuse

1. AMD validation；
2. multi-GPU photon partition；
3. temporal guiding；
4. ReSTIR candidate/reuse [Experimental until WYSIWYG parity]；
5. robust history invalidation；
6. optional invocation reordering。

## F.6 Caustica 2.0+

1. Light Tracing；
2. Lightweight BDPT；
3. Full BDPT；
4. VCM；
5. ReSTIR BDPT / GRIS；
6. spectral caustics；
7. volume/SSS advanced transport。

---

# 附录 G：最终审核结论

V2 的主方向正确，但要成为“完整开发方案”，必须补上三类此前低估的问题：

1. **Sampling/Estimator 正确性闭环**：credit assignment、kernel normalization、numerical robustness、RR 与 guiding 的独立性；
2. **Production Renderer 完整性**：材质 ABI、纹理参数、geometry coverage、motion blur、AOV/denoiser/history；
3. **Vulkan 工程现实**：OptiX/Vulkan interop、双份 AS/VRAM、AS 生命周期、capability matrix、multi-GPU。

因此最终架构不应理解为“给 Caustica 加一个 Vulkan raygen”，而应理解为：

```text
            Blender / Cycles
                  │
        Canonical PhotonScene
                  │
         Photon Closure IR
                  │
       Light Transport Core
                  │
       Guided Adaptive SPPM
                  │
     ┌────────────┼────────────┐
     │            │            │
   OptiX       Vulkan RT      HIP RT
     │            │            │
   NVIDIA     NVIDIA/AMD       AMD
```

而长期的 BDPT / VCM / ReSTIR 应建立在这个统一 Transport Core 上，而不是分别维护三套材质、场景和 PDF 逻辑。


---

# 附录 G：WYSIWYG 设计总则（V4 修订）

## G.1 一句话定义

> **Preview = Final Render 的低采样版本，而不是另一种渲染模式。**

## G.2 相同项

| 模块 | Viewport | Final |
|---|---|---|
| Integrator | Guided Adaptive SPPM | Guided Adaptive SPPM |
| Photon Guiding | 相同 | 相同 |
| Light/Caster/Direction PDF | 相同 | 相同 |
| BSDF/Medium | 相同 | 相同 |
| Measurement Point | 相同 | 相同 |
| Radius 更新 | 相同 | 相同 |
| ESS/Adaptive shrink | 相同 | 相同 |
| Anisotropic Kernel | 相同 | 相同 |
| Energy accounting | 相同 | 相同 |
| AOV / Light Group | 相同 | 相同 |
| Photon Count | 相同 | 相同 |
| Batch/index/seed 规则 | 相同 | 相同 |
| Caustica 物理预算配置 | 同一套 | 同一套 |

## G.3 允许不同项

| 模块 | Viewport | Final |
|---|---|---|
| Blender 样本设置 | Viewport Samples | Render Samples |
| Photon Count 设置 | 相同 | 相同 |
| 每 sample/work 的 batch 规则 | 相同 | 相同 |
| Caustica 专用预算层 | 不存在 | 不存在 |
| Refresh cadence | 高频 | 低频/最终输出 |
| Display denoise | 可开启 | 可开启/高质量 |
| 物理 estimator | **不得不同** | **不得不同** |

当 Viewport Samples 与 Render Samples 相等时，总 photon generations 与统计演化也必须相等。Viewport/Final 的宿主身份不得参与 Photon Count、batch size、Guide beta、radius shrink 或 kernel 的计算。

## G.4 最终产品体验

用户在 Rendered View 中看到：

```text
玻璃焦散的位置
焦散大体形状
相对亮度
颜色
反射/折射关系
```

F12 后应该保持这些关系，只是：

```text
噪声更少
边缘更稳定
细节更完整
```

而不是重新得到另一张“更正确但不同”的图。

这条原则应作为 Caustica 后续 Vulkan、ReSTIR、BDPT、VCM 等所有新模块的统一验收条件。
