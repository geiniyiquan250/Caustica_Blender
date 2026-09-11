# CyclesPlus 本次更新说明

# CyclesPlus Update Notes

> 关联提交 / Related commit：`ba91f58c23d`
>
> 状态 / Status：已推送 / Pushed

---

## English

This update normalizes the CyclesPlus downstream integration so official Blender/Cycles updates can be merged with less risk.

### Source Isolation

- Standardized personal feature regions using paired `CyclesPlus Begin/End` markers across Cycles source files.
- Added or completed guards for `WITH_CYCLES_SPPM_CAUSTICS` and `WITH_DLSS`.
- Marked standalone personal implementation files with file-level CyclesPlus ownership boundaries.
- Covered surface caustics, volumetric caustics, GPU photon tracing, OptiX photon pipelines, glass dispersion, DLSS viewport denoising, UI integration, and build integration.
- Kept existing feature-enabled behavior unchanged except where disabled-feature compilation required structural fixes.

### Build and Feature Guards

- Added clear CyclesPlus boundaries around DLSS and photon-related CMake entries.
- Ensured personal declarations, calls, fields, and kernel entries are protected by their corresponding feature macros.
- Fixed `KernelShader` padding when `WITH_CYCLES_SPPM_CAUSTICS=OFF`.
- Fixed preprocessing boundaries in `session.cpp` so feature-disabled builds do not produce unbalanced code blocks.
- Added guards around DLSS render scheduler, denoiser selection, CUDA/OptiX queues, and device capability checks.

### UI and Integration

- Added annotation boundaries around photon caustics properties, material/light/world controls, render pass registration, and DLSS preview controls.
- Marked CyclesPlus changes in synchronization, shader upload, film passes, integrator fields, and render scheduler integration.

### Documentation

- Added and organized downstream maintenance, synchronization, feature mapping, marker inventory, architecture review, and CPU/GPU hybrid caustics planning documents.
- Documented that paper and external implementations provide scheduling principles only. Actual performance must be validated by fixed-scene wall-clock measurements.

### Validation

- Full Release build completed with CUDA, OptiX, and DLSS enabled.
- Blender installed successfully to `E:\Blender_Shuimeng_5.2`.
- CUDA, CPU, and OptiX devices were detected after startup.
- A separate build with feature macros disabled successfully compiled the Cycles session target.
- `git diff --check` reported no remaining source whitespace errors.

### Repository Cleanliness

- Build directories, backups, release packages, caches, temporary logs, `.bak` files, and interrupted-save files were excluded from the commit.

---

## 中文

本次更新对 CyclesPlus 个人下游分支进行了长期维护规范化，主要目的是降低未来合并 Blender/Cycles 官方更新时的冲突和维护风险。

### 源码隔离

- 在 Cycles 源码中统一使用成对的 `CyclesPlus Begin/End` 标记个人功能区域。
- 补齐个人功能对应的 `WITH_CYCLES_SPPM_CAUSTICS` 和 `WITH_DLSS` 编译宏保护。
- 对整文件属于个人实现的文件增加文件级 CyclesPlus 归属边界。
- 覆盖表面焦散、体积焦散、GPU 光子追踪、OptiX 光子管线、玻璃色散、DLSS 视口降噪、UI 接入和构建接入。
- 功能开启时的原有行为保持不变；只有关闭功能时无法编译的部分进行了结构修正。

### 构建与功能开关

- 为 DLSS 和光子相关 CMake 内容增加明确的 CyclesPlus 边界。
- 确保个人声明、函数调用、结构字段和设备内核入口受到对应宏保护。
- 修复 `WITH_CYCLES_SPPM_CAUSTICS=OFF` 时 `KernelShader` 的对齐填充。
- 修正 `session.cpp` 的预处理边界，避免关闭功能后出现过早 `#endif` 和代码块不平衡。
- 为 DLSS 调度器、降噪器选择、CUDA/OptiX 队列和设备能力检查补齐宏保护。

### UI 与接入点

- 为光子焦散属性、材质/灯光/世界设置、渲染通道注册和 DLSS 视口控制增加标准标记。
- 标记同步、着色器上传、Film 通道、积分器字段和渲染调度中的 CyclesPlus 修改。

### 文档

- 新增和整理个人下游分支同步、功能映射、代码隔离标记清单、架构评估和 CPU/GPU 混合焦散规划文档。
- 明确说明论文和外部项目只提供异构调度原则，最终性能必须通过固定场景整帧墙钟实测确认。

### 验证结果

- CUDA、OptiX、DLSS 全部启用的完整 Release 构建成功。
- 已安装至 `E:\Blender_Shuimeng_5.2`，并确认 CUDA、CPU、OptiX 设备正常识别。
- 单独关闭功能宏后，Cycles Session 目标编译成功。
- `git diff --check` 无剩余源码空白格式错误。

### 仓库清理

- 构建目录、备份、发布包、缓存、临时日志、`.bak` 文件和系统生成的临时锁文件均未提交。
