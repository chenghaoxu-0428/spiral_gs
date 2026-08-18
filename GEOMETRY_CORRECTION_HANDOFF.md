# Real Spiral 几何修正实验交接记录

更新时间：2026-08-15  
当前分支：`fix/real-cylindrical-detector-geometry`

> **2026-08-15 更新：P0/P1 实验已完成，结果与结论见 [GEOMETRY_SOURCE_SHIFT_RESULTS.md](GEOMETRY_SOURCE_SHIFT_RESULTS.md)。**
> 一句话结论：逐视角 source shift 按 CT-PD 语义注入并重训后无实质收益（angle/z +0.07 dB，radial→DSD 反而 −0.48 dB 2D），不应作为正式修正；解析主点 u=−0.28125 与经验 −0.275 等价（差 0.025 dB）；建议正式化 u 主点修正即可。source shift 注入机制已提交为 `model.geometry` 配置，可复现。

本文用于把当前账号的实验上下文交接给后续 agent。重点是 `real/ldctl004/spiral/ntrain1000`，并记录已经验证、未验证以及不应重复使用的结果。

## 一、当前问题和坐标约定

实拍数据参考左手坐标系。当前 real 数据链路涉及：

1. DICOM 投影读取、转置、圆柱 detector 展平和 z/angle 生成。
2. `coord_left=True` 下的投影/相机方向处理。
3. FDK 初始化点云。
4. FaCT-GS 训练时的 Camera 投影矩阵。
5. `vol_pred`/`vol_gt` 的显示切片。

已确认的左手 FDK 修正位于 `data_preprocess/norm_pipeline.py`：在左手坐标系 FDK 后把采样点的 x 坐标镜像回训练体坐标：

```python
xyz[:, 0] = 2 * scaled_scanner["offOrigin"][0] - xyz[:, 0]
```

这解决的是 FDK 初始化体方向问题；它不等价于 detector 主点或逐视角源焦点修正。

## 二、最重要的实验结论（按重要性排序）

### 1. 水平 detector 主点偏移是已确认的主要几何错误

对 real `ldctl004` 的最终模型做投影域几何扫描，得到：

- `u_pixels=-0.275` 最优。
- v 主点、角度、z、pitch、DSO、DSD 的零偏移最优。
- 在原模型上把预测投影按 `u=-0.275 px` 对齐，测试 PSNR 从约 38.48 dB 提升到约 40.48 dB。

随后在训练和评估 Camera 中同时加入 `u=-0.275 px` 并完整训练 30000 步：

| 模型 | PSNR2D | SSIM2D | PSNR3D | SSIM3D |
|---|---:|---:|---:|---:|
| real 原几何 | 38.479 | 0.97985 | 20.228 | 0.650 |
| real + u=-0.275 px 重训 | **41.360** | **0.9849** | 20.135 | 0.649 |

结论：主点修正带来 **+2.881 dB**，是目前最明确、收益最大的几何修正。3D 没有改善，说明它主要改善投影几何自洽性，不能解决 `vol_gt` 与真实投影之间的体数据/方向/配准问题。

重要复现说明：本次 `u=-0.275 px` 是运行时 Camera 矩阵修正，不是已经提交到共享 `graphics_utils.py` 的正式实现。因此换 agent 后若要复现，必须显式注入该修正；当前 `scanner.offDetector` 仍为 `[0, 0]`。

### 2. 匹配真实采样几何的 syn 证明模型能力不是首要瓶颈

用真实数据的 DSO、DSD、angle、z、train/test split 生成了 matched-syn：

- 数据：`data/syn/ldctl004/spiral/ntrain1000/r2gs_real_geometry`
- 模型：`models/syn/ldctl004/spiral/ntrain1000/factgs_matched_real_geometry`
- 30000 步测试：PSNR2D **47.434 dB**，SSIM2D **0.99827**。

同样的采样轨迹和模型结构在理想投影上可以达到高指标，因此 real 与 syn 的差距不是由螺旋采样数量、网络结构或基本 Gaussian 表达能力单独造成的。

### 3. 20 视角强制过拟合进一步排除了“模型容量不足”

构造了只含 20 个均匀 real 视角、train=test 的诊断集：

- 数据：`data/real/ldctl004/spiral/ntrain20/r2gs_cyl_detector_overfit`
- 模型：`models/real/ldctl004/spiral/ntrain20/factgs_cyl_detector_uoffset_overfit20`
- 10000 步：PSNR2D **50.351 dB**，SSIM2D **0.9987**。

这不是泛化实验，而是容量/优化能力测试。结论是：算法可以把一小组真实投影拟合得非常好；完整 1000 视角只能得到约 41 dB，主导限制是不同视角无法被同一套静态体和理想几何同时解释。

### 4. DICOM source dynamics 中确实存在 flying focal spot 信息

原始 DICOM 路径：

`/opt/data/private/data/ldctl004/LDCT-and-Projection-data/L004/08-21-2018-NA-NA-10971/1.000000-Full dose projections-24362`

共读取到 32786 张投影。`data_preprocess/dict.txt` 定义了以下关键私有字段：

| 字段 | tag | 观测值 | 当前流程 |
|---|---|---:|---|
| `DetectorCentralElement` | `(7031,1033)` | `[369.625, 32.5]`，恒定 | 未使用 |
| `SourceAngularPositionShift` | `(7033,100B)` | 0.000333/0.000336 rad，交替 | 未使用 |
| `SourceAxialPositionShift` | `(7033,100C)` | 0/-0.66 mm，交替 | 未使用 |
| `SourceRadialDistanceShift` | `(7033,100D)` | 0/+5.45 mm，交替 | 未使用 |
| `FlyingFocalSpotMode` | `(7033,100E)` | `FFSZ` | 未使用 |

`FFSZ` 很像 z-flying focal spot：相邻视角使用两个不同焦点位置。当前 `norm_pipeline.py` 和 `dicom_spiral_process.m` 只读取 nominal 的 `DetectorFocalCenter*` 和 `ConstantRadialDistance`，没有使用上述 source shift。

### DetectorCentralElement 的维度解释（关键结论）

原始 DICOM Pixel Data 是 `Rows=736, Columns=64`，而 CT-PD detector geometry 是 `NumberofDetectorRows=64, NumberofDetectorColumns=736`。这是投影图像存储方向和 detector 几何方向的转置；当前流程的 `image.T` 正好把原始 `[736,64]` 转为 detector 方向 `[64,736]`，之后再按 4 倍下采样为训练使用的 `[16,184]`。

CT-PD 对 `(7031,1033) DetectorCentralElement` 的定义是 `(Column X, Row Y)` 的 detector element 索引，不是 `[Rows, Columns]` 尺寸，也不是毫米坐标。当前值 `[369.625,32.5]` 应解释为：X 对应 736 个 detector columns，Y 对应 64 个 detector rows。Y 的几何中心为 `(64+1)/2=32.5`；X 的几何中心为 `(736+1)/2=368.5`，因此 X 偏移为 `1.125` 个原始 detector element。下采样因子为 4 后：

```text
1.125 / 4 = 0.28125 training-detector pixel
```

这与投影扫描得到的经验最优 `u=-0.275 px` 几乎完全一致；负号来自当前左手坐标、列方向和投影矩阵符号约定。因此 `DetectorCentralElement` 很可能就是静态水平主点偏移的 DICOM 来源。这里不应把中心坐标乘 2；应先按 `(Column X, Row Y)` 解释，再按实际 detector 下采样比例转换。

这验证了用户提出的“跨视角 source geometry”方向是有实际 DICOM 依据的。

### 5. source shift 能解释一部分残差，但不是全部主因

将 DICOM 投影按 angle/z 与当前 1500 个 test 投影精确对齐后，比较主点修正模型的逐视角 PSNR：

- source shift 为 0：平均 **41.476 dB**
- source shift 非 0：平均 **41.245 dB**
- 两组差异约 **0.23 dB**

按两种 source 状态分别估计残差均值图，再做 oracle 组校正：

- 基线：**41.360 dB**
- 两组残差分别校正：**41.521 dB**
- oracle 收益：约 **+0.161 dB**
- 两组均值解释残差能量约 **10.6%**

这说明 DICOM source shift 不是伪字段，确实有可测影响；但即使理想地按两种状态消除组偏差，也只解释一小部分 real/syn 差距。

注意：这是对已经按 nominal geometry 训练好的模型做的残差诊断，不是动态 source geometry 重训结果。真正验证 source shift 是否能提高最终重建，需要把每视角 source shift 注入训练 Camera 后重新训练。

### 6. 其他简单全局几何参数没有证据支持

在正确 `u=-0.275 px` 的基础上，对 400 个测试视角复扫：

- angle：0 最优
- z offset：0 最优
- pitch scale：1.0 最优
- DSO scale：1.0 最优
- DSD scale：1.0 最优
- v offset：0 最优

因此目前没有证据表明存在第二个简单的全局角度、z、DSO/DSD 或 v 主点偏移。

### 7. 增益/偏置不是主要因素

对 real 投影做全局 gain/bias 拟合：

- 全局 gain 约 1.001985
- 全局 bias 约 0.000344
- 全局校正反而约 -0.025 dB
- 逐视角 oracle affine 也只有约 +0.158 dB

而项目 2D metric 本身会对每个投影独立 max-normalize，所以整体曝光比例已经大部分被指标消除。

## 三、残差分析的正确解释

主点修正后的残差统计：

- 固定 detector 残差图占比：约 **9.0%**，修正后收益约 **+0.446 dB**。
- 残差与物体边缘相关性：约 **0.333**。
- 残差与 z 的相关性：约 **0.163**。
- 低频残差能量：约 **55.9%**。
- 前十个 SVD 模式解释约 **52.3%**。

“55.9% 低频”表示残差在 detector 图像空间中具有较强的低空间频率结构，**不能直接等同于 55.9% 来自 source shift 或跨视角误差**。目前已能从 DICOM source state 解释的 oracle 残差能量约 10.6%。

更准确的判断是：低频结构是“系统误差的表现形式”，但具体来源仍混合了：

- 圆柱 detector 展平到平面 detector 的插值/中心定义误差；
- flying focal spot 等逐视角源轨迹；
- 真实扫描中的散射、束硬化、残余平场/暗场误差；
- `vol_gt` 与真实投影不完全自洽；
- 可能的运动或扫描期间非静态变化。

## 四、已知无效或不应复用的结果

1. `.../eval/step_030000/projection_pairs.npz` 是一次错误的运行时投影矩阵入口产物，复渲染时没有正确注入 Camera 层的主点修正。不要用于残差分析。
2. 应使用：
   `models/real/ldctl004/spiral/ntrain1000/factgs_cyl_detector_uoffset_m0275/eval/step_030000/projection_pairs_corrected.npz`
3. 之前未缩放 z 的 source-shift 渲染结果也已作废。`readCTameras` 的实际 Camera z 必须乘场景缩放因子 `scene_scale=2/max(sVoxel)`；DICOM mm 转 scene unit 后还要再乘该缩放。
4. `vol_pred` 的最终显示切片翻转与投影 Camera 几何修正是两条独立链路，不能用切片显示方向代替训练几何修正。

## 五、后续实验优先级

### P0：动态 source geometry 重训（最有价值）

把每个 DICOM projection 的以下字段写入 `meta_data.json` 或独立 sidecar：

```text
source_angular_position_shift
source_axial_position_shift
source_radial_distance_shift
flying_focal_spot_mode
```

训练 Camera 时按每视角修正：

- angle 加/减 `SourceAngularPositionShift`；
- z 加/减 `SourceAxialPositionShift`；
- source radial shift 同时测试对 DSO 和 DSD 的物理解释；
- 保留已验证的 `u=-0.275 px`。

建议至少比较三种模型：

1. nominal geometry + u offset；
2. source shift 只改 angle/z；
3. source shift 改 angle/z/radial（分别测试 DSD 是否随 source radial shift 反向变化）。

每种训练 30000 步，比较最终 PSNR2D、SSIM2D、逐视角 PSNR 分布和 3D 指标。只有重训后有稳定收益，才能把 source shift 定性为可操作的主修正。

### P1：把圆柱展平中心和 DICOM `DetectorCentralElement` 做物理映射

当前 `DetectorCentralElement=[369.625,32.5]` 恒定，不能解释逐视角差异，但可能帮助解释静态 `u=-0.275 px`：

- 明确 raw detector 行列方向；
- 明确 `DetectorCentralElement` 的坐标单位和原点；
- 推导圆柱 detector 的 tangent-plane principal point；
- 检查 flatten 前后 detector center 是否应使用 `(N-1)/2`、`N/2` 或 DICOM center element；
- 用解析映射替代经验 `-0.275 px`，再做一次完整重训。

### P1：逐视角残差与 source state/angle/z 的联合回归

不要只比较两组均值。对每个 test view 的残差统计量回归：

- source state；
- angle 的周期项；
- z；
- detector row/column 低频分量；
- exposure/timestamp/photon statistics（如果确认其 VR 和语义）。

目标是判断剩余低频误差是否是 source state、z 或角度的可预测函数。

### P2：真实投影预处理一致性核查

DICOM flags 在抽样投影中均为 `YES`：beam hardening、gain、dark field、flat field、bad pixel、scatter、log。但这些是数据声明，不代表残余误差为零。建议：

- 比较 raw/processed projection 的 histogram 和低频背景；
- 按 source state 比较 detector 平场和边缘背景；
- 检查 `PhotonStatistics`、`Timestamp` 是否与 residual magnitude 相关；
- 不要将已经 `LogFlag=YES` 的数据再次 log transform。

### P2：`vol_gt` 与投影自洽性单独验证

即使 2D projection fit 已经改善，3D 指标仍接近 20 dB，说明真实 `vol_gt` 不是可靠的严格投影真值。建议：

- 用 corrected geometry 对 `vol_gt` 前向投影；
- 与 DICOM projection 做逐视角 residual；
- 分离 volume mismatch、projection preprocessing mismatch 和 camera mismatch；
- 暂时不要用 3D PSNR 单独判断 Camera 修正是否有效。

### P3：训练时长/超参数

主点修正模型 25k→30k 仅约 +0.29 dB，说明训练时长不是当前第一优先级。只有 P0/P1 几何实验完成后，再测试 60k 或 loss 权重；否则容易用更多迭代拟合系统误差。

## 六、关键文件和结果路径

- 当前分支：`fix/real-cylindrical-detector-geometry`
- 几何/预处理代码：[data_preprocess/norm_pipeline.py](/opt/data/private/spiral_gs/data_preprocess/norm_pipeline.py)
- DICOM 字典：[data_preprocess/dict.txt](/opt/data/private/spiral_gs/data_preprocess/dict.txt)
- DICOM MATLAB 流程：[data_preprocess/dicom_spiral_process.m](/opt/data/private/spiral_gs/data_preprocess/dicom_spiral_process.m)
- matched-syn 生成器：[data_preprocess/generate_matched_syn.py](/opt/data/private/spiral_gs/data_preprocess/generate_matched_syn.py)
- 投影对导出：[experiments/helpers/render_projection_pairs.py](/opt/data/private/spiral_gs/experiments/helpers/render_projection_pairs.py)
- 几何扫描：[experiments/helpers/scan_projection_geometry.py](/opt/data/private/spiral_gs/experiments/helpers/scan_projection_geometry.py)
- real 主点修正模型：[factgs_cyl_detector_uoffset_m0275](/opt/data/private/spiral_gs/models/real/ldctl004/spiral/ntrain1000/factgs_cyl_detector_uoffset_m0275)
- real 主点修正最终 2D 指标：[eval2d_render_test.yml](/opt/data/private/spiral_gs/models/real/ldctl004/spiral/ntrain1000/factgs_cyl_detector_uoffset_m0275/eval/step_030000/eval2d_render_test.yml)
- corrected 投影对：[projection_pairs_corrected.npz](/opt/data/private/spiral_gs/models/real/ldctl004/spiral/ntrain1000/factgs_cyl_detector_uoffset_m0275/eval/step_030000/projection_pairs_corrected.npz)
- matched-syn 模型：[factgs_matched_real_geometry](/opt/data/private/spiral_gs/models/syn/ldctl004/spiral/ntrain1000/factgs_matched_real_geometry)
- 20-view 过拟合模型：[factgs_cyl_detector_uoffset_overfit20](/opt/data/private/spiral_gs/models/real/ldctl004/spiral/ntrain20/factgs_cyl_detector_uoffset_overfit20)

## 七、给后续 agent 的一句话摘要

先保留并正式实现已经验证的 `u=-0.275 px`；DICOM 的 `FFSZ` 和三个 `Source*Shift` 字段确实存在并解释约 0.16 dB 的 oracle 残差收益，但不是 55.9% 低频残差的全部来源。下一步最高价值实验是：给每个 Camera 注入 source angular/axial/radial shift，在相同主点修正下完整重训并比较三种 source radial/DSD 解释。
