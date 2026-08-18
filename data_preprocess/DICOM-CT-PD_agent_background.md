# DICOM-CT-PD：CT 投影数据与几何字段速查

> 用途：作为 CT / 螺旋 CT 重建 Agent 的背景知识。
> 来源：`DICOM-CTPD User Manual, Version 3, February 2020`。
> 本文只保留与 **投影读取、扫描轨迹、source/detector 几何、螺旋扫描、预处理状态和重建** 直接相关的信息。

---

## 1. DICOM-CT-PD 是什么

DICOM-CT-PD 是一种厂商无关的 CT 投影数据格式，用扩展 DICOM Header 保存 CT 重建所需的扫描信息。

一个 CT scan 由一系列 DICOM-CT-PD 文件组成：

- **一个文件 = 一个 projection/view**
- 每个 projection 对应一个特定的：
  - 机架角度；
  - z 位置 / table position；
  - source 状态；
  - detector readout。
- 文件由两部分组成：
  1. **Header**：扫描轨迹、source/detector geometry、tube 参数、校正状态等；
  2. **PixelData**：该 view 的完整 detector projection。

该格式支持：

- third-generation CT geometry；
- cylindrical detector；
- spherical detector；
- flat detector；
- axial scan；
- helical scan；
- flying focal spot (FFS)；
- 多 source / 多 tube voltage / 多 detector layer / 多 energy bin。

---

# 2. 最重要的坐标系定义

## 2.1 DICOM-CT-PD 使用左手圆柱坐标系

扫描几何首先定义在圆柱坐标：

\[
(\rho,\phi,z)
\]

这是一个 **left-handed coordinate system**。

坐标系与 patient 绑定。为了描述 table translation，手册等价地认为：

- patient/table 静止；
- source 和 detector 相对于 patient 运动。

### z 轴

`z` 方向：

- 垂直于 gantry rotation plane；
- 从 patient table base 指向 gantry。

### 方位角 φ

从 patient table 一侧观察 gantry：

- `φ = 0` 位于 **12 点钟方向**；
- `φ` 沿 **逆时针方向增加**。

---

## 2.2 圆柱坐标 → 笛卡尔坐标

DICOM-CT-PD 给出的 Cartesian system 同样是 **左手系**：

\[
x=-\rho\sin\phi
\]

\[
y=\rho\cos\phi
\]

\[
z=z
\]

因此不要直接套通常右手系的：

\[
x=\rho\cos\phi,\quad y=\rho\sin\phi
\]

否则很容易造成：

- view angle 方向反转；
- detector 左右翻转；
- reconstructed volume 镜像；
- source trajectory 与 detector trajectory 错位。

---

# 3. Detector 几何

## 3.1 Detector shape

重要字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7029,100B)` | `DetectorShape` | `CYLINDRICAL` / `SPHERICAL` / `FLAT` |

对于真实 CT，尤其需要检查是否为：

```text
CYLINDRICAL
```

不要默认 detector 是 flat panel。

---

## 3.2 Detector 尺寸

| Tag | Attribute | 含义 |
|---|---|---|
| `(7029,1010)` | `NumberofDetectorRows` | detector row 数 |
| `(7029,1011)` | `NumberofDetectorColumns` | detector column 数 |
| `(7029,1002)` | `DetectorElementTransverseSpacing` | 单个 detector column 的横向宽度，单位 mm |
| `(7029,1006)` | `DetectorElementAxialSpacing` | 单个 detector row 的轴向高度，单位 mm |

记：

```text
Nrow = NumberofDetectorRows
Ncol = NumberofDetectorColumns

dCol = DetectorElementTransverseSpacing
dRow = DetectorElementAxialSpacing
```

其中：

- column 方向主要对应 transaxial / fan direction；
- row 方向对应 axial / z direction。

---

# 4. Detector focal center

DICOM-CT-PD 并不是简单给 detector center 的 Cartesian position，而是定义了一个 **detector focal center**。

对于 cylindrical detector：

- detector focal center 与 isocenter 对齐；
- 在 z 方向对应 detector rows 中心；
- 同时是 detector arc 在 transverse plane 中的圆弧焦点。

其圆柱坐标为：

\[
(\rho_0,\phi_0,z_0)
\]

字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7031,1001)` | `DetectorFocalCenterAngularPosition` | `φ0`，rad |
| `(7031,1002)` | `DetectorFocalCenterAxialPosition` | `z0`，mm |
| `(7031,1003)` | `DetectorFocalCenterRadialDistance` | `ρ0`，mm |

转换为 Cartesian：

\[
x_d=-\rho_0\sin\phi_0
\]

\[
y_d=\rho_0\cos\phi_0
\]

\[
z_d=z_0
\]

---

# 5. Source 位置：不要直接把 DetectorFocalCenter 当 source

这是 DICOM-CT-PD 几何建模里最容易出错的地方之一。

正常情况下 focal spot 可能与 detector focal center 重合，但也允许存在偏移：

\[
(\Delta\rho,\Delta\phi,\Delta z)
\]

因此真实 focal spot/source 的圆柱坐标应为：

\[
\rho_s=\rho_0+\Delta\rho
\]

\[
\phi_s=\phi_0+\Delta\phi
\]

\[
z_s=z_0+\Delta z
\]

对应字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7033,100B)` | `SourceAngularPositionShift` | `Δφ`，rad |
| `(7033,100C)` | `SourceAxialPositionShift` | `Δz`，mm |
| `(7033,100D)` | `SourceRadialDistanceShift` | `Δρ`，mm |

然后再按：

\[
x_s=-\rho_s\sin\phi_s
\]

\[
y_s=\rho_s\cos\phi_s
\]

\[
z_s=z_s
\]

得到 source Cartesian position。

> **Agent 注意：**
>
> `DetectorFocalCenterAngularPosition` / `AxialPosition` / `RadialDistance`
> 并不应无条件直接作为真实 source position。
> 必须检查 Source*Shift 字段。

---

# 6. Detector central element

重要字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7031,1033)` | `DetectorCentralElement` | `(Column X, Row Y)` |
| `(7031,1031)` | `ConstantRadialDistance` | `d0`，detector focal center 到 central detector element 的距离 |

`DetectorCentralElement = (Column X, Row Y)`：

表示连接：

```text
detector focal center → isocenter
```

的直线击中 detector 的位置。

它允许是 **非整数索引**。

例如：

```text
(Column 369.625, Row 32.5)
```

表示：

- 横向位置在 column 369 到 370 之间的 0.625 处；
- 轴向位置在 row 32 与 33 正中间。

因此 detector principal point / central ray **不一定落在某个实际 detector pixel center 上**。

这是从 DICOM 几何构造精确 ray geometry 时的重要参数。

---

# 7. Detector 元素索引方向

当 detector 位于 6 点钟方向，并从 patient table 一侧观察：

- 最远的一排定义为 `Row 1`；
- 最左边的一列定义为 `Column 1`。

实现投影读入和几何 ray mapping 时，需要特别检查：

- array row index 是否需要 flip；
- detector column 是否需要 flip；
- 自己代码里的 `(u,v)` 定义是否与 DICOM 的 `(column,row)` 一致。

不要根据图像“看起来差不多”来猜方向。

---

# 8. 螺旋扫描 / Helical CT

## 8.1 判断扫描类型

| Tag | Attribute | 值 |
|---|---|---|
| `(7037,1009)` | `TypeofProjectionData` | `"AXIAL"` / `"HELICAL"` |

如果为：

```text
HELICAL
```

则 source/detector 随 view angle 变化的同时，`z0` 也发生变化。

---

## 8.2 z0 是螺旋轨迹的重要变量

字段：

```text
(7031,1002) DetectorFocalCenterAxialPosition
```

即：

\[
z_0
\]

手册指出：

- 如果 projection number 增大时 `z0` 减小，则 patient/table 在 acquisition 中向 gantry 方向移动；
- 反之亦然。

因此对一组投影：

```python
theta_i = DetectorFocalCenterAngularPosition[i]
z_i     = DetectorFocalCenterAxialPosition[i]
```

`(theta_i, z_i)` 的联合变化可以直接描述螺旋 scan trajectory。

不要只使用：

```text
projection_index -> angle
```

然后自行假设 z 线性变化；实际几何应优先读取每个 projection 的 `z0`。

---

## 8.3 Spiral Pitch

| Tag | Attribute | 含义 |
|---|---|---|
| `(0018,9311)` | `SpiralPitchFactor` | table feed per rotation / total collimation width |

定义：

\[
pitch=
\frac{\text{table feed per rotation}}
{\text{total collimation width}}
\]

Pitch 可用于：

- 检查轨迹是否合理；
- 估计每圈 axial translation；
- 和逐投影 `z0` 做 consistency check。

但在精确 reconstruction geometry 中，优先使用 **逐 projection 的实际 angle 和 z position**。

---

# 9. Projection geometry 类型

| Tag | Attribute | 含义 |
|---|---|---|
| `(7037,100A)` | `TypeofProjectionGeometry` | `"FANBEAM"` for third-generation CT geometry |

这里的 `FANBEAM` 是第三代 CT acquisition geometry 的格式描述。

即使 detector 有多个 rows、可以进行 3D reconstruction，也可能仍标记为：

```text
FANBEAM
```

不要简单把该字符串理解成“只有单排 detector 的二维 fan-beam CT”。

---

# 10. Flying Focal Spot (FFS)

字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7033,100E)` | `FlyingFocalSpotMode` | focal spot 模式 |

可能值：

```text
FFSNONE
FFSZ
FFSXY
FFSXYZ
```

含义：

- `FFSNONE`：无 flying focal spot；
- `FFSZ`：沿 axial/z 方向移动；
- `FFSXY`：transverse plane 内移动；
- `FFSXYZ`：同时存在 transverse 和 axial movement。

使用 FFS 时：

- focal spot 在 anode surface 上周期移动；
- 不同 focal spot position 会有不同 `(Δρ, Δφ, Δz)`；
- **每一个 focal spot position 对应独立 projection，并存成独立 DICOM-CT-PD 文件。**

因此不要把连续两个文件简单认为只是普通角度步进。

---

# 11. 每圈 projection 数

| Tag | Attribute | 含义 |
|---|---|---|
| `(7033,1013)` | `NumberofSourceAngularSteps` | 每完整 rotation 的 projection 数 |

可用于：

```text
angle_step ≈ 2π / NumberofSourceAngularSteps
```

但精确 angle 仍应使用逐 projection：

```text
DetectorFocalCenterAngularPosition
```

而不是自行从 instance number 推 angle。

---

# 12. 时间信息

| Tag | Attribute | 含义 |
|---|---|---|
| `(7033,1067)` | `Timestamp` | absolute time，ms |
| `(0018,1150)` | `ExposureTime` | gantry rotation time，ms |

Timestamp 对：

- 动态 CT；
- 检查投影顺序；
- FFS；
- 非均匀采样；
- trajectory debugging

都很有价值。

---

# 13. Projection PixelData

一个 DICOM-CT-PD 文件对应一个完整 detector readout。

PixelData：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7FE0,0010)` | `PixelData` | projection data |

原始存储为：

```text
unsigned 16-bit
```

projection data matrix 代表完整 detector：

```text
Nrow × Ncol
```

手册描述的线性排列顺序是 row-major：

```text
前 Ncol 个值       -> Row 1, Column 1 ... Column Ncol
接下来 Ncol 个值   -> Row 2
...
```

---

# 14. PixelData 必须先进行 Rescale

读取出的原始整数不能直接作为真实 projection value。

字段：

| Tag | Attribute |
|---|---|
| `(0028,1052)` | `RescaleIntercept` |
| `(0028,1053)` | `RescaleSlope` |

转换：

\[
P =
P_{\text{readout}}\cdot
\text{RescaleSlope}
+
\text{RescaleIntercept}
\]

即：

```python
projection = raw.astype(float) * RescaleSlope + RescaleIntercept
```

**做任何 reconstruction / normalization 前，应先完成这一步。**

---

# 15. Projection 是否已经做 log transform

字段：

| Tag | Attribute | 值 |
|---|---|---|
| `(7039,1009)` | `LogFlag` | `"YES"` / `"NO"` |

必须检查。

如果：

```text
LogFlag == YES
```

说明数据已经 logarithmically transformed。

不要再次执行：

\[
-\log(I/I_0)
\]

否则会对数据进行二次 log，彻底破坏 Beer-Lambert line integral。

如果：

```text
LogFlag == NO
```

需要进一步结合数据定义和 air/intensity normalization 判断是否需要自行转为 line integral。

---

# 16. Projection correction flags

实际数据使用前建议记录以下状态：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7039,1003)` | `BeamHardeningCorrectionFlag` | beam hardening correction |
| `(7039,1004)` | `GainCorrectionFlag` | detector gain calibration |
| `(7039,1005)` | `DarkFieldCorrectionFlag` | dark/background subtraction |
| `(7039,1006)` | `FlatFieldCorrectionFlag` | flood field / bowtie / heel effect correction |
| `(7039,1007)` | `BadPixelCorrectionFlag` | bad pixel correction |
| `(7039,1008)` | `ScatterCorrectionFlag` | scatter correction |
| `(7039,1009)` | `LogFlag` | log transform |

均通常为：

```text
YES / NO
```

这些字段对实拍数据异常分析很重要。

例如：

- 未做 scatter correction → 投影存在低频 bias；
- 未做 beam hardening correction → 高密度结构附近非线性明显；
- 未做 flat-field correction → detector spatial gain pattern 残留；
- 未做 bad pixel correction → 投影出现固定 detector artifacts。

---

# 17. Photon statistics

字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(7033,1065)` | `PhotonStatistics` | 每个 detector column 的 incident photon estimate |

它描述：

```text
Column 1 ... Column M
```

方向上的 photon distribution。

手册中的 photon number 是根据 air scans 的 transmission variance 估计的 **noise equivalent quanta**。

其变化：

- 沿 detector columns 给出；
- 忽略 detector rows 之间的变化；
- 会考虑 scanner/filter/kVp；
- 并根据该 projection 的 tube current 缩放。

适合用于：

- weighted least squares；
- Poisson/noise-aware loss；
- projection reliability weighting。

---

# 18. Tube / acquisition 参数

对 reconstruction 或数据诊断较重要的标准字段：

| Tag | Attribute | 含义 |
|---|---|---|
| `(0018,0060)` | `KVP` | tube peak voltage |
| `(0018,1151)` | `XrayTubeCurrent` | mA |
| `(0018,1150)` | `ExposureTime` | gantry rotation time，ms |
| `(0018,0090)` | `DataCollectionDiameter` | scan field of view，mm |
| `(0018,9311)` | `SpiralPitchFactor` | spiral pitch |

---

# 19. Water attenuation coefficient 与 HU

字段：

| Tag | Attribute |
|---|---|
| `(7041,1001)` | `WaterAttenuationCoefficient` |

记水的 calibration coefficient 为：

\[
\mu_\text{water}
\]

线性衰减系数转换为 CT number / HU：

\[
HU =
1000
\frac{\mu-\mu_\text{water}}
{\mu_\text{water}}
\]

如果 projection 已做 beam-hardening correction，则该 calibration factor 对应 beam-hardening correction energy 下的 water attenuation coefficient。

注意：

**该 Tag 主要用于重建出的 attenuation coefficient volume → HU 的标定，不是对 projection pixel 直接做 HU 转换。**

---

# 20. 多 source / 多 spectrum / 多 detector layer

字段：

| Tag | Attribute |
|---|---|
| `(7033,1061)` | `NumberofSpectra` |
| `(7033,1063)` | `SpectrumIndex` |

一个 DICOM-CT-PD series 可以对应：

- 一个 x-ray source；
- 一个 tube potential；
- 一个 detector layer；
- 一个 energy threshold / energy bin。

在以下数据中必须特别注意 series 的拆分：

- dual-source CT；
- fast kVp switching；
- dual-layer detector；
- photon-counting / energy-resolved scan。

不能把不同 `SpectrumIndex` 的 projection 不加区分地混到同一个普通单源 reconstruction geometry 中。

---

# 21. DICOM-CT-PD Header 读取注意事项

DICOM-CT-PD 包含 private tags。

普通 DICOM reader 如果没有加载对应 dictionary，可能：

- 无法识别 private tags；
- 将字段显示为 unknown；
- 导致关键 scanner geometry 丢失。

手册特别指出：

- 应使用 DICOM-CT-PD dictionary；
- 不支持 custom dictionary 的 DICOM viewer 不适合完整检查 Header。

同时应注意 VR 数据类型，例如：

```text
FL = 32-bit float
US = unsigned short
DS = decimal string
CS = coded string
```

读取后不要因为错误 cast 导致 geometry precision 丢失。

---

# 22. Agent 推荐的单 projection 几何解析流程

对于每一个 DICOM-CT-PD projection：

```python
# 1. detector focal center
rho0 = DetectorFocalCenterRadialDistance
phi0 = DetectorFocalCenterAngularPosition
z0   = DetectorFocalCenterAxialPosition

# 2. source offset
drho = SourceRadialDistanceShift
dphi = SourceAngularPositionShift
dz   = SourceAxialPositionShift

# 3. true focal spot
rho_s = rho0 + drho
phi_s = phi0 + dphi
z_s   = z0 + dz

# 4. left-handed cylindrical -> Cartesian
source_x = -rho_s * sin(phi_s)
source_y =  rho_s * cos(phi_s)
source_z =  z_s

# 5. detector geometry
shape = DetectorShape
nrow  = NumberofDetectorRows
ncol  = NumberofDetectorColumns
drow  = DetectorElementAxialSpacing
dcol  = DetectorElementTransverseSpacing

central_element = DetectorCentralElement
d0 = ConstantRadialDistance

# 6. projection
raw = PixelData
proj = raw * RescaleSlope + RescaleIntercept

# 7. check preprocessing state
log_flag = LogFlag
```

之后再根据：

```text
DetectorShape
DetectorCentralElement
ConstantRadialDistance
dCol
dRow
```

构造 detector element positions / rays。

---

# 23. 螺旋 CT Agent 推荐处理方式

对于整个 series，首先构造：

```python
views = [
    {
        "theta": DetectorFocalCenterAngularPosition,
        "z": DetectorFocalCenterAxialPosition,
        "rho": DetectorFocalCenterRadialDistance,
        "source_shift": (d_rho, d_phi, d_z),
        "timestamp": Timestamp,
        "projection": scaled_pixel_data,
    }
]
```

然后：

1. 按 acquisition / projection order 排序；
2. 检查 `theta` 是否随 projection 连续变化；
3. 对 angle 做 unwrap；
4. 检查 `z(theta)`；
5. 检查每圈 z displacement；
6. 与 `SpiralPitchFactor` 做 consistency check；
7. 检查 FFS 是否造成交替 source position；
8. 不要默认 DSO/DSD、source z 或 detector center 在整个 scan 中严格不变；
9. 构造 ray 时使用 **每个 view 自己的 geometry**。

---

# 24. 与常规 CBCT / 3DGS 几何模型对接时必须检查

将 DICOM-CT-PD geometry 接入 NAF、SAX-NeRF、R²Gaussian、3DGS-style CT renderer 时，重点检查：

### 24.1 Handedness

DICOM-CT-PD：

```text
left-handed
```

而很多 graphics / NeRF / 3DGS pipeline 默认：

```text
right-handed
```

必须显式做坐标系转换。

### 24.2 Detector shape

真实 scanner 可能：

```text
CYLINDRICAL
```

而多数 neural CT 项目的默认模型通常是：

```text
FLAT detector
```

二者的 pixel → ray direction 映射不同。

### 24.3 Principal ray

不要假设 detector center 为：

```python
(cx, cy) = ((W-1)/2, (H-1)/2)
```

优先读取：

```text
DetectorCentralElement
```

其值允许是非整数。

### 24.4 Source position

不要简单使用 detector focal center。

应考虑：

```text
SourceAngularPositionShift
SourceAxialPositionShift
SourceRadialDistanceShift
```

### 24.5 Helical trajectory

螺旋 geometry 应由每个 projection 的：

```text
(theta_i, z_i)
```

实际字段构造。

不要仅使用：

```python
z = pitch * theta
```

作为真实 scanner geometry 的替代。

---

# 25. 最重要字段：Agent 最小必读集合

如果只允许 Agent 记住一组字段，优先以下这些：

```text
# Projection
(7FE0,0010) PixelData
(0028,1052) RescaleIntercept
(0028,1053) RescaleSlope
(7039,1009) LogFlag

# Detector
(7029,1010) NumberofDetectorRows
(7029,1011) NumberofDetectorColumns
(7029,1002) DetectorElementTransverseSpacing
(7029,1006) DetectorElementAxialSpacing
(7029,100B) DetectorShape
(7031,1033) DetectorCentralElement
(7031,1031) ConstantRadialDistance

# View geometry
(7031,1001) DetectorFocalCenterAngularPosition
(7031,1002) DetectorFocalCenterAxialPosition
(7031,1003) DetectorFocalCenterRadialDistance

# True source offset
(7033,100B) SourceAngularPositionShift
(7033,100C) SourceAxialPositionShift
(7033,100D) SourceRadialDistanceShift
(7033,100E) FlyingFocalSpotMode

# Helical scan
(7037,1009) TypeofProjectionData
(7037,100A) TypeofProjectionGeometry
(0018,9311) SpiralPitchFactor
(7033,1013) NumberofSourceAngularSteps
(7033,1067) Timestamp

# Physical / calibration
(7033,1065) PhotonStatistics
(7041,1001) WaterAttenuationCoefficient

# Correction state
(7039,1003) BeamHardeningCorrectionFlag
(7039,1004) GainCorrectionFlag
(7039,1005) DarkFieldCorrectionFlag
(7039,1006) FlatFieldCorrectionFlag
(7039,1007) BadPixelCorrectionFlag
(7039,1008) ScatterCorrectionFlag
```

---

# 26. 一句话总结

DICOM-CT-PD 的核心不是“一个 DICOM 文件里有一张 projection”，而是：

> **每个 projection 文件同时提供该 view 的真实 detector readout 和足够重建 source–detector ray geometry 的参数；对螺旋 CT，应逐 view 使用 angle、axial position、source shift、detector shape 和 detector central element 构造真实射线，而不是套用理想 circular flat-detector CBCT 几何。**
