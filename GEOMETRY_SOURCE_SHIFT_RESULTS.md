# P0/P1 几何修正实验结果报告

日期：2026-08-15  
分支：`fix/real-cylindrical-detector-geometry`  
数据：`data/real/ldctl004/spiral/ntrain1000/r2gs_cyl_detector`（1000 train / 1500 test）  
对比基准：`factgs_cyl_detector_uoffset_m0275`（旧 runtime 注入 u=−0.275，2D 41.360 dB）

## 一、本次新增的可复现基础设施

1. **侧车数据** `data_preprocess/extract_source_shifts.py` → 数据集内 `source_shifts.json`：
   - 32786 张 DICOM 全部为 `FFSZ`；三个 shift 字段严格逐视角交替（周期 2，各状态 16393 例）。
   - angular：{0.000333, 0.000336} rad，状态差 3e-6 rad ≈ 0.08 µm，**可忽略**。
   - axial：{0, −0.66} mm；radial：{0, +5.45} mm；两字段状态**反相关**（一致率 0.000）。
   - 对齐验证：与 `meta_data.json` 的 angle/z_shift 完全一致（`_split_indices(32786, 1000, 1500, 0)`）。
2. **已提交的相机注入机制**（`model.geometry` Hydra 组）：
   - `u_offset_px`：水平主点偏移（`camera_utils.loadCam` 改 `projection_matrix[2,0]` 并重算 `full_proj_transform`）。
   - `source_shift_mode: off | angle_z | angle_z_radial`：按 CT-PD 语义逐视角加 Source*Shift（angle 加 rad、z 加场景单位轴向 shift、radial 按 `radial_mode: dso|dsd` 处理，dsd 模式同步重算 FoV）。
   - 训练与评估共用同一相机路径；`train_recon.py` 落盘 `geometry_used.yml`。
   - 复现验证：committed u=−0.275 与旧 runtime 基线差 **0.001 dB**（41.359 vs 41.360）。

## 二、最终对比（30000 步）

| 模型 | PSNR2D | SSIM2D | PSNR3D | SSIM3D | Δ2D vs A | 训练时长 |
|---|---:|---:|---:|---:|---:|---:|
| A0 旧 runtime u=−0.275 | 41.360 | 0.98489 | 20.135 | 0.6492 | — | 277.9 s |
| A committed u=−0.275 | 41.359 | 0.98483 | 20.138 | 0.6495 | 0.000 | 286.5 s |
| **B u + angle/z（ax=+1）** | **41.432** | **0.98535** | 20.144 | 0.6493 | **+0.073** | 290.1 s |
| C u + angle/z/radial（→DSD，−1） | 40.882 | 0.98358 | **20.320** | **0.6545** | −0.477 | 278.6 s |
| D analytic u=−0.28125 | 41.334 | 0.98447 | 20.135 | 0.6495 | −0.025 | 284.8 s |

C 复用 `factgs_cyl_detector_dynamic_fsf`（渲染交叉验证：committed 几何下 40.867 dB ≈ 其记录 40.882 dB，确认其训练几何即 angle_z_radial + radial→DSD(sign −1)，无需重训）。

## 三、逐视角 / 状态分析

| 模型 | stateA | stateB | gap(B−A) | 回归 R²（state/sin/cos(angle)/z） |
|---|---:|---:|---:|---:|
| A committed | 41.454 | 41.263 | −0.192 | 0.046 |
| B angle_z | 41.550 | 41.314 | −0.236 | 0.050 |
| C angle_z_radial | 40.932 | 40.832 | −0.100 | 0.053 |
| D analytic u | 41.446 | 41.223 | −0.223 | 0.044 |

- 训练前 400 视角符号扫描（固定旧基线模型重渲染）：nominal+u 41.546；angle_z ax=+1 41.463；radial→DSD(−1) 41.376；radial→DSO 明显更差（39.0–40.3）。CT-PD 加法语义（ax=+1）确认最优符号。
- 所有模型（含注入 shift 的 B/C）state B 仍比 state A 差约 0.1–0.24 dB；逐视角 PSNR 对 state/angle/z 的线性回归 R² 仅 0.044–0.053。source shift 字段即使按 CT-PD 语义逐视角注入并重训，也无法消除 state 相关残差。

## 四、结论

1. **P0 主结论：逐视角 source shift 不是主要瓶颈，不建议作为正式修正。**
   - B（只改 angle/z）在 25000/30000 步分别 +0.119/+0.073 dB，方向稳定但量级很小，远小于 u 主点修正的 +2.88 dB，也低于 handoff 预估的 oracle 上限（+0.161 dB，约兑现一半）。
   - C（含 radial→DSD）2D 反而 −0.477 dB；3D 的 +0.185 dB 提升发生在 `vol_gt` 与投影不自洽的背景下，不能作为采纳依据。
   - FFSZ 字段真实且有可测影响，但按最合理物理解释重训后没有实质收益，说明 55.9% 的低频残差能量主要来源仍在别处（圆柱展平插值/中心定义、散射/束硬化、`vol_gt` 失配、扫描内非静态变化）。
2. **P1 解析主点映射成立：** u=−0.28125（DetectorCentralElement 1.125/4 下采样）与经验最优 −0.275 仅差 0.025 dB，证明 DICOM 解析映射可替代经验扫描值。
3. **正式化建议：** 将 u=−0.275 px（或解析 −0.28125 px）设为 real 数据默认；source shift 保持 off；B 的 angle/z 注入可作为可选物理项保留（+0.07 dB，代价极小），但不应默认开启。
4. **未完成的可选后续**（按 handoff P1/P2/P3）：逐视角残差更深层联合回归；圆柱展平中心定义与 raw/processed 预处理一致性核查；用 corrected geometry 对 `vol_gt` 前向投影做自洽性分离；60k 步/超参实验仍不优先。

## 五、关键文件

- 提取脚本：`data_preprocess/extract_source_shifts.py`；侧车：`.../r2gs_cyl_detector/source_shifts.json`
- 注入实现：`fact_gs/r2_gaussian/dataset/dataset_readers.py`、`fact_gs/r2_gaussian/utils/camera_utils.py`、`fact_gs/r2_gaussian/dataset/__init__.py`、`config/model/model_default_{recon,test}.yaml`、`train_recon.py`
- 分析工具：`experiments/helpers/scan_source_shift_signs.py`、`experiments/helpers/analyze_variant_results.py`
- 新模型：`factgs_cyl_detector_uoffset_committed`（A）、`factgs_cyl_detector_uoffset_ffsz_anglez`（B）、`factgs_cyl_detector_uoffset_m028125`（D）；C 沿用 `factgs_cyl_detector_dynamic_fsf`
- 训练日志：`training_logs/ffsz-{uoffset-committed,anglez,analyticu}-20260815/`

## 六、低频残差能量归因（2026-08-15 第二轮诊断）

对 u=−0.275 模型（`factgs_cyl_detector_uoffset_m0275`，`projection_pairs_corrected.npz`，test 1500 视角）做残差分解与物理假设检验。

### 频带与静态性

- 低频（σ_f=0.055 cyc/px 高斯低通）占残差能量 **53.7%**；静态残差图 train/test 相关 **0.993**（真实静态）。
- 静态图（≈7.4% 总能量）为沿 u 的平滑"微笑"bow（中心 −0.006，corr(u²)=0.80），行无关。
- **跨病人验证**：ldctc001（全分辨率 888 列，未下采样，独立模型/数据集）的静态列轮廓与 ldctl004 归一化相关 **0.88**，bow 形状一致（corr(u²)=0.96）→ 该模式是扫描链系统误差，不是病人/下采样/模型产物。

### 假设检验结果（全部在真实数据上直接验证）

| 假设 | 检验方式 | 结果 |
|---|---|---|
| 圆柱展平切点应取真实中心元素（c=−0.28 px） | 从 DICOM 按参数化切点重新展平 GT 扫描 c∈[−1,+0.5] | **否定**：一致参数化（c+相机 u）比经验最优差 ~2 dB |
| 展平半径 ≠ ConstantRadialDistance | 展平半径扫描 R/dsd∈[0.85,1.15] | **否定**：PSNR 最优在 1.0；bow 对 R 不敏感 |
| 先下采样后展平造成失真 | 全分辨率展平→下采样 vs 现顺序 | 顺序差 43 dB 图像差；对模型残差 bow 无改善（模型已适应现顺序） |
| 阳极脚跟/静态列增益 | 全局逐列乘性增益（自由 + 多项式）拟合 | **否定**：+0.003 dB，轮廓幅度 ±0.0007 |
| 束硬化（r∝P²） | P² 顺序最小二乘归因 | **否定**：0.14%（低频带） |
| 散射（r∝LP(P)） | 低通 P 归因 | **否定**：0.09% |
| v 向扇形 cos(γ) 因子 | 无 fan 展平重生成 | **否定**：−0.06 dB，z 依赖模式不变 |
| source state（FFSZ） | 状态交互项归因 | 很小：低频带 1.5%，重训收益 +0.07 dB |
| z 相关 | 逐视角 PSNR/残差 std 对 z 相关 | ldctl004 相关 −0.16/0.23；**ldctc001 相关 −0.05** → 解剖特异，非几何 |

### 归因结论

低频残差（53.7%）的可归因构成：

1. **静态 detector 系统模式 ~7-9% 能量**（oracle 上限约 +0.45 dB）：跨病人、跨分辨率一致的平滑 u-bow，加性而非乘性（增益拟合否定）→ 最可能为扫描链残余暗场/平场类加性系统误差（DICOM 校正标志 ≠ 残余为零）。**这是唯一可通过数据预处理可操作回收的部分。**
2. **每视角尺度项 ~7%**：指标逐视图 max 归一化留下的尺度歧义（非模型误差）。
3. **z/解剖项 ~5%**：病人特异（另一病人无此相关）→ 解剖/运动，几何无法修正。
4. **剩余 ~30% 低频 + 46% 高频**：逐视角拟合误差（噪声、小运动、模型容量），无系统性几何来源。

### 建议

- 几何修正已挖到底：u 主点 −0.275/解析 −0.28125 是唯一有效修正；所有其他几何参数（含 source shift 的三种解释）无收益。
- 若要再压 2D 指标：唯一剩余杠杆是静态 detector 加性校正图（+0.45 dB 上限），可用残差均值图估计后作为预处理或后处理固定校正；注意避免用测试集估计校正图（应只用 train 或独立校正扫描）。
- 3D 指标（~20 dB）受 `vol_gt` 与投影不自洽限制，与 2D 几何修正无关（见 handoff P2）。

## 七、圆柱 detector→flat 重采样对指标的影响（2026-08-15 第三轮）

### 干净合成实验（matched-syn 数据，与真值 flat 投影直接比较）

| 预处理顺序 | 与真值的偏差 |
|---|---:|
| A 当前（先 4× 下采样@184 再展平） | **30.82 dB（2.9% RMS 失真）** |
| B 先展平@原生分辨率再下采样 | **61.54 dB（0.08% RMS）** |
| C 无展平纯下采样（参照） | 68.24 dB |

A 顺序的失真主要来自：184 粗网格上切平面重映射的亚像素线性插值（映射局部拉伸 ~8.7%，位移 ~1px），而 B 在原生 736 列分辨率展平、插值误差被随后的 4× 下采样平均掉。

### 真实数据重训（`r2gs_cyl_detector_flatfirst`，flatten_before_subsample=true，u=−0.275，30000 步）

| 评估 | PSNR2D | SSIM2D | PSNR3D |
|---|---:|---:|---:|
| A 数据 + A 模型（旧顺序，现基线） | **41.360** | 0.9849 | 20.138 |
| B 数据 + B 模型（先展平） | 40.473 | 0.9832 | 19.933 |
| 交叉：A 模型评 B 数据 | 39.246 | — | — |
| 交叉：B 模型评 A 数据 | 41.188 | — | — |

分解：B 数据是"更难"的参考（A 模型跨评 −2.1 dB）；B 模型本身与 A 模型能力相当（评 A 数据仅 −0.15 dB）。u 扫描确认 B 数据最优 u 仍为 −0.30~−0.275。

### 结论

1. **旧顺序（先下采样后展平）的插值失真对 2D 指标是"有利失真"**：它平滑了参考投影，使 41.36 dB 比真实切平面拟合精度虚高约 0.9 dB；换用更忠实的先展平顺序后，诚实数字约为 40.3–40.5 dB。重建本身质量不变（交叉评估证实）。
2. 报告数字时的口径：当前 41.36 含 ~1 dB 预处理失真红利；若追求"指标=真实投影拟合精度"，应改用 flatten_before_subsample 数据并报 ~40.5 dB。
3. 之前的静态 u-bow 中约一半是旧顺序展平插值伪影（顺序差图像的 bow 含量 0.0175 vs 总 bow 0.0337），剩余为扫描链系统误差。
4. 实现已提交：`norm_pipeline.load_real_projections(flatten_before_subsample=...)` + 配置 `real_ldctl004_spiral_ntrain1000_r2gs_cyl_detector_flatfirst.yml` + 数据集/模型 `*_flatfirst*`。默认行为不变。
