# 统一数据预处理（norm）

入口：

```bash
python data_preprocess/norm_pipeline.py \
  --config data_preprocess/configs/norm_pipeline.example.yml
```

先检查配置而不读取数据或调用 GPU：

```bash
python data_preprocess/norm_pipeline.py \
  --config data_preprocess/configs/your.yml \
  --validate-only
```

`dataset_type: syn` 读取 `raw_gt` DICOM 切片，由 YAML scanner/spiral 参数正投影。
仿真螺旋直接使用配置的 `z_start` 和 `z_end`，不会针对投影截断自动收缩范围；
`raw_gt` 也可指向已预处理的 `.npy` 体数据；管线会验证形状与有限值，并按需重采样。
`dataset_type: real` 同时读取 `raw_proj`，并通过 pydicom 数值 tag 直接解析 Siemens
CT-PD 私有几何字段。spiral 始终生成；stitch 默认关闭，仅在配置
`stitch.enabled: true` 时生成，并使用独立的 `stitch.n_train/n_test`。

real 数据的 DSD、DSO、探测器尺寸、pitch 和每圈采样数来自投影 DICOM；YAML
保留体素分辨率等重建设置。默认按照全部投影的 z 最小值/最大值自动更新
`scanner.sVoxel` 和 `scanner.offOrigin`，行为由 `real.*` 配置控制。

省略 `n_test` 时，管线默认使用 `ceil(1.5 * n_train)` 个测试视角；显式配置
`n_test` 时以配置值为准。stitch 的 `n_test` 同样相对于 stitch 自己的
`n_train` 计算。

所有配置文件统一放在 `data_preprocess/configs/`。配置中的 `output_root` 建议设置为
`data`，输出目录遵循与训练结果相同的层级：

```text
data/{real|syn}/{organ}/{spiral|stitch}/ntrain{N}/{model}/
  vol_gt.npy
  init_{model}.npy
  meta_data.json
  proj_train/*.npy
  proj_test/*.npy
```

`init_*.npy` 是用训练集投影通过 `ct_utils.py` 的 `recon_volume` 进行 FDK 后采样的
`[x,y,z,density]` 点云；螺旋轨迹会同时传入各视角的 z 位移。它不是
`vol_gt.npy` 的复制品。FDK/正投影依赖 TIGRE 和 CUDA。
预处理初始化与训练期 `model.init_mode=intensity` 共用同一实现：按训练场景尺度
缩放投影与几何，对 FDK 体做非负裁剪、p99.5 归一化和 `[0, 1]` 裁剪，再从
阈值以上体素无放回均匀采样。`init.density_threshold: auto` 对应训练默认阈值
`0.05`；数值阈值仍受支持。
