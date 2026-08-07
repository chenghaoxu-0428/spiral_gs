# Spiral-GS

Spiral-GS combines the helical CT data path developed in
`r2_gaussian_spiral` with the tiled CUDA rasterizer and voxelizer from
FaCT-GS. The source projects remain unchanged; this directory is an
independent integration workspace.

## Preserved spiral behavior

- Per-projection `z_shift` from `meta_data.json`.
- Helical camera motion through `angle2pose(DSO, angle, z_shift)`.
- `coord_left` projection/camera conversion.
- Spiral DICOM preprocessing, dataset generation, conversion, tuning, and
  experiment helpers under `spiral_tools/`.
- Circular datasets remain compatible because missing `z_shift` defaults to 0.

## GPU acceleration

- `gs_ct_rasterizer`: tiled projection rasterization and fused backward pass.
- `gs_voxelizer`: tiled volume voxelization and fused backward pass.
- `fused_ssim`: fused 2D/3D SSIM.
- `fused_3d_tv`: fused 3D total variation.
- FaCT-GS step-based training, initialization, densification, profiling, and
  evaluation pipeline.

## Quality-first defaults

The default reconstruction profile intentionally restores the initialization,
loss, learning-rate scale, capacity, and optimization budget used by
`r2_gaussian_spiral`: dataset `init_*.npy` when available, legacy SSIM,
30,000 optimization steps,
densification every 100 steps from step 500 through step 15,000, a density
pruning threshold of `1e-5`, DSSIM weight `0.25`, and an absolute ceiling of
500,000 Gaussians. The fast FaCT-GS CUDA kernels remain enabled, with
quality-safe 3-sigma tile bounds. Do not compare
quality against the old run at only 20,000 steps or with the FaCT-GS 1.1x
Gaussian cap.

For a faster exploratory run, override the quality budget explicitly, e.g.:

```bash
python train_recon.py optim.steps=20000 \
  optim.use_fused_ssim=true \
  optim.max_num_gaussians_absolute=100000 \
  model.data_source_path=/path/to/data model.model_path=/path/to/output
```

## Run with the configured environment

```bash
conda activate fact-gs
cd /home/chenghaoxu/Documents/spiral_gs
export MPLCONFIGDIR=/tmp/spiral-gs-matplotlib

python train_recon.py \
  model.data_source_path=/path/to/spiral/dataset \
  model.model_path=/path/to/output
```

For an isolated environment, create `spiral-gs` from `environment.yml` and
install the submodules as described in the upstream README.

## Dataset contract

The dataset uses the standard r2_gaussian `meta_data.json` layout. Spiral
datasets add `z_shift` to each `proj_train` and `proj_test` entry. The value
uses the same physical unit as the scanner geometry before scene scaling.

## Validation

```bash
python -m pytest tests/test_spiral_geometry.py
```

## Layout

- `fact_gs/`: accelerated runtime and r2_gaussian-derived model.
- `fact_gs/r2_gaussian/dataset/dataset_readers.py`: circular + helical camera
  loading.
- `spiral_tools/data_preprocess/`: DICOM spiral preprocessing.
- `spiral_tools/data_generator_usr/`: synthetic and real spiral generation.
- `spiral_tools/scripts/`: migrated experiment/conversion scripts. Legacy
  scripts invoking `train.py` are retained as references and need conversion
  to `train_recon.py`/Hydra before direct use.
