# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Spiral-GS integrates helical CT data paths (from `r2_gaussian_spiral`) with the tiled CUDA rasterizer and voxelizer from FaCT-GS for fast CT reconstruction via Gaussian Splatting. The project uses Hydra for configuration, PyTorch 2.4.1 + CUDA 12.1, and seven git submodules that provide CUDA kernels and utilities.

## Essential commands

```bash
# Environment
conda activate fact-gs
export GLM_HOME="$(pwd)/fact_gs/submodules/glm"   # required by rasterizer/voxelizer submodules
export MPLCONFIGDIR=/tmp/spiral-gs-matplotlib

# Training — reconstruction (projection-driven)
python train_recon.py \
  model.data_source_path=/path/to/spiral/dataset \
  model.model_path=/path/to/output

# Training — volume fitting (direct 3D supervision)
python train_volume.py \
  model.data_source_path=/path/to/dataset \
  model.model_path=/path/to/output

# Fast exploratory run
python train_recon.py optim.steps=20000 optim.use_fused_ssim=true \
  optim.max_num_gaussians_absolute=100000 \
  model.data_source_path=/path/to/data model.model_path=/path/to/output

# Evaluate a trained model
python test_model.py model.model_path=/path/to/model model.data_source_path=/path/to/data

# Run tests
python -m pytest tests/test_spiral_geometry.py -v
python -m pytest tests/ -v
```

## Architecture

### Top-level scripts

- **`train_recon.py`** — Main reconstruction training: renders X-ray projections through the Gaussian model, compares against ground-truth projections with L1 + SSIM loss, and voxelizes for 3D evaluation. Uses `SceneRecon` for data, `rasterize_proj` for rendering, and `voxelize_vol` for 3D metrics.
- **`train_volume.py`** — Direct volume fitting: optimizes Gaussians against a ground-truth 3D volume (no projection rendering). Uses `SceneVol` and `voxelize_vol`.
- **`test_model.py`** — Offline evaluation of a saved checkpoint against both projection and volume metrics.

### `fact_gs/` — Core library

- **`fact_gs/__init__.py`** exposes the two fused CUDA operations: `rasterize_proj` (tiled projection rendering with fused backward pass) and `voxelize_vol` (tiled volume voxelization with fused backward pass).
- **`fact_gs/r2_gaussian/gaussian/`** — `GaussianModel` (density, position, scale, rotation parameters per Gaussian) and initialization factories (`initialize_gaussian`, `initialize_gaussian_from_proj`, `initialize_gaussian_from_vol`, `initialize_gaussian_from_prior`).
- **`fact_gs/r2_gaussian/dataset/`** — `SceneRecon` (projection-driven), `SceneVol` (volume-driven), and `dataset_readers.py` which handles both circular and helical camera loading with per-projection `z_shift`.
- **`fact_gs/r2_gaussian/utils/`** — Loss functions (L1, SSIM, 3D TV, frequency-magnitude), image metrics (PSNR/SSIM for 2D projections and 3D volumes), and camera utilities.
- **`fact_gs/submodules/`** — Git submodules: `gs_ct_rasterizer` (projection CUDA kernel), `gs_voxelizer` (voxelization CUDA kernel), `fused_ssim`, `fused_3d_tv`, `TIGRE` (tomographic reconstruction toolkit), `simple_knn`, `glm`.
- **`fact_gs/utils/`** — Volume I/O, quantization, profiling, Gaussian visualization helpers.

### `config/` — Hydra configuration

Hierarchical config with defaults specified per config file:
- **`default_recon.yaml`** → model: `model_default_recon`, optim: `optim_default_recon`, eval: `eval_default`
- **`default_volume.yaml`** → model: `model_default_volume`, optim: `optim_compress_volume`
- **`model/`** — Gaussian count, init mode, data paths, density thresholds, scale bounds.
- **`optim/`** — Step count, learning rates (position/density/scaling/rotation with linear decay), loss weights (lambda_dssim, lambda_tv, lambda_frequency), densification schedule, early stopping (SSIM threshold or time limit).
- **`eval/`** — Evaluation cadence (`every_n_steps`), visualization toggles.

Override any parameter at the CLI: `model.num_gaussians=100000 optim.steps=20000`.

### `spiral_tools/` — Spiral-specific utilities

- **`data_preprocess/`** — DICOM spiral preprocessing (Python + MATLAB scripts for helical scan conversion).
- **`data_generator_usr/`** — Synthetic and real spiral dataset generation, including NAF-format converters.
- **`scripts/`** — Experiment launchers, parameter sweeps, baseline comparison scripts, and format conversion utilities. Legacy scripts referencing `train.py` need conversion to `train_recon.py`/Hydra before use.

### `data_preprocess/` — DICOM preprocessing pipeline

Contains `norm_pipeline.py` (normalization helpers) and MATLAB scripts (`dicom_spiral_process.m`, `CAT.m`) for helical DICOM conversion.

### `plugins/` — Plugin system

`plugins/training/` contains training pipeline plugins (extensible hook points for the training loop).

## Key technical constraints

- **Import order matters**: `import tigre` must come before `import torch` to avoid GPU initialization errors.
- **`GLM_HOME`** must be set to `fact_gs/submodules/glm` before installing or running the submodule CUDA packages.
- **Coordinate system**: `coord_left` projection/camera convention; datasets with `coord_left: true` in scanner config get a volume flip along axis 0 during evaluation.
- **Spiral data**: each projection entry in `meta_data.json` carries a `z_shift` value (same physical unit as scanner geometry). Circular datasets work because missing `z_shift` defaults to 0.
- **Quality-first defaults**: 30,000 steps, densification every 100 steps from step 500 through step 15,000, density pruning threshold `1e-5`, DSSIM weight `0.25`, absolute ceiling of 500,000 Gaussians.
- **Initialization**: `auto` mode looks for `init_<dataset>.npy` in the data directory; if absent, falls back to FDK-based uniform sampling ("intensity" mode). Set `save_generated_init: True` to cache the FDK init for future runs.
- **This is an integration workspace** — the source submodule projects (r2_gaussian, gs-ct-rasterizer, etc.) remain unchanged; all integration logic lives in this repo's top-level scripts and configs.

## Dataset contract

Datasets use the r2_gaussian `meta_data.json` layout with projections under `proj_train`/`proj_test`. Spiral datasets add `z_shift` to each projection entry. The scanner config block (`scanner_cfg`) defines `offOrigin`, `nVoxel`, `sVoxel`, `dVoxel`, and optional `coord_left`.
