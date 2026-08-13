# FaCT-GS runtime contract

## Repository and environment

The project root contains `train_recon.py`, `train_volume.py`, `test_model.py`, `config/`, and `fact_gs/`. `environment.yml` defines the reference `spiral-gs` environment: Python 3.10, PyTorch 2.4.1, CUDA 12.1, Hydra 1.3, and OmegaConf 2.3. Treat versions as the tested baseline, not proof that every other compatible build is invalid.

CUDA training depends on TIGRE plus these importable packages/extensions:

- `simple_knn._C`
- `fused_ssim`
- `fused_3d_tv`
- `gs_ct_rasterizer`
- `gs_voxelizer`

Use the exact interpreter that will launch training. Record Python, PyTorch/CUDA, driver, GPU, NVCC, compiler, and import errors before proposing a rebuild. Installing packages, rebuilding extensions, changing CUDA links or shell profiles, and deleting build artifacts require an explicit fix request.

## Hydra configuration

Training uses Hydra defaults in `config/` and dot-list overrides. Reconstruction defaults to `default_recon`; volume fitting to `default_volume`; compression to `compress_volume`; prior reconstruction to `fromPrior_recon`. Hydra overrides are applied in order and use grouped keys such as:

- `model.data_source_path`, `model.model_path`, `model.num_gaussians`
- `model.init_mode`, `model.prior_path`, `model.vol_name`
- `optim.steps`, learning-rate fields, loss weights, densification controls, early-stop/time-limit fields
- `eval.*`, `profile.*`, and reconstruction `tensorboard.*`

`--config-name` selects a preset and is not a dot-list override. The working directory remains unchanged because project configs set `hydra.job.chdir: false`.

## Data and artifacts

Both entry points use R2-Gaussian-style `meta_data.json`. Reconstruction consumes projections and supports `auto`, `gradient`, `intensity`, `precomputed`, and `prior` initialization. `auto` loads `init_<dataset>.npy` when available or creates an FDK-based initialization. `prior` requires a `point_cloud.pickle` from volume fitting.

Volume fitting requires `model.vol_name` (normally `vol_prior` or `vol_gt`) as `.npy` or `.tiff`. Outputs may include `ckpt/`, `point_cloud/step_<N>/point_cloud.pickle`, reconstructed TIFF/preview files, evaluation YAML, a sibling `<model>_metrics_final.yml`, and reconstruction TensorBoard events under `<model_path>/tensorboard`.

Reconstruction `<model>_metrics_final.yml` written by `train_recon.py` includes `psnr_3d`, `ssim_3d`, test-split `psnr_2d` / `ssim_2d`, and `time_training_seconds`. Volume fitting writes the same schema; `psnr_2d` / `ssim_2d` may be null when projection metrics are unavailable.

`optim.steps` is the global optimization step count. In reconstruction, an evaluation iteration is a complete sweep of training projections; for volume fitting, step and iteration are equivalent.
