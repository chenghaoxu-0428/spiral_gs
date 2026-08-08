---
name: validate-fact-gs-training
description: Run a read-only preflight for FaCT-GS training. Use before or while debugging train_recon.py, train_volume.py, Hydra presets/overrides, cold or prior initialization, volume compression, CUDA/PyTorch/TIGRE and compiled FaCT-GS extensions, dataset paths, model outputs, GPUs, early stopping, or optimization and densification settings.
---

# Validate FaCT-GS Training

## Workflow

1. Identify the exact Python, target (`recon` or `volume`), Hydra config name, ordered overrides, source data, model output, initialization mode, and GPU. Do not train during preflight.
2. From the repository root, run:
   ```bash
   <python> <skill-dir>/scripts/preflight.py --repo . --target recon \
     --config-name default_recon --gpu 0 \
     --override model.data_source_path=<data> \
     --override model.model_path=<output>
   ```
   Use target `volume` and config `default_volume` or `compress_volume` as appropriate. Pass overrides in command order.
3. Treat exit code 2 as blocking. Resolve every `[FAIL]`; inspect every relevant `[WARN]`.
4. Report status, failures, warnings, confirmed environment/effective Hydra parameters, and the exact safe next command.
5. Start training only when explicitly requested.

## Validation rules

- Require repository markers `train_recon.py`, `train_volume.py`, `config/`, and `fact_gs/`.
- Compose the selected Hydra config rather than parsing one YAML in isolation. Validate `model.data_source_path`, `model.model_path`, and positive `model.num_gaussians` and `optim.steps`.
- Require `meta_data.json`. For reconstruction, verify projection directories and initialization/prior requirements. For volume fitting, verify the configured `model.vol_name` volume.
- Check `0 < model.scale_min < model.scale_max`, non-negative loss weights and time limit, valid learning-rate schedules, and densification intervals/fractions when enabled.
- Warn on non-empty output directories. Do not create, delete, compile, install, or alter environment variables during preflight.
- Validate CUDA with the selected Python and import `tigre`, `simple_knn._C`, `fused_ssim`, `fused_3d_tv`, `gs_ct_rasterizer`, and `gs_voxelizer` according to the selected target. Read `references/fact-gs-contract.md` before remediation.
