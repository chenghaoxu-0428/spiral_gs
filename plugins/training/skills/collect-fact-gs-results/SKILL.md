---
name: collect-fact-gs-results
description: Collect and validate FaCT-GS reconstruction, volume-fitting, compression, warm-start, scaling, initialization, and ablation results. Use to extract PSNR/SSIM, projection metrics, training time, Gaussian/model size, convergence curves, final volumes and point clouds from FaCT-GS YAML outputs, model directories, TensorBoard events, logs, and experiments/helpers collectors.
---

# Collect FaCT-GS Results

## Collect evidence

1. Identify the task type, effective Hydra configuration, dataset, model directory, final step, stopping reason, and evaluation split.
2. Inspect, in order:
   - `<model-parent>/<model-name>_metrics_final.yml`
   - evaluation YAML files and reconstructed volumes under the model directory
   - `point_cloud/step_*/point_cloud.pickle`, `ckpt/`, and the PTY log
   - `<model_path>/tensorboard` for reconstruction
   - the matching `experiments/helpers/collect_*_results.py` for a paper study
3. Prefer the repository's study-specific collector when its expected layout matches. For normalized spiral comparisons, inspect `spiral_tools/scripts/collect_r2_results.py` before use.
4. Record each metric's exact source and status (`observed`, `derived`, or `missing`). Never substitute training loss for evaluation quality.

## Required record

- dataset/case, acquisition geometry and split size when known
- task (`recon`, `volume-prior`, or `compression`), config name, initialization mode, and method label
- final step/iteration and stopping reason
- From sibling `<model>_metrics_final.yml` when present: `psnr_3d`, `ssim_3d`, `psnr_2d`, `ssim_2d`, and `time_training_seconds`
- Reconstruction `metrics_final.yml` must include test-split 2D projection PSNR/SSIM as `psnr_2d` / `ssim_2d` (sourced from `render_test_*` evaluation). Treat missing 2D keys as incomplete recon output, not optional.
- Volume-prior/compression may leave `psnr_2d` / `ssim_2d` null when projection evaluation was not run
- training time in seconds and minutes, converted once from `time_training_seconds` or `training_time_seconds`
- final Gaussian count and model/compressed size when requested and supported
- paths to the final volume, point cloud, config/run log, and every metric source

Keep unavailable real-scan 3D ground-truth metrics as `/`. Do not silently recompute metrics with another normalization or replace the project's axis-wise R2-compatible 3D SSIM protocol.

## Validate and export

Validate numeric finiteness, units, task type, final checkpoint correspondence, and comparable evaluation splits. TensorBoard curves should use cumulative elapsed training minutes when timing is available; label step-based fallbacks. Write CSV or JSON plus a concise summary beside the experiment results, without overwriting curated paper outputs unless approved.
