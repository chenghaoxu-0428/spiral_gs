---
name: automate-fact-gs-training
description: Automate end-to-end FaCT-GS experiments in this repository. Use when starting, running, resuming, or monitoring CT reconstruction, volume-prior fitting, volume compression, warm-start reconstruction, one of the paper experiment scripts, or a batch of FaCT-GS training jobs. Coordinate the spiral-gs environment, R2-compatible data preparation, Hydra overrides, GPU scheduling, preflight validation, independent-terminal live progress, and result collection.
---

# Automate FaCT-GS Training

## Resolve the run

1. Work from the repository containing `train_recon.py`, `train_volume.py`, `config/`, and `fact_gs/`.
2. Resolve the exact Python executable. Prefer the `spiral-gs` Conda environment from `environment.yml`, but inspect installed environments and require an explicit choice before an expensive run. Use the same interpreter throughout.
3. Classify the task:
   - reconstruction: `train_recon.py` with default `default_recon`
   - volume prior: `train_volume.py` with default `default_volume`
   - volume compression: `train_volume.py --config-name compress_volume`
   - warm-start reconstruction: first fit a prior, then run `train_recon.py --config-name fromPrior_recon model.prior_path=<point_cloud.pickle>`
   - paper study: inspect the matching `experiments/*.sh` and its helper before execution
4. Invoke `prepare-fact-gs-dataset` when data is missing or uncertain. Both training entry points consume the R2-Gaussian `meta_data.json` layout; volume fitting also requires the configured `model.vol_name` as `.npy` or `.tiff`.
5. Express run changes as Hydra overrides, for example:
   ```bash
   <python> train_recon.py model.data_source_path=<data> model.model_path=<output> optim.steps=30000
   ```
   Use `--config-name <name>` only to select a complete preset. Do not pass R2-Gaussian argparse flags such as `-s`, `-m`, or `--iterations`.

## Validate and launch

1. Inspect GPUs with `nvidia-smi`. For independent jobs, assign at most one job per selected GPU by default. A single FaCT-GS process is not distributed across multiple GPUs.
2. Resolve every task's target, config name, ordered Hydra overrides, dataset, output directory, initialization mode, and GPU. Preserve user-specified paths. Never mix a new run into a non-empty model directory without explicit reuse approval.
3. Invoke `validate-fact-gs-training` for every distinct command with the chosen interpreter. Treat `[FAIL]` as blocking and review relevant warnings.
4. Show the exact command and obtain confirmation immediately before launching training.
5. Set `CUDA_VISIBLE_DEVICES=<physical-index>` for isolation. The process then sees its selected GPU as local device 0.
6. After the user confirms the dataset and exact command, launch training in a **new graphical terminal window**, never in Codex's own PTY/background exec session. Use the bundled launcher:
   ```bash
   <python> <skill-dir>/scripts/launch_training_terminal.py \
     --cwd <repo> \
     --run-dir training_logs/<unique-run-name> \
     --title "FaCT-GS: <run-name>" \
     --env CUDA_VISIBLE_DEVICES=<physical-gpu> \
     -- <python> train_recon.py <ordered-hydra-overrides...>
   ```
   Request GUI permission when required. The new terminal must display the native `tqdm` output in real time and remain independent of Codex's terminal/output. Do not pipe training into Codex, duplicate its progress bar, or launch it with a Codex-owned persistent PTY.
7. Read `<run-dir>/terminal.log`, `<run-dir>/pid`, and `<run-dir>/exit_code` from Codex. Poll at least every 30–60 seconds while monitoring; immediately report NaN/Inf, CUDA errors, OOM, or a non-zero exit. Treat a missing `exit_code` with a live PID as running. Do not infer success merely because the graphical terminal closed.
8. Report one compact status line using the latest native progress and evaluation output:
   `Train: <step>/<steps> (<percent>%) | loss <loss> | ssim3d <value> | psnr3d <value> | ssim2d <value> | psnr2d <value>`
   Keep the latest evaluation values until replaced. Use `?` before first observation. For reconstruction only, `<model_path>/tensorboard` is a fallback when PTY output is unavailable.
9. Continue monitoring until completion when requested, then invoke `collect-fact-gs-results`.

## Project-specific guardrails

- `optim.steps` counts optimization/rasterization passes. Reconstruction iterations reported by evaluation are full camera sweeps; volume fitting has `iteration == step`.
- Prefer `model.init_mode=auto` for cold-start reconstruction. It uses `init_<dataset>.npy` when present and otherwise generates/caches an R2-compatible FDK initialization. Use `prior` only with a readable `point_cloud.pickle`.
- TensorBoard is configured for reconstruction through `tensorboard.enabled`; do not assume volume fitting emits event files.
- Early exits may be intentional when `optim.ssim3d_early_stop=true` or `optim.training_time_limit_seconds>0`.
- Give every launch a new, non-existing `<run-dir>` so logs and status cannot be confused with an earlier attempt. Keep logs outside `model.model_path`, which must remain empty before training.
- If no supported GUI terminal is installed or no display is available, stop and explain the blocker. Do not silently fall back to Codex's PTY; use it only when the user explicitly requests that fallback.
- For `experiments/*.sh`, inspect `MAIN_ROOT`, generated model paths, and the companion result collector before launch. Do not rewrite a curated study into ad-hoc commands unless requested.
- After multiple comparable runs, offer `summarize-fact-gs-results`.
