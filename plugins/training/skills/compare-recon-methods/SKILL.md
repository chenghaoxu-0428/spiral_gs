---
name: compare-recon-methods
description: Collect CT reconstruction volumes and metrics from FDK, SAX-NeRF (NAF, Lineformer, IntraTomo), and FaCT-GS, then draw a multi-method slice plus ROI comparison with plot_roi.py. Use when gathering baseline vs FaCT-GS results, plotting ROI comparison grids, or assembling FDK/NAF/Lineformer/IntraTomo/FaCT-GS/GT figures.
---

# Compare Reconstruction Methods

Collect finished volumes and metrics, then draw one figure with every method's slice and two ROIs. Do not retrain during this skill.

## Resolve inputs

1. Work from the repository containing `plot_roi.py`, `train_recon.py`, and `data/`.
2. Require an R2-Gaussian dataset directory with `meta_data.json` and `vol_gt.npy`.
3. Default method order (left to right): `fdk`, `naf`, `lineformer`, `intratomo`, `fact-gs`, `gt`.
4. Resolve paths from `data/{real|syn}/{organ}/{spiral|stitch}/ntrain<N>/<model>/` unless the user gave explicit locations:

| Method | Volume | Metrics |
| --- | --- | --- |
| FDK | `<dataset>/fdk_vol.npy` | optional 3D PSNR/SSIM vs `vol_gt.npy` via `metric_vol` |
| NAF / Lineformer / IntraTomo | latest `<sax-root>/output/.../<method>/eval/epoch_*/image_pred.npy` | that epoch's `stats.txt` (`proj_ssim`, `proj_psnr`, `psnr_3d`, `ssim_3d`) and `training_time_sec.txt` |
| FaCT-GS | latest `<model>/eval/step_*/vol_pred.tiff` | sibling `<model>_metrics_final.yml` (`psnr_2d`, `ssim_2d`, `psnr_3d`, `ssim_3d`, `time_training_seconds`) |
| GT | `<dataset>/vol_gt.npy` | none |

5. FDK must be the preprocessing helical FDK volume (`data_preprocess/export_fdk_vol.py` / `reconstruct_fdk_volume`), not a copy of `vol_gt.npy` and not a sampled `init_*.npy` point cloud.
6. SAX-NeRF runs live in `/opt/data/private/sax-nerf` unless `SAX_NERF_ROOT` is set. Wait until each method has `eval/epoch_*` plus `image_pred.npy` and `stats.txt`.
7. Keep FaCT-GS 2D metrics from `render_test_*`. Do not recompute FaCT-GS 3D SSIM with another protocol. SAX numbers stay as written in `stats.txt`. FDK 3D metrics are derived from `metric_vol` and must be labeled as such.

## Collect and plot

From the repository root:

The collector inserts the repository root onto `sys.path` so `fact_gs` imports work even when the script is invoked by path rather than as a module.

```bash
<python> plugins/training/skills/compare-recon-methods/scripts/collect_and_plot.py \
  --dataset data/syn/ldctl004/spiral/ntrain500/r2gs \
  --sax-root /opt/data/private/sax-nerf \
  --factgs-model models/syn/ldctl004/spiral/ntrain500/factgs_spiralfdk \
  --output output/syn/ldctl004/spiral/ntrain500/comparison \
  --axis 2 --slice-idx 128 \
  --roi-red 176,76,48,48 --roi-blue 108,120,48,48
```

Default abdomen ROIs (`ldctl004` axial `axis=2`, slice `128`) match `plot_roi.py`. Change slice/ROI only when the anatomy requires it.

Outputs written beside `--output`:

- `metrics.json` / `metrics.csv`
- `roi_comparison.png` (one row: full slice + two ROI zooms per method). Caption lists 2D PSNR/SSIM, 3D PSNR/SSIM, and training time; missing values (FDK 2D, GT, FDK time) are `/`. FDK 3D metrics are derived vs `vol_gt` with `metric_vol`; SAX and FaCT-GS stay as written in `stats.txt` / `metrics_final.yml`. Integer TIFF volumes such as FaCT-GS `vol_pred.tiff` are scaled to `[0, 1]`. Default display (`window=match_gt`) histogram-matches every method to GT so brightness is comparable; `per_method` uses independent percentiles, `gt` uses a shared GT window.

## Report

Record each metric's source (`observed` from YAML/`stats.txt`, `derived` for FDK 3D, or `missing`). Missing 2D SSIM for FDK and GT is expected. Print a compact table of method, 2D SSIM, PSNR/SSIM 3D, training time, and the figure path. Do not overwrite curated paper figures unless the user asked.
