"""Compare geometry-variant training results and analyze per-source-state PSNR.

Reads the final eval YAMLs of each trained model, aligns per-view PSNR with
the source-dynamics sidecar, and reports:
  - summary table (2D/3D metrics, training time);
  - per-source-state mean PSNR (state from the alternating axial shift);
  - per-view PSNR regression on source state / angle periodic terms / z.

Usage:
  python experiments/helpers/analyze_variant_results.py \
      --data <dataset dir> --models <model dir> [<model dir> ...] \
      [--labels label1 label2 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))


def load_metrics(model: Path):
    eval2d = yaml.safe_load((model / "eval/step_030000/eval2d_render_test.yml").read_text())
    eval3d = yaml.safe_load((model / "eval/step_030000/eval3d.yml").read_text())
    metrics = {}
    final = model.parent / f"{model.name}_metrics_final.yml"
    if final.exists():
        metrics = yaml.safe_load(final.read_text())
    return {
        "psnr_2d": float(eval2d["psnr_2d"]),
        "ssim_2d": float(eval2d["ssim_2d"]),
        "psnr_3d": float(eval3d["psnr_3d"]),
        "ssim_3d": float(eval3d["ssim_3d"]),
        "time_s": float(metrics.get("time_training_seconds", np.nan)),
        "psnr_2d_projs": np.asarray(eval2d["psnr_2d_projs"], dtype=np.float64),
    }


def state_from_sidecar(data: Path, split: str = "test") -> np.ndarray:
    sidecar = json.loads((data / "source_shifts.json").read_text())
    shifts = sidecar["shifts"][split]
    # State B = nonzero axial shift (the FFSZ-deflected spot).
    return np.asarray([1.0 if abs(s["axial"]) > 1e-9 else 0.0 for s in shifts])


def per_state_stats(psnr_projs: np.ndarray, state: np.ndarray) -> dict:
    a = psnr_projs[state == 0.0]
    b = psnr_projs[state == 1.0]
    return {
        "n": len(psnr_projs),
        "mean_stateA": float(a.mean()),
        "mean_stateB": float(b.mean()),
        "gap_B_minus_A": float(b.mean() - a.mean()),
    }


def regression(psnr_projs: np.ndarray, state: np.ndarray, angles: np.ndarray, z: np.ndarray) -> dict:
    """Linear regression of per-view PSNR on source state, angle periodics, z."""
    x = np.stack(
        [np.ones_like(psnr_projs), state, np.sin(angles), np.cos(angles), z],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(x, psnr_projs, rcond=None)
    resid = psnr_projs - x @ coef
    r2 = 1.0 - float((resid ** 2).sum() / ((psnr_projs - psnr_projs.mean()) ** 2).sum())
    return {"r2": r2, "state_coef": float(coef[1]), "sin_coef": float(coef[2]),
            "cos_coef": float(coef[3]), "z_coef": float(coef[4])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    meta = json.loads((args.data / "meta_data.json").read_text())
    angles = np.asarray([e["angle"] for e in meta["proj_test"]])
    z = np.asarray([e["z_shift"] for e in meta["proj_test"]])
    state = state_from_sidecar(args.data)
    labels = args.labels or [m.name for m in args.models]

    rows = []
    for label, model in zip(labels, args.models):
        metrics = load_metrics(model)
        per = per_state_stats(metrics["psnr_2d_projs"], state)
        reg = regression(metrics["psnr_2d_projs"], state, angles, z)
        rows.append({**metrics, "label": label, **per, **reg})

    print("summary:")
    print(f"{'model':<58} {'psnr2d':>7} {'ssim2d':>7} {'psnr3d':>7} {'ssim3d':>7} {'stateA':>7} {'stateB':>7} {'gap':>7} {'regR2':>6}")
    for r in rows:
        print(
            f"{r['label']:<58} {r['psnr_2d']:7.3f} {r['ssim_2d']:7.5f} {r['psnr_3d']:7.3f} "
            f"{r['ssim_3d']:7.4f} {r['mean_stateA']:7.3f} {r['mean_stateB']:7.3f} "
            f"{r['gap_B_minus_A']:+7.3f} {r['r2']:6.3f}"
        )
    print("\nregression (per-view PSNR vs state/sin/cos(angle)/z):")
    for r in rows:
        print(
            f"  {r['label']:<58} state={r['state_coef']:+.3f} sin={r['sin_coef']:+.3f} "
            f"cos={r['cos_coef']:+.3f} z={r['z_coef']:+.3f} R2={r['r2']:.3f}"
        )
    print(f"\ntraining time (s): " + ", ".join(f"{r['label']}={r['time_s']:.1f}" for r in rows))


if __name__ == "__main__":
    main()
