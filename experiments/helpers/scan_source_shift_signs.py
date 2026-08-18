"""Scan sign/mode interpretations of CT-PD per-view source shifts.

For a fixed trained model, re-render test views with candidate per-view
source-dynamics corrections (built through the committed
``model.geometry`` camera path) and report mean per-view PSNR, so the best
sign convention for ``Source*Shift`` can be chosen empirically before
retraining.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fact_gs import rasterize_proj
from fact_gs.r2_gaussian.dataset import SceneRecon
from fact_gs.r2_gaussian.gaussian import GaussianModel


def score_candidate(data: Path, model: Path, step: int, geometry: dict, n_views: int) -> float:
    scene = SceneRecon(
        SimpleNamespace(
            data_source_path=str(data),
            model_path=str(model),
            eval=True,
            geometry=geometry,
        ),
        shuffle=False,
    )
    gaussians = GaussianModel(None)
    gaussians.load_ply(model / f"point_cloud/step_{step}/point_cloud.pickle")
    cameras = scene.getTestCameras()
    stride = max(1, len(cameras) // n_views)
    values = []
    with torch.no_grad():
        for camera in cameras[::stride][:n_views]:
            gt = camera.original_image[0]
            pred = rasterize_proj(camera, gaussians)["render"][0]
            gt = gt / gt.max()
            pred = pred / pred.max()
            values.append(float(-10 * torch.log10(torch.mean((gt - pred) ** 2))))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--step", default=30000, type=int)
    parser.add_argument("--views", default=400, type=int)
    args = parser.parse_args()

    base = {
        "u_offset_px": -0.275,
        "source_shift_mode": "angle_z",
        "radial_mode": "dso",
        "angular_sign": 1.0,
        "axial_sign": 1.0,
        "radial_sign": 1.0,
    }
    candidates = {
        "nominal (u only)": {**base, "source_shift_mode": "off"},
    }
    for axial_sign in (1.0, -1.0):
        candidates[f"angle_z axial={axial_sign:+.0f}"] = {
            **base,
            "source_shift_mode": "angle_z",
            "axial_sign": axial_sign,
        }
        for radial_mode in ("dso", "dsd"):
            for radial_sign in (1.0, -1.0):
                candidates[
                    f"angle_z_radial axial={axial_sign:+.0f} radial({radial_mode})={radial_sign:+.0f}"
                ] = {
                    **base,
                    "source_shift_mode": "angle_z_radial",
                    "axial_sign": axial_sign,
                    "radial_mode": radial_mode,
                    "radial_sign": radial_sign,
                }
    results = []
    for name, geometry in candidates.items():
        psnr = score_candidate(args.data, args.model, args.step, geometry, args.views)
        results.append((psnr, name))
        print(f"{psnr:8.3f} dB  {name}", flush=True)
    results.sort(reverse=True)
    print("\nbest:")
    for psnr, name in results[:3]:
        print(f"  {psnr:8.3f} dB  {name}")


if __name__ == "__main__":
    main()
