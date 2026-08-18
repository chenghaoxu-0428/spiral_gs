"""Render a saved FaCT-GS model and export aligned GT/prediction pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fact_gs import rasterize_proj
from fact_gs.r2_gaussian.dataset import SceneRecon
from fact_gs.r2_gaussian.gaussian import GaussianModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--step", default=30000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    scene = SceneRecon(
        SimpleNamespace(
            data_source_path=args.data,
            model_path=str(args.model),
            eval=True,
        ),
        shuffle=False,
    )
    gaussians = GaussianModel(None)
    gaussians.load_ply(
        args.model / f"point_cloud/step_{args.step}/point_cloud.pickle"
    )

    payload = {}
    with torch.no_grad():
        for split, cameras in (
            ("train", scene.getTrainCameras()),
            ("test", scene.getTestCameras()),
        ):
            gt, pred, angles, z = [], [], [], []
            for camera in cameras:
                gt.append(camera.original_image[0].cpu().numpy())
                pred.append(rasterize_proj(camera, gaussians)["render"][0].cpu().numpy())
                angles.append(float(camera.angle))
                z.append(float(camera.z_shift))
            payload[f"gt_{split}"] = np.stack(gt).astype(np.float32)
            payload[f"pred_{split}"] = np.stack(pred).astype(np.float32)
            payload[f"angle_{split}"] = np.asarray(angles, dtype=np.float32)
            payload[f"z_{split}"] = np.asarray(z, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
