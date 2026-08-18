"""Coarse projection-domain geometry scan for a saved reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from fact_gs import rasterize_proj
from fact_gs.r2_gaussian.dataset import SceneRecon
from fact_gs.r2_gaussian.dataset.cameras import Camera
from fact_gs.r2_gaussian.dataset.dataset_readers import angle2pose
from fact_gs.r2_gaussian.gaussian import GaussianModel


def adjusted_camera(base, scanner, angle, z, kind, value):
    dso = float(scanner["DSO"])
    dsd = float(scanner["DSD"])
    if kind == "angle_deg":
        angle += math.radians(value)
    elif kind == "z_offset":
        z += value
    elif kind == "pitch_scale":
        z = float(scanner["offOrigin"][2]) + value * (
            z - float(scanner["offOrigin"][2])
        )
    elif kind == "dso_scale":
        dso *= value
    elif kind == "dsd_scale":
        dsd *= value

    c2w = angle2pose(dso, angle, z)
    w2c = np.linalg.inv(c2w)
    fov_x = 2 * math.atan2(float(scanner["sDetector"][1]) / 2, dsd)
    fov_y = 2 * math.atan2(float(scanner["sDetector"][0]) / 2, dsd)
    camera = Camera(
        colmap_id=base.colmap_id,
        scanner_cfg=scanner,
        R=w2c[:3, :3].T,
        T=w2c[:3, 3],
        angle=angle,
        mode=base.mode,
        FoVx=fov_x,
        FoVy=fov_y,
        image=base.original_image,
        image_name=base.image_name,
        uid=base.uid,
        z_shift=z,
    )
    if kind in {"u_pixels", "v_pixels"}:
        axis = 0 if kind == "u_pixels" else 1
        size = camera.image_width if axis == 0 else camera.image_height
        camera.projection_matrix[2, axis] = 2 * value / size
        camera.full_proj_transform = (
            camera.world_view_transform.unsqueeze(0)
            .bmm(camera.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
    return camera


def score(cameras, gaussians, scanner, frames, kind, value, ids):
    values = []
    with torch.no_grad():
        for i in ids:
            base = cameras[i]
            camera = adjusted_camera(
                base,
                scanner,
                float(frames[i]["angle"]),
                float(base.z_shift),
                kind,
                value,
            )
            gt = base.original_image[0]
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
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    scene = SceneRecon(
        SimpleNamespace(
            data_source_path=str(args.data), model_path=str(args.model), eval=True
        ),
        shuffle=False,
    )
    gaussians = GaussianModel(None)
    gaussians.load_ply(
        args.model / f"point_cloud/step_{args.step}/point_cloud.pickle"
    )
    with (args.data / "meta_data.json").open() as handle:
        frames = json.load(handle)["proj_test"]
    cameras = scene.getTestCameras()
    ids = np.unique(np.rint(np.linspace(0, len(cameras) - 1, args.samples)).astype(int))
    scanner = scene.scanner_cfg
    grid = {
        "angle_deg": [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
        "z_offset": [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02],
        "pitch_scale": [0.996, 0.998, 1.0, 1.002, 1.004],
        "dso_scale": [0.995, 0.998, 1.0, 1.002, 1.005],
        "dsd_scale": [0.995, 0.998, 1.0, 1.002, 1.005],
        "u_pixels": [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
        "v_pixels": [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
    }
    result = {
        kind: [
            {"value": value, "psnr": score(cameras, gaussians, scanner, frames, kind, value, ids)}
            for value in values
        ]
        for kind, values in grid.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
