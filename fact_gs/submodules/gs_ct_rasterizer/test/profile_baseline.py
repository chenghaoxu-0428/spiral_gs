"""
Utility script for profiling the xray_gaussian_rasterization_voxelization baseline with Nsight.

Run this script under Nsight Systems/Compute to capture the forward and/or backward
passes at a specific image size and number of Gaussians, e.g.:

    nsys profile -o baseline_forward python test/profile_baseline.py --passes forward
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn import functional as F

import utils
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def build_rasterizer(camera: utils.TestCamera) -> GaussianRasterizer:
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        scale_modifier=1,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        campos=camera.camera_center,
        prefiltered=False,
        mode=int(camera.mode),
        debug=False,
    )
    return GaussianRasterizer(settings)


def sample_gaussians(num_gaussians: int, volume, device, requires_grad: bool):
    pos3d_t, scale3d_t, quat_t, intensity_t = utils.random_gauss_init(
        num_gaussians, volume, device=device
    )

    if requires_grad:
        pos3d_t.requires_grad_()
        scale3d_t.requires_grad_()
        quat_t.requires_grad_()
        intensity_t.requires_grad_()

    return pos3d_t, scale3d_t, quat_t, intensity_t


def profile_forward(rasterizer: GaussianRasterizer, args, camera, volume) -> None:
    pos3d, scale3d, quat, density = sample_gaussians(
        args.num_gaussians, volume, camera.camera_center.device, requires_grad=False
    )
    means2d = torch.zeros_like(pos3d)

    torch.cuda.synchronize()
    rasterizer(
        means3D=pos3d,
        means2D=means2d,
        opacities=density,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    torch.cuda.synchronize()


def profile_backward(rasterizer: GaussianRasterizer, args, camera, volume, target_image) -> None:
    pos3d, scale3d, quat, density = sample_gaussians(
        args.num_gaussians, volume, camera.camera_center.device, requires_grad=True
    )
    means2d = torch.zeros_like(pos3d)

    rendered, _ = rasterizer(
        means3D=pos3d,
        means2D=means2d,
        opacities=density,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    loss = F.mse_loss(rendered, target_image)

    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the baseline rasterizer.")
    parser.add_argument("--image-size", type=int, default=512, help="Square image size.")
    parser.add_argument(
        "--num-gaussians", type=int, default=50000, help="Number of Gaussians to render."
    )
    parser.add_argument(
        "--passes",
        choices=["forward", "backward"],
        nargs="+",
        default=["forward", "backward"],
        help="Which passes to execute for profiling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for NumPy randomness to keep profiles reproducible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = utils.create_test_camera(
        image_height=args.image_size, image_width=args.image_size
    ).to(device)
    rasterizer = build_rasterizer(camera)
    target_image = utils.random_target_image(
        args.image_size, args.image_size, 1, device=device
    )
    test_volume = utils.generate_test_volume((192, 192, 192))

    for pass_name in args.passes:
        if pass_name == "forward":
            profile_forward(rasterizer, args, camera, test_volume)
        elif pass_name == "backward":
            profile_backward(rasterizer, args, camera, test_volume, target_image)


if __name__ == "__main__":
    main()
