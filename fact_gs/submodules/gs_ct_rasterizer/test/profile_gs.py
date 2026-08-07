"""
Utility script for profiling the gs_ct_rasterizer implementation with Nsight.

Example usage:

    ncu -o ours_backward python test/profile_gs.py --passes backward
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.nn import functional as F

from gs_ct_rasterizer import optim_to_render, rasterize
import utils


def sample_gaussians(num_gaussians: int, volume, device, requires_grad: bool):
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(
        num_gaussians, volume, device=device
    )

    if requires_grad:
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        intensity = intensity.requires_grad_()

    return pos3d, scale3d, quat, intensity


def profile_forward(args, camera, volume):
    device = camera.camera_center.device
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, volume, device, requires_grad=False
    )
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)

    torch.cuda.synchronize()
    pos2d_buffer = torch.empty(
        (*pos3d.shape[:-1], 2), device=pos3d.device, dtype=pos3d.dtype
    )
    pos2d, conics, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        intensity,
        camera.world_view_transform,
        camera.full_proj_transform,
        tanfovx,
        tanfovy,
        camera.image_height,
        camera.image_width,
        camera.mode,
        pos2d_buffer=pos2d_buffer,
    )

    rasterize.rasterize_gaussians(
        pos2d,
        conics,
        intensity,
        tile_min,
        tile_max,
        num_tiles_hit,
        camera.image_height,
        camera.image_width,
        use_per_gaussian_backward=True,
    )
    torch.cuda.synchronize()


def profile_backward(args, camera, volume, target_image):
    device = camera.camera_center.device
    pos3d, scale3d, quat, intensity = sample_gaussians(
        args.num_gaussians, volume, device, requires_grad=True
    )
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)

    pos2d_buffer = torch.empty(
        (*pos3d.shape[:-1], 2), device=pos3d.device, dtype=pos3d.dtype
    ).requires_grad_(pos3d.requires_grad)
    pos2d, conics, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        intensity,
        camera.world_view_transform,
        camera.full_proj_transform,
        tanfovx,
        tanfovy,
        camera.image_height,
        camera.image_width,
        camera.mode,
        pos2d_buffer=pos2d_buffer,
    )
    rendered = rasterize.rasterize_gaussians(
        pos2d,
        conics,
        intensity,
        tile_min,
        tile_max,
        num_tiles_hit,
        camera.image_height,
        camera.image_width,
        use_per_gaussian_backward=True,
    ).permute(2, 0, 1)
    loss = F.mse_loss(rendered, target_image)

    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the gs_ct_rasterizer pipeline.")
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
        help="Seed for numpy/torch randomness to keep inputs reproducible.",
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
    target_image = utils.random_target_image(
        args.image_size, args.image_size, 1, device=device
    )
    test_volume = utils.generate_test_volume((192, 192, 192))

    for pass_name in args.passes:
        if pass_name == "forward":
            profile_forward(args, camera, test_volume)
        elif pass_name == "backward":
            profile_backward(args, camera, test_volume, target_image)


if __name__ == "__main__":
    main()
