"""
Torch profiler entry point for the baseline xray Gaussian rasterizer.
Mirrors test/profile_torch.py but swaps in the reference implementation so the
CPU/GPU traces can be inspected with TensorBoard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

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
    pos3d, scale3d, quat, intensity = utils.random_gauss_init(
        num_gaussians, volume, device=device
    )
    if requires_grad:
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        intensity = intensity.requires_grad_()
    return pos3d, scale3d, quat, intensity


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_forward(args, rasterizer: GaussianRasterizer, camera, volume):
    device = camera.camera_center.device
    with record_function("gaussian_sampling"):
        pos3d, scale3d, quat, intensity = sample_gaussians(
            args.num_gaussians, volume, device, requires_grad=False
        )
    cuda_sync()
    with record_function("gaussian_setup"):
        means2d = torch.zeros_like(pos3d)
    cuda_sync()
    rasterizer(
        means3D=pos3d,
        means2D=means2d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    cuda_sync()


def run_backward(args, rasterizer: GaussianRasterizer, camera, volume, target_image):
    device = camera.camera_center.device
    with record_function("gaussian_sampling"):
        pos3d, scale3d, quat, intensity = sample_gaussians(
            args.num_gaussians, volume, device, requires_grad=True
        )
    cuda_sync()
    with record_function("gaussian_setup"):
        means2d = torch.zeros_like(pos3d)
    cuda_sync()
    rendered, _ = rasterizer(
        means3D=pos3d,
        means2D=means2d,
        opacities=intensity,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    loss = F.mse_loss(rendered, target_image)
    loss.backward()
    cuda_sync()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Torch profiler driver for the baseline rasterizer."
    )
    parser.add_argument("--image-size", type=int, default=192, help="Square image size.")
    parser.add_argument(
        "--num-gaussians", type=int, default=5000, help="Number of Gaussians to render."
    )
    parser.add_argument(
        "--passes",
        choices=["forward", "backward"],
        nargs="+",
        default=["forward", "backward"],
        help="Which passes to execute under the profiler.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy/torch.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before tracing.")
    parser.add_argument("--cycles", type=int, default=2, help="Active iterations to capture.")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profiles/torch_baseline",
        help="Where to store TensorBoard trace files.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes in the profiler trace.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Track memory usage during profiling.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Include stack traces for profiled operations.",
    )
    return parser.parse_args()


def build_profiler(args):
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    log_dir = Path(args.profile_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return profile(
        activities=activities,
        schedule=schedule(
            wait=0,
            warmup=max(0, args.warmup),
            active=max(1, args.cycles),
            repeat=1,
        ),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        on_trace_ready=tensorboard_trace_handler(str(log_dir)),
    )


def run_with_profiler(fn, warmup: int, cycles: int, prof):
    total_steps = max(0, warmup) + max(0, cycles)
    for _ in range(total_steps):
        fn()
        prof.step()


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

    with build_profiler(args) as prof:
        for pass_name in args.passes:
            if pass_name == "forward":
                run_with_profiler(
                    lambda: run_forward(args, rasterizer, camera, test_volume),
                    args.warmup,
                    args.cycles,
                    prof,
                )
            elif pass_name == "backward":
                run_with_profiler(
                    lambda: run_backward(
                        args, rasterizer, camera, test_volume, target_image
                    ),
                    args.warmup,
                    args.cycles,
                    prof,
                )
    print(f"Torch profiler traces stored in: {Path(args.profile_dir).resolve()}")


if __name__ == "__main__":
    main()
