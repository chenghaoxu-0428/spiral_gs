"""
Benchmark rasterization speed against the xray_gaussian_rasterization_voxelization baseline.

This script measures forward and backward runtimes while varying either the
image resolution or the number of Gaussians. Results are saved as two plots
for quick visual comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from gs_ct_rasterizer import optim_to_render, rasterize
import utils
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_baseline_rasterizer(camera: utils.TestCamera) -> GaussianRasterizer:
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


def _forward_baseline_run(
    rasterizer: GaussianRasterizer,
    gaussians,
) -> float:
    pos3d, scale3d, quat, density = gaussians
    means2d = torch.zeros_like(pos3d)

    _sync()
    start_time = time()
    rasterizer(
        means3D=pos3d,
        means2D=means2d,
        opacities=density,
        scales=scale3d,
        rotations=quat,
        cov3D_precomp=None,
    )
    _sync()
    end_time = time()
    return end_time - start_time


def _forward_gs_run(
    camera: utils.TestCamera,
    gaussians,
) -> float:
    pos3d, scale3d, quat, density = gaussians
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)
    pos2d_buffer = torch.empty(
        (*pos3d.shape[:-1], 2), device=pos3d.device, dtype=pos3d.dtype
    )

    _sync()
    start_time = time()
    pos2d, conics_mu, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        density,
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
        conics_mu,
        density,
        tile_min,
        tile_max,
        num_tiles_hit,
        camera.image_height,
        camera.image_width,
        use_per_gaussian_backward=True,
    )
    _sync()
    end_time = time()
    return end_time - start_time


def _backward_baseline_run(
    rasterizer: GaussianRasterizer,
    gaussians,
    target_image: torch.Tensor,
) -> float:
    pos3d, scale3d, quat, density = gaussians
    pos3d = pos3d.requires_grad_()
    scale3d = scale3d.requires_grad_()
    quat = quat.requires_grad_()
    density = density.requires_grad_()
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

    _sync()
    start_time = time()
    loss.backward()
    _sync()
    end_time = time()
    return end_time - start_time


def _backward_gs_run(
    camera: utils.TestCamera,
    gaussians,
    target_image: torch.Tensor,
) -> float:
    pos3d, scale3d, quat, density = gaussians
    pos3d = pos3d.requires_grad_()
    scale3d = scale3d.requires_grad_()
    quat = quat.requires_grad_()
    density = density.requires_grad_()
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)

    pos2d_buffer = torch.empty(
        (*pos3d.shape[:-1], 2), device=pos3d.device, dtype=pos3d.dtype
    ).requires_grad_(pos3d.requires_grad)
    pos2d, conics_mu, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pos3d,
        scale3d,
        quat,
        density,
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
        conics_mu,
        density,
        tile_min,
        tile_max,
        num_tiles_hit,
        camera.image_height,
        camera.image_width,
        use_per_gaussian_backward=True,
    ).permute(2, 0, 1)
    loss = F.mse_loss(rendered, target_image)

    _sync()
    start_time = time()
    loss.backward()
    _sync()
    end_time = time()
    return end_time - start_time


def average_runtime(fn, repetitions: int, warmup: int) -> float:
    """Average runtime (seconds) while discarding the warmup iterations."""
    total = 0.0
    for idx in range(repetitions + warmup):
        duration = fn()
        if idx >= warmup:
            total += duration
    return total / repetitions


def benchmark_case(
    image_size: int,
    num_gaussians: int,
    repetitions: int,
    warmup: int,
):
    """Measure forward/backward times for one pair of (image_size, num_gaussians)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = utils.create_test_camera(image_height=image_size, image_width=image_size).to(
        device
    )
    rasterizer = build_baseline_rasterizer(camera)
    target_image = utils.random_target_image(image_size, image_size, 1, device=device)
    test_volume = utils.generate_test_volume((96, 128, 96))

    def sample_gaussians():
        return utils.random_gauss_init(num_gaussians, test_volume, device=device)

    forward_baseline = average_runtime(
        lambda: _forward_baseline_run(rasterizer, sample_gaussians()),
        repetitions,
        warmup,
    )
    forward_gs = average_runtime(
        lambda: _forward_gs_run(camera, sample_gaussians()),
        repetitions,
        warmup,
    )
    backward_baseline = average_runtime(
        lambda: _backward_baseline_run(rasterizer, sample_gaussians(), target_image),
        repetitions,
        warmup,
    )
    backward_gs = average_runtime(
        lambda: _backward_gs_run(camera, sample_gaussians(), target_image),
        repetitions,
        warmup,
    )

    return {
        "forward": {
            "gs_ct_rasterizer": forward_gs * 1000.0,
            "baseline": forward_baseline * 1000.0,
        },
        "backward": {
            "gs_ct_rasterizer": backward_gs * 1000.0,
            "baseline": backward_baseline * 1000.0,
        },
    }


def plot_results(
    x_values,
    metrics,
    xlabel: str,
    title: str,
    output_path: Path,
):
    """Create a comparison plot and save it."""
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    forward_color = np.array([57, 106, 177]) / 255
    backward_color = np.array([166, 23, 23]) / 255

    plt.plot(
        x_values,
        metrics["forward"]["baseline"],
        c=forward_color,
        linestyle="dashed",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["forward"]["gs_ct_rasterizer"],
        c=forward_color,
        linestyle="solid",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["backward"]["baseline"],
        c=backward_color,
        linestyle="dashed",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x_values,
        metrics["backward"]["gs_ct_rasterizer"],
        c=backward_color,
        linestyle="solid",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    plt.plot(
        [],
        [],
        label="Baseline backward",
        c=backward_color,
        linestyle="dashed",
        linewidth=2,
    )
    plt.plot(
        [],
        [],
        label="Ours backward",
        c=backward_color,
        linestyle="solid",
        linewidth=2,
    )
    plt.plot(
        [],
        [],
        label="Baseline forward",
        c=forward_color,
        linestyle="dashed",
        linewidth=2,
    )
    plt.plot(
        [],
        [],
        label="Ours forward",
        c=forward_color,
        linestyle="solid",
        linewidth=2,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, which="major", axis="x", alpha=0.3)
    ax.grid(True, which="major", axis="y", alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel("Execution time (ms)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def prepare_metric_storage():
    return {
        "forward": {"baseline": [], "gs_ct_rasterizer": []},
        "backward": {"baseline": [], "gs_ct_rasterizer": []},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare rasterization runtimes.")
    parser.add_argument("--repetitions", type=int, default=15, help="Measurement iterations per data point.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations to discard.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_out_speed"),
        help="Directory used to store the generated plots.",
    )
    parser.add_argument(
        "--image-sizes",
        type=int,
        nargs="+",
        default=[128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192],
        help="Image resolutions evaluated when varying the rasterization grid size.",
    )
    parser.add_argument(
        "--gaussian-counts",
        type=int,
        nargs="+",
        default=[1000, 2500, 5000, 7500, 10000, 20000, 40000, 80000, 160000, 320000],
        help="Gaussian counts evaluated when varying the number of primitives.",
    )
    parser.add_argument(
        "--fixed-image-size",
        type=int,
        default=1024,
        help="Image size used when sweeping Gaussian counts.",
    )
    parser.add_argument(
        "--fixed-gaussian-count",
        type=int,
        default=20000,
        help="Number of Gaussians used when sweeping image sizes.",
    )
    parser.add_argument(
        "--save-to-npy",
        type=Path,
        default=None,
        help="If set, skip plotting and save the collected metrics to this NumPy file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_to_npy is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.save_to_npy.parent.mkdir(parents=True, exist_ok=True)

    image_metrics = prepare_metric_storage()
    gaussian_metrics = prepare_metric_storage()

    print("=== Sweep: varying image size ===")
    for image_size in args.image_sizes:
        measurements = benchmark_case(
            image_size=image_size,
            num_gaussians=args.fixed_gaussian_count,
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
        print(
            f"Image {image_size}: "
            f"baseline forward {measurements['forward']['baseline']:.2f} ms, "
            f"Ours forward {measurements['forward']['gs_ct_rasterizer']:.2f} ms | "
            f"baseline backward {measurements['backward']['baseline']:.2f} ms, "
            f"Ours backward {measurements['backward']['gs_ct_rasterizer']:.2f} ms"
        )
        for phase in ("forward", "backward"):
            for method in ("baseline", "gs_ct_rasterizer"):
                image_metrics[phase][method].append(measurements[phase][method])

    print("\n=== Sweep: varying number of Gaussians ===")
    for num_gaussians in args.gaussian_counts:
        measurements = benchmark_case(
            image_size=args.fixed_image_size,
            num_gaussians=num_gaussians,
            repetitions=args.repetitions,
            warmup=args.warmup,
        )
        print(
            f"Gaussians {num_gaussians}: "
            f"baseline forward {measurements['forward']['baseline']:.2f} ms, "
            f"Ours forward {measurements['forward']['gs_ct_rasterizer']:.2f} ms | "
            f"baseline backward {measurements['backward']['baseline']:.2f} ms, "
            f"Ours backward {measurements['backward']['gs_ct_rasterizer']:.2f} ms"
        )
        for phase in ("forward", "backward"):
            for method in ("baseline", "gs_ct_rasterizer"):
                gaussian_metrics[phase][method].append(measurements[phase][method])

    if args.save_to_npy is not None:
        benchmark_data = {
            "image_sizes": args.image_sizes,
            "gaussian_counts": args.gaussian_counts,
            "fixed_image_size": args.fixed_image_size,
            "fixed_gaussian_count": args.fixed_gaussian_count,
            "image_metrics": image_metrics,
            "gaussian_metrics": gaussian_metrics,
        }
        np.save(args.save_to_npy, benchmark_data, allow_pickle=True)
        print(f"\nSaved benchmark data to: {args.save_to_npy}")
    else:
        image_plot = args.output_dir / "speed_vs_image.png"
        gaussian_plot = args.output_dir / "speed_vs_gaussians.png"
        plot_results(
            x_values=args.image_sizes,
            metrics=image_metrics,
            xlabel="Image size (pixels)",
            title=f"Runtime vs. image size (Gaussians={args.fixed_gaussian_count})",
            output_path=image_plot,
        )
        plot_results(
            x_values=args.gaussian_counts,
            metrics=gaussian_metrics,
            xlabel="Number of Gaussians",
            title=f"Runtime vs. Gaussians (image={args.fixed_image_size}^2)",
            output_path=gaussian_plot,
        )

        print(f"\nSaved image sweep plot to: {image_plot}")
        print(f"Saved Gaussian sweep plot to: {gaussian_plot}")


if __name__ == "__main__":
    main()
