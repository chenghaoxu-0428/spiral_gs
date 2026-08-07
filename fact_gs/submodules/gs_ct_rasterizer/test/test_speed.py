"""Speed comparison between gs_ct_rasterizer and the reference implementation."""

from __future__ import annotations

from time import time

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


def _build_reference(camera: utils.TestCamera) -> GaussianRasterizer:
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


def test_speed():
    # Test parameters
    image_size = 512
    num_gaussians = 50000
    repetitions = 30
    warmup = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = utils.create_test_camera(image_height=image_size, image_width=image_size).to(
        device
    )
    reference_rasterizer = _build_reference(camera)
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)
    test_volume = utils.generate_test_volume((256, 256, 256))

    def sample_gaussians():
        return utils.random_gauss_init(
            num_gaussians, test_volume, device=device, anisotropicScale=True, rotation=True
        )

    # Reference rasterizer speed
    speed_reference = 0.0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, density = sample_gaussians()
        means2d = torch.zeros_like(pos3d)

        _sync()
        start_time = time()
        reference_rasterizer(
            means3D=pos3d,
            means2D=means2d,
            opacities=density,
            scales=scale3d,
            rotations=quat,
            cov3D_precomp=None,
        )
        _sync()
        end_time = time()
        if i >= warmup:
            speed_reference += end_time - start_time

    speed_reference /= repetitions
    print(
        f"Reference rasterizer forward pass average time over {repetitions} runs: "
        f"{speed_reference * 1000:.2f} ms"
    )

    # gs_ct_rasterizer speed (split into optim_to_render and rasterize passes)
    speed_total = 0.0
    speed_optim = 0.0
    speed_rasterize = 0.0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, density = sample_gaussians()
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
        _sync()
        mid_time = time()

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

        if i >= warmup:
            speed_total += end_time - start_time
            speed_optim += mid_time - start_time
            speed_rasterize += end_time - mid_time

    speed_total /= repetitions
    speed_optim /= repetitions
    speed_rasterize /= repetitions
    print(
        f"gs_ct_rasterizer forward pass average time over {repetitions} runs: {speed_total * 1000:.2f} ms"
    )
    print(f"  -  of which optim_to_render: {speed_optim * 1000:.2f} ms")
    print(f"  -  of which rasterize_gaussians: {speed_rasterize * 1000:.2f} ms")


def test_speed_backwards():
    # Test parameters
    image_size = 256
    num_gaussians = 5000
    repetitions = 30
    warmup = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = utils.create_test_camera(image_height=image_size, image_width=image_size).to(
        device
    )
    reference_rasterizer = _build_reference(camera)
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)
    test_volume = utils.generate_test_volume((96, 128, 96))
    target_image = utils.random_target_image(image_size, image_size, 1, device=device)

    def sample_gaussians():
        return utils.random_gauss_init(
            num_gaussians, test_volume, device=device, anisotropicScale=True, rotation=True
        )

    speed_reference = 0.0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, density = sample_gaussians()
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        density = density.requires_grad_()
        means2d = torch.zeros_like(pos3d)

        rendered, _ = reference_rasterizer(
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
        if i >= warmup:
            speed_reference += end_time - start_time

    speed_reference /= repetitions
    print(
        f"Reference rasterizer backward pass average time over {repetitions} runs: "
        f"{speed_reference * 1000:.2f} ms"
    )

    speed_gs = 0.0
    for i in range(repetitions + warmup):
        pos3d, scale3d, quat, density = sample_gaussians()
        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        density = density.requires_grad_()

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
        if i >= warmup:
            speed_gs += end_time - start_time

    speed_gs /= repetitions
    print(
        f"gs_ct_rasterizer backward pass average time over {repetitions} runs: {speed_gs * 1000:.2f} ms"
    )


if __name__ == "__main__":
    test_speed()
    test_speed_backwards()
    torch.cuda.empty_cache()
