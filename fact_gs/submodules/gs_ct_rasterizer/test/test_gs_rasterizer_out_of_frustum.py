"""Stress test for gaussians falling outside the camera frustum."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import skimage.io
import torch
from torch.nn import functional as F

from gs_ct_rasterizer import optim_to_render, rasterize
import utils

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def build_reference_rasterizer(camera: utils.TestCamera) -> GaussianRasterizer:
    tanfovx, tanfovy = utils.camera_tan_fovs(camera)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
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
    return GaussianRasterizer(raster_settings=settings)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    img = (arr * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(path, img)


def test_rasterizer_matches_reference_out_of_frustum():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_shape = (220, 220, 220)
    test_volume = utils.generate_test_volume(vol_shape)
    print(f"Initialized oversized test volume with shape {vol_shape}")

    num_gaussians = 3000
    xy_extent = 2.5
    depth_range = (0.2, 0.8)
    scenarios = (
        ("parallel_out_of_frustum", 0),
        ("cone_out_of_frustum", 1),
    )

    for scenario_name, camera_mode in scenarios:
        print(f"\n=== Scenario: {scenario_name} ===")
        camera = utils.create_test_camera(
            image_height=192,
            image_width=192,
            mode=camera_mode,
            fov_y_degrees=35.0 if camera_mode else 50.0,
        ).to(device)
        reference_rasterizer = build_reference_rasterizer(camera)

        pos3d, scale3d, quat, density = utils.random_gauss_init(
            num_gaussians=num_gaussians,
            vol=test_volume,
            device=device,
            xy_extent=xy_extent,
            depth_range=depth_range,
        )

        outside_mask = (
            (pos3d.detach()[:, 0].abs() > 1.1) | (pos3d.detach()[:, 1].abs() > 1.1)
        )
        outside_ratio = float(outside_mask.float().mean().item())
        if outside_ratio <= 0.5:
            raise AssertionError("Expected at least half the gaussians to be outside the frustum")
        print(f"{outside_ratio * 100:.1f}% of gaussians start outside the view frustum")

        pos3d = pos3d.requires_grad_()
        scale3d = scale3d.requires_grad_()
        quat = quat.requires_grad_()
        density = density.requires_grad_()

        # Baseline rendering
        print("Running reference rasterizer...")
        means2d = torch.zeros_like(pos3d).requires_grad_(pos3d.requires_grad)
        ref_img, ref_radii = reference_rasterizer(
            means3D=pos3d,
            means2D=means2d,
            opacities=density,
            scales=scale3d,
            rotations=quat,
            cov3D_precomp=None,
        )

        # Our rendering
        print("Running gs_ct_rasterizer...")
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
        ours_img = rasterize.rasterize_gaussians(
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

        out_dir = Path("test_out/rasterizer_out_of_frustum") / scenario_name
        _save_image(ours_img, out_dir / "ours.png")
        _save_image(ref_img, out_dir / "baseline.png")
        print("Saved rasterized images.")

        torch.testing.assert_close(ours_img, ref_img, rtol=1e-3, atol=5e-2)
        print("Forward pass outputs match.")

        target = utils.random_target_image(
            camera.image_height, camera.image_width, ours_img.shape[0], device=device
        )

        params = (pos3d, scale3d, quat, density)
        param_names = ("pos3d", "scale3d", "quat", "density")

        print("Running backward pass (ours)...")
        for tensor in params:
            tensor.grad = None
        ours_loss = F.mse_loss(ours_img, target)
        ours_loss.backward()
        ours_grads = []
        for tensor in params:
            grad = tensor.grad
            if grad is None:
                raise AssertionError(f"Missing ours gradient for {scenario_name}")
            ours_grads.append(grad.detach().clone())

        print("Running backward pass (baseline)...")
        for tensor in params:
            tensor.grad = None
        reference_loss = F.mse_loss(ref_img, target)
        reference_loss.backward()
        reference_grads = []
        for tensor in params:
            grad = tensor.grad
            if grad is None:
                raise AssertionError(f"Missing reference gradient for {scenario_name}")
            reference_grads.append(grad.detach().clone())

        for ours_grad, ref_grad, name in zip(ours_grads, reference_grads, param_names):
            print(f"Validating gradient for {name}")
            torch.testing.assert_close(ours_grad, ref_grad, rtol=1e-4, atol=1e-3)
            assert torch.isfinite(ours_grad).all()
            assert torch.isfinite(ref_grad).all()

        print("Validating pos2d_buffer gradients")
        torch.testing.assert_close(pos2d_buffer.grad, means2d.grad[:, :2], rtol=1e-4, atol=1e-3)

        print(f"Scenario {scenario_name}: all checks passed.")


if __name__ == "__main__":
    test_rasterizer_matches_reference_out_of_frustum()
