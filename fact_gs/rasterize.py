import sys
import torch
import math

from gs_ct_rasterizer import optim_to_render, rasterize

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian import GaussianModel
from fact_gs.r2_gaussian.dataset.cameras import Camera


def _quality_safe_tile_bounds(pos2d, conics_mu, radii, height, width):
    """Build conservative 3-sigma tile bounds without density culling.

    This Torch-side correction also works with an already-installed upstream
    FaCT-GS extension, so quality does not depend on whether the local CUDA
    source has been rebuilt yet.
    """
    block_x = block_y = 16
    grid_x = (width + block_x - 1) // block_x
    grid_y = (height + block_y - 1) // block_y

    # conics_mu stores inverse covariance [A, B, C, mu].
    inv_det = conics_mu[:, 0] * conics_mu[:, 2] - conics_mu[:, 1].square()
    valid = (radii > 0) & torch.isfinite(inv_det) & (inv_det > 0)
    safe_det = torch.where(valid, inv_det, torch.ones_like(inv_det))
    cov_xx = torch.clamp_min(conics_mu[:, 2] / safe_det, 0.0)
    cov_yy = torch.clamp_min(conics_mu[:, 0] / safe_det, 0.0)
    radius_x = 3.0 * torch.sqrt(cov_xx) + 0.5
    radius_y = 3.0 * torch.sqrt(cov_yy) + 0.5

    min_x = torch.clamp((pos2d[:, 0] - radius_x) / block_x, 0, grid_x).to(torch.int32)
    min_y = torch.clamp((pos2d[:, 1] - radius_y) / block_y, 0, grid_y).to(torch.int32)
    max_x = torch.clamp(
        (pos2d[:, 0] + radius_x + block_x - 1) / block_x, 0, grid_x
    ).to(torch.int32)
    max_y = torch.clamp(
        (pos2d[:, 1] + radius_y + block_y - 1) / block_y, 0, grid_y
    ).to(torch.int32)
    zeros = torch.zeros_like(min_x)
    tile_min = torch.stack(
        [torch.where(valid, min_x, zeros), torch.where(valid, min_y, zeros)], dim=1
    ).contiguous()
    tile_max = torch.stack(
        [torch.where(valid, max_x, zeros), torch.where(valid, max_y, zeros)], dim=1
    ).contiguous()
    num_tiles_hit = (
        (tile_max[:, 0] - tile_min[:, 0]) * (tile_max[:, 1] - tile_min[:, 1])
    ).contiguous()
    return tile_min, tile_max, num_tiles_hit


def rasterize_proj(
    viewpoint_camera: Camera,
    pc: GaussianModel,
):
    """Render a single projection image from the Gaussian model.

    Args:
        viewpoint_camera: Camera definition that stores transforms, FoV and mode.
        pc: Gaussian model containing learnable position/orientation/density tensors.

    Returns:
        Dictionary with rendered image, intermediate buffers and visibility flags.
    """
    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    # The upstream extension derives its initial tile bound from density.  Feed
    # a large proxy only for visibility preprocessing, then rasterize with the
    # real density below.  This prevents low-density Gaussians losing gradients.
    extent_proxy = torch.full_like(pc.get_density, 1.0e10)
    pos2d, conics_mu, radii, tile_min, tile_max, num_tiles_hit = optim_to_render.optim_to_render(
        pc.get_xyz,
        pc.get_scaling,
        pc.get_rotation,
        extent_proxy,
        viewpoint_camera.world_view_transform,
        viewpoint_camera.full_proj_transform,
        tanfovx,
        tanfovy,
        viewpoint_camera.image_height,
        viewpoint_camera.image_width,
        viewpoint_camera.mode,
        pos2d_buffer=torch.zeros_like(pc.get_xyz[:, :2], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda").contiguous(),
    )

    if pos2d.requires_grad:
        pos2d.retain_grad()

    tile_min, tile_max, num_tiles_hit = _quality_safe_tile_bounds(
        pos2d,
        conics_mu,
        radii,
        viewpoint_camera.image_height,
        viewpoint_camera.image_width,
    )

    rendered_image = rasterize.rasterize_gaussians(
        pos2d,
        conics_mu,
        pc.get_density,
        tile_min,
        tile_max,
        num_tiles_hit,
        viewpoint_camera.image_height,
        viewpoint_camera.image_width,
        use_per_gaussian_backward=True,
    ).permute(2, 0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": pos2d,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
