"""Voxelizer python-cuda bindings with Python-based Gaussian sorting"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gs_ct_rasterizer.cuda as _C

from .bin_and_sort_gaussians import bin_and_sort_gaussians


def rasterize_gaussians(
    pos2d: Float[Tensor, "*batch 2"],
    conics_mu: Float[Tensor, "*batch 4"],
    intensities: Float[Tensor, "*batch channels"],
    tile_min: Int[Tensor, "*batch 2"],
    tile_max: Int[Tensor, "*batch 2"],
    num_tiles_hit: Int[Tensor, "*batch"],
    img_h: int,
    img_w: int,
    use_per_gaussian_backward: bool = True,
) -> Tensor:
    """Voxelizes 3D Gaussians into a 3D voxel grid.

    Args:
        pos2d: (*batch, 2) Tensor of 2D positions of Gaussians in image space.
        conics_mu: (*batch, 4) Tensor of conic parameters and mu (integration factor) values for visualization.
        intensities: (*batch, *channels) Tensor of intensity values for Gaussians.
        tile_min: (*batch, 2) minimum tile indices intersected by each Gaussian.
        tile_max: (*batch, 2) maximum tile indices intersected by each Gaussian.
        num_tiles_hit: (*batch) number of tiles overlapped by each Gaussian.
        img_h/w: Height and width of the output image in pixels.
        use_per_gaussian_backward: f True use the per-Gaussian backward CUDA kernel. Per-pixel otherwise.

    Returns:
        Image grid of shape (img_h, img_w) with accumulated Gaussian values.
    """

    return _RasterizeGaussians.apply(
        pos2d.contiguous(),
        conics_mu.contiguous(),
        intensities.contiguous(),
        tile_min.contiguous(),
        tile_max.contiguous(),
        num_tiles_hit.contiguous(),
        img_h,
        img_w,
        use_per_gaussian_backward,
    )


class _RasterizeGaussians(Function):

    @staticmethod
    def forward(
        ctx,
        pos2d: Float[Tensor, "*batch 2"],
        conics_mu: Float[Tensor, "*batch 4"],
        intensities: Float[Tensor, "*batch channels"],
        tile_min: Int[Tensor, "*batch 2"],
        tile_max: Int[Tensor, "*batch 2"],
        num_tiles_hit: Int[Tensor, "*batch"],
        img_h: int,
        img_w: int,
        use_per_gaussian_backward: bool = False,
    ) -> Tensor:
        # Check if intensities has channel dimension
        if intensities.dim() == 1:
            intensities = intensities.unsqueeze(-1)

        num_gaussians = pos2d.size(-2)
        img_size = (img_h, img_w)

        if num_gaussians == 0:
            out_img = torch.zeros(
                vol_size_voxel[0],
                vol_size_voxel[1],
                vol_size_voxel[2],
                intensities.shape[-1],
                device=pos3d_radii.device,
                dtype=pos3d_radii.dtype,
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=pos3d_radii.device, dtype=torch.int32)
            tile_bins = torch.zeros(0, 3, device=pos3d_radii.device, dtype=torch.int32)
            ctx.num_intersects = 0
        else:
            cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
            num_intersects = cum_tiles_hit[-1].item()
            ctx.num_intersects = num_intersects
            ctx.img_h = img_h
            ctx.img_w = img_w
            ctx.use_per_gaussian_backward = use_per_gaussian_backward

            if num_intersects < 1:
                
                out_img = (
                    torch.zeros(img_h, img_w, intensities.shape[-1], device=pos2d.device)
                )
                gaussian_ids_sorted = torch.zeros(0, 1, device=pos2d.device, dtype=torch.int32)
                tile_bins =           torch.zeros(0, 3, device=pos2d.device, dtype=torch.int32)
            else:

                gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
                    num_gaussians,
                    num_intersects,
                    tile_min,
                    tile_max,
                    cum_tiles_hit,
                    img_size,
                )

                out_img = _C.rasterize_forward(
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    pos2d,
                    conics_mu,
                    intensities)

        ctx.save_for_backward(
            pos2d,
            conics_mu,
            intensities,
            gaussian_ids_sorted,
            tile_bins,
        )

        return out_img

    @staticmethod
    def backward(ctx, vol_grad):
      
        img_h = ctx.img_h
        img_w = ctx.img_w
        
        num_intersects = ctx.num_intersects
        pos2d, conics_mu, intensities, gaussian_ids_sorted, tile_bins = ctx.saved_tensors

        if num_intersects < 1:
            pos2d_grad = torch.zeros_like(pos2d)
            conics_mu_grad = torch.zeros_like(conics_mu)
            intensities_grad = torch.zeros_like(intensities)
        else:
            if ctx.use_per_gaussian_backward:
                pos2d_grad, conics_mu_grad, intensities_grad = _C.rasterize_backward_per_gaussian(
                    (img_h, img_w),
                    gaussian_ids_sorted,
                    tile_bins,
                    pos2d,
                    conics_mu,
                    intensities,
                    vol_grad.contiguous(),
                )
            else:
                pos2d_grad, conics_mu_grad, intensities_grad = _C.rasterize_backward(
                    (img_h, img_w),
                    gaussian_ids_sorted,
                    tile_bins,
                    pos2d,
                    conics_mu,
                    intensities,
                    vol_grad.contiguous(),
                )

        return (
            pos2d_grad,
            conics_mu_grad,
            intensities_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
