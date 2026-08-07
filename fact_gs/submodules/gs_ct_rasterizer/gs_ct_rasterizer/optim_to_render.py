"""Rasterizer python-cuda bindings with Python-based Gaussian sorting"""


from typing import Optional, Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gs_ct_rasterizer.cuda as _C


def optim_to_render(
    pos3d: Float[Tensor, "*batch 3"],
    scale3d: Float[Tensor, "*batch 3"],
    quat: Float[Tensor, "*batch 4"],
    intensities: Float[Tensor, "*batch"],
    viewmatrix: Float[Tensor, "4 4"],
    projmatrix: Float[Tensor, "4 4"],
    tan_fovx: float,
    tan_fovy: float,
    image_height: int,
    image_width: int,
    mode: int,
    pos2d_buffer: Float[Tensor, "*batch 2"],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Converts Gaussians from optimization-oriented parameters to rendering-oriented.
    Args:
        pos3d: (*batch, 3) Tensor of 3D positions of Gaussians (0-1 range, will be multiplied by volume dimensions).
        scale3d: (*batch, 3) Tensor of scale parameters of Gaussians.
        quat: (*batch, 4) Tensor of quaternion orientations of Gaussians.
        intensities: (*batch, *channels) Tensor of per-gaussian intensity/opacity values
            used when estimating a gaussian's projected extent.
        viewmatrix: (4, 4) Tensor of camera view matrix.
        projmatrix: (4, 4) Tensor of camera projection matrix.
        tan_fovx/tan_fovy: Tangent of the field of view angles in x and y directions.
        image_height/image_width: Height and width of the output image in pixels.
        mode: Mode of operation (0: parallel beam, 1: cone beam).
        pos2d_buffer: Tensor (same shape/device/dtype as the returned 2D positions)
            that will be overwritten with the output positions. Allows for collecting 2D position gradients
            that are used during training for gaussian densification.
    Returns:
        (*batch, 2) Tensor of Gaussian 2D positions in image space.
        (*batch, 4) Tensor of conic parameters and mu (integration factor) values for visualization.
        (*batch, 1) Tensor of minimum enclosing radii for each gaussian.
        (*batch, 2) Tensor of minimum tile indices (x, y) intersected by each gaussian.
        (*batch, 2) Tensor of maximum tile indices (x, y) intersected by each gaussian.
        (*batch,) Tensor storing the number of tiles hit by each gaussian.
    """
    if pos2d_buffer.shape != (*pos3d.shape[:-1], 2):
        raise ValueError(
            f"pos2d_buffer must match shape (*batch, 2); "
            f"got {pos2d_buffer.shape} vs expected {(*pos3d.shape[:-1], 2)}"
        )
    if pos2d_buffer.device != pos3d.device:
        raise ValueError("pos2d_buffer must be on the same device as pos3d")
    if pos2d_buffer.dtype != pos3d.dtype:
        raise ValueError("pos2d_buffer must share dtype with pos3d")

    return _OptimToRender.apply(
        pos3d.contiguous(),
        scale3d.contiguous(),
        quat.contiguous(),
        intensities.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        tan_fovx,
        tan_fovy,
        image_height,
        image_width,
        mode,
        pos2d_buffer,
    )


class _OptimToRender(Function):
    @staticmethod
    def forward(
        ctx,
        pos3d: Float[Tensor, "*batch 3"],
        scale3d: Float[Tensor, "*batch 3"],
        quat: Float[Tensor, "*batch 4"],
        intensities: Float[Tensor, "*batch"],
        viewmatrix: Float[Tensor, "4 4"],
        projmatrix: Float[Tensor, "4 4"],
        tan_fovx: float,
        tan_fovy: float,
        image_height: int,
        image_width: int,
        mode: int,
        pos2d_buffer: Float[Tensor, "*batch 2"],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        num_gaussians = pos3d.size(-2)
        if num_gaussians < 1 or pos3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for pos3d: {pos3d.shape}")
        (
            pos2d_out,
            conics_mu_out,
            radii_out,
            tile_min_out,
            tile_max_out,
            num_tiles_hit_out,
        ) = _C.optim_to_render_forward(
            num_gaussians,
            pos3d,
            scale3d,
            quat,
            intensities,
            viewmatrix,
            projmatrix,
            tan_fovx,
            tan_fovy,
            image_height,
            image_width,
            mode,
        )

        ctx.num_gaussians = num_gaussians
        ctx.viewmatrix = viewmatrix
        ctx.projmatrix = projmatrix
        ctx.tan_fovx = tan_fovx
        ctx.tan_fovy = tan_fovy
        ctx.image_height = image_height
        ctx.image_width = image_width
        ctx.mode = mode

        if not pos2d_buffer.is_contiguous():
            raise ValueError("pos2d_buffer must be contiguous")
        if pos2d_buffer.shape != pos2d_out.shape:
            raise ValueError(
                f"pos2d_buffer shape mismatch: expected {pos2d_out.shape}, "
                f"got {pos2d_buffer.shape}"
            )
        pos2d_buffer.copy_(pos2d_out)
        pos2d_return = pos2d_buffer

        ctx.save_for_backward(
            pos3d,
            scale3d,
            quat,
            radii_out
        )

        return (
            pos2d_return,
            conics_mu_out,
            radii_out,
            tile_min_out,
            tile_max_out,
            num_tiles_hit_out,
        )

    @staticmethod
    def backward(
        ctx,
        pos2d_grad_in: Float[Tensor, "*batch 4"],
        conics_mu_grad_in: Float[Tensor, "*batch 4"],
        radii_grad_in: Float[Tensor, "*batch 1"],
        tile_min_grad_in: Optional[Tensor],
        tile_max_grad_in: Optional[Tensor],
        num_tiles_grad_in: Optional[Tensor],
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # Tile/tile-count gradients are not used in backward pass.
        _ = (tile_min_grad_in, tile_max_grad_in, num_tiles_grad_in, radii_grad_in)

        pos3d, scale3d, quat, radii_out = ctx.saved_tensors

        grad_pos3d, grad_scale3d, grad_quat = _C.optim_to_render_backward(
            ctx.num_gaussians,
            pos3d,
            scale3d,
            quat,
            ctx.viewmatrix,
            ctx.projmatrix,
            ctx.tan_fovx,
            ctx.tan_fovy,
            ctx.image_height,
            ctx.image_width,
            ctx.mode,
            radii_out,
            pos2d_grad_in,
            conics_mu_grad_in,
        )

        # Only first 3 inputs require gradients
        grad_inputs = [
            grad_pos3d,
            grad_scale3d,
            grad_quat,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            pos2d_grad_in,
        ]

        return tuple(grad_inputs)
