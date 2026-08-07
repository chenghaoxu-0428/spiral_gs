"""CUDA fused 3D total variation implementation."""

from __future__ import annotations

import torch

from torch import Tensor

if not torch.cuda.is_available():
    raise RuntimeError("fused-3d-tv requires CUDA")

from fused_3d_tv_cuda import tv3d_forward as _tv3d_forward, tv3d_backward as _tv3d_backward


class _TV3DAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume: Tensor) -> Tensor:
        """Runs the fused 3D total variation kernel and saves inputs for backward.

        Args:
            ctx: Autograd context used to stash the volume tensor.
            volume: Input tensor shaped [B, C, D, H, W] stored on CUDA.

        Returns:
            Scalar tensor containing the accumulated TV3D loss.
        """
        volume = volume.contiguous()
        tv = _tv3d_forward(volume)
        ctx.save_for_backward(volume)
        return tv

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Computes gradients of the fused TV3D loss w.r.t. the input volume.

        Args:
            ctx: Autograd context with saved forward tensors.
            grad_output: Upstream gradient for the scalar TV loss.

        Returns:
            Tensor of gradients that matches the shape of the original volume.
        """
        (volume,) = ctx.saved_tensors
        grad_volume = _tv3d_backward(volume, grad_output.contiguous())
        return grad_volume


def tv3d_loss(volume: Tensor, reduction: str = "sum") -> Tensor:
    """Compute 3D total variation loss for a 5D tensor.

    Args:
        volume: Tensor shaped [B, C, D, H, W].
        reduction: only "sum" is supported to match the reference implementation (ignored otherwise).

    Returns:
        Scalar tensor representing the fused total variation loss.
    """
    if reduction != "sum":
        raise ValueError("Only sum reduction is supported")

    if volume.dim() != 5:
        raise ValueError("Expected volume shaped [B, C, D, H, W]")
    if volume.dtype != torch.float32:
        raise TypeError("Only float32 inputs are supported")
    if not volume.is_cuda:
        raise TypeError("tv3d_loss expects a CUDA tensor")

    return _TV3DAutograd.apply(volume)


__all__ = ["tv3d_loss"]
