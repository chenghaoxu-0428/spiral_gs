import time

import torch
import pytest

from fused_3d_tv import tv3d_loss


def reference_tv3d(volume: torch.Tensor) -> torch.Tensor:
    """Reference implementation of TV3D used to validate CUDA kernels.

    Args:
        volume: CUDA tensor shaped [B, C, D, H, W].

    Returns:
        Scalar tensor containing the isotropic TV3D loss.
    """
    dx = torch.abs(torch.diff(volume, dim=2))
    dy = torch.abs(torch.diff(volume, dim=3))
    dz = torch.abs(torch.diff(volume, dim=4))
    return dx.sum() + dy.sum() + dz.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_forward_matches_reference():
    device = torch.device("cuda")
    x = torch.randn(2, 3, 10, 10, 10, device=device, dtype=torch.float32)
    fused = tv3d_loss(x.clone().requires_grad_(True))
    reference = reference_tv3d(x)
    torch.testing.assert_close(fused, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backward_matches_reference():
    device = torch.device("cuda")
    inp = torch.randn(1, 2, 10, 10, 10, device=device, dtype=torch.float32, requires_grad=True)
    ref_inp = inp.detach().clone().requires_grad_(True)

    fused = tv3d_loss(inp)
    ref = reference_tv3d(ref_inp)

    fused.backward()
    ref.backward()

    torch.testing.assert_close(inp.grad, ref_inp.grad, atol=1e-5, rtol=1e-5)
