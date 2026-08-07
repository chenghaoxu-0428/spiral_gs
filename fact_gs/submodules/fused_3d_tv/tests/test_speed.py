import time

import torch
import pytest

from fused_3d_tv import tv3d_loss


def reference_tv3d(volume: torch.Tensor) -> torch.Tensor:
    """Reference implementation of TV3D used for speed comparisons.

    Args:
        volume: CUDA tensor shaped [B, C, D, H, W].

    Returns:
        Scalar tensor containing the isotropic TV3D loss.
    """
    dx = torch.abs(torch.diff(volume, dim=2))
    dy = torch.abs(torch.diff(volume, dim=3))
    dz = torch.abs(torch.diff(volume, dim=4))
    return dx.sum() + dy.sum() + dz.sum()

def _measure(func, base_tensor, repeats):
    """Measure forward/backward runtime for a TV3D implementation.

    Args:
        func: Callable that takes a tensor and returns a scalar loss.
        base_tensor: Tensor used as the source for cloning per repeat.
        repeats: Number of timing iterations to execute.

    Returns:
        Tuple of (forward_mean_seconds, backward_mean_seconds).
    """
    fwd_total = 0.0
    bwd_total = 0.0
    for _ in range(repeats):
        tensor = base_tensor.clone().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = func(tensor)
        torch.cuda.synchronize()
        fwd_total += time.perf_counter() - t0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_total += time.perf_counter() - t0

    return fwd_total / repeats, bwd_total / repeats

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_forward_backward_speed_scaling():
    """Ensure the fused kernel is never slower than the reference by >20%."""
    device = torch.device("cuda")
    sizes = [8, 12, 16, 24, 32, 64, 128, 256]
    repeats = 10

    fused_forward = []
    fused_backward = []
    ref_forward = []
    ref_backward = []

    # warmup to avoid first-call overheads
    warmup = torch.randn(1, 1, 8, 8, 8, device=device, requires_grad=True)
    tv3d_loss(warmup).backward()
    warmup_ref = warmup.detach().clone().requires_grad_(True)
    reference_tv3d(warmup_ref).backward()

    for size in sizes:
        base = torch.randn(1, 1, size, size, size, device=device, dtype=torch.float32)
        fused_fwd, fused_bwd = _measure(tv3d_loss, base, repeats)
        ref_fwd, ref_bwd = _measure(reference_tv3d, base, repeats)

        fused_forward.append(fused_fwd)
        fused_backward.append(fused_bwd)
        ref_forward.append(ref_fwd)
        ref_backward.append(ref_bwd)

    header = (
        f"{'Size':>6} | {'Fused Fwd':>10} | {'Ref Fwd':>10} | "
        f"{'Fused Bwd':>10} | {'Ref Bwd':>10}"
    )
    print(header)
    print("-" * len(header))
    for idx, size in enumerate(sizes):
        print(
            f"{size:6d} | "
            f"{fused_forward[idx]*1000:10.3f} | "
            f"{ref_forward[idx]*1000:10.3f} | "
            f"{fused_backward[idx]*1000:10.3f} | "
            f"{ref_backward[idx]*1000:10.3f}"
        )

    for f_fwd, r_fwd in zip(fused_forward, ref_forward):
        assert f_fwd <= r_fwd * 1.2
    for f_bwd, r_bwd in zip(fused_backward, ref_backward):
        assert f_bwd <= r_bwd * 1.2

if __name__ == "__main__":
    test_forward_backward_speed_scaling()
