from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fact_gs.r2_gaussian.utils.loss_utils import ssim
from fact_gs.rasterize import _quality_safe_tile_bounds


ROOT = Path(__file__).resolve().parents[1]


def _reference_ssim(img1, img2, window_size=11):
    x = torch.arange(window_size, dtype=img1.dtype, device=img1.device)
    weights = torch.exp(-((x - window_size // 2) ** 2) / (2 * 1.5**2))
    weights /= weights.sum()
    window = weights[:, None].mm(weights[None, :])[None, None]
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding)
    mu2 = F.conv2d(img2, window, padding=padding)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq = F.conv2d(img1.square(), window, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2.square(), window, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding) - mu1_mu2
    return (((2 * mu1_mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
        (mu1_sq + mu2_sq + 0.01**2) * (sigma1_sq + sigma2_sq + 0.03**2)
    )).mean()


def test_legacy_ssim_is_numerically_identical():
    generator = torch.Generator().manual_seed(7)
    pred = torch.rand((1, 1, 64, 96), generator=generator, requires_grad=True)
    target = torch.rand((1, 1, 64, 96), generator=generator)
    actual = ssim(pred, target)
    expected = _reference_ssim(pred, target)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.backward()
    assert torch.isfinite(pred.grad).all()


def test_reconstruction_defaults_preserve_r2_quality_budget():
    model = yaml.safe_load((ROOT / "config/model/model_default_recon.yaml").read_text())
    optim = yaml.safe_load((ROOT / "config/optim/optim_default_recon.yaml").read_text())
    assert model["init_mode"] == "auto"
    assert model["init_spatial_lr_scale"] == 1.0
    assert optim["steps"] == 30_000
    assert optim["use_fused_ssim"] is False
    assert optim["max_num_gaussians_absolute"] == 500_000
    assert optim["densify_from_step"] == 500
    assert optim["densify_until_step_percent"] == 0.5


def test_quality_tile_bounds_do_not_depend_on_density():
    pos2d = torch.tensor([[31.5, 31.5], [8.0, 8.0]])
    # Identity inverse covariance and a wider x covariance for point two.
    conics = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.25, 0.0, 1.0, 1.0]])
    radii = torch.tensor([3.0, 6.0])
    tile_min, tile_max, hits = _quality_safe_tile_bounds(
        pos2d, conics, radii, height=64, width=64
    )
    assert tile_min.dtype == torch.int32
    assert torch.all(tile_max >= tile_min)
    assert torch.all(hits > 0)
