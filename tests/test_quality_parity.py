from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fact_gs.r2_gaussian.utils.loss_utils import frequency_magnitude_loss, ssim
from fact_gs.r2_gaussian.utils.image_utils import metric_vol
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


def test_volume_ssim_matches_r2_three_axis_slice_average():
    generator = torch.Generator().manual_seed(11)
    target = torch.rand((5, 6, 7), generator=generator)
    pred = torch.rand((5, 6, 7), generator=generator)
    expected_axes = []
    for axis in range(3):
        values = []
        for index in range(target.shape[axis]):
            target_slice = target.select(axis, index)
            pred_slice = pred.select(axis, index)
            values.append(ssim(target_slice[None, None], pred_slice[None, None]))
        expected_axes.append(torch.stack(values).mean().item())

    actual, actual_axes = metric_vol(target, pred, "ssim")
    assert actual_axes == expected_axes
    assert actual == sum(expected_axes) / 3


def test_optional_frequency_loss_is_stable_and_differentiable():
    target = torch.zeros((1, 32, 34))
    target[:, ::2, ::2] = 1
    identical = target.clone().requires_grad_(True)
    zero_loss = frequency_magnitude_loss(identical, target, highpass_cutoff=0.1)
    torch.testing.assert_close(zero_loss, torch.zeros_like(zero_loss), atol=1e-7, rtol=0)

    pred = torch.zeros_like(target, requires_grad=True)
    loss = frequency_magnitude_loss(pred, target, highpass_cutoff=0.1)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_reconstruction_defaults_preserve_r2_quality_budget():
    model = yaml.safe_load((ROOT / "config/model/model_default_recon.yaml").read_text())
    optim = yaml.safe_load((ROOT / "config/optim/optim_default_recon.yaml").read_text())
    assert model["init_mode"] == "auto"
    assert model["init_spatial_lr_scale"] == 1.0
    assert model["density_rescale"] == 0.15
    assert model["density_init_scale"] == 1.0
    assert model["init_seed"] == 0
    assert model["save_generated_init"] is True
    assert optim["steps"] == 30_000
    assert optim["use_fused_ssim"] is False
    assert optim["lambda_frequency"] == 0.0
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
