#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from math import exp

try:
    from fused_3d_tv import tv3d_loss as _fused_tv3d_loss
except (ImportError, RuntimeError):
    _fused_tv3d_loss = None


def tv_3d_loss(vol, reduction_value=1):
    if _fused_tv3d_loss is not None and vol.is_cuda:
        tv = _fused_tv3d_loss(vol.clone().unsqueeze(0).unsqueeze(0))
    else:
        # CPU/reference fallback, also useful for numerical regression tests.
        tv = (
            torch.abs(torch.diff(vol, dim=0)).sum()
            + torch.abs(torch.diff(vol, dim=1)).sum()
            + torch.abs(torch.diff(vol, dim=2)).sum()
        )

    return tv/reduction_value


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def frequency_magnitude_loss(pred, target, highpass_cutoff=0.1, eps=1e-6):
    """Compare log Fourier magnitudes above a normalized radial cutoff."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Frequency loss requires equal shapes, got {pred.shape} and {target.shape}."
        )
    cutoff = float(max(0.0, min(1.0, highpass_cutoff)))
    pred_spectrum = torch.fft.rfft2(pred.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    pred_magnitude = torch.log1p(torch.abs(pred_spectrum) + eps)
    target_magnitude = torch.log1p(torch.abs(target_spectrum) + eps)

    height, width = pred.shape[-2:]
    fy = torch.fft.fftfreq(height, device=pred.device).abs() / 0.5
    fx = torch.fft.rfftfreq(width, device=pred.device) / 0.5
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    mask = radius >= cutoff
    if not torch.any(mask):
        return pred_magnitude.new_zeros(())
    return torch.abs(pred_magnitude[..., mask] - target_magnitude[..., mask]).mean()


def _ssim_window(window_size, channel, dtype, device):
    weights = torch.tensor(
        [exp(-((x - window_size // 2) ** 2) / (2 * 1.5**2)) for x in range(window_size)],
        dtype=dtype,
        device=device,
    )
    weights = weights / weights.sum()
    window = weights[:, None].mm(weights[None, :])[None, None]
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11):
    """Legacy r2_gaussian SSIM, kept for quality-baseline parity."""
    channel = img1.size(-3)
    window = _ssim_window(window_size, channel, img1.dtype, img1.device)
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    return (((2 * mu1_mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
        (mu1_sq + mu2_sq + 0.01**2) * (sigma1_sq + sigma2_sq + 0.03**2)
    )).mean()
