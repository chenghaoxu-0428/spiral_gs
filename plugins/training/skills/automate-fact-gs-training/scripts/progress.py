#!/usr/bin/env python3
"""Print one FaCT-GS reconstruction TensorBoard progress snapshot."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest(accumulator, tag):
    values = accumulator.Scalars(tag) if tag in accumulator.Tags().get("scalars", []) else []
    return values[-1] if values else None


def value_text(item, precision=".5g"):
    return format(item.value, precision) if item is not None else "?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--total-steps", required=True, type=int)
    args = parser.parse_args()

    event_path = args.model_path / "tensorboard"
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    loss = latest(accumulator, "loss/loss_total")
    psnr3d = latest(accumulator, "reconstruction/psnr_3d")
    ssim3d = latest(accumulator, "reconstruction/ssim_3d")
    psnr2d = latest(accumulator, "projection/render_test_psnr_2d")
    ssim2d = latest(accumulator, "projection/render_test_ssim_2d")
    observed = [item for item in (loss, psnr3d, ssim3d, psnr2d, ssim2d) if item]
    current = max((item.step for item in observed), default=0)
    total = max(1, args.total_steps)
    percent = min(100.0, current / total * 100)
    print(
        f"Train: {current}/{total} ({percent:.2f}%) | loss {value_text(loss)} | "
        f"ssim3d {value_text(ssim3d, '.4f')} | psnr3d {value_text(psnr3d, '.3f')} | "
        f"ssim2d {value_text(ssim2d, '.4f')} | psnr2d {value_text(psnr2d, '.3f')}"
    )


if __name__ == "__main__":
    main()
