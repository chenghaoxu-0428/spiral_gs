"""Quantify the interpolation loss of the cylindrical->flat resampling.

1. Round-trip test on real GT: flatten -> unflatten -> flatten and measure
   the pure resampling error at the training resolution (16x184) vs full
   detector resolution (64x888, ldctc001).
2. Model-fit test on synthetic data: evaluate the matched-syn model (trained
   on pure flat projections, 47.43 dB) against its own GT after the same
   flatten round-trip. The PSNR drop isolates the metric impact of the
   resampling step itself.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))


def flatten_maps(dsd, du, dv, rows, cols):
    """Forward flatten maps (same as norm_pipeline.flatten_cylindrical_detector)."""
    flat_width = 2 * dsd * math.tan(cols * du / (2 * dsd))
    duf = flat_width / cols
    u = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2) * duf
    v = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2) * dv
    gamma = np.arctan(u / dsd)
    map_x = np.broadcast_to(
        (dsd * gamma / du + (cols - 1) / 2)[None, :], (rows, cols)
    ).astype(np.float32)
    map_y = (v[:, None] * np.cos(gamma)[None, :] / dv + (rows - 1) / 2).astype(np.float32)
    return map_x, map_y


def unflatten_maps(map_x, map_y, rows, cols):
    """Inverse maps: for each cylindrical pixel, the flat coordinate to read.

    Forward maps: flat[i] samples cyl[map_x[i]] with
    map_y[j, i] = (j - c) * cos(gamma_i) + c. Inverse: for each cylindrical
    pixel (j, i) read flat[(j - c) / cos(gamma_i) + c, inv_x[i]].
    """
    x = np.arange(cols, dtype=np.float32)
    inv_x = np.interp(x, map_x[0], x)  # flat coord for each cylindrical column
    center = (rows - 1) / 2
    j = np.arange(rows, dtype=np.float32)
    cos_gamma = (map_y[:, :] - center) / (j[:, None] - center)
    inv_y = (j[:, None] - center) / cos_gamma + center
    mx = np.broadcast_to(inv_x[None, :], (rows, cols)).astype(np.float32)
    return mx, inv_y.astype(np.float32)


def remap_stack(imgs, map_x, map_y):
    return np.stack(
        [cv2.remap(i, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in imgs]
    )


def psnr(a, b):
    a = a / np.maximum(a.max(axis=(1, 2)), 1e-9)[:, None, None]
    b = b / np.maximum(b.max(axis=(1, 2)), 1e-9)[:, None, None]
    return -10 * math.log10(float(np.mean((a - b) ** 2)))


def roundtrip(flat, dsd, du, dv):
    """flatten -> unflatten -> flatten round trip; returns (psnr, rms)."""
    rows, cols = flat.shape[1:]
    mx, my = flatten_maps(dsd, du, dv, rows, cols)
    ux, uy = unflatten_maps(mx, my, rows, cols)
    cyl = remap_stack(flat, ux, uy)   # flat -> cylindrical (1 interpolation)
    back = remap_stack(cyl, mx, my)   # cylindrical -> flat (1 interpolation)
    return psnr(flat, back), float(np.sqrt(np.mean((flat - back) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path,
                        help="npz with gt_test/pred_test (test views only)")
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    pairs = np.load(args.pairs)
    gt = pairs["gt_test"]
    meta = json.loads((args.dataset / "meta_data.json").read_text())
    scanner = meta["scanner"]
    dsd = float(scanner["DSD"])
    n_det = scanner["nDetector"]
    # detector spacing in scene units
    d_det = scanner.get("dDetector")
    if d_det is None:
        d_det = [s / n for s, n in zip(scanner["sDetector"], n_det)]
    dv, du = float(d_det[0]), float(d_det[1])

    # Stored GT is in camera convention (coord_left col-flip); flip back to
    # pipeline convention for the resampling test, then flip again.
    coord_left = bool(scanner.get("coord_left", False))
    g = gt[:, :, ::-1].copy() if coord_left else gt
    rt_psnr, rt_rms = roundtrip(g, dsd, du, dv)
    print(f"roundtrip (n={len(gt)}, {n_det[0]}x{n_det[1]}): "
          f"PSNR {rt_psnr:.2f} dB, RMS {rt_rms:.6f} (normalized-scale)")

    if "pred_test" in pairs:
        pred = pairs["pred_test"]
        back = None
        mx, my = flatten_maps(dsd, du, dv, *g.shape[1:])
        ux, uy = unflatten_maps(mx, my, *g.shape[1:])
        cyl = remap_stack(g, ux, uy)
        back = remap_stack(cyl, mx, my)
        back = back[:, :, ::-1] if coord_left else back
        print(f"model fit vs original GT : {psnr(gt, pred):.3f} dB")
        print(f"model fit vs roundtrip GT: {psnr(back, pred):.3f} dB")


if __name__ == "__main__":
    main()
