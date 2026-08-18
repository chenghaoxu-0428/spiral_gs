"""Scan the cylindrical-flatten tangent point and test the bow hypothesis.

The dominant low-frequency residual mode is a u-quadratic bow. Hypothesis:
the flatten maps the cylindrical detector onto a tangent plane centered at the
array geometric center ``(cols-1)/2``, while the true central element
(DICOM DetectorCentralElement) sits ~1.125 raw elements away. The linear part
of the resulting warp was absorbed by the trained camera u-offset; the
remaining quadratic term ``~ (c/dsd) * u^2 * dI/du`` is the observed bow.

This script regenerates GT projections from raw DICOM for the dataset's test
views, re-flattens them with a parameterized tangent point ``c`` (px in the
downsampled grid), compensates the linear part by translating the model's
prediction accordingly, and reports PSNR + bow energy per ``c``.

Usage:
  python experiments/helpers/scan_flatten_center.py \
      --dicom-root <raw proj dir> --data <dataset dir> --pairs <pairs npz> \
      [--views 400]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom

sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_preprocess.norm_pipeline import (
    PRIVATE_TAGS,
    _dicom_files,
    _value,
)


def load_cylindrical(dicom_root: Path, indices, object_scale=50.0, proj_rescale=400.0, proj_subsample=4):
    """Load the selected DICOM views as downsampled cylindrical projections,
    reproducing norm_pipeline.load_real_projections per-view."""
    files = _dicom_files(dicom_root)
    records = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        records.append((int(_value(ds, "InstanceNumber", default=0)), path))
    records.sort(key=lambda x: x[0])
    ordered = [p for _, p in records]

    images = []
    for idx in indices:
        path = ordered[idx]
        ds = pydicom.dcmread(path)
        slope = float(_value(ds, "RescaleSlope", default=1.0))
        intercept = float(_value(ds, "RescaleIntercept", default=0.0))
        image = np.asarray(ds.pixel_array, dtype=np.float32) * slope + intercept
        image = image.T.astype(np.float32) / float(proj_rescale) * float(object_scale)
        image[image < 0] = 0
        if proj_subsample != 1:
            h_ori, w_ori = image.shape
            image = cv2.resize(image, (w_ori // proj_subsample, h_ori // proj_subsample))
        images.append(image)
    projs = np.stack(images)
    first = pydicom.dcmread(ordered[indices[0]])
    spacing_u = float(_value(first, "DetectorElementTransverseSpacing", PRIVATE_TAGS["spacing_u"]))
    spacing_v = float(_value(first, "DetectorElementAxialSpacing", PRIVATE_TAGS["spacing_v"]))
    dsd = float(_value(first, "ConstantRadialDistance", PRIVATE_TAGS["dsd"])) / 1000.0 * object_scale
    du = spacing_u / 1000.0 * object_scale * proj_subsample
    dv = spacing_v / 1000.0 * object_scale * proj_subsample
    return projs, dsd, du, dv


def flatten_centered(projs: np.ndarray, dsd: float, du: float, dv: float, c: float):
    """Flatten with tangent point at flat coordinate ``c`` (px), matching
    norm_pipeline.flatten_cylindrical_detector (including the v fan factor)."""
    rows, cols = projs.shape[1:]
    flat_width = 2 * dsd * math.tan(cols * du / (2 * dsd))
    duf = flat_width / cols
    u = (np.arange(cols, dtype=np.float32) - (cols - 1) / 2 - c) * duf
    v = (np.arange(rows, dtype=np.float32) - (rows - 1) / 2) * dv
    gamma = np.arctan(u / dsd)
    map_x = np.broadcast_to(
        (dsd * gamma / du + (cols - 1) / 2)[None, :], (rows, cols)
    ).astype(np.float32)
    map_y = (v[:, None] * np.cos(gamma)[None, :] / dv + (rows - 1) / 2).astype(np.float32)
    out = np.stack(
        [
            cv2.remap(proj, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for proj in projs
        ]
    )
    return out


def translate(imgs: np.ndarray, px: float):
    """Shift images horizontally by fractional pixels (linear part)."""
    out = np.empty_like(imgs)
    m = np.float32([[1, 0, px], [0, 1, 0]])
    for i, img in enumerate(imgs):
        out[i] = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    return out


def psnr_stack(a: np.ndarray, b: np.ndarray):
    a = a / np.maximum(a.max(axis=(1, 2)), 1e-9)[:, None, None]
    b = b / np.maximum(b.max(axis=(1, 2)), 1e-9)[:, None, None]
    mse = float(np.mean((a - b) ** 2))
    return -10 * math.log10(mse)


def bow_energy(resid: np.ndarray):
    """Energy of the residual explained by the static u-quadratic pattern."""
    n, h, w = resid.shape
    uu = np.linspace(-1, 1, w).astype(np.float32)
    pat = np.tile(uu**2, (h, 1)).reshape(-1).astype(np.float64)
    # Orthogonalize against const and linear-u.
    one = np.ones_like(pat)
    lin = np.tile(uu, (h, 1)).reshape(-1).astype(np.float64)
    pat = pat - one * (one @ pat) / (one @ one)
    pat = pat - lin * (lin @ pat) / (lin @ lin)
    pat = pat / np.linalg.norm(pat)
    rv = resid.reshape(n, -1).astype(np.float64)
    coef = rv @ pat
    return float(np.mean(coef**2)), float(np.mean(np.abs(coef)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--views", default=400, type=int)
    args = parser.parse_args()

    sidecar = json.loads((args.data / "source_shifts.json").read_text())
    test_idx = sidecar["dicom_indices"]["test"]
    step = max(1, len(test_idx) // args.views)
    sel = test_idx[::step][: args.views]
    print(f"loading {len(sel)} DICOM views...", flush=True)
    cyl, dsd, du, dv = load_cylindrical(args.dicom_root, sel)
    flip = bool(sidecar["z_flip"])
    if flip:
        cyl = cyl[:, ::-1, :].copy()

    pairs = np.load(args.pairs)
    pred = pairs["pred_test"][::step][: args.views]

    # Sanity: flatten at c=0 must reproduce the stored GT closely. The stored
    # images live in the camera-load convention (coord_left column flip).
    gt0 = flatten_centered(cyl, dsd, du, dv, 0.0)
    stored = pairs["gt_test"][::step][: args.views]
    print(f"c=0 vs stored GT: PSNR {psnr_stack(gt0, stored):.2f} dB (no flip)")
    print(f"c=0 vs stored GT (col-flip): PSNR {psnr_stack(gt0[:, :, ::-1], stored):.2f} dB (pipeline sanity)", flush=True)

    print(f"\n{'c (px)':>10} | {'PSNR (best sign)':>18} | {'bow energy':>10} | {'bow |coef|':>10}")
    results = []
    for c in (-1.0, -0.75, -0.53125, -0.5, -0.28125, -0.25, 0.0, 0.25, 0.5):
        gt = flatten_centered(cyl, dsd, du, dv, c)[:, :, ::-1]
        best = None
        for s in (+1.0, -1.0):
            pred_s = translate(pred, s * (c + 0.275))
            score = psnr_stack(gt, pred_s)
            if best is None or score > best[0]:
                resid = gt / np.maximum(gt.max(axis=(1, 2)), 1e-9)[:, None, None] \
                    - pred_s / np.maximum(pred_s.max(axis=(1, 2)), 1e-9)[:, None, None]
                best = (score, s, resid)
        score, s, resid = best
        bow_e, bow_m = bow_energy(resid)
        results.append((c, score, s, bow_e, bow_m))
        print(
            f"{c:>10.5g} | {score:>16.3f} (s={s:+.0f}) | {bow_e:>10.6f} | {bow_m:>10.6f}",
            flush=True,
        )
    best_c = max(results, key=lambda t: t[1])
    print(f"\nbest PSNR at c={best_c[0]:.5g} ({best_c[1]:.3f} dB, sign {best_c[2]:+.0f})")
    min_bow = min(results, key=lambda t: t[3])
    print(f"min bow energy at c={min_bow[0]:.5g} ({min_bow[3]:.6f})")


if __name__ == "__main__":
    main()
