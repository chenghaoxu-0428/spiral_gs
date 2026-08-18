"""Test whether decimate-then-flatten vs flatten-then-decimate matters.

The current pipeline downsamples the cylindrical projections 4x (cv2.resize)
BEFORE the tangent-plane remap. Remapping a decimated signal introduces
smooth, u-dependent sub-pixel interpolation distortion that the Gaussian
model cannot reproduce. This script regenerates GT both ways from raw DICOM
and compares them against the trained model's predictions (PSNR + u-quadratic
bow energy).

Usage:
  python experiments/helpers/test_flatten_order.py \
      --dicom-root <raw proj dir> --data <dataset dir> --pairs <pairs npz>
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
from scan_flatten_center import bow_energy, flatten_centered, load_cylindrical


def load_raw(dicom_root: Path, indices, object_scale=50.0, proj_rescale=400.0):
    files = _dicom_files(dicom_root)
    records = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        records.append((int(_value(ds, "InstanceNumber", default=0)), path))
    records.sort(key=lambda x: x[0])
    ordered = [p for _, p in records]
    images = []
    for idx in indices:
        ds = pydicom.dcmread(ordered[idx])
        slope = float(_value(ds, "RescaleSlope", default=1.0))
        intercept = float(_value(ds, "RescaleIntercept", default=0.0))
        image = np.asarray(ds.pixel_array, dtype=np.float32) * slope + intercept
        image = image.T.astype(np.float32) / float(proj_rescale) * float(object_scale)
        image[image < 0] = 0
        images.append(image)
    projs = np.stack(images)
    first = pydicom.dcmread(ordered[indices[0]])
    spacing_u = float(_value(first, "DetectorElementTransverseSpacing", PRIVATE_TAGS["spacing_u"]))
    spacing_v = float(_value(first, "DetectorElementAxialSpacing", PRIVATE_TAGS["spacing_v"]))
    dsd = float(_value(first, "ConstantRadialDistance", PRIVATE_TAGS["dsd"])) / 1000.0 * object_scale
    du = spacing_u / 1000.0 * object_scale
    dv = spacing_v / 1000.0 * object_scale
    return projs, dsd, du, dv


def psnr_stack(a, b):
    a = a / np.maximum(a.max(axis=(1, 2)), 1e-9)[:, None, None]
    b = b / np.maximum(b.max(axis=(1, 2)), 1e-9)[:, None, None]
    return -10 * math.log10(float(np.mean((a - b) ** 2)))


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
    flip = bool(sidecar["z_flip"])

    raw, dsd, du, dv = load_raw(args.dicom_root, sel)
    if flip:
        raw = raw[:, ::-1, :].copy()

    # Variant A: current pipeline order (downsample 4x, then flatten).
    down = np.stack([cv2.resize(img, (184, 16)) for img in raw])
    gt_a = flatten_centered(down, dsd, du * 4, dv * 4, 0.0)[:, :, ::-1]

    # Variant B: flatten at full 736-column resolution, then downsample 4x.
    flat_full = flatten_centered(raw, dsd, du, dv, 0.0)
    gt_b = np.stack([cv2.resize(img, (184, 16)) for img in flat_full])[:, :, ::-1]

    pairs = np.load(args.pairs)
    stored = pairs["gt_test"][::step][: args.views]
    pred = pairs["pred_test"][::step][: args.views]

    print(f"A (down-then-flatten, current) vs stored GT: {psnr_stack(gt_a, stored):.2f} dB (sanity)", flush=True)
    print(f"A vs B (order difference): {psnr_stack(gt_a, gt_b):.2f} dB", flush=True)
    print(f"A vs pred: {psnr_stack(gt_a, pred):.3f} dB", flush=True)
    print(f"B (flatten-then-down) vs pred: {psnr_stack(gt_b, pred):.3f} dB", flush=True)

    def bow(resid):
        e, m = bow_energy(resid)
        return e

    ga = gt_a / np.maximum(gt_a.max(axis=(1, 2)), 1e-9)[:, None, None]
    gb = gt_b / np.maximum(gt_b.max(axis=(1, 2)), 1e-9)[:, None, None]
    q = pred / np.maximum(pred.max(axis=(1, 2)), 1e-9)[:, None, None]
    print(f"bow energy  A vs pred: {bow(ga - q):.6f}", flush=True)
    print(f"bow energy  B vs pred: {bow(gb - q):.6f}", flush=True)
    print(f"bow energy  A vs B    : {bow(ga - gb):.6f} (bow content of the order difference)", flush=True)


if __name__ == "__main__":
    main()
