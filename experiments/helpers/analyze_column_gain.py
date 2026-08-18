"""Fit a static per-column multiplicative gain profile to the residual.

Tests the hypothesis that the dominant low-frequency residual (a smooth,
row-independent u-quadratic bow) is a residual detector flat-field / anode
heel effect: a fixed multiplicative gain g(u) along the detector columns.
The per-view max-normalization of the metric does not remove spatial gain
profiles, so they survive as systematic residual structure.

Reports: explained variance of the global smooth g(u) fit, the recovered
profile shape, and the PSNR gain when pred is corrected by g(u). Also fits
per-source-state profiles to separate FFSZ state effects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))


def psnr(a, b):
    a = a / np.maximum(a.max(axis=(1, 2)), 1e-9)[:, None, None]
    b = b / np.maximum(b.max(axis=(1, 2)), 1e-9)[:, None, None]
    return -10 * math.log10(float(np.mean((a - b) ** 2)))


def fit_column_gain(resid, pred, smooth):
    """Least-squares per-column gain: resid[u] ~ g[u] * pred[u].

    ``smooth`` > 1 fits a polynomial (degree smooth) in column index instead
    of a free per-column gain.
    """
    n, h, w = pred.shape
    if smooth:
        uu = np.linspace(-1, 1, w)
        design = np.stack([uu**k for k in range(smooth + 1)], axis=1)  # (w, k)
        # Build normal equations across all pixels.
        pw = pred.transpose(0, 2, 1).reshape(-1, w)  # (n*h, w)
        rw = resid.transpose(0, 2, 1).reshape(-1, w)
        lhs = design.T @ (pw.T @ pw) @ design
        rhs = design.T @ (pw * rw).sum(axis=0)
        coef = np.linalg.solve(lhs, rhs)
        gain = design @ coef
    else:
        pw = pred.transpose(0, 2, 1).reshape(-1, w)
        rw = resid.transpose(0, 2, 1).reshape(-1, w)
        gain = (rw * pw).sum(axis=0) / np.maximum((pw * pw).sum(axis=0), 1e-12)
    return gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()

    pairs = np.load(args.pairs)
    gt = pairs["gt_test"]
    pred = pairs["pred_test"]
    sidecar = json.loads((args.data / "source_shifts.json").read_text())
    state = np.asarray(
        [1.0 if abs(s["axial"]) > 1e-9 else 0.0 for s in sidecar["shifts"]["test"]]
    )

    g = gt / np.maximum(gt.max(axis=(1, 2)), 1e-9)[:, None, None]
    p = pred / np.maximum(pred.max(axis=(1, 2)), 1e-9)[:, None, None]
    r = g - p
    base = psnr(g, p)
    print(f"baseline PSNR: {base:.3f} dB", flush=True)

    for deg in (1, 2, 3, 5, None):
        gain = fit_column_gain(r, p, deg)
        label = f"poly{deg}" if deg else "free184"
        p_corr = p * (1.0 + gain[None, None, :])
        # renormalize per view as the metric does
        score = psnr(g, p_corr)
        explained = 1.0 - float(((r - (p_corr - p)) ** 2).sum() / (r**2).sum())
        print(f"gain({label}): PSNR {score:.3f} dB (d={score-base:+.3f}), "
              f"residual variance explained {explained*100:.2f}%", flush=True)
        if deg == 2:
            print(f"  delta profile[::32] = {np.round(gain[::32], 6).tolist()}")

    # Per-source-state gain profiles (poly2), to isolate FFSZ state effects.
    for st, name in ((0.0, "stateA"), (1.0, "stateB")):
        m = state == st
        gain = fit_column_gain(r[m], p[m], 2)
        p_corr = p[m] * (1.0 + gain[None, None, :])
        score = psnr(g[m], p_corr)
        print(f"per-{name} poly2 gain: PSNR {score:.3f} dB "
              f"(base {psnr(g[m], p[m]):.3f})", flush=True)

    # Combined: per-state poly2 gain.
    gain = np.empty((2, pred.shape[2]))
    for st in (0, 1):
        m = state == st
        gain[int(st)] = fit_column_gain(r[m], p[m], 2)
    p_corr = p * (1.0 + gain[state.astype(int), None, :])
    score = psnr(g, p_corr)
    print(f"per-state poly2 gain: PSNR {score:.3f} dB (d={score-base:+.3f})", flush=True)


if __name__ == "__main__":
    main()
