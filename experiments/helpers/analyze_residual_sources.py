"""Attribute the projection-domain residual to physical candidate sources.

Loads exported GT/prediction pairs (projection_pairs_corrected.npz), computes
per-view max-normalized residuals (the project 2D metric convention), then
regresses the residual stack on a Gram-Schmidt-orthogonalized candidate basis:

  du, dv        : spatial derivatives of pred  -> residual u/v misalignment
  P             : pred itself                  -> gain-like / linear scale
  P2            : squared pred                 -> beam-hardening quadratic
  LP, HP        : low/high-pass split of pred  -> scatter / structure terms
  row_i         : fixed detector-row patterns  -> flat-field/v-dependent bias
  col pattern   : fixed low-order u patterns   -> flatten/static detector bias
  state x const : source-state fixed residual  -> FFSZ state bias

Reports incremental R^2 per candidate, split into low/high frequency bands of
the residual, plus SVD mode shapes and per-view coefficient correlations with
angle/z/source-state.

Usage:
  python experiments/helpers/analyze_residual_sources.py \
      --pairs <projection_pairs_corrected.npz> \
      --data <dataset dir> [--split test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))


def lowpass_mask(shape, sigma_cycles=0.055):
    """Gaussian low-pass mask in the frequency domain."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.exp(-(fx**2 + fy**2) / (2 * sigma_cycles**2))


def split_bands(x, mask):
    """Split x into low/high frequency parts along the last two axes."""
    spec = np.fft.rfft2(x, axes=(-2, -1))
    low = np.fft.irfft2(spec * mask[:, : x.shape[-1] // 2 + 1], s=x.shape[-2:], axes=(-2, -1))
    return low, x - low


def sequential_attribution(cand_columns, target, total_var):
    """Sequential least-squares attribution of a target vector.

    Each candidate column list is regressed (lstsq) against the current
    residual, the fit is removed, and the explained-variance gain is recorded.
    Order-sensitive by design: earlier groups absorb shared variance.
    """
    resid = target.astype(np.float32).copy()
    gains = []
    for name, cols in cand_columns:
        x = np.stack([c.astype(np.float32) for c in cols], axis=1)
        coef, *_ = np.linalg.lstsq(x, resid, rcond=None)
        fit = x @ coef
        gain = float((fit**2).sum() / total_var)
        resid = resid - fit
        gains.append((name, gain))
    return gains, resid




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print("[1/7] loading pairs...", flush=True)
    pairs = np.load(args.pairs)
    gt = pairs[f"gt_{args.split}"]
    pred = pairs[f"pred_{args.split}"]
    angles = pairs[f"angle_{args.split}"]
    z = pairs[f"z_{args.split}"]
    n, h, w = gt.shape
    print(f"{args.split}: {n} views, {h}x{w}", flush=True)

    # Per-view max normalization (project metric convention).
    print("[2/7] normalizing...", flush=True)
    g = gt / np.maximum(gt.max(axis=(1, 2)), 1e-9)[:, None, None]
    p = pred / np.maximum(pred.max(axis=(1, 2)), 1e-9)[:, None, None]
    r = g - p  # residual, metric space
    total_var = float((r**2).sum())

    # Frequency split of the residual.
    print("[3/7] frequency split...", flush=True)
    mask = lowpass_mask((h, w))
    r_low, r_high = split_bands(r, mask)
    frac_low = float((r_low**2).sum() / total_var)
    print(f"residual energy: total {total_var:.4f} | low-freq {frac_low*100:.1f}% | high-freq {(1-frac_low)*100:.1f}%", flush=True)

    # Per-view scalar regressors.
    state = np.zeros(n)
    sidecar_path = Path(args.data) / "source_shifts.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        shifts = sidecar["shifts"][args.split]
        state = np.asarray([1.0 if abs(s["axial"]) > 1e-9 else 0.0 for s in shifts])
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    zc = z - z.mean()

    print("[4/6] building candidate bases...", flush=True)
    # Spatial candidate bases (vectorized over all pixels).
    def vec(x):
        return x.reshape(n * h * w).astype(np.float32)

    pv = vec(p)
    du = np.gradient(p, axis=2).reshape(-1)
    dv = np.gradient(p, axis=1).reshape(-1)
    p2 = (p - p.mean(axis=(1, 2), keepdims=True)) ** 2
    p2v = p2.reshape(-1)
    p_low, p_high = split_bands(p, mask)
    lp = p_low.reshape(-1)
    hp = p_high.reshape(-1)
    ones = np.ones(n * h * w)

    # Fixed row pattern (per-vrow dummies) and low-order column pattern.
    row_basis = []
    for vrow in range(h):
        m = np.zeros((h, w), dtype=np.float32)
        m[vrow, :] = 1.0
        row_basis.append(np.tile(m, (n, 1, 1)).reshape(-1))
    # Low-order u polynomials (constant + linear + quadratic along u).
    uu = np.linspace(-1, 1, w).astype(np.float32)
    col_basis = []
    for deg in range(3):
        m = np.tile(uu**deg, (h, 1))
        col_basis.append(np.tile(m, (n, 1, 1)).reshape(-1))

    # Interactions: state x const, state x P, z x const, z x P, sin/cos x const.
    tiled = lambda s: np.repeat(s, h * w)

    groups = [
        ("const", [ones]),
        ("du", [du]),
        ("dv", [dv]),
        ("P", [pv]),
        ("P2", [p2v]),
        ("LP(P)", [lp]),
        ("HP(P)", [hp]),
        ("u-poly(3)", col_basis),
        ("row-pattern(16)", row_basis),
        ("state x [1,P]", [tiled(state) * ones, tiled(state) * pv]),
        ("z x [1,P]", [tiled(zc) * ones, tiled(zc) * pv]),
        ("sin/cos x 1", [tiled(sin_a) * ones, tiled(cos_a) * ones]),
    ]

    print("[5/6] attributing...", flush=True)
    rv = vec(r)
    print(f"\nincremental R^2 (order-sensitive sequential least squares):")
    gains, resid = sequential_attribution(groups, rv, total_var)
    explained = 0.0
    for name, gain in gains:
        explained += gain
        print(f"  {name:<22} {gain*100:6.2f}%  (cumulative {explained*100:6.2f}%)")
    print(f"  unexplained: {(1-explained)*100:.2f}%")

    # Same attribution restricted to the low-freq residual band.
    print(f"\nincremental R^2 on low-freq residual only ({frac_low*100:.1f}% of energy):")
    rl = vec(r_low)
    gains_low, _ = sequential_attribution(groups, rl, float(rl @ rl))
    explained_low = 0.0
    for name, gain in gains_low:
        explained_low += gain
        print(f"  {name:<22} {gain*100:6.2f}%  (cumulative {explained_low*100:6.2f}%)")

    print("[6/6] SVD of residual stack...", flush=True)
    # SVD of residual stack: describe top modes.
    rmat = r.reshape(n, -1)
    uu_, s_, vv_ = np.linalg.svd(rmat, full_matrices=False)
    cum = np.cumsum(s_**2) / (s_**2).sum()
    print(f"\nSVD: top-1 {cum[0]*100:.1f}%, top-3 {cum[2]*100:.1f}%, top-10 {cum[9]*100:.1f}%")
    for k in range(3):
        mode = vv_[k].reshape(h, w)
        row_prof = mode.mean(axis=1)
        col_prof = mode.mean(axis=0)
        col_corr_u = np.corrcoef(col_prof, np.linspace(-1, 1, w))[0, 1]
        print(
            f"  mode{k}: row-profile std {row_prof.std():.4f}, "
            f"col-profile corr(u) {col_corr_u:+.3f}, "
            f"col-profile corr(u^2) {np.corrcoef(col_prof, np.linspace(-1,1,w)**2)[0,1]:+.3f}, "
            f"row range {row_prof.min():+.4f}..{row_prof.max():+.4f}"
        )
        # correlation of per-view mode coefficient with angle/z/state
        c = uu_[:, k] * s_[k]
        print(
            f"         coef corr: angle {np.corrcoef(c, angles)[0,1]:+.3f}, "
            f"z {np.corrcoef(c, z)[0,1]:+.3f}, state {np.corrcoef(c, state)[0,1]:+.3f}"
        )


if __name__ == "__main__":
    main()
