#!/usr/bin/env python3
"""Export the preprocessing FDK volume next to vol_gt.npy.

Uses the same helical FDK + intensity normalization as dataset initialization
(``reconstruct_fdk_volume`` / ``fdk_point_cloud``). Writes ``fdk_vol.npy`` in
the dataset folder with the same dtype/layout as ``vol_gt.npy``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# TIGRE must be imported before torch (pulled in via ct_utils).
import tigre  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_preprocess.norm_pipeline import reconstruct_fdk_volume  # noqa: E402


def load_train_views(root: Path):
    meta_path = root / "meta_data.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing meta_data.json: {meta_path}")
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    entries = meta["proj_train"]
    if not entries:
        raise ValueError(f"no proj_train entries in {meta_path}")
    projs = np.stack(
        [np.load(root / item["file_path"]) for item in entries], axis=0
    ).astype(np.float32)
    angles = np.asarray([item["angle"] for item in entries], dtype=np.float32)
    z = np.asarray([item.get("z_shift", 0.0) for item in entries], dtype=np.float32)
    return projs, angles, z, meta["scanner"], meta


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/syn/ldctl004/spiral/ntrain500/r2gs"),
        help="R2-Gaussian dataset directory containing meta_data.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path (default: <data>/fdk_vol.npy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.data.expanduser().resolve()
    output = (args.output or (root / "fdk_vol.npy")).expanduser().resolve()
    projs, angles, z, scanner, _meta = load_train_views(root)
    print(
        f"FDK from {len(projs)} train views, proj shape={tuple(projs.shape[1:])}, "
        f"nVoxel={scanner.get('nVoxel')}"
    )
    volume = reconstruct_fdk_volume(projs, angles, z, scanner).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, volume)
    print(
        f"Wrote {output} shape={tuple(volume.shape)} "
        f"range=[{float(volume.min()):.6g}, {float(volume.max()):.6g}]"
    )


if __name__ == "__main__":
    main()
