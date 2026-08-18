"""Extract per-view CT-PD source dynamics shifts and align them to a dataset.

Reads the SourceDynamicsModule fields (SourceAngularPositionShift,
SourceAxialPositionShift, SourceRadialDistanceShift, FlyingFocalSpotMode)
from every raw DICOM projection in the same order used by
``norm_pipeline.load_real_projections`` (sorted by InstanceNumber), then maps
them onto an existing dataset's train/test views via ``_split_indices``.

The output sidecar stores shift values in the *dataset convention* units:
  - ``angular_shift``: radians, added to the per-view ``angle``
  - ``axial_shift``:   scene units (mm / 1000 * object_scale, z-flip applied),
                       added to the per-view ``z_shift``
  - ``radial_shift``:  scene units (mm / 1000 * object_scale), added to
                       ``DSO`` (or ``DSD``, per interpretation)

Alignment is validated against the existing ``meta_data.json`` before writing.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pydicom

from norm_pipeline import (
    PRIVATE_TAGS,
    _dicom_files,
    _split_indices,
    _value,
)


def _cs_string(ds: pydicom.Dataset, tag):
    """Read a CS (code string) private tag, handling implicit-VR bytes."""
    if tag in ds:
        value = ds[tag].value
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="replace").strip()
        return str(value).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="existing r2gs dataset dir (contains meta_data.json)")
    parser.add_argument("--object-scale", required=True, type=float)
    parser.add_argument("--n-train", required=True, type=int)
    parser.add_argument("--n-test", required=True, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for path in _dicom_files(args.dicom_root):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        instance = int(_value(ds, "InstanceNumber", default=len(records)))
        angle = float(_value(ds, "DetectorFocalCenterAngularPosition", PRIVATE_TAGS["angle"]))
        z_mm = float(_value(ds, "DetectorFocalCenterAxialPosition", PRIVATE_TAGS["table_z"]))
        ang_shift = float(_value(ds, "source_angular_shift", PRIVATE_TAGS["source_angular_shift"], default=0.0))
        ax_shift_mm = float(_value(ds, "source_axial_shift", PRIVATE_TAGS["source_axial_shift"], default=0.0))
        rad_shift_mm = float(_value(ds, "source_radial_shift", PRIVATE_TAGS["source_radial_shift"], default=0.0))
        ffs_mode = _cs_string(ds, PRIVATE_TAGS["flying_focal_spot_mode"])
        records.append((instance, angle, z_mm, ang_shift, ax_shift_mm, rad_shift_mm, ffs_mode))
    records.sort(key=lambda x: x[0])
    n = len(records)
    print(f"read {n} DICOM projections")

    instances = np.asarray([r[0] for r in records])
    angles = np.asarray([r[1] for r in records], dtype=np.float64)
    z_mm = np.asarray([r[2] for r in records], dtype=np.float64)
    ang_shift = np.asarray([r[3] for r in records], dtype=np.float64)
    ax_shift_mm = np.asarray([r[4] for r in records], dtype=np.float64)
    rad_shift_mm = np.asarray([r[5] for r in records], dtype=np.float64)
    ffs_modes = [r[6] for r in records]

    # Diagnostic: per-view state pattern and cross-field correlation.
    for name, values in (
        ("angular", ang_shift),
        ("axial(mm)", ax_shift_mm),
        ("radial(mm)", rad_shift_mm),
    ):
        uniq = np.unique(values)
        print(f"{name}: unique values {uniq[:8]} (n={len(uniq)})")
        if len(uniq) == 2:
            state = (values == uniq[1]).astype(int)
            print(f"  state pattern: {state[:16].tolist()}...")
            print(f"  state 0 count {int((state == 0).sum())}, state 1 count {int(state.sum())}")
    if len(np.unique(ax_shift_mm)) == 2 and len(np.unique(rad_shift_mm)) == 2:
        s_ax = (ax_shift_mm == np.unique(ax_shift_mm)[1]).astype(int)
        s_rad = (rad_shift_mm == np.unique(rad_shift_mm)[1]).astype(int)
        agree = float(np.mean(s_ax == s_rad))
        print(f"axial/radial state agreement: {agree:.4f}")
    print(f"ffs modes: {sorted(set(ffs_modes))}")

    # Reproduce the pipeline's z convention (flip when DICOM z increases with
    # instance index) and scene-unit conversion.
    z_scene = z_mm / 1000.0 * args.object_scale
    flip = len(z_scene) > 1 and float(np.median(np.diff(z_scene))) > 0
    if flip:
        z_scene *= -1
        ax_shift_mm_scene = ax_shift_mm * (-1.0)
    else:
        ax_shift_mm_scene = ax_shift_mm
    axial_shift = ax_shift_mm_scene / 1000.0 * args.object_scale
    radial_shift = rad_shift_mm / 1000.0 * args.object_scale
    print(f"z flip applied: {flip}")

    train_idx, test_idx = _split_indices(n, args.n_train, args.n_test, args.seed)

    # Validate alignment against the existing dataset.
    meta = json.loads((args.dataset / "meta_data.json").read_text())
    ok_angle = True
    ok_z = True
    for split, idx in (("train", train_idx), ("test", test_idx)):
        entries = meta[f"proj_{split}"]
        for j, view in enumerate(idx):
            if not np.isclose(angles[view], entries[j]["angle"], atol=1e-5):
                ok_angle = False
                print(f"MISMATCH angle {split}[{j}] dicom={angles[view]:.6f} meta={entries[j]['angle']:.6f}")
            if not np.isclose(z_scene[view], entries[j]["z_shift"], atol=1e-4):
                ok_z = False
                print(f"MISMATCH z {split}[{j}] dicom={z_scene[view]:.6f} meta={entries[j]['z_shift']:.6f}")
    print(f"alignment check: angle {'OK' if ok_angle else 'FAIL'}, z {'OK' if ok_z else 'FAIL'}")
    if not (ok_angle and ok_z):
        raise SystemExit("alignment validation failed; not writing sidecar")

    payload = {
        "schema_version": 1,
        "n_dicom": n,
        "object_scale": args.object_scale,
        "z_flip": bool(flip),
        "n_train": args.n_train,
        "n_test": args.n_test,
        "seed": args.seed,
        "flying_focal_spot_mode": sorted(set(ffs_modes)),
        "dicom_indices": {"train": train_idx.tolist(), "test": test_idx.tolist()},
        "shifts": {},
    }
    for split, idx in (("train", train_idx), ("test", test_idx)):
        payload["shifts"][split] = [
            {
                "angular": float(ang_shift[v]),
                "axial": float(axial_shift[v]),
                "radial": float(radial_shift[v]),
                "ffs_mode": ffs_modes[v],
            }
            for v in idx
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
