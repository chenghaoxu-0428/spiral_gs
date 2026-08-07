#!/usr/bin/env python3
"""Extract spiral CT projections and geometry from Siemens DICOM CT-PD data.

Python port of ``dicom_spiral_process.m``.  The output layout and metadata are
kept compatible with the downstream r2_gaussian pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from scipy.io import savemat


# Exact tags taken from the MATLAB dictionary shipped with the CT-PD manual.
PRIVATE_TAGS = {
    "DetectorElementTransverseSpacing": Tag(0x7029, 0x1002),
    "DetectorElementAxialSpacing": Tag(0x7029, 0x1006),
    "DetectorFocalCenterAngularPosition": Tag(0x7031, 0x1001),
    "DetectorFocalCenterAxialPosition": Tag(0x7031, 0x1002),
    "DetectorFocalCenterRadialDistance": Tag(0x7031, 0x1003),
    "ConstantRadialDistance": Tag(0x7031, 0x1031),
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def _resolve_input_dir(path: Path, name_fragment: str, label: str) -> Path:
    """Resolve a CLI path, with project-data auto-discovery as a fallback."""
    if path.is_absolute() and path.is_dir():
        return path
    if not path.is_absolute():
        for base in (Path.cwd(), SCRIPT_DIR):
            candidate = (base / path).resolve()
            if candidate.is_dir():
                return candidate

    data_root = PROJECT_DIR / "data"
    fragment = name_fragment.casefold()
    matches = sorted(
        candidate
        for candidate in data_root.rglob("*")
        if candidate.is_dir() and fragment in candidate.name.casefold()
    ) if data_root.is_dir() else []
    if len(matches) == 1:
        print(f"Auto-detected {label}: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        choices = "\n  ".join(str(item) for item in matches)
        raise FileNotFoundError(
            f"Configured {label} directory does not exist: {path}\n"
            f"Multiple candidates were found; select one with --{label}:\n  {choices}"
        )
    raise FileNotFoundError(
        f"Configured {label} directory does not exist: {path}\n"
        f"No directory containing {name_fragment!r} was found under {data_root}."
    )


def _value(ds: Dataset, name: str, default: Any = None, *, required: bool = False) -> Any:
    """Read a standard keyword or one of the CT-PD private tags."""
    value = getattr(ds, name, None)
    if value is None and name in PRIVATE_TAGS:
        element = ds.get(PRIVATE_TAGS[name])
        value = None if element is None else element.value
        # With an implicit-VR transfer syntax pydicom cannot infer the VR of
        # vendor-private tags and returns their raw bytes.  dict.txt declares
        # all six tags used here as FL (32-bit IEEE floating point).
        if isinstance(value, (bytes, bytearray)):
            if len(value) == 0 or len(value) % 4 != 0:
                raise ValueError(
                    f"Private DICOM field {name!r} has invalid FL byte length "
                    f"{len(value)}"
                )
            endian = "<" if getattr(ds, "is_little_endian", True) else ">"
            decoded = struct.unpack(f"{endian}{len(value) // 4}f", value)
            value = decoded[0] if len(decoded) == 1 else decoded
    if value is None:
        if required:
            tag = PRIVATE_TAGS.get(name, name)
            raise KeyError(f"Required DICOM field {name!r} ({tag}) is missing")
        return default
    return value


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        raise ValueError(f"DICOM field {name!r} is empty")
    return float(array[0])


def _dicom_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.dcm") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .dcm files were found under: {root}")
    return files


def _read(path: Path, *, pixels: bool = False) -> Dataset:
    return pydicom.dcmread(path, stop_before_pixels=not pixels)


def process(
    dicom_root: Path,
    image_root: Path,
    save_root: Path,
    *,
    bake_coord_left: bool = True,
    transpose_image: bool = True,
    negate_source_angle: bool = False,
    intensity_scale: float = 6.0,
    apply_intensity_bake: bool = False,
    flip_detector_columns: bool = False,
) -> Path:
    """Convert projection DICOM files and return the generated JSON path."""
    slice_files = _dicom_files(image_root)
    first_slice = _read(slice_files[0])
    slice_thickness = _scalar(_value(first_slice, "SliceThickness", 0.0), "SliceThickness")
    rows = int(_scalar(_value(first_slice, "Rows", 0), "Rows"))
    columns = int(_scalar(_value(first_slice, "Columns", 0), "Columns"))
    pixel_spacing = np.asarray(
        _value(first_slice, "PixelSpacing", required=True), dtype=np.float64
    ).reshape(-1)
    if pixel_spacing.size < 2:
        raise ValueError("PixelSpacing must contain row and column spacing")

    slice_z = np.asarray(
        [_scalar(_value(_read(path), "SliceLocation", 0.0), "SliceLocation") for path in slice_files],
        dtype=np.float64,
    )
    svo = [
        float(rows * pixel_spacing[0]),
        float(columns * pixel_spacing[1]),
        float(np.max(slice_z) - np.min(slice_z) + slice_thickness),
    ]

    projection_files = _dicom_files(dicom_root)
    indexed: list[tuple[float, Path, Dataset]] = []
    for fallback_index, path in enumerate(projection_files, start=1):
        info = _read(path)
        instance = _scalar(_value(info, "InstanceNumber", fallback_index), "InstanceNumber")
        indexed.append((instance, path, info))
    indexed.sort(key=lambda item: (item[0], str(item[1])))

    projection_dir = save_root / "proj"
    projection_dir.mkdir(parents=True, exist_ok=True)
    projections: list[dict[str, Any]] = []
    scanner: dict[str, Any] | None = None

    for index, (_, path, header) in enumerate(indexed, start=1):
        ds = _read(path, pixels=True)
        slope = _scalar(_value(ds, "RescaleSlope", 1.0), "RescaleSlope")
        intercept = _scalar(_value(ds, "RescaleIntercept", 0.0), "RescaleIntercept")
        image = ds.pixel_array.astype(np.float64) * slope + intercept
        raw_angle = _scalar(
            _value(header, "DetectorFocalCenterAngularPosition", required=True),
            "DetectorFocalCenterAngularPosition",
        )

        if bake_coord_left:
            if transpose_image:
                image = image.T
            if flip_detector_columns:
                image = np.fliplr(image)
            if apply_intensity_bake:
                image *= intensity_scale
            angle = -raw_angle + 2.0 * math.pi if negate_source_angle else raw_angle
        else:
            angle = raw_angle

        file_stem = f"{index:04d}"
        # MATLAB save(..., '-v7') produces a level-5 MAT file, as scipy does here.
        savemat(projection_dir / f"{file_stem}.mat", {"img": image}, format="5")
        projections.append(
            {
                "file_stem": file_stem,
                "original_file": path.name,
                "angle_rad": float(angle),
                "table_z_mm": _scalar(
                    _value(header, "DetectorFocalCenterAxialPosition", required=True),
                    "DetectorFocalCenterAxialPosition",
                ),
            }
        )

        if scanner is None:
            transverse = _value(header, "DetectorElementTransverseSpacing")
            axial = _value(header, "DetectorElementAxialSpacing")
            if transverse is not None and axial is not None:
                spacing = np.asarray(
                    [_scalar(transverse, "DetectorElementTransverseSpacing"),
                     _scalar(axial, "DetectorElementAxialSpacing")],
                    dtype=np.float64,
                )
            elif transverse is not None:
                spacing = np.asarray(transverse, dtype=np.float64).reshape(-1)
            elif axial is not None:
                spacing = np.asarray(axial, dtype=np.float64).reshape(-1)
            else:
                fallback_spacing = _value(header, "PixelSpacing")
                if fallback_spacing is None:
                    raise KeyError("Detector spacing and PixelSpacing are both missing")
                spacing = np.asarray(fallback_spacing, dtype=np.float64).reshape(-1)
            if bake_coord_left and transpose_image and spacing.size >= 2:
                spacing = spacing[[1, 0]]

            scanner = {
                "DSO_mm": _scalar(
                    _value(header, "DetectorFocalCenterRadialDistance", required=True),
                    "DetectorFocalCenterRadialDistance",
                ),
                "DSD_mm": _scalar(
                    _value(header, "ConstantRadialDistance", required=True),
                    "ConstantRadialDistance",
                ),
                "detector_pixel_size_mm": spacing.astype(float).tolist(),
                "detector_pixels": list(image.shape),
                "mode": "cone",
                "r2_gaussian_coord_left_baked": bool(bake_coord_left),
            }
            if bake_coord_left:
                scanner["r2_gaussian_coord_left"] = False
                scanner["r2_preprocess"] = {
                    "negate_source_angle_plus_2pi": bool(negate_source_angle),
                    "intensity_scale": float(intensity_scale if apply_intensity_bake else 1.0),
                    "fliplr_detector_columns": bool(flip_detector_columns),
                }

        if index % 50 == 0 or index == len(indexed):
            print(f"Saved {index}/{len(indexed)} projections")

    geometry = {
        "scanner": scanner,
        "projections": projections,
        "svo": svo,
        "notes": {
            "generated_with": Path(__file__).name,
            "dictionary": "dict.txt (private tag numbers embedded in this script)",
            "r2_gaussian_coord_left_baked": bool(bake_coord_left),
        },
    }
    json_path = save_root / "scanner_geometry.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(geometry, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Scanner geometry saved to {json_path}")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path,
                        default=Path("1.000000-Full dose projections-24362"))
    parser.add_argument("--image-root", type=Path,
                        default=Path("1.000000-Full Dose Images-63186"))
    parser.add_argument("--save-root", type=Path, default=Path("SPIRAL_processed_t"))
    parser.add_argument("--no-bake-coord-left", action="store_true")
    parser.add_argument("--no-transpose", action="store_true")
    parser.add_argument("--negate-source-angle", action="store_true")
    parser.add_argument("--intensity-scale", type=float, default=6.0)
    parser.add_argument("--apply-intensity-bake", action="store_true")
    parser.add_argument("--flip-detector-columns", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dicom_root = _resolve_input_dir(args.dicom_root, "full dose projections", "dicom-root")
    image_root = _resolve_input_dir(args.image_root, "full dose images", "image-root")
    save_root = args.save_root if args.save_root.is_absolute() else SCRIPT_DIR / args.save_root
    process(
        dicom_root,
        image_root,
        save_root,
        bake_coord_left=not args.no_bake_coord_left,
        transpose_image=not args.no_transpose,
        negate_source_angle=args.negate_source_angle,
        intensity_scale=args.intensity_scale,
        apply_intensity_bake=args.apply_intensity_bake,
        flip_detector_columns=args.flip_detector_columns,
    )


if __name__ == "__main__":
    main()
