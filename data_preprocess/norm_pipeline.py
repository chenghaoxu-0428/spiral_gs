"""Unified DICOM -> spiral/stitch R2-Gaussian dataset pipeline.

The module deliberately imports TIGRE lazily: ``--validate-only`` and the
pure DICOM/stitch helpers can therefore be used on machines without CUDA.
Lengths in scanner YAML and metadata are expressed in the R2-Gaussian scene
unit; raw DICOM scanner lengths are converted by ``mm / 1000 * object_scale``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import yaml
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parents[1]))


PRIVATE_TAGS = {
    "detector_rows": (0x7029, 0x1010),
    "detector_cols": (0x7029, 0x1011),
    "spacing_u": (0x7029, 0x1002),
    "spacing_v": (0x7029, 0x1006),
    "angle": (0x7031, 0x1001),
    "table_z": (0x7031, 0x1002),
    "dso": (0x7031, 0x1003),
    "dsd": (0x7031, 0x1031),
    "samples_per_rotation": (0x7033, 0x1013),
}

PRIVATE_VRS = {
    PRIVATE_TAGS["detector_rows"]: "US",
    PRIVATE_TAGS["detector_cols"]: "US",
    PRIVATE_TAGS["spacing_u"]: "FL",
    PRIVATE_TAGS["spacing_v"]: "FL",
    PRIVATE_TAGS["angle"]: "FL",
    PRIVATE_TAGS["table_z"]: "FL",
    PRIVATE_TAGS["dso"]: "FL",
    PRIVATE_TAGS["dsd"]: "FL",
    PRIVATE_TAGS["samples_per_rotation"]: "US",
}


def _value(ds: pydicom.Dataset, keyword: str, tag=None, default=None):
    value = getattr(ds, keyword, None)
    if value is None and tag is not None and tag in ds:
        value = ds[tag].value
    # With an implicit-VR transfer syntax pydicom exposes unknown private
    # elements as bytes. Their VRs are declared in data_preprocess/dict.txt.
    if isinstance(value, bytes) and tag in PRIVATE_VRS:
        byte_order = "<" if ds.is_little_endian is not False else ">"
        fmt = {"FL": "f", "US": "H"}[PRIVATE_VRS[tag]]
        item_size = struct.calcsize(fmt)
        if len(value) % item_size:
            raise ValueError(f"Invalid byte length for private DICOM tag {tag}: {len(value)}")
        decoded = struct.unpack(byte_order + fmt * (len(value) // item_size), value)
        value = decoded[0] if len(decoded) == 1 else decoded
    return default if value is None else value


def _dicom_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    readable = []
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if "SOPClassUID" in ds or "Rows" in ds:
                readable.append(path)
        except Exception:
            continue
    if not readable:
        raise FileNotFoundError(f"No readable DICOM files under {root}")
    return readable


def load_gt_dicom(
    root: Path, target_shape: tuple[int, int, int], xy_invert: bool = False
) -> tuple[np.ndarray, dict]:
    """Restore DICOM slices with process_raw_data.py-compatible orientation."""
    records = []
    for path in _dicom_files(root):
        ds = pydicom.dcmread(path)
        pos = _value(ds, "ImagePositionPatient")
        z = float(pos[2]) if pos is not None and len(pos) >= 3 else float(
            _value(ds, "SliceLocation", default=_value(ds, "InstanceNumber", default=0))
        )
        slope = float(_value(ds, "RescaleSlope", default=1.0))
        intercept = float(_value(ds, "RescaleIntercept", default=0.0))
        instance = int(_value(ds, "InstanceNumber", default=len(records)))
        # Keep process_raw_data.py semantics: stable acquisition/file order,
        # followed by its explicit z-axis reversal below.
        records.append(
            (instance, path.name, z, np.asarray(ds.pixel_array, dtype=float) * slope + intercept, ds)
        )
    records.sort(key=lambda x: (x[0], x[1]))
    volume = np.stack([r[3] for r in records], axis=-1)
    volume = volume[:, :, ::-1]
    volume = volume.clip(-1000.0, 2000.0)
    hu_min = float(volume.min())
    hu_max = float(volume.max())
    if hu_max <= hu_min:
        raise ValueError(
            f"Cannot normalize constant DICOM volume: min=max={hu_min} under {root}"
        )
    volume = (volume - hu_min) / (hu_max - hu_min)
    ds0 = records[0][4]
    spacing_xy = np.asarray(_value(ds0, "PixelSpacing", default=[1.0, 1.0]), dtype=float)
    if len(records) > 1:
        spacing_z = float(np.median(np.abs(np.diff([r[2] for r in records]))))
    else:
        spacing_z = float(_value(ds0, "SliceThickness", default=1.0))
    zoom = np.asarray(target_shape, dtype=float) / np.asarray(volume.shape)
    # scipy.ndimage.zoom defaults to cubic interpolation (order=3) in the
    # original process_raw_data.py; keep that behavior exactly.
    volume = ndimage.zoom(volume, zoom, order=3, mode="nearest").astype(np.float32)
    volume = volume.clip(0.0, 1.0)
    if xy_invert:
        volume = volume[::-1, ::-1, :].copy()
    return volume, {
        "source_shape": list(records[0][3].shape) + [len(records)],
        "spacing_mm": [float(spacing_xy[0]), float(spacing_xy[1]), spacing_z],
        "hu_window": [-1000.0, 2000.0],
        "hu_range_after_window": [hu_min, hu_max],
        "normalized_range": [float(volume.min()), float(volume.max())],
    }


def load_gt_source(
    source: Path, target_shape: tuple[int, int, int], xy_invert: bool = False
) -> tuple[np.ndarray, dict]:
    if source.suffix.lower() != ".npy":
        return load_gt_dicom(source, target_shape, xy_invert)
    volume = np.load(source).astype(np.float32)
    if volume.shape != target_shape:
        zoom = np.asarray(target_shape, dtype=float) / np.asarray(volume.shape)
        volume = ndimage.zoom(volume, zoom, order=3, mode="nearest").astype(np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Ground-truth volume contains NaN or Inf: {source}")
    if xy_invert:
        volume = volume[::-1, ::-1, :].copy()
    return volume, {
        "source_type": "npy",
        "source_shape": list(np.load(source, mmap_mode="r").shape),
        "normalized_range": [float(volume.min()), float(volume.max())],
    }


def load_real_projections(root: Path, object_scale: float, proj_rescale: float, proj_subsample: int = 1):
    """Python equivalent of dicom_spiral_process.m (no MAT intermediary).

    Parameters
    ----------
    proj_subsample : int
        Pixel downsampling factor (default 1 = no downsampling).
        When > 1, each projection is resized by 1/proj_subsample via cv2.resize
        and detector spacing is scaled up by the same factor to keep the
        physical detector size invariant (matching generate_data_usr.py).
    """
    records = []
    for path in _dicom_files(root):
        ds = pydicom.dcmread(path)
        instance = int(_value(ds, "InstanceNumber", default=len(records)))
        angle = float(_value(ds, "DetectorFocalCenterAngularPosition", PRIVATE_TAGS["angle"]))
        z_mm = float(_value(ds, "DetectorFocalCenterAxialPosition", PRIVATE_TAGS["table_z"]))
        slope = float(_value(ds, "RescaleSlope", default=1.0))
        intercept = float(_value(ds, "RescaleIntercept", default=0.0))
        image = np.asarray(ds.pixel_array, dtype=np.float32) * slope + intercept
        # dicom_spiral_process.m transposes detector data before export.
        image = image.T.astype(np.float32) / float(proj_rescale) * float(object_scale)
        image[image < 0] = 0
        # Projection pixel downsampling (reference: generate_data_usr.py).
        if proj_subsample != 1:
            h_ori, w_ori = image.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            image = cv2.resize(image, (w_new, h_new))
        records.append((instance, angle, z_mm, image, ds, path.name))
    records.sort(key=lambda x: x[0])
    first = records[0][4]
    z_scene = np.asarray([x[2] for x in records], dtype=np.float32) / 1000 * object_scale
    # Preserve the established real-data convention.
    flip = len(z_scene) > 1 and float(np.median(np.diff(z_scene))) > 0
    if flip:
        z_scene *= -1
    projs = np.stack([np.flip(x[3], axis=0) if flip else x[3] for x in records])
    spacing_mm = [
        float(_value(first, "DetectorElementAxialSpacing", PRIVATE_TAGS["spacing_v"])),
        float(_value(first, "DetectorElementTransverseSpacing", PRIVATE_TAGS["spacing_u"])),
    ]
    # Scale detector spacing to preserve physical detector size after
    # downsampling (matching generate_data_usr.py: spacing *= proj_subsample).
    spacing_mm = [s * proj_subsample for s in spacing_mm]
    scanner = {
        "mode": "cone",
        "DSO": float(_value(first, "DetectorFocalCenterRadialDistance", PRIVATE_TAGS["dso"])) / 1000 * object_scale,
        "DSD": float(_value(first, "ConstantRadialDistance", PRIVATE_TAGS["dsd"])) / 1000 * object_scale,
        "nDetector": list(projs.shape[1:]),
        "sDetector": (np.asarray(projs.shape[1:]) * np.asarray(spacing_mm) / 1000 * object_scale).tolist(),
        "samples_per_rotation": int(_value(first, "NumberofSourceAngularSteps", PRIVATE_TAGS["samples_per_rotation"])),
        "pitch": float(_value(first, "SpiralPitchFactor", default=1.0)),
    }
    return projs, np.asarray([x[1] for x in records]), z_scene, scanner, flip


def _geometry(scanner: dict, z_shifts=None):
    import tigre

    geo = tigre.geometry(mode=scanner["mode"])
    geo.DSD, geo.DSO = scanner["DSD"], scanner["DSO"]
    geo.nDetector = np.asarray(scanner["nDetector"])
    geo.sDetector = np.asarray(scanner["sDetector"])
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.asarray(scanner["nVoxel"])[::-1]
    geo.sVoxel = np.asarray(scanner["sVoxel"])[::-1]
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    base = np.asarray(scanner["offOrigin"], dtype=np.float64)[::-1]
    if z_shifts is None:
        geo.offOrigin = base
    else:
        geo.offOrigin = np.repeat(base[None, :], len(z_shifts), axis=0)
        geo.offOrigin[:, 0] -= np.asarray(z_shifts)
    off_det = scanner.get("offDetector", [0, 0])
    geo.offDetector = np.asarray([off_det[1], off_det[0], 0])
    geo.accuracy = scanner.get("accuracy", 0.5)
    geo.filter = scanner.get("filter")
    return geo


def synthesize_projections(volume, scanner, cfg):
    import tigre
    from tigre.utilities import CTnoise

    spiral = cfg["spiral"]
    spr = int(spiral["sample_per_rotation"])
    if cfg.get("trajectory", "spiral") == "circular":
        count = int(cfg["n_train"]) + int(cfg["n_test"])
        angles = np.deg2rad(float(spiral.get("angle_start", 0))) + np.arange(count) * 2 * np.pi / count
        z = np.zeros(count, dtype=np.float32)
        trajectory = {"z_step": 0.0, "rotations": 1.0}
    else:
        dsd, dso = float(scanner["DSD"]), float(scanner["DSO"])
        collimation = 2 * dso * math.tan(float(scanner["sDetector"][0]) / (2 * dsd))
        dz = float(spiral["pitch"]) * collimation / spr
        z0, z1 = float(spiral["z_start"]), float(spiral["z_end"])
        if dz == 0 or (z1 - z0) * dz < 0:
            raise ValueError("spiral pitch direction does not reach z_end from z_start")
        count = int(math.floor((z1 - z0) / dz + 1e-9)) + 1
        z = np.clip(z0 + np.arange(count, dtype=np.float64) * dz,
                    min(z0, z1), max(z0, z1))
        angles = np.deg2rad(float(spiral.get("angle_start", 0))) + np.arange(count) * 2 * np.pi / spr
        trajectory = {
            "collimation_width": collimation, "z_step": dz,
        }
    geo = _geometry(scanner, z)
    projs = tigre.Ax(np.transpose(volume, (2, 1, 0)).copy(), geo, np.mod(angles, 2 * np.pi))[:, ::-1, :]
    if scanner.get("noise", False):
        projs = CTnoise.add(projs, Poisson=float(scanner["possion_noise"]), Gaussian=np.asarray(scanner["gaussian_noise"]))
        projs[projs < 0] = 0
    return projs.astype(np.float32), angles, z, trajectory


def stitch_projections(projs, angles, z, samples_per_rotation, detector_row_size):
    """CAT.m-style physical z placement; returns one tall projection per angle slot."""
    order = np.argsort(z, kind="stable")
    projs, angles, z = projs[order], angles[order], z[order]
    spr = int(samples_per_rotation)
    starts = np.rint((z - z[0]) / detector_row_size).astype(int)
    height = int(starts.max() + projs.shape[1])
    canvas = np.zeros((spr, height, projs.shape[2]), dtype=np.float32)
    slot_angles = np.zeros(spr, dtype=float)
    for i, (proj, start) in enumerate(zip(projs, starts)):
        slot = i % spr
        if i < spr:
            slot_angles[slot] = angles[i]
        canvas[slot, start : start + proj.shape[0]] = proj
    coverage = np.zeros(height, dtype=int)
    for start in starts:
        coverage[start : start + projs.shape[1]] += 1
    # Keep the union. Missing rays remain explicit zeros, matching CAT.m.
    nonzero = np.flatnonzero(coverage)
    lo, hi = int(nonzero[0]), int(nonzero[-1] + 1)
    return canvas[:, lo:hi], slot_angles, {"crop_rows": [lo, hi], "raw_views": len(projs)}


def _indices(n: int, count: int) -> np.ndarray:
    if count < 1 or count > n:
        raise ValueError(f"requested {count} views from {n} available views")
    return np.unique(np.rint(np.linspace(0, n - 1, count)).astype(int))


def _split_indices(n, n_train, n_test, seed):
    train = _indices(n, n_train)
    remaining = np.setdiff1d(np.arange(n), train)
    if n_test > len(remaining):
        raise ValueError(f"n_test={n_test}, but only {len(remaining)} non-training views remain")
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(remaining, n_test, replace=False))
    return train, test


def fdk_point_cloud(projs, angles, z, scanner, output, n_points, threshold, density_rescale, seed):
    from fact_gs.r2_gaussian.utils.ct_utils import (
        normalize_fdk_volume, recon_volume, sample_intensity_volume,
    )

    scene_scale = 2 / max(scanner["sVoxel"])
    scaled_scanner = copy.deepcopy(scanner)
    for key in ("dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin",
                "offDetector", "DSD", "DSO"):
        scaled_scanner[key] = (np.asarray(scaled_scanner[key]) * scene_scale).tolist()
    scaled_projs = projs * scene_scale
    scaled_angles = angles
    if scaled_scanner.get("coord_left", False):
        scaled_projs = scaled_projs[:, :, ::-1].copy()
        scaled_projs *= float(scaled_scanner.get("coord_left_projection_scale", 7.0))
        scaled_angles = -angles

    volume = recon_volume(
        scaled_projs, scaled_angles, _geometry(scaled_scanner),
        recon_method="fdk", z_shifts=z * scene_scale,
    )
    xyz, density = sample_intensity_volume(
        normalize_fdk_volume(volume),
        0.05 if threshold is None or str(threshold).lower() == "auto" else float(threshold),
        n_points, scaled_scanner, density_rescale, seed=seed,
    )
    np.save(output, np.concatenate([xyz, density[:, None]], axis=1).astype(np.float32))


def write_dataset(root, kind, projs, angles, z, scanner, volume, cfg, provenance):
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "vol_gt.npy", volume.astype(np.float32))
    train, test = _split_indices(len(projs), int(cfg["n_train"]), int(cfg["n_test"]), int(cfg["seed"]))
    payload = {}
    for split, ids in (("train", train), ("test", test)):
        folder = root / f"proj_{split}"
        folder.mkdir(exist_ok=True)
        payload[f"proj_{split}"] = []
        for j, idx in enumerate(ids):
            rel = Path(f"proj_{split}") / f"proj_{split}_{j:04d}.npy"
            np.save(root / rel, projs[idx].astype(np.float32))
            payload[f"proj_{split}"].append({"file_path": rel.as_posix(), "angle": float(angles[idx]), "z_shift": float(z[idx])})
    scanner = copy.deepcopy(scanner)
    scanner["dDetector"] = (np.asarray(scanner["sDetector"]) / np.asarray(scanner["nDetector"])).tolist()
    scanner["dVoxel"] = (np.asarray(scanner["sVoxel"]) / np.asarray(scanner["nVoxel"])).tolist()
    init_name = f"init_{root.name}.npy"
    metadata = {
        "schema_version": 1, "dataset_type": cfg["dataset_type"], "projection_type": kind,
        "scanner": scanner, "vol": "vol_gt.npy", "init": init_name,
        "bbox": (np.asarray([[-.5], [.5]]) * np.asarray(scanner["sVoxel"]) + np.asarray(scanner["offOrigin"])).tolist(),
        **payload, "preprocess": provenance,
    }
    with (root / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    init_path = root / init_name
    fdk_point_cloud(projs[train], angles[train], z[train], scanner, init_path, int(cfg["init"]["n_points"]), cfg["init"].get("density_threshold", "auto"), float(cfg["init"]["density_rescale"]), int(cfg["seed"]))


def validate_config(cfg):
    required = ["dataset_type", "organ", "model", "raw_gt", "output_root", "n_train", "scanner"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if cfg["dataset_type"] not in {"real", "syn"}:
        raise ValueError("dataset_type must be real or syn")
    if cfg["dataset_type"] == "real" and not cfg.get("raw_proj"):
        raise ValueError("real dataset requires raw_proj")
    if cfg.get("trajectory", "spiral") not in {"spiral", "circular"}:
        raise ValueError("trajectory must be spiral or circular")
    cfg.setdefault("n_test", math.ceil(1.5 * int(cfg["n_train"])))
    stitch = cfg.get("stitch", {})
    if stitch.get("enabled", False) and "n_train" not in stitch:
        raise ValueError("stitch.enabled=true requires stitch.n_train")


def dataset_output_path(output_root: Path, cfg: dict, kind: str, n_train=None) -> Path:
    """Return the hierarchical dataset directory used by the training workflow."""
    if kind not in {"spiral", "circular", "stitch"}:
        raise ValueError(f"unsupported projection type: {kind}")
    count = int(cfg["n_train"] if n_train is None else n_train)
    return (
        Path(output_root)
        / str(cfg["dataset_type"])
        / str(cfg["organ"])
        / kind
        / f"ntrain{count}"
        / str(cfg["model"])
    )


def _set_real_volume_bounds(scanner: dict, z_shifts: np.ndarray, cfg: dict) -> dict:
    """Port generate_data_usr.py's default real helical volume-bound logic."""
    real_cfg = cfg.get("real", {})
    if not real_cfg.get("auto_svoxel_from_zshift", True) or not len(z_shifts):
        return {"enabled": False}
    z_lower = float(np.min(z_shifts))
    z_upper = float(np.max(z_shifts))
    z_span = max(z_upper - z_lower, float(real_cfg.get("min_svoxel_span", 1e-6)))
    z_center = (z_lower + z_upper) / 2
    if real_cfg.get("equal_xyz_span", True):
        scanner["sVoxel"] = [z_span, z_span, z_span]
    else:
        scanner["sVoxel"][2] = z_span
    scanner["offOrigin"][2] = z_center
    return {
        "enabled": True,
        "z_shift_range": [z_lower, z_upper],
        "z_span": z_span,
        "z_center": z_center,
        "equal_xyz_span": bool(real_cfg.get("equal_xyz_span", True)),
    }


def run(config_path: Path, validate_only=False):
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    if validate_only:
        return
    target = tuple(int(x) for x in cfg["scanner"]["nVoxel"])
    volume, gt_info = load_gt_source(
        Path(cfg["raw_gt"]), target, bool(cfg.get("gt_xy_invert", False))
    )
    scanner = copy.deepcopy(cfg["scanner"])
    if cfg["dataset_type"] == "syn":
        projs, angles, z, trajectory = synthesize_projections(volume, scanner, cfg)
        flip = False
    else:
        proj_subsample = int(cfg.get("proj_subsample", 1))
        projs, angles, z, dicom_scanner, flip = load_real_projections(
            Path(cfg["raw_proj"]), float(cfg["object_scale"]), float(cfg["proj_rescale"]), proj_subsample
        )
        scanner.update(dicom_scanner)
        trajectory = {}
    scanner.setdefault("coord_left", False)
    scanner.setdefault("offOrigin", [0, 0, 0])
    scanner.setdefault("offDetector", [0, 0])
    scanner.setdefault("accuracy", 0.5)
    scanner.setdefault("filter", None)
    real_bounds = (
        _set_real_volume_bounds(scanner, z, cfg)
        if cfg["dataset_type"] == "real"
        else {"enabled": False}
    )
    out = Path(cfg["output_root"])
    provenance = {"config": str(config_path), "gt": gt_info, "trajectory": trajectory, "real_z_convention_flipped": flip, "real_volume_bounds": real_bounds, "proj_subsample": int(cfg.get("proj_subsample", 1))}
    kind = str(cfg.get("trajectory", "spiral"))
    write_dataset(dataset_output_path(out, cfg, kind), kind, projs, angles, z, scanner, volume, cfg, provenance)
    stitch_cfg_raw = cfg.get("stitch", {})
    if stitch_cfg_raw.get("enabled", False):
        spr = int(cfg.get("spiral", {}).get("sample_per_rotation", scanner.get("samples_per_rotation", 0)))
        if spr < 1:
            raise ValueError("sample_per_rotation is required for stitch output")
        drow = float(scanner["sDetector"][0]) / int(scanner["nDetector"][0])
        stitched, stitched_angles, stitch_info = stitch_projections(projs, angles, z, spr, drow)
        stitch_scanner = copy.deepcopy(scanner)
        stitch_scanner["nDetector"] = list(stitched.shape[1:])
        stitch_scanner["sDetector"][0] = stitched.shape[1] * drow
        stitched_z = np.zeros(len(stitched), dtype=np.float32)
        stitch_cfg = copy.deepcopy(cfg)
        stitch_cfg["n_train"] = int(stitch_cfg_raw["n_train"])
        stitch_cfg["n_test"] = int(
            stitch_cfg_raw.get(
                "n_test", math.ceil(1.5 * int(stitch_cfg["n_train"]))
            )
        )
        write_dataset(dataset_output_path(out, stitch_cfg, "stitch"), "stitch", stitched, stitched_angles, stitched_z, stitch_scanner, volume, stitch_cfg, {**provenance, "stitch": stitch_info})


def main():
    parser = argparse.ArgumentParser(description="Generate normalized spiral and stitch datasets")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.validate_only)


if __name__ == "__main__":
    main()
