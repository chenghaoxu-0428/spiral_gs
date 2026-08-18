"""Generate ideal projections on an existing real dataset's exact cameras."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from norm_pipeline import _geometry, fdk_point_cloud


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with (args.source / "meta_data.json").open(encoding="utf-8") as handle:
        source_meta = json.load(handle)
    scanner = copy.deepcopy(source_meta["scanner"])
    volume = np.load(args.source / source_meta["vol"]).astype(np.float32)
    if volume.shape != tuple(scanner["nVoxel"]) or not np.isfinite(volume).all():
        raise ValueError("Source volume shape/finiteness does not match scanner metadata")

    import tigre

    args.output.mkdir(parents=True)
    np.save(args.output / "vol_gt.npy", volume)
    meta = copy.deepcopy(source_meta)
    meta["dataset_type"] = "syn"
    meta["init"] = f"init_{args.output.name}.npy"
    meta["preprocess"] = {
        "matched_syn_source": str(args.source),
        "projection_model": "tigre.Ax",
        "exact_source_camera_splits": True,
    }

    generated = {}
    for split in ("train", "test"):
        frames = source_meta[f"proj_{split}"]
        angles = np.asarray([frame["angle"] for frame in frames], dtype=np.float32)
        z = np.asarray([frame.get("z_shift", 0.0) for frame in frames], dtype=np.float32)
        projs = tigre.Ax(
            np.transpose(volume, (2, 1, 0)).copy(),
            _geometry(scanner, z),
            np.mod(angles, 2 * np.pi),
        )[:, ::-1, :].astype(np.float32)

        # Invert dataset_readers.py's coord_left input transform so this
        # diagnostic exercises the identical real-data camera/loading path.
        stored = projs
        if scanner.get("coord_left", False):
            stored = stored[:, :, ::-1] / float(
                scanner.get("coord_left_projection_scale", 7.0)
            )

        folder = args.output / f"proj_{split}"
        folder.mkdir()
        meta[f"proj_{split}"] = []
        for i, (frame, proj) in enumerate(zip(frames, stored)):
            rel = Path(f"proj_{split}/proj_{split}_{i:04d}.npy")
            np.save(args.output / rel, proj)
            meta[f"proj_{split}"].append({
                "file_path": rel.as_posix(),
                "angle": float(frame["angle"]),
                "z_shift": float(frame.get("z_shift", 0.0)),
            })
        generated[split] = (stored, angles, z)

    with (args.output / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    train_projs, train_angles, train_z = generated["train"]
    fdk_point_cloud(
        train_projs,
        train_angles,
        train_z,
        scanner,
        args.output / meta["init"],
        n_points=50000,
        threshold="auto",
        density_rescale=0.15,
        seed=0,
    )


if __name__ == "__main__":
    main()
