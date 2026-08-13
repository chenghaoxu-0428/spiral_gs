"""Save the training split's normalized helical FDK volume."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fact_gs.r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", default="vol_fdk_train.npy")
    args = parser.parse_args()

    metadata = json.loads((args.dataset / "meta_data.json").read_text())
    train = metadata["proj_train"]
    projections = np.stack([np.load(args.dataset / item["file_path"]) for item in train])
    angles = np.asarray([item["angle"] for item in train])
    z_shifts = np.asarray([item.get("z_shift", 0.0) for item in train])
    volume = recon_volume(
        projections,
        angles,
        get_geometry_tigre(metadata["scanner"]),
        "fdk",
        z_shifts=z_shifts,
    )
    volume = np.clip(volume, 0.0, None)
    volume /= np.percentile(volume, 99.5) + 1e-12
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)
    assert volume.shape == tuple(metadata["scanner"]["nVoxel"])
    assert np.isfinite(volume).all()
    output = args.dataset / args.output
    np.save(output, volume)
    print(f"Saved {output}: shape={volume.shape}, range=[{volume.min()}, {volume.max()}]")


if __name__ == "__main__":
    main()
