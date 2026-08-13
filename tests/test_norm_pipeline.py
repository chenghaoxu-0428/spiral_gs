from pathlib import Path

import numpy as np

from data_preprocess import norm_pipeline
from data_preprocess.norm_pipeline import dataset_output_path


def test_circular_dataset_path():
    cfg = {"dataset_type": "syn", "organ": "aorta", "model": "r2gs", "n_train": 100}
    assert dataset_output_path(Path("data"), cfg, "circular") == Path(
        "data/syn/aorta/circular/ntrain100/r2gs"
    )


def test_fdk_initialization_uses_training_views(tmp_path, monkeypatch):
    captured = {}

    def capture(projs, angles, z, *_args):
        captured["projs"], captured["angles"], captured["z"] = projs, angles, z

    monkeypatch.setattr(norm_pipeline, "fdk_point_cloud", capture)
    projs = np.arange(6, dtype=np.float32)[:, None, None]
    angles = np.arange(6, dtype=np.float32)
    z = angles + 10
    scanner = {
        "nDetector": [1, 1], "sDetector": [1, 1],
        "nVoxel": [1, 1, 1], "sVoxel": [1, 1, 1], "offOrigin": [0, 0, 0],
    }
    cfg = {
        "n_train": 3, "n_test": 2, "seed": 0, "dataset_type": "syn",
        "init": {"n_points": 1, "density_rescale": 1},
    }

    norm_pipeline.write_dataset(
        tmp_path / "sample", "spiral", projs, angles, z, scanner,
        np.zeros((1, 1, 1), dtype=np.float32), cfg, {},
    )

    expected = np.array([0, 2, 5])
    np.testing.assert_array_equal(captured["projs"], projs[expected])
    np.testing.assert_array_equal(captured["angles"], angles[expected])
    np.testing.assert_array_equal(captured["z"], z[expected])


def test_intensity_initialization_normalizes_density_and_scene_coordinates():
    from fact_gs.r2_gaussian.utils.ct_utils import (
        normalize_fdk_volume,
        sample_intensity_volume,
    )

    volume = normalize_fdk_volume(np.arange(27, dtype=np.float32).reshape(3, 3, 3))
    scanner = {
        "dVoxel": [0.5, 0.5, 0.5],
        "sVoxel": [1.5, 1.5, 1.5],
        "offOrigin": [0, 0, 0],
    }
    xyz, density = sample_intensity_volume(volume, 0.05, 10, scanner, 0.15, seed=0)

    assert np.all(xyz >= -0.75) and np.all(xyz <= 0.75)
    assert np.all(density > 0.05 * 0.15) and np.all(density <= 0.15)

