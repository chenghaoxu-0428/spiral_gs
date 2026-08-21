from pathlib import Path

import numpy as np

from data_preprocess import norm_pipeline
from data_preprocess.norm_pipeline import dataset_output_path


def test_circular_dataset_path():
    cfg = {"dataset_type": "syn", "organ": "aorta", "model": "r2gs", "n_train": 100}
    assert dataset_output_path(Path("data"), cfg, "circular") == Path(
        "data/syn/aorta/circular/ntrain100/r2gs"
    )


def test_cylindrical_detector_is_rebinned_to_tangent_plane():
    source = np.tile(np.arange(5, dtype=np.float32), (3, 1))[None]
    flat, size = norm_pipeline.flatten_cylindrical_detector(source, 10.0, [1.0, 2.0])

    assert flat.shape == source.shape
    np.testing.assert_allclose(flat[0, 1], [0, 0.938, 2, 3.062, 4], atol=0.01)
    np.testing.assert_allclose(size, [3.0, 20 * np.tan(0.5)], rtol=1e-6)


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


def test_preprocessed_coord_left_init_reflects_x(tmp_path, monkeypatch):
    from fact_gs.r2_gaussian.utils import ct_utils

    monkeypatch.setattr(ct_utils, "recon_volume", lambda *_args, **_kwargs: np.ones((1, 1, 1)))
    monkeypatch.setattr(
        ct_utils, "sample_intensity_volume",
        lambda *_args, **_kwargs: (np.array([[0.25, 0.0, 0.0]]), np.ones(1)),
    )
    scanner = {
        "nDetector": [1, 1], "sDetector": [1, 1], "dDetector": [1, 1],
        "nVoxel": [1, 1, 1], "sVoxel": [2, 2, 2], "dVoxel": [2, 2, 2],
        "offOrigin": [0, 0, 0], "offDetector": [0, 0], "DSD": 2, "DSO": 1,
        "mode": "cone", "coord_left": True,
    }

    output = tmp_path / "init.npy"
    norm_pipeline.fdk_point_cloud(
        np.ones((1, 1, 1)), np.zeros(1), np.zeros(1), scanner,
        output, 1, "auto", 1, 0,
    )

    np.testing.assert_allclose(np.load(output)[0, :3], [-0.25, 0, 0])


def test_detector_central_element_maps_to_analytic_u_offset():
    off, info = norm_pipeline.detector_off_from_central_element(
        [369.625, 32.5], [64, 736], 4, [0.2189445495605469, 0.2571678638458252],
    )
    np.testing.assert_allclose(info["u_offset_px"], -0.28125)
    np.testing.assert_allclose(info["v_offset_px"], 0.0)
    np.testing.assert_allclose(off[0], -0.28125 * 0.2571678638458252)
    np.testing.assert_allclose(off[1], 0.0)


def test_missing_detector_central_element_leaves_offdetector_zero():
    off, info = norm_pipeline.detector_off_from_central_element(
        None, [64, 736], 4, [1.0, 1.0],
    )
    assert off == [0.0, 0.0]
    assert info == {}


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
