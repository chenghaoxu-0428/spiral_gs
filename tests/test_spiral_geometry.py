import json
from pathlib import Path

import numpy as np

from fact_gs.r2_gaussian.dataset.dataset_readers import angle2pose
from fact_gs.r2_gaussian.utils.graphics_utils import getProjectionMatrix


def test_angle2pose_preserves_per_view_z_translation():
    pose = angle2pose(2.0, np.pi / 3, z_shift=-0.75)
    np.testing.assert_allclose(
        pose[:3, 3], [1.0, np.sqrt(3.0), -0.75], rtol=1e-6, atol=1e-6
    )


def test_original_spiral_metadata_contains_nonconstant_z_shift():
    metadata_path = Path(
        "/home/chenghaoxu/Documents/r2_gaussian_spiral-data/001/real_dataset/ldct_001/meta_data.json"
    )
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shifts = np.asarray([x.get("z_shift", 0.0) for x in metadata["proj_train"]])
    assert shifts.size > 1
    assert np.ptp(shifts) > 0


def test_zero_shift_is_backward_compatible():
    old_pose = angle2pose(3.0, 0.25)
    explicit_zero_pose = angle2pose(3.0, 0.25, 0.0)
    np.testing.assert_array_equal(old_pose, explicit_zero_pose)


def test_projection_matrix_bakes_offdetector_as_principal_point():
    width = 184
    u_offset_px = -0.28125
    d_u = 0.2571678638458252
    s_u = width * d_u
    P = getProjectionMatrix(
        fovX=0.8,
        fovY=0.2,
        mode=1,
        scanner_cfg={
            "sDetector": [3.5, s_u],
            "offDetector": [u_offset_px * d_u, 0.0],
        },
    )
    # Camera stores P.T, so NDC u-shift is P[0, 2].
    np.testing.assert_allclose(P[0, 2].item(), 2.0 * u_offset_px / width)
    np.testing.assert_allclose(P[1, 2].item(), 0.0)
