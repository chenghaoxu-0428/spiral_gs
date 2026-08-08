import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INITIALIZE_PATH = ROOT / "fact_gs/r2_gaussian/gaussian/initialize.py"


def _load_sample_vol_without_cuda_dependencies():
    """Load the pure NumPy sampler without importing TIGRE/CUDA extensions."""
    tree = ast.parse(INITIALIZE_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sample_vol"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    from scipy.ndimage import sobel

    namespace["sobel"] = sobel
    exec(compile(module, str(INITIALIZE_PATH), "exec"), namespace)
    return namespace["sample_vol"]


def test_uniform_initialization_matches_r2_gaussian_sampling():
    sample_vol = _load_sample_vol_without_cuda_dependencies()
    vol = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6) / 100
    scanner = {
        "offOrigin": [0.1, -0.2, 0.3],
        "dVoxel": [0.5, 0.25, 0.125],
        "sVoxel": [2.0, 1.25, 0.75],
    }
    threshold = 0.2
    count = 12
    valid_indices = np.argwhere(vol > threshold)
    expected_indices = valid_indices[
        np.random.RandomState(0).choice(len(valid_indices), count, replace=False)
    ]

    positions, densities = sample_vol(
        vol, threshold, count, scanner, "intensity", 0.15, seed=0
    )

    expected_positions = (
        expected_indices * np.asarray(scanner["dVoxel"])
        - np.asarray(scanner["sVoxel"]) / 2
        + np.asarray(scanner["offOrigin"])
    )
    np.testing.assert_allclose(positions, expected_positions)
    np.testing.assert_allclose(
        densities,
        vol[tuple(expected_indices.T)] * 0.15,
    )


def test_uniform_initialization_is_repeatable_and_rejects_auto():
    sample_vol = _load_sample_vol_without_cuda_dependencies()
    vol = np.ones((4, 4, 4), dtype=np.float32)
    scanner = {"offOrigin": [0, 0, 0], "dVoxel": [1, 1, 1], "sVoxel": [4, 4, 4]}
    first = sample_vol(vol, 0.05, 10, scanner, "intensity", 0.15, seed=7)
    second = sample_vol(vol, 0.05, 10, scanner, "intensity", 0.15, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    try:
        sample_vol(vol, 0.05, 10, scanner, "auto", 0.15)
    except ValueError as error:
        assert "Unknown volume sampling mode" in str(error)
    else:
        raise AssertionError("Unknown sampling modes must fail explicitly")


def test_auto_fallback_passes_the_resolved_uniform_mode():
    source = (ROOT / "train_recon.py").read_text()
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "init_mode" for target in node.targets)
    ]
    assert any(
        isinstance(node.value, ast.IfExp)
        and isinstance(node.value.orelse, ast.Constant)
        and node.value.orelse.value == "intensity"
        for node in assignments
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "initialize_gaussian_from_proj"
    ]
    assert any(
        any(
            keyword.arg == "init_mode"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "init_mode"
            for keyword in call.keywords
        )
        for call in calls
    )
