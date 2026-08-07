"""Utility helpers shared across rasterization tests."""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import skimage.io
import torch
from torch import Tensor


def _get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize(vec: Tensor) -> Tensor:
    return vec / (vec.norm() + 1e-8)


def _look_at(eye: Tensor, target: Tensor, up: Tensor) -> Tensor:
    forward = _normalize(target - eye)
    right = _normalize(torch.cross(up, forward, dim=-1))
    true_up = torch.cross(forward, right, dim=-1)

    view = torch.eye(4, dtype=torch.float32, device=eye.device)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[0, 3] = -torch.dot(right, eye)
    view[1, 3] = -torch.dot(true_up, eye)
    view[2, 3] = -torch.dot(forward, eye)
    return view


def _projection_matrix(FoVx: float, FoVy: float, mode: int) -> Tensor:
    if mode == 0:
        return torch.eye(4, dtype=torch.float32)

    znear, zfar = 0.01, 100.0
    tan_half_y = math.tan(FoVy * 0.5)
    tan_half_x = math.tan(FoVx * 0.5)

    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right

    proj = torch.zeros((4, 4), dtype=torch.float32)
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = (right + left) / (right - left)
    proj[1, 2] = (top + bottom) / (top - bottom)
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    return proj


@dataclass
class TestCamera:
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    mode: int
    world_view_transform: Tensor
    full_proj_transform: Tensor
    camera_center: Tensor

    def to(self, device: torch.device) -> "TestCamera":
        return TestCamera(
            self.image_height,
            self.image_width,
            self.FoVx,
            self.FoVy,
            self.mode,
            self.world_view_transform.to(device),
            self.full_proj_transform.to(device),
            self.camera_center.to(device),
        )


def create_test_camera(
    image_height: int = 256,
    image_width: int = 256,
    fov_y_degrees: float = 50.0,
    mode: int = 0,
    device: str | torch.device | None = None,
    camera_origin: Iterable[float] = (0.0, 0.0, -2.0),
    camera_target: Iterable[float] = (0.0, 0.0, 0.0),
    up: Iterable[float] = (0.0, 1.0, 0.0),
) -> TestCamera:
    """Construct a simple cone-beam (pinhole) or parallel-beam camera for testing."""

    dev = _get_device(device)
    eye = torch.as_tensor(camera_origin, dtype=torch.float32, device=dev)
    target = torch.as_tensor(camera_target, dtype=torch.float32, device=dev)
    up_vec = torch.as_tensor(up, dtype=torch.float32, device=dev)

    if mode == 0:
        FoVy = FoVx = 1.0
        view_raw = _look_at(eye, target, up_vec)
        proj_raw = torch.eye(4, dtype=torch.float32, device=dev)
    else:
        FoVy = math.radians(fov_y_degrees)
        aspect = image_width / max(image_height, 1)
        FoVx = 2.0 * math.atan(math.tan(FoVy * 0.5) * aspect)
        view_raw = _look_at(eye, target, up_vec)
        proj_raw = _projection_matrix(FoVx, FoVy, mode).to(dev)

    view = view_raw.transpose(0, 1).contiguous()
    proj = proj_raw.transpose(0, 1).contiguous()
    full_proj = view @ proj
    return TestCamera(
        image_height=image_height,
        image_width=image_width,
        FoVx=FoVx,
        FoVy=FoVy,
        mode=mode,
        world_view_transform=view,
        full_proj_transform=full_proj,
        camera_center=eye.clone(),
    )


def camera_focal_lengths(camera: TestCamera) -> Tuple[float, float]:
    if camera.mode == 0:
        return 1.0, 1.0
    fx = camera.image_width / (2.0 * math.tan(camera.FoVx * 0.5))
    fy = camera.image_height / (2.0 * math.tan(camera.FoVy * 0.5))
    return float(fx), float(fy)


def camera_tan_fovs(camera: TestCamera) -> Tuple[float, float]:
    if camera.mode == 0:
        return 1.0, 1.0
    return math.tan(camera.FoVx * 0.5), math.tan(camera.FoVy * 0.5)


def generate_random_gaussians(
    num_gaussians: int,
    xy_extent: float = 0.9,
    depth_range: Tuple[float, float] = (0.3, 1.2),
    scale_range: Tuple[float, float] = (0.01, 0.05),
    feature_dim: int = 1,
    device: str | torch.device | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample gaussian parameters that sit inside the camera frustum."""

    dev = _get_device(device)
    num = int(num_gaussians)
    xy = (torch.rand((num, 2), device=dev) * 2.0 - 1.0) * xy_extent
    depth_min, depth_max = depth_range
    depth = torch.rand((num, 1), device=dev) * (depth_max - depth_min) + depth_min
    pos3d = torch.cat([xy, depth], dim=-1)

    scale_min, scale_max = scale_range
    scale3d = torch.rand((num, 3), device=dev) * (scale_max - scale_min) + scale_min
    quat = torch.zeros((num, 4), device=dev)
    quat[:, 0] = 1.0
    intensity = torch.rand((num, feature_dim), device=dev)
    return pos3d, scale3d, quat, intensity


def random_target_image(
    height: int,
    width: int,
    channels: int = 1,
    device: str | torch.device | None = None,
) -> Tensor:
    """Generate a random target image tensor used for gradient checks."""

    dev = _get_device(device)
    return torch.rand((channels, height, width), device=dev)


def generate_test_volume(vol_size: int | Iterable[int]) -> np.ndarray:
    """Create a synthetic 3D volume filled with random soft spheres."""

    if isinstance(vol_size, int):
        shape = (vol_size, vol_size, vol_size)
    else:
        dims = tuple(vol_size)
        if len(dims) != 3:
            raise ValueError("vol_size must have three elements")
        shape = dims

    vol = np.ones(shape, dtype=np.float32) * 0.05
    num_spheres = max(5, shape[0] // 4)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for _ in range(num_spheres):
        center = np.array(
            [
                np.random.randint(0, shape[0]),
                np.random.randint(0, shape[1]),
                np.random.randint(0, shape[2]),
            ]
        )
        radius = np.random.randint(max(5, shape[0] // 12), max(6, shape[0] // 5))
        mask = (
            (zz - center[0]) ** 2
            + (yy - center[1]) ** 2
            + (xx - center[2]) ** 2
            <= radius**2
        )
        vol[mask] += np.random.uniform(0.3, 0.8)
    return np.clip(vol, 0.0, 1.0)


def save_axis_sums_as_images(volume: np.ndarray, path_prefix: str) -> None:
    """Persist axial/coronal/sagittal slices for quick inspection."""

    base_dir = os.path.dirname(path_prefix)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)


    slices = {
        "xy": volume.sum(axis=0),
        "xz": volume.sum(axis=1),
        "yz": volume.sum(axis=2),
    }
    # Normalize slices to [0, 1]
    for name in slices:
        data = slices[name]
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            slices[name] = (data - data_min) / (data_max - data_min)
        else:
            slices[name] = data - data_min
    for name, data in slices.items():
        img = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)
        skimage.io.imsave(f"{path_prefix}_{name}.png", img)


def random_gauss_init(
    num_gaussians: int,
    vol: np.ndarray,
    device: str | torch.device | None = None,
    xy_extent: float = 0.9,
    depth_range: Tuple[float, float] = (0.3, 1.2),
    scale_value: float = 0.04,
    feature_dim: int = 1,
    anisotropicScale: bool = False,
    rotation: bool = False,
):
    """Sample Gaussians informed by a reference volume for visualization tests.

    Args:
        anisotropicScale: Stretch/compress axes to better exercise gradient paths.
        rotation: Apply random rotations (only if ``anisotropicScale`` is enabled).
    """

    dev = _get_device(device)
    num = int(num_gaussians)
    vol_tensor = torch.as_tensor(vol, dtype=torch.float32, device=dev)
    vol_shape = torch.tensor(vol_tensor.shape, device=dev, dtype=torch.float32)

    pos_norm = torch.rand((num, 3), device=dev)
    idx = torch.clamp((pos_norm * (vol_shape - 1)).long(), min=0)
    intensity = vol_tensor[idx[:, 0], idx[:, 1], idx[:, 2]].unsqueeze(-1)
    if feature_dim > 1:
        intensity = intensity.repeat(1, feature_dim)

    pos3d = torch.empty_like(pos_norm)
    pos3d[:, 0] = (pos_norm[:, 2] * 2.0 - 1.0) * xy_extent
    pos3d[:, 1] = (pos_norm[:, 1] * 2.0 - 1.0) * xy_extent
    depth_min, depth_max = depth_range
    pos3d[:, 2] = pos_norm[:, 0] * (depth_max - depth_min) + depth_min

    scale3d = torch.full((num, 3), scale_value, device=dev)
    if anisotropicScale:
        # Encourage non-axis aligned gradients when testing the CUDA kernels.
        scale3d[:, 0] = scale3d[:, 0] * 1.5
        scale3d[:, 1] = scale3d[:, 1] * 0.5

    quat = torch.zeros((num, 4), device=dev)
    quat[:, 0] = 1.0
    if rotation:
        if not anisotropicScale:
            warnings.warn(
                "random_gauss_init rotation requested without anisotropicScale; "
                "skipping rotation for parity with voxelizer tests."
            )
        else:
            angles = torch.rand((num, 3), device=dev) * 360.0
            angles_rad = torch.deg2rad(angles)
            cx = torch.cos(angles_rad[:, 0] * 0.5)
            sx = torch.sin(angles_rad[:, 0] * 0.5)
            cy = torch.cos(angles_rad[:, 1] * 0.5)
            sy = torch.sin(angles_rad[:, 1] * 0.5)
            cz = torch.cos(angles_rad[:, 2] * 0.5)
            sz = torch.sin(angles_rad[:, 2] * 0.5)

            quat[:, 0] = cx * cy * cz + sx * sy * sz
            quat[:, 1] = sx * cy * cz - cx * sy * sz
            quat[:, 2] = cx * sy * cz + sx * cy * sz
            quat[:, 3] = cx * cy * sz - sx * sy * cz

    return pos3d, scale3d, quat, intensity
