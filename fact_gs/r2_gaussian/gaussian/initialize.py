import os
import sys
import os.path as osp
import numpy as np
import copy
from scipy.ndimage import sobel

import tigre.utilities.gpu as gpu

sys.path.append("./")
from fact_gs.r2_gaussian.gaussian.gaussian_model import GaussianModel
from fact_gs.r2_gaussian.utils.general_utils import t2a
from fact_gs.r2_gaussian.utils.graphics_utils import fetchPly
from fact_gs.r2_gaussian.utils.system_utils import searchForMaxStep
from fact_gs.r2_gaussian.dataset import SceneRecon, SceneVol
from fact_gs.r2_gaussian.utils.ct_utils import (
    get_geometry_tigre,
    normalize_fdk_volume,
    recon_volume,
    sample_intensity_volume,
)


def initialize_gaussian(gaussians: GaussianModel, model_args, loaded_step=None):
    if loaded_step:
        if loaded_step == -1:
            loaded_step = searchForMaxStep(
                osp.join(model_args.model_path, "point_cloud")
            )
        step_path = os.path.join(
            model_args.model_path,
            "point_cloud",
            f"step_{loaded_step}",
            "point_cloud.pickle",
        )
        legacy_path = os.path.join(
            model_args.model_path,
            "point_cloud",
            f"iteration_{loaded_step}",
            "point_cloud.pickle",
        )
        ply_path = None
        if osp.exists(step_path):
            ply_path = step_path
        elif osp.exists(legacy_path):
            ply_path = legacy_path
            print(
                f"[compat] Found legacy checkpoint at iteration_{loaded_step}; consider renaming to step_{loaded_step}."
            )
        if ply_path is None:
            raise FileNotFoundError(
                f"Cannot find saved Gaussians for step {loaded_step} in {model_args.model_path}."
            )
        gaussians.load_ply(ply_path)
        print("Loading trained model at step {}".format(loaded_step))
    else:
        if osp.exists(osp.join(model_args.data_source_path, "meta_data.json")):
            ply_path = osp.join(
                model_args.data_source_path, "init_" + osp.basename(model_args.data_source_path) + ".npy"
            )
        elif model_args.data_source_path.split(".")[-1] in ["pickle", "pkl"]:
            ply_path = osp.join(
                osp.dirname(model_args.data_source_path),
                "init_" + osp.basename(model_args.data_source_path).split(".")[0] + ".npy",
            )
        else:
            raise ValueError("Could not recognize scene type!")

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        # r2_gaussian_spiral uses 1.0 here.  FaCT-GS used 2.0, which silently
        # doubled every optimizer learning rate and changes the 30k-step
        # convergence trajectory even when the same init point cloud is used.
        spatial_lr_scale = float(getattr(model_args, "init_spatial_lr_scale", 1.0))
        gaussians.create_from_pcd(xyz, density, spatial_lr_scale)

    return loaded_step

def initialize_gaussian_from_prior(gaussians: GaussianModel, model_args):
    ply_path = model_args.prior_path
    assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
    print("Loading trained model from prior at {}".format(ply_path))
    gaussians.load_ply(ply_path, spatial_lr_scale=1.0)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians.")
    

def sample_vol(
    vol,
    density_thresh,
    n_points,
    scanner_cfg,
    init_mode,
    density_rescale,
    density_init_scale=1.0,
    seed=0,
):
    """Sample points from a volume for initialization."""
    
    density_mask = vol > density_thresh
    valid_indices = np.argwhere(density_mask)

    assert (valid_indices.shape[0] >= n_points), "Valid voxels less than target number of sampling. Check threshold"

    offOrigin = np.array(scanner_cfg["offOrigin"])
    dVoxel = np.array(scanner_cfg["dVoxel"])
    sVoxel = np.array(scanner_cfg["sVoxel"])

    rng = np.random.RandomState(seed)

    if init_mode == 'intensity':
        print("Uniformly sampling foreground voxels (R2-Gaussian initialization).")

        sampled_indices = valid_indices[
            rng.choice(len(valid_indices), n_points, replace=False)
        ]

    elif init_mode == 'gradient':
        print("Random sampling in an intensity volume using gradient as probability.")
        gz = sobel(vol, 0)
        gy = sobel(vol, 1)
        gx = sobel(vol, 2)
        g_norm = np.linalg.norm(np.stack([gz, gy, gx], axis=0), ord=2, axis=0).astype(np.float32)
        g_norm = g_norm/g_norm.max()
        # g_norm = np.power(g_norm, 2.0)
        g_norm = g_norm[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] 

        g_norm = g_norm / g_norm.sum()

        sampled_indices = valid_indices[
            rng.choice(len(valid_indices), n_points, replace=False, p=g_norm)
        ]

    else:
        raise ValueError(
            f"Unknown volume sampling mode '{init_mode}'. Expected 'intensity' or 'gradient'."
        )

    sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]

    # R2-Gaussian uses density_rescale directly.  FaCT-GS historically applied
    # an additional hard-coded 0.25 here, which makes an otherwise identical
    # initialization four times dimmer and breaks R2 quality parity.
    sampled_densities = sampled_densities * density_rescale * density_init_scale

    return sampled_positions, sampled_densities

def initialize_gaussian_from_proj(
    gaussians: GaussianModel,
    model_args,
    optim_args,
    scene: SceneRecon,
    init_mode=None,
):
    # Calculate initial gaussian count
    n_points = int(model_args.num_gaussians)
    print(f"Initialize {n_points} out of total {model_args.num_gaussians} Gaussians.")
    print("Using FDK reconstruction for initialization.")

    listGpuNames = gpu.getGpuNames()
    if len(listGpuNames) == 0:
        print("Tigre error: No gpu found")
    gpuids = gpu.getGpuIds(listGpuNames[0])

    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    z_shifts = np.stack([t2a(cam.z_shift) for cam in train_cameras], axis=0)
    if np.any(z_shifts != 0):
        print(
            f"Spiral acquisition detected (|z_shift| max {np.abs(z_shifts).max():.3f} "
            "scene units); using helical per-view FDK."
        )
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry_tigre(scanner_cfg)

    # vol = algs.fdk(projs_train, geo, angles_train, gpuids=gpuids)
    vol = recon_volume(
        projs_train, angles_train, copy.deepcopy(geo), recon_method="fdk",
        z_shifts=z_shifts,
    )
    print("Reconstruction finished")

    vol = normalize_fdk_volume(vol)

    resolved_init_mode = init_mode or model_args.init_mode
    if resolved_init_mode == "intensity":
        sampled_positions, sampled_densities = sample_intensity_volume(
            vol, model_args.density_thresh, n_points, scanner_cfg,
            model_args.density_rescale,
            getattr(model_args, "density_init_scale", 1.0),
            getattr(model_args, "init_seed", 0),
        )
    else:
        sampled_positions, sampled_densities = sample_vol(
            vol, model_args.density_thresh, n_points, scanner_cfg,
            resolved_init_mode, model_args.density_rescale,
            getattr(model_args, "density_init_scale", 1.0),
            getattr(model_args, "init_seed", 0),
        )

    # Only the R2-equivalent uniform initializer owns the canonical
    # init_<dataset>.npy name.  An explicitly requested gradient experiment
    # must not overwrite the cold-start baseline used by later auto runs.
    if (
        resolved_init_mode == "intensity"
        and getattr(model_args, "save_generated_init", True)
    ):
        dataset_name = osp.basename(osp.normpath(model_args.data_source_path))
        init_path = osp.join(
            model_args.data_source_path, f"init_{dataset_name}.npy"
        )
        if not osp.exists(init_path):
            point_cloud = np.concatenate(
                [sampled_positions, sampled_densities[:, None]], axis=1
            )
            np.save(init_path, point_cloud)
            print(f"Saved reproducible initialization to {init_path}.")

    gaussians.create_from_pcd(
        sampled_positions,
        sampled_densities[:,None],
        float(getattr(model_args, "init_spatial_lr_scale", 1.0)),
    )

def initialize_gaussian_from_vol(gaussians: GaussianModel, model_args, optim_args, scene: SceneVol):
    # Calculate initial gaussian count
    n_points = int(model_args.num_gaussians)
    print(f"Initialize {n_points} out of total {model_args.num_gaussians} Gaussians.")

    vol = scene.vol_gt.cpu().numpy()
    print("Using prior volume for initialization.")
    scanner_cfg = scene.scanner_cfg

    sampled_positions, sampled_densities = sample_vol(vol, 
                                                model_args.density_thresh, 
                                                n_points, 
                                                scanner_cfg, 
                                                model_args.init_mode,
                                                model_args.density_rescale,
                                                getattr(model_args, "density_init_scale", 1.0),
                                                getattr(model_args, "init_seed", 0))

    gaussians.create_from_pcd(sampled_positions, sampled_densities[:,None], 1.0)
