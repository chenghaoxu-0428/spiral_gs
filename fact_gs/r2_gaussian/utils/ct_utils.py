import os
import sys
import copy
import numpy as np
import tigre
import os.path as osp
import yaml
import time
import tigre.algorithms as algs
from tqdm import trange
from tigre.utilities.Atb import Atb
from tigre.utilities.filtering import filtering
from tigre.utilities.im3Dnorm import im3DNORM
import matplotlib.pyplot as plt

sys.path.append("./")
from fact_gs.r2_gaussian.utils.image_utils import metric_vol


def recon_volume(projs, angles, geo, recon_method, z_shifts=None):
    """Reconstruct ct with traditional methods.

    Args:
        projs: Stacked projections, shape ``(n_views, n_rows, n_cols)``.
        angles: Per-view rotation angles in radians.
        geo: Base TIGRE geometry (scalar attributes, offOrigin as ``(3,)``).
        recon_method: ``"fdk"`` or ``"cgls"``.
        z_shifts: Optional per-view source z translation (same units as
            ``geo``, i.e. scene units). When given and not all zero, uses the
            helical per-view offOrigin FDK that matches the spiral camera
            model of ``dataset_readers.angle2pose`` (source z = z_shift,
            detector parallel to itself); otherwise keeps the legacy
            circular path unchanged.
    """
    if z_shifts is not None and np.any(np.asarray(z_shifts) != 0):
        if recon_method != "fdk":
            raise NotImplementedError(
                "Helical reconstruction is only supported for the 'fdk' method."
            )
        vol = fdk_helical(projs, angles, z_shifts, geo)
    elif recon_method == "fdk":
        vol = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif recon_method == "cgls":
        vol, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise ValueError("Unsupported reconstruction method")
    vol = np.transpose(vol, (2, 1, 0))
    return vol


def normalize_fdk_volume(vol):
    """Match the training-time FDK intensity normalization."""
    vol = np.clip(vol, 0.0, None)
    vol = vol / (np.percentile(vol, 99.5) + 1e-12)
    return np.clip(vol, 0.0, 1.0)


def sample_intensity_volume(
    vol, density_thresh, n_points, scanner_cfg, density_rescale,
    density_init_scale=1.0, seed=0,
):
    """Uniformly sample normalized foreground voxels for initialization."""
    valid_indices = np.argwhere(vol > density_thresh)
    assert len(valid_indices) >= n_points, (
        "Valid voxels less than target number of sampling. Check threshold"
    )
    rng = np.random.RandomState(seed)
    sampled_indices = valid_indices[
        rng.choice(len(valid_indices), n_points, replace=False)
    ]
    sampled_positions = (
        sampled_indices * np.asarray(scanner_cfg["dVoxel"])
        - np.asarray(scanner_cfg["sVoxel"]) / 2
        + np.asarray(scanner_cfg["offOrigin"])
    )
    sampled_densities = vol[tuple(sampled_indices.T)]
    sampled_densities *= density_rescale * density_init_scale
    return sampled_positions, sampled_densities


def geo_with_per_view_offorigin(base_geo, z_shifts):
    """Return a TIGRE geometry with per-view ``offOrigin`` z offsets.

    A spiral acquisition moves the source/detector along z by ``z_shift``
    while the volume stays fixed. TIGRE cannot shift the source along z, so
    instead we shift the reconstruction volume by ``-z_shift`` per view:
    the ray geometry is then identical (source at z=0, detector parallel to
    itself). TIGRE's axis 0 is z, so the offset goes into column 0.

    Args:
        base_geo: Base TIGRE geometry with scalar attributes.
        z_shifts: Per-view source z translations (scene units, shape ``(n,)``).

    Returns:
        Deep-copied geometry with ``offOrigin`` of shape ``(n, 3)``. ``DSD``
        and ``DSO`` are scalarized to plain 1-element arrays because the
        hand-rolled FDK path needs scalar values.
    """
    geo = copy.deepcopy(base_geo)
    # Per-view helical geometry may carry vector DSD/DSO; the FDK path needs scalars.
    for attr in ("DSD", "DSO"):
        val = getattr(geo, attr, None)
        if val is not None:
            scalar = float(np.asarray(val, dtype=np.float64).reshape(-1)[0])
            setattr(geo, attr, np.array([scalar], dtype=np.float32))
    z = np.asarray(z_shifts, dtype=np.float32).reshape(-1)
    base_origin = np.asarray(base_geo.offOrigin, dtype=np.float32).reshape(1, 3)
    geo.offOrigin = base_origin + np.stack(
        [-z, np.zeros_like(z), np.zeros_like(z)], axis=1
    )
    return geo


def fdk_helical(projs, angles, z_shifts, geo, filter_name=None):
    """Helical FDK via per-view ``offOrigin`` (cos weight -> filtering -> Atb).

    Matches ``algs.fdk`` bit-for-bit when ``z_shifts`` is all zero, and
    reproduces the spiral camera model of the training rasterizer (verified
    against ``tigre.Ax`` on synthetic spiral data). ``filter_name`` follows
    TIGRE's ``filtering`` convention (``None`` = default Ram-Lak).

    Returns the volume in TIGRE layout ``[z, y, x]``; ``recon_volume``
    applies the final transpose to storage layout.
    """
    geo = geo_with_per_view_offorigin(geo, z_shifts)
    ang = np.asarray(angles, dtype=np.float32).reshape(-1)
    geo.check_geo(ang)
    geo.checknans()
    geo.angles = ang
    geo.filter = filter_name

    # Same detector orientation convention as the legacy circular path:
    # image rows are flipped to match TIGRE's v axis.
    p = projs.astype(np.float32, copy=True)[:, ::-1, :]

    # Cone-beam cosine weight (hand-rolled because algs.fdk does not accept
    # a vectorized offOrigin geometry).
    xv = (
        np.arange(-geo.nDetector[1] / 2 + 0.5, 1 + geo.nDetector[1] / 2 - 0.5)
        * geo.dDetector[1]
    )
    yv = (
        np.arange(-geo.nDetector[0] / 2 + 0.5, 1 + geo.nDetector[0] / 2 - 0.5)
        * geo.dDetector[0]
    )
    yy, xx = np.meshgrid(xv, yv)
    dsd0 = float(np.asarray(geo.DSD).reshape(-1)[0])
    w = dsd0 / np.sqrt(dsd0**2 + xx**2 + yy**2)
    proj_weighted = np.zeros(p.shape, dtype=np.float32)
    np.multiply(p, w, out=proj_weighted)
    proj_filt = filtering(proj_weighted, geo, geo.angles, parker=False)
    return Atb(proj_filt, geo, geo.angles, "FDK")


def get_geometry_tigre(cfg):
    """For TIGRE only."""
    if cfg["mode"] == "parallel":
        geo = tigre.geometry(mode="parallel", nVoxel=np.array(cfg["nVoxel"][::-1]))
    elif cfg["mode"] == "cone":
        geo = tigre.geometry(mode="cone")
    else:
        raise NotImplementedError("Unsupported scanner mode!")

    geo.DSD = cfg["DSD"]  # Distance Source Detector
    geo.DSO = cfg["DSO"]  # Distance Source Origin
    # Detector parameters
    geo.nDetector = np.array(cfg["nDetector"])  # number of pixels
    geo.sDetector = np.array(cfg["sDetector"])  # size of each pixel
    geo.dDetector = geo.sDetector / geo.nDetector  # total size of the detector
    # Image parameters
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])  # number of voxels
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])  # size of each voxel
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # total size of the image
    # Offsets
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])  # Offset of image from origin
    geo.offDetector = np.array(
        [cfg["offDetector"][1], cfg["offDetector"][0], 0]
    )  # Offset of Detector
    # Auxiliary
    geo.accuracy = cfg["accuracy"]  # Accuracy of FWD proj
    # Mode
    geo.filter = cfg["filter"]
    return geo


def run_ct_recon_algs(projs, angles, geo, ct_gt, save_path, method):
    print("Run {}...".format(method))
    save_path = osp.join(save_path, method)
    slice_save_path = osp.join(save_path, "slice_{}".format(method))
    os.makedirs(slice_save_path, exist_ok=True)
    start_time = time.time()

    if method == "fdk":
        ct_pred = algs.fdk(projs[:, ::-1, :], geo, angles)
    elif method == "sart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE"]
        blcks = 10
        order = "ordered"
        ct_pred, _ = algs.sart(
            projs[:, ::-1, :],
            geo,
            angles,
            20,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=True,
        )
    elif method == "ossart":
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True
        qualmeas = ["RMSE"]
        blcks = 10
        order = "ordered"
        ct_pred, qualityOSSART = algs.ossart(
            projs[:, ::-1, :],
            geo,
            angles,
            20,
            lmbda=lmbda,
            lmbda_red=lambdared,
            verbose=verbose,
            Quameasopts=qualmeas,
            computel2=False,
            blocksize=blcks,
            OrderStrategy=order,
        )
    elif method == "asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        ct_pred = algs.asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
        )
    elif method == "os_asd_pocs":
        epsilon = (
            im3DNORM(
                tigre.Ax(algs.fdk(projs[:, ::-1, :], geo, angles), geo, angles)
                - projs[:, ::-1, :],
                2,
            )
            * 0.15
        )
        alpha = 0.002
        ng = 20
        lmbda = 1
        lambdared = 0.9999
        alpha_red = 0.95
        ratio = 0.94
        verb = True
        order = "ordered"
        blcks = 10
        ct_pred = algs.os_asd_pocs(
            projs[:, ::-1, :],
            geo,
            angles,
            10,  # these are very important
            tviter=ng,
            maxl2err=epsilon,
            alpha=alpha,  # less important.
            lmbda=lmbda,
            lmbda_red=lambdared,
            rmax=ratio,
            verbose=verb,
            OrderStrategy=order,
            blocksize=blcks,
        )
    elif method == "cgls":
        ct_pred, _ = algs.cgls(projs[:, ::-1, :], geo, angles, 60, computel2=True)
    else:
        raise NotImplementedError("Unsupported reconstruction method!")
    ct_pred = ct_pred.transpose((2, 1, 0))

    duration = time.time() - start_time
    psnr_3d, _ = metric_vol(ct_gt, ct_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(ct_gt, ct_pred, "ssim")

    np.save(osp.join(save_path, "ct_gt.npy"), ct_gt)
    np.save(osp.join(save_path, "ct_pred.npy"), ct_pred)

    n_slice = ct_gt.shape[2]
    for i_slice in trange(n_slice, desc="[{}] Save slice".format(method), leave=False):
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_gt.png".format(i_slice)),
            ct_gt[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        plt.imsave(
            osp.join(slice_save_path, "{0:05d}_pred.png".format(i_slice)),
            ct_pred[:, :, i_slice],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    report_dict = {
        "method": method,
        "psnr_3d": psnr_3d,
        "ssim_3d": float(ssim_3d),
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "duration (sec)": duration,
        "duration (min)": duration / 60,
    }
    with open(osp.join(save_path, "eval_3d.yml"), "w") as f:
        yaml.dump(report_dict, f, default_flow_style=False, sort_keys=False)

    print("[{}] psnr_3d: {}, ssim_3d: {}".format(method, psnr_3d, ssim_3d))
    return report_dict, ct_pred, ct_gt
