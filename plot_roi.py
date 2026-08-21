import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import tifffile
except ImportError:
    tifffile = None


# ROI 框/边框颜色（主图框与局部图边框共用，改这里即可同步）
COLOR_ROI_RED = "red"
COLOR_ROI_BLUE = "royalblue"


def load_volume(path):
    """
    支持:
        .npy
        .tif / .tiff

    返回:
        volume: np.ndarray, shape = [D, H, W] 或类似三维数组
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        volume = np.load(path)

    elif ext in [".tif", ".tiff"]:
        if tifffile is None:
            raise ImportError(
                "读取 TIFF 需要安装 tifffile:\n"
                "pip install tifffile"
            )
        volume = tifffile.imread(path)

    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"期望输入为 3D volume，但实际 shape={volume.shape}"
        )

    # FaCT-GS vol_pred.tiff is uint8 in [0, 255] (see save_volume);
    # npy volumes from the dataset / SAX-NeRF are already float in ~[0, 1].
    if np.issubdtype(volume.dtype, np.integer):
        volume = volume.astype(np.float32) / float(np.iinfo(volume.dtype).max)
    else:
        volume = volume.astype(np.float32)
    return volume


def get_slice(volume, axis=0, slice_idx=None):
    """
    axis:
        0 -> volume[z, :, :]
        1 -> volume[:, y, :]
        2 -> volume[:, :, x]
    """
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    if axis == 0:
        image = volume[slice_idx, :, :]
    elif axis == 1:
        image = volume[:, slice_idx, :]
    elif axis == 2:
        image = volume[:, :, slice_idx]
    else:
        raise ValueError("axis 必须是 0 / 1 / 2")

    return image, slice_idx


def crop_roi(image, roi):
    """
    roi = (x, y, w, h)

    注意:
        x: 列坐标
        y: 行坐标

    ROI 超出图像范围时直接报错（而不是静默裁剪）。
    """
    x, y, w, h = roi
    ih, iw = image.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError(f"ROI 宽高必须为正数: {roi}")
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        raise ValueError(f"ROI {roi} 超出图像范围 (W={iw}, H={ih})")

    return image[y:y + h, x:x + w]


def _percentile_window(image, percentile):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, list(percentile))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _match_histogram(source, reference):
    """Map source intensities onto the reference histogram (display only)."""
    src = np.asarray(source, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    try:
        from skimage.exposure import match_histograms
        return match_histograms(src, ref).astype(np.float32)
    except Exception:
        src_lo, src_hi = _percentile_window(src, (1.0, 99.0))
        ref_lo, ref_hi = _percentile_window(ref, (1.0, 99.0))
        scale = (ref_hi - ref_lo) / max(src_hi - src_lo, 1e-8)
        return (src - src_lo) * scale + ref_lo


def _layout_rects(image, red_crop, blue_crop, pad, gap_main_roi, gap_between_rois,
                  fig_width, line_width):
    """
    以归一化坐标（figure 分数）计算三个 axes 的矩形位置。

    关键不变式:
      - 每个 axes 的宽高比 == 其所显示图像的宽高比，
        因此 imshow(aspect="equal") 时图像恰好填满 axes，内部无留白；
      - 下方两个 ROI 的总宽度 + 间隔 == 主图宽度，天然对齐；
      - ROI 的彩色边框（spine，居中于 axes 边缘）会向外溢出半个线宽，
        因此 ROI axes 向内缩进半个线宽，使边框外缘与主图左右边缘精确重合。

    返回 (rect_main, rect_red, rect_blue, total_h)，total_h 为整个 figure 的
    归一化总高度（figure 宽度恒为 1）。
    """
    ih, iw = image.shape[:2]
    rh, rw = red_crop.shape[:2]
    bh, bw = blue_crop.shape[:2]

    # 主图占满整行
    main_w = 1.0 - 2 * pad
    main_h = main_w * ih / iw

    # ROI 行: 按像素宽度比例分配（像素尺寸相同的 ROI 显示得一样大）
    roi_total_w = main_w - gap_between_rois
    red_w = roi_total_w * rw / (rw + bw)
    blue_w = roi_total_w * bw / (rw + bw)
    red_h = red_w * rh / rw
    blue_h = blue_w * bh / bw

    bottom_h = max(red_h, blue_h)
    total_h = main_h + gap_main_roi + bottom_h + 2 * pad

    # 半线宽 (point -> figure 宽度分数; 1pt = 1/72 in)
    hl = line_width / (2.0 * 72.0 * fig_width)

    # 上面所有量都以「figure 宽度」为单位；而 add_axes 的 y/h 是 figure 高度的
    # 分数，figure 高度 = total_h x 宽度，所以这里统一除以 total_h 换算。
    # ROI axes 向内缩进 hl，使 spine 外缘与主图边缘对齐。
    main_rect = [pad, (pad + bottom_h + gap_main_roi) / total_h, main_w, main_h / total_h]
    red_rect = [pad + hl, (pad + bottom_h - red_h + hl) / total_h,
                red_w - 2 * hl, (red_h - 2 * hl) / total_h]
    blue_rect = [pad + red_w + gap_between_rois + hl,
                 (pad + bottom_h - blue_h + hl) / total_h,
                 blue_w - 2 * hl, (blue_h - 2 * hl) / total_h]

    return main_rect, red_rect, blue_rect, total_h


def _offset_rect(rect, x0, y0, sx, sy):
    """Map a column-local add_axes rect into figure coordinates."""
    x, y, w, h = rect
    return [x0 + x * sx, y0 + y * sy, w * sx, h * sy]


def _draw_slice_panel(
    fig,
    rect_main,
    rect_red,
    rect_blue,
    image,
    red_crop,
    blue_crop,
    roi_red,
    roi_blue,
    imshow_kwargs,
    line_width,
    hide_axis=True,
):
    ax_main = fig.add_axes(rect_main)
    ax_red = fig.add_axes(rect_red)
    ax_blue = fig.add_axes(rect_blue)

    ax_main.imshow(image, **imshow_kwargs)
    x, y, w, h = roi_red
    ax_main.add_patch(Rectangle(
        (x, y), w, h,
        linewidth=line_width,
        edgecolor=COLOR_ROI_RED,
        facecolor="none",
        clip_on=False,
    ))
    x, y, w, h = roi_blue
    ax_main.add_patch(Rectangle(
        (x, y), w, h,
        linewidth=line_width,
        edgecolor=COLOR_ROI_BLUE,
        facecolor="none",
        clip_on=False,
    ))

    ax_red.imshow(red_crop, **imshow_kwargs)
    ax_blue.imshow(blue_crop, **imshow_kwargs)

    for ax, color in [(ax_red, COLOR_ROI_RED), (ax_blue, COLOR_ROI_BLUE)]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(line_width)
            spine.set_edgecolor(color)

    if hide_axis:
        for ax in (ax_main, ax_red, ax_blue):
            ax.set_xticks([])
            ax.set_yticks([])

    for spine in ax_main.spines.values():
        spine.set_visible(False)

    return ax_main, ax_red, ax_blue


def read_eval_stats(stats_path):
    """Read numeric key: value pairs from an eval stats.txt."""
    metrics = {}
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                continue
    return metrics


def read_proj_ssim(stats_path):
    """Read 2D projection SSIM from an eval stats.txt."""
    metrics = read_eval_stats(stats_path)
    if "proj_ssim" not in metrics:
        raise KeyError(f"proj_ssim not found in {stats_path}")
    return metrics["proj_ssim"]


def find_latest_eval_dir(run_dir):
    eval_root = os.path.join(run_dir, "eval")
    epoch_dirs = [
        os.path.join(eval_root, name)
        for name in os.listdir(eval_root)
        if name.startswith("epoch_") and os.path.isdir(os.path.join(eval_root, name))
    ]
    if not epoch_dirs:
        raise FileNotFoundError(f"No eval/epoch_* under {run_dir}")

    def epoch_num(path):
        name = os.path.basename(path)
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(epoch_dirs, key=epoch_num)


def _fmt_metric(value, digits):
    if value is None:
        return "/"
    return f"{float(value):.{digits}f}"


def _fmt_time(seconds):
    if seconds is None:
        return "/"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} h"


def _copy_metrics(dst, rec):
    if not rec:
        return dst
    if rec.get("ssim") is not None:
        dst["ssim_2d"] = rec["ssim"]
    if rec.get("ssim_2d") is not None:
        dst["ssim_2d"] = rec["ssim_2d"]
    for key in ("psnr_2d", "psnr_3d", "ssim_3d", "time_training_seconds"):
        if rec.get(key) is not None:
            dst[key] = rec[key]
    return dst


def plot_methods_with_rois(
    methods,
    output_path="method_roi_comparison.png",
    axis=0,
    slice_idx=None,
    roi_red=(40, 40, 40, 40),
    roi_blue=(120, 90, 40, 40),
    cmap="gray",
    interpolation="nearest",
    vmin=None,
    vmax=None,
    percentile=(0.5, 99.5),
    window="match_gt",
    fig_width_per_col=3.4,
    col_gap=0.06,
    caption_h=1.08,
    dpi=300,
    line_width=1.8,
    pad=0.02,
    gap_main_roi=0.03,
    gap_between_rois=0.025,
    hide_axis=True,
    ssim_decimals=3,
):
    """
    将多种重建结果排成一行，每列为主切片 + 两个 ROI，列下标注
    2D/3D PSNR、SSIM 与训练时间（缺失项显示为 /）。

        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  method0 │ │  method1 │ │  method2 │
        │   ROIs   │ │   ROIs   │ │   ROIs   │
        └──────────┘ └──────────┘ └──────────┘
         name / SSIM  name / SSIM  name / SSIM

    methods: list of dicts with keys
        volume_path (required)
        label (required)
        ssim / psnr_2d / ssim_2d / ssim_3d / psnr_3d / time_training_seconds
        stats_path (optional; fills 2D from proj_* and 3D from ssim_3d/psnr_3d)

    window:
        match_gt    histogram-match every column to GT, then share the GT window
        per_method  each column uses its own percentile range
        gt          all columns share the GT slice percentile range
        shared      all columns share one percentile range over every slice
    Explicit vmin/vmax override the automatic window.
    """
    if not methods:
        raise ValueError("methods 不能为空")

    panels = []
    for method in methods:
        volume = load_volume(method["volume_path"])
        image, used_slice = get_slice(volume, axis, slice_idx)
        ssim_2d = method.get("ssim_2d", method.get("ssim"))
        psnr_2d = method.get("psnr_2d")
        ssim_3d = method.get("ssim_3d")
        psnr_3d = method.get("psnr_3d")
        time_s = method.get("time_training_seconds")
        if method.get("stats_path"):
            stats = read_eval_stats(method["stats_path"])
            if ssim_2d is None:
                ssim_2d = stats.get("proj_ssim")
            if psnr_2d is None:
                psnr_2d = stats.get("proj_psnr")
            if ssim_3d is None:
                ssim_3d = stats.get("ssim_3d")
            if psnr_3d is None:
                psnr_3d = stats.get("psnr_3d")
        panels.append({
            "label": method["label"],
            "image": image,
            "volume_shape": volume.shape,
            "slice_idx": used_slice,
            "psnr_2d": psnr_2d,
            "ssim_2d": ssim_2d,
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "time_training_seconds": time_s,
            "red_crop": crop_roi(image, roi_red),
            "blue_crop": crop_roi(image, roi_blue),
        })

    mode = (window or "match_gt").strip().lower()
    if mode not in ("match_gt", "per_method", "gt", "shared"):
        raise ValueError("window 必须是 match_gt / per_method / gt / shared")

    gt_panels = [p for p in panels if str(p["label"]).strip().upper() == "GT"]
    if mode == "match_gt":
        if not gt_panels:
            raise ValueError("window=match_gt 需要 methods 中包含 label='GT' 的一列")
        gt_image = gt_panels[0]["image"]
        for panel in panels:
            if str(panel["label"]).strip().upper() == "GT":
                continue
            panel["image"] = _match_histogram(panel["image"], gt_image)
            panel["red_crop"] = crop_roi(panel["image"], roi_red)
            panel["blue_crop"] = crop_roi(panel["image"], roi_blue)

    shared_vmin, shared_vmax = vmin, vmax
    if (shared_vmin is None or shared_vmax is None) and mode != "per_method":
        if mode in ("gt", "match_gt"):
            source = gt_panels[0]["image"] if gt_panels else None
        else:
            source = None
        if source is None:
            finite = np.concatenate([
                p["image"][np.isfinite(p["image"])].ravel() for p in panels
            ])
        else:
            finite = source[np.isfinite(source)]
        auto_vmin, auto_vmax = np.percentile(finite, list(percentile)) \
            if finite.size else (0.0, 1.0)
        if shared_vmin is None:
            shared_vmin = auto_vmin
        if shared_vmax is None:
            shared_vmax = auto_vmax

    for panel in panels:
        if vmin is None and vmax is None and mode == "per_method":
            lo, hi = _percentile_window(panel["image"], percentile)
        else:
            lo, hi = shared_vmin, shared_vmax
        panel["vmin"] = lo
        panel["vmax"] = hi

    n = len(panels)
    col_w = fig_width_per_col
    fig_w = n * col_w + (n - 1) * col_gap
    local_main, local_red, local_blue, total_h = _layout_rects(
        panels[0]["image"], panels[0]["red_crop"], panels[0]["blue_crop"],
        pad, gap_main_roi, gap_between_rois, col_w, line_width,
    )
    col_h = col_w * total_h
    fig_h = col_h + caption_h
    sx = col_w / fig_w
    sy = col_h / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, panel in enumerate(panels):
        x0 = i * (col_w + col_gap) / fig_w
        y0 = caption_h / fig_h
        imshow_kwargs = dict(
            cmap=cmap,
            vmin=panel["vmin"],
            vmax=panel["vmax"],
            interpolation=interpolation,
            aspect="equal",
        )
        _draw_slice_panel(
            fig,
            _offset_rect(local_main, x0, y0, sx, sy),
            _offset_rect(local_red, x0, y0, sx, sy),
            _offset_rect(local_blue, x0, y0, sx, sy),
            panel["image"],
            panel["red_crop"],
            panel["blue_crop"],
            roi_red,
            roi_blue,
            imshow_kwargs,
            line_width,
            hide_axis=hide_axis,
        )
        caption = "\n".join([
            panel["label"],
            f"2D PSNR = {_fmt_metric(panel.get('psnr_2d'), 2)}",
            f"2D SSIM = {_fmt_metric(panel.get('ssim_2d'), ssim_decimals)}",
            f"3D PSNR = {_fmt_metric(panel.get('psnr_3d'), 2)}",
            f"3D SSIM = {_fmt_metric(panel.get('ssim_3d'), ssim_decimals)}",
            f"Time = {_fmt_time(panel.get('time_training_seconds'))}",
        ])
        cx = (i * (col_w + col_gap) + 0.5 * col_w) / fig_w
        fig.text(
            cx,
            0.90 * caption_h / fig_h,
            caption,
            ha="center",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
        )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Methods      : {[p['label'] for p in panels]}")
    print(f"Volume shape : {panels[0]['volume_shape']}")
    print(f"Axis         : {axis}")
    print(f"Slice index  : {panels[0]['slice_idx']}")
    print(f"Red ROI      : {roi_red}")
    print(f"Blue ROI     : {roi_blue}")
    windows = [f"{p['label']}[{p['vmin']:.4g}, {p['vmax']:.4g}]" for p in panels]
    print(f"Window       : {mode}")
    print(f"Intensity    : {', '.join(windows)}")
    print(f"2D PSNR      : {[p['psnr_2d'] for p in panels]}")
    print(f"2D SSIM      : {[p['ssim_2d'] for p in panels]}")
    print(f"3D PSNR      : {[p['psnr_3d'] for p in panels]}")
    print(f"3D SSIM      : {[p['ssim_3d'] for p in panels]}")
    print(f"Time (s)     : {[p['time_training_seconds'] for p in panels]}")
    print(f"Saved to     : {output_path}")
    return output_path


def plot_slice_with_rois(
    volume_path,
    output_path="slice_roi_comparison.png",
    axis=0,
    slice_idx=None,

    # ROI 格式: (x, y, width, height)
    roi_red=(40, 40, 40, 40),
    roi_blue=(120, 90, 40, 40),

    cmap="gray",
    interpolation="nearest",   # nearest=忠实显示原始像素块; 想平滑可试 "bicubic"(人造过渡)

    # 显示范围
    vmin=None,
    vmax=None,
    percentile=(0.5, 99.5),

    # 图像显示
    fig_width=4.0,          # 单位: 英寸; 高度由图像宽高比自动推导
    dpi=300,                # 保存分辨率: PNG 像素数 = 英寸尺寸 x dpi

    # ROI 框
    line_width=1.8,

    # 布局(归一化, 相对 figure 宽度)
    pad=0.01,               # 四周留白(容纳框线)
    gap_main_roi=0.02,      # 主图与 ROI 行的间距
    gap_between_rois=0.02,  # 两个 ROI 之间的间距

    # 是否隐藏所有坐标轴
    hide_axis=True,
):
    """
    生成：
        ┌───────────────┐
        │               │
        │   主切片      │
        │  □红   □蓝    │
        │               │
        ├───────┬───────┤
        │ 红ROI │ 蓝ROI │
        └───────┴───────┘

    下方两个 ROI 的总宽度与主图完全对齐（左右边缘一致）。
    实现方式：不用 GridSpec，而是按像素宽高比手动计算每个 axes 的
    归一化矩形，使图像恰好填满 axes、无 aspect 留白。
    """

    volume = load_volume(volume_path)
    image, slice_idx = get_slice(volume, axis, slice_idx)

    # --------------------------------------------------
    # 强度 window
    # --------------------------------------------------
    if vmin is None or vmax is None:
        finite = image[np.isfinite(image)]
        p_low, p_high = percentile
        auto_vmin, auto_vmax = np.percentile(finite, [p_low, p_high]) \
            if finite.size else (0.0, 1.0)

        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    # --------------------------------------------------
    # ROI crop
    # --------------------------------------------------
    red_crop = crop_roi(image, roi_red)
    blue_crop = crop_roi(image, roi_blue)

    # --------------------------------------------------
    # Layout: 手动按像素比例摆放 axes，保证上下对齐
    # --------------------------------------------------
    rect_main, rect_red, rect_blue, total_h = _layout_rects(
        image, red_crop, blue_crop,
        pad, gap_main_roi, gap_between_rois,
        fig_width, line_width,
    )

    fig = plt.figure(figsize=(fig_width, fig_width * total_h))
    imshow_kwargs = dict(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        aspect="equal",
    )
    _draw_slice_panel(
        fig, rect_main, rect_red, rect_blue,
        image, red_crop, blue_crop, roi_red, roi_blue,
        imshow_kwargs, line_width, hide_axis=hide_axis,
    )

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    fig_h = fig_width * total_h
    print(f"Volume shape : {volume.shape}")
    print(f"Axis         : {axis}")
    print(f"Slice index  : {slice_idx}")
    print(f"Red ROI      : {roi_red}")
    print(f"Blue ROI     : {roi_blue}")
    print(f"Intensity    : [{vmin:.6g}, {vmax:.6g}]")
    print(f"Figure size  : {fig_width:.2f} x {fig_h:.2f} in")
    print(f"PNG size     : ~{round(fig_width * dpi)} x {round(fig_h * dpi)} px "
          f"(dpi={dpi}, 保存时 tight 裁剪会略小)")
    print(f"Saved to     : {output_path}")


def parse_roi_arg(text):
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    return tuple(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot multi-method slice + ROI comparison")
    parser.add_argument(
        "--dataset",
        default="data/syn/ldctl004/spiral/ntrain500/r2gs",
        help="R2-Gaussian dataset directory (vol_gt.npy / fdk_vol.npy)",
    )
    parser.add_argument(
        "--sax-output",
        default="/opt/data/private/sax-nerf/output/syn/ldctl004/spiral/ntrain500",
    )
    parser.add_argument(
        "--factgs-eval",
        default="models/syn/ldctl004/spiral/ntrain500/factgs_spiralfdk/eval/step_030000",
    )
    parser.add_argument(
        "--factgs-metrics",
        default="models/syn/ldctl004/spiral/ntrain500/factgs_spiralfdk_metrics_final.yml",
    )
    parser.add_argument(
        "--output",
        default="output/syn/ldctl004/spiral/ntrain500/comparison/roi_comparison.png",
    )
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--slice-idx", type=int, default=128)
    parser.add_argument("--roi-red", default="176,76,48,48")
    parser.add_argument("--roi-blue", default="108,120,48,48")
    parser.add_argument(
        "--window",
        default="match_gt",
        choices=["match_gt", "per_method", "gt", "shared"],
        help="Display window: histogram-match to GT (default), per-method percentile, GT window, or shared",
    )
    args = parser.parse_args()

    factgs_metrics = {}
    if os.path.isfile(args.factgs_metrics):
        with open(args.factgs_metrics, "r") as handle:
            for line in handle:
                key = line.split(":", 1)[0].strip() if ":" in line else ""
                if key in ("psnr_2d", "ssim_2d", "psnr_3d", "ssim_3d", "time_training_seconds"):
                    try:
                        factgs_metrics[key] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

    methods = [
        {
            "volume_path": os.path.join(args.dataset, "fdk_vol.npy"),
            "label": "FDK",
        },
    ]
    metrics_by_label = {}
    metrics_json = os.path.join(os.path.dirname(args.output), "metrics.json")
    if os.path.isfile(metrics_json):
        import json
        for rec in json.loads(open(metrics_json, encoding="utf-8").read()):
            metrics_by_label[rec.get("label")] = rec
    _copy_metrics(methods[0], metrics_by_label.get("FDK"))
    for name, label in (
        ("naf", "NAF"),
        ("lineformer", "Lineformer"),
        ("intratomo", "IntraTomo"),
    ):
        eval_dir = find_latest_eval_dir(os.path.join(args.sax_output, name))
        item = {
            "volume_path": os.path.join(eval_dir, "image_pred.npy"),
            "stats_path": os.path.join(eval_dir, "stats.txt"),
            "label": label,
        }
        _copy_metrics(item, metrics_by_label.get(label))
        methods.append(item)
    factgs_item = {
        "volume_path": os.path.join(args.factgs_eval, "vol_pred.tiff"),
        "label": "FaCT-GS",
        **factgs_metrics,
    }
    _copy_metrics(factgs_item, metrics_by_label.get("FaCT-GS"))
    methods.append(factgs_item)
    methods.append({
        "volume_path": os.path.join(args.dataset, "vol_gt.npy"),
        "label": "GT",
    })

    plot_methods_with_rois(
        methods,
        output_path=args.output,
        axis=args.axis,
        slice_idx=args.slice_idx,
        roi_red=parse_roi_arg(args.roi_red),
        roi_blue=parse_roi_arg(args.roi_blue),
        cmap="gray",
        percentile=(0.5, 99.5),
        window=args.window,
        dpi=300,
    )
