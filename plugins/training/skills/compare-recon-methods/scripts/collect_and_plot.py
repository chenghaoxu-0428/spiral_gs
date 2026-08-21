#!/usr/bin/env python3
"""Collect FDK / SAX-NeRF / FaCT-GS volumes and draw a slice+ROI comparison."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAYOUT_RE = re.compile(
    r"(?:^|/)(?P<kind>real|syn)/(?P<organ>[^/]+)/(?P<acq>spiral|stitch)/"
    r"(?P<ntrain>ntrain\d+)(?:/[^/]+)?/?$"
)

METHOD_ORDER = ("fdk", "naf", "lineformer", "intratomo", "fact-gs", "gt")
LABELS = {
    "fdk": "FDK",
    "naf": "NAF",
    "lineformer": "Lineformer",
    "intratomo": "IntraTomo",
    "fact-gs": "FaCT-GS",
    "gt": "GT",
}


def load_plot_roi(repo: Path):
    path = repo / "plot_roi.py"
    spec = importlib.util.spec_from_file_location("plot_roi", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_kv(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.is_file():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if not val or val.startswith("-") or val.startswith("[") or val.startswith("{"):
            continue
        try:
            metrics[key] = float(val)
        except ValueError:
            metrics[key] = val
    return metrics


def latest_sax_eval(run_dir: Path) -> Path:
    eval_root = run_dir / "eval"
    epoch_dirs = [
        p for p in eval_root.iterdir()
        if p.is_dir() and p.name.startswith("epoch_")
    ] if eval_root.is_dir() else []
    if not epoch_dirs:
        raise FileNotFoundError(f"No eval/epoch_* under {run_dir}")

    def epoch_num(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(epoch_dirs, key=epoch_num)


def latest_factgs_eval(model_dir: Path) -> Path:
    eval_root = model_dir / "eval"
    step_dirs = [
        p for p in eval_root.iterdir()
        if p.is_dir() and p.name.startswith("step_")
    ] if eval_root.is_dir() else []
    if not step_dirs:
        raise FileNotFoundError(f"No eval/step_* under {model_dir}")

    def step_num(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    return max(step_dirs, key=step_num)


def infer_layout(dataset: Path) -> dict[str, str] | None:
    match = LAYOUT_RE.search(dataset.as_posix())
    if not match:
        return None
    return match.groupdict()


def parse_roi(text: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def maybe_compute_fdk_3d(gt_path: Path, fdk_path: Path) -> dict[str, float]:
    import numpy as np
    from fact_gs.r2_gaussian.utils.image_utils import metric_vol

    gt = np.load(gt_path)
    pred = np.load(fdk_path)
    psnr_3d, _ = metric_vol(gt, pred, "psnr")
    ssim_3d, _ = metric_vol(gt, pred, "ssim")
    return {"psnr_3d": float(psnr_3d), "ssim_3d": float(ssim_3d)}


def collect_records(args) -> list[dict[str, Any]]:
    dataset = args.dataset.resolve()
    layout = infer_layout(dataset)
    sax_output = args.sax_output
    if sax_output is None:
        if layout is None:
            raise ValueError("cannot infer SAX output; pass --sax-output")
        sax_output = (
            args.sax_root
            / "output"
            / layout["kind"]
            / layout["organ"]
            / layout["acq"]
            / layout["ntrain"]
        )
    sax_output = sax_output.resolve()

    factgs_model = args.factgs_model
    if factgs_model is None and layout is not None:
        factgs_model = (
            args.repo
            / "models"
            / layout["kind"]
            / layout["organ"]
            / layout["acq"]
            / layout["ntrain"]
            / "factgs_spiralfdk"
        )
    factgs_model = factgs_model.resolve() if factgs_model is not None else None

    records: list[dict[str, Any]] = []
    for name in METHOD_ORDER:
        rec: dict[str, Any] = {
            "method": name,
            "label": LABELS[name],
            "volume_path": None,
            "stats_path": None,
            "ssim": None,
            "psnr_2d": None,
            "psnr_3d": None,
            "ssim_3d": None,
            "time_training_seconds": None,
            "source": None,
        }
        if name == "fdk":
            rec["volume_path"] = str(dataset / "fdk_vol.npy")
            rec["source"] = "dataset/fdk_vol.npy (preprocessing FDK)"
        elif name in ("naf", "lineformer", "intratomo"):
            eval_dir = latest_sax_eval(sax_output / name)
            stats = parse_kv(eval_dir / "stats.txt")
            rec["volume_path"] = str(eval_dir / "image_pred.npy")
            rec["stats_path"] = str(eval_dir / "stats.txt")
            rec["ssim"] = stats.get("proj_ssim")
            rec["psnr_2d"] = stats.get("proj_psnr")
            rec["psnr_3d"] = stats.get("psnr_3d")
            rec["ssim_3d"] = stats.get("ssim_3d")
            tfile = sax_output / name / "training_time_sec.txt"
            if tfile.is_file():
                try:
                    rec["time_training_seconds"] = float(tfile.read_text().strip())
                except ValueError:
                    pass
            rec["source"] = str(eval_dir)
        elif name == "fact-gs":
            if factgs_model is None or not factgs_model.is_dir():
                raise FileNotFoundError("FaCT-GS model directory not found")
            eval_dir = latest_factgs_eval(factgs_model)
            sibling = factgs_model.parent / f"{factgs_model.name}_metrics_final.yml"
            stats = parse_kv(sibling)
            if not stats:
                stats = parse_kv(eval_dir / "eval2d_render_test.yml")
                stats.update(parse_kv(eval_dir / "eval3d.yml"))
            rec["volume_path"] = str(eval_dir / "vol_pred.tiff")
            rec["ssim"] = stats.get("ssim_2d")
            rec["psnr_2d"] = stats.get("psnr_2d")
            rec["psnr_3d"] = stats.get("psnr_3d")
            rec["ssim_3d"] = stats.get("ssim_3d")
            rec["time_training_seconds"] = stats.get("time_training_seconds")
            rec["source"] = str(eval_dir)
        else:
            rec["volume_path"] = str(dataset / "vol_gt.npy")
            rec["source"] = "dataset/vol_gt.npy"

        path = Path(rec["volume_path"])
        if not path.is_file():
            raise FileNotFoundError(f"{name} volume missing: {path}")
        records.append(rec)

    if args.compute_fdk_3d:
        fdk = next(r for r in records if r["method"] == "fdk")
        gt = next(r for r in records if r["method"] == "gt")
        fdk.update(maybe_compute_fdk_3d(Path(gt["volume_path"]), Path(fdk["volume_path"])))
        fdk["source"] = f"{fdk['source']}; 3D metrics vs vol_gt via metric_vol"
    return records


def write_tables(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    fields = [
        "method", "label", "psnr_3d", "ssim_3d", "psnr_2d", "ssim",
        "time_training_seconds", "volume_path", "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    md_path = output_dir / "metrics.md"
    lines = [
        "| method | 2D SSIM | PSNR 2D | PSNR 3D | SSIM 3D | time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rec in records:
        def cell(key: str, digits: int | None = 3) -> str:
            val = rec.get(key)
            if val is None or val == "":
                return "/"
            if isinstance(val, (int, float)) and digits is not None:
                return f"{float(val):.{digits}f}"
            return str(val)
        lines.append(
            f"| {rec['label']} | {cell('ssim')} | {cell('psnr_2d', 2)} | "
            f"{cell('psnr_3d', 2)} | {cell('ssim_3d')} | {cell('time_training_seconds', 1)} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/syn/ldctl004/spiral/ntrain500/r2gs"),
    )
    parser.add_argument("--sax-root", type=Path, default=Path("/opt/data/private/sax-nerf"))
    parser.add_argument("--sax-output", type=Path, default=None)
    parser.add_argument("--factgs-model", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/syn/ldctl004/spiral/ntrain500/comparison"),
    )
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--slice-idx", type=int, default=128)
    parser.add_argument("--roi-red", type=parse_roi, default=(176, 76, 48, 48))
    parser.add_argument("--roi-blue", type=parse_roi, default=(108, 120, 48, 48))
    parser.add_argument("--compute-fdk-3d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--figure-name", default="roi_comparison.png")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    dataset = args.dataset if args.dataset.is_absolute() else repo / args.dataset
    args.dataset = dataset
    output_dir = args.output if args.output.is_absolute() else repo / args.output
    if args.factgs_model is not None and not args.factgs_model.is_absolute():
        args.factgs_model = repo / args.factgs_model

    records = collect_records(args)
    write_tables(records, output_dir)

    plot_roi = load_plot_roi(repo)
    methods = []
    for rec in records:
        item = {"volume_path": rec["volume_path"], "label": rec["label"]}
        if rec.get("stats_path"):
            item["stats_path"] = rec["stats_path"]
        if rec.get("ssim") is not None:
            item["ssim_2d"] = rec["ssim"]
        if rec.get("psnr_2d") is not None:
            item["psnr_2d"] = rec["psnr_2d"]
        if rec.get("ssim_3d") is not None:
            item["ssim_3d"] = rec["ssim_3d"]
        if rec.get("psnr_3d") is not None:
            item["psnr_3d"] = rec["psnr_3d"]
        if rec.get("time_training_seconds") is not None:
            item["time_training_seconds"] = rec["time_training_seconds"]
        methods.append(item)

    figure = output_dir / args.figure_name
    plot_roi.plot_methods_with_rois(
        methods,
        output_path=str(figure),
        axis=args.axis,
        slice_idx=args.slice_idx,
        roi_red=args.roi_red,
        roi_blue=args.roi_blue,
        cmap="gray",
        percentile=(0.5, 99.5),
        window="match_gt",
        dpi=args.dpi,
    )
    print("Metrics:")
    for rec in records:
        ssim = rec["ssim"]
        psnr3d = rec["psnr_3d"]
        ssim3d = rec["ssim_3d"]
        print(
            f"  {rec['label']:<12} 2D SSIM={ssim if ssim is not None else '/'}  "
            f"PSNR3D={psnr3d if psnr3d is not None else '/'}  "
            f"SSIM3D={ssim3d if ssim3d is not None else '/'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
