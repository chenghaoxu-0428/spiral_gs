#!/usr/bin/env python3
"""Read-only FaCT-GS Hydra/CUDA training preflight."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)

    def emit(self) -> int:
        for message in self.info:
            print(f"[INFO] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.failures:
            print(f"[FAIL] {message}")
        if self.failures:
            print(f"RESULT: FAIL ({len(self.failures)} blocking, {len(self.warnings)} warnings)")
            return 2
        result = f"PASS WITH WARNINGS ({len(self.warnings)} warnings)" if self.warnings else "PASS"
        print(f"RESULT: {result}")
        return 0


def compose_config(repo: Path, config_name: str, overrides: list[str], report: Report) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except Exception as exc:
        report.fail(f"Cannot import Hydra/OmegaConf with selected Python: {exc}")
        return {}
    normalized = config_name.removesuffix(".yaml")
    try:
        with initialize_config_dir(version_base=None, config_dir=str(repo / "config")):
            cfg = compose(config_name=normalized, overrides=overrides)
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:
        report.fail(f"Cannot compose Hydra config {normalized!r}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(data, dict):
        report.fail("Composed Hydra configuration is not a mapping")
        return {}
    report.note(f"Hydra config: {normalized}; overrides: {overrides or '[]'}")
    return data


def nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def path_from_repo(repo: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo / path


def positive(value: Any, label: str, report: Report) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        report.fail(f"{label} must be positive, got {value!r}")


def run_text(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:
        return False, str(exc)
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def validate_data(repo: Path, cfg: dict[str, Any], target: str, report: Report) -> None:
    source_value = nested(cfg, "model", "data_source_path")
    model_value = nested(cfg, "model", "model_path")
    if not source_value:
        report.fail("Missing model.data_source_path")
    else:
        source = path_from_repo(repo, source_value)
        if not source.is_dir():
            report.fail(f"Dataset directory does not exist: {source}")
        else:
            metadata_path = source / "meta_data.json"
            if not metadata_path.is_file():
                report.fail(f"Dataset metadata not found: {metadata_path}")
            else:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    report.note(f"Dataset metadata keys: {', '.join(sorted(metadata)[:16])}")
                except Exception as exc:
                    report.fail(f"Invalid meta_data.json: {exc}")
            if target == "recon":
                train_dirs = [p for p in (source / "proj_train", source / "train") if p.is_dir()]
                if not train_dirs:
                    report.warn("No conventional proj_train/ or train/ directory found; verify paths encoded by meta_data.json")
                init_mode = nested(cfg, "model", "init_mode")
                if init_mode == "precomputed" and not list(source.glob("init_*.npy")):
                    report.fail("model.init_mode=precomputed but no init_*.npy exists in the dataset")
                if init_mode == "prior":
                    prior = nested(cfg, "model", "prior_path")
                    if not prior or not path_from_repo(repo, prior).is_file():
                        report.fail(f"model.init_mode=prior requires a readable model.prior_path, got {prior!r}")
            else:
                vol_name = nested(cfg, "model", "vol_name")
                candidates = [source / f"{vol_name}.npy", source / f"{vol_name}.tiff", source / f"{vol_name}.tif"]
                if not vol_name or not any(path.is_file() for path in candidates):
                    report.fail(f"Volume target {vol_name!r} not found as .npy/.tif/.tiff under {source}")
    if not model_value:
        report.fail("Missing model.model_path")
    else:
        model_path = path_from_repo(repo, model_value)
        if model_path.exists() and not model_path.is_dir():
            report.fail(f"model.model_path exists but is not a directory: {model_path}")
        elif model_path.is_dir() and any(model_path.iterdir()):
            report.warn(f"model.model_path is non-empty: {model_path}")


def validate_parameters(cfg: dict[str, Any], target: str, report: Report) -> None:
    positive(nested(cfg, "model", "num_gaussians"), "model.num_gaussians", report)
    steps = nested(cfg, "optim", "steps")
    positive(steps, "optim.steps", report)
    scale_min, scale_max = nested(cfg, "model", "scale_min"), nested(cfg, "model", "scale_max")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (scale_min, scale_max)) or not 0 < scale_min < scale_max:
        report.fail(f"Require 0 < model.scale_min < model.scale_max, got {scale_min!r}, {scale_max!r}")
    for key in ("lambda_dssim", "lambda_tv", "lambda_frequency", "training_time_limit_seconds"):
        value = nested(cfg, "optim", key, default=0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            report.fail(f"optim.{key} must be non-negative, got {value!r}")
    for prefix in ("position", "density", "scaling", "rotation"):
        initial = nested(cfg, "optim", f"{prefix}_lr_init")
        final = nested(cfg, "optim", f"{prefix}_lr_final")
        positive(initial, f"optim.{prefix}_lr_init", report)
        positive(final, f"optim.{prefix}_lr_final", report)
        if isinstance(initial, (int, float)) and isinstance(final, (int, float)) and final > initial:
            report.warn(f"optim.{prefix}_lr_final ({final}) exceeds initial value ({initial})")
    if nested(cfg, "optim", "densify_gaussians"):
        interval = nested(cfg, "optim", "densification_interval")
        start = nested(cfg, "optim", "densify_from_step")
        until = nested(cfg, "optim", "densify_until_step_percent")
        positive(interval, "optim.densification_interval", report)
        if not isinstance(start, (int, float)) or start < 0:
            report.fail(f"optim.densify_from_step must be non-negative, got {start!r}")
        if not isinstance(until, (int, float)) or not 0 < until <= 1:
            report.fail(f"optim.densify_until_step_percent must be in (0, 1], got {until!r}")
        if isinstance(steps, (int, float)) and isinstance(start, (int, float)) and isinstance(until, (int, float)) and start >= steps * until:
            report.warn("Densification start is not earlier than its configured end")
    if target == "recon" and nested(cfg, "optim", "lambda_tv", default=0) > 0:
        positive(nested(cfg, "optim", "tv_vol_size"), "optim.tv_vol_size", report)


def validate_environment(gpu: int, target: str, cfg: dict[str, Any], report: Report) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        report.note(f"CUDA_VISIBLE_DEVICES={visible}")
    smi = shutil.which("nvidia-smi")
    if smi:
        ok, output = run_text([smi, "--query-gpu=index,name,driver_version,memory.total,memory.free", "--format=csv,noheader"])
        (report.note if ok else report.warn)("nvidia-smi: " + (output.replace("\n", " | ") or "no output"))
    else:
        report.warn("nvidia-smi is unavailable")
    nvcc = shutil.which("nvcc")
    if nvcc:
        ok, output = run_text([nvcc, "--version"])
        (report.note if ok else report.warn)("NVCC: " + (output.splitlines()[-1] if output else "no output"))
    else:
        report.warn("nvcc is unavailable; existing extensions may still import")
    try:
        import torch
        report.note(f"PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
        if not torch.cuda.is_available():
            report.fail("torch.cuda.is_available() is false")
        elif gpu < 0 or gpu >= torch.cuda.device_count():
            report.fail(f"GPU {gpu} is outside visible range 0..{torch.cuda.device_count() - 1}")
        else:
            props = torch.cuda.get_device_properties(gpu)
            report.note(f"Selected GPU {gpu}: {props.name}; capability {props.major}.{props.minor}; {props.total_memory / 2**30:.1f} GiB")
            try:
                if torch.ones(1, device=f"cuda:{gpu}").item() == 1:
                    report.note("Minimal CUDA tensor check passed")
            except Exception as exc:
                report.fail(f"Minimal CUDA tensor check failed: {exc}")
    except Exception as exc:
        report.fail(f"Cannot import PyTorch: {exc}")
    modules = ["tigre", "simple_knn._C", "fused_ssim", "gs_voxelizer"]
    if target == "recon":
        modules.append("gs_ct_rasterizer")
        if nested(cfg, "optim", "lambda_tv", default=0) > 0:
            modules.append("fused_3d_tv")
    for module in modules:
        try:
            importlib.import_module(module)
            report.note(f"Import passed: {module}")
        except Exception as exc:
            report.fail(f"Import failed for {module}: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="FaCT-GS repository root")
    parser.add_argument("--target", choices=("recon", "volume"), required=True)
    parser.add_argument("--config-name", help="Hydra config name without .yaml")
    parser.add_argument("--override", action="append", default=[], help="Hydra dot-list override; repeat in command order")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index visible to this process")
    args = parser.parse_args()

    report = Report()
    repo = Path(args.repo).expanduser().resolve()
    report.note(f"Python: {sys.executable} ({sys.version.split()[0]})")
    report.note(f"Repository: {repo}")
    for marker in ("train_recon.py", "train_volume.py", "config", "fact_gs"):
        if not (repo / marker).exists():
            report.fail(f"Missing repository marker: {marker}")
    default_config = "default_recon" if args.target == "recon" else "default_volume"
    config_name = args.config_name or default_config
    cfg = compose_config(repo, config_name, args.override, report)
    if cfg:
        validate_data(repo, cfg, args.target, report)
        validate_parameters(cfg, args.target, report)
        selected = {
            "target": args.target,
            "config_name": config_name,
            "model": nested(cfg, "model"),
            "optim": nested(cfg, "optim"),
            "eval": nested(cfg, "eval"),
        }
        report.note("Effective parameters: " + json.dumps(selected, ensure_ascii=False, sort_keys=True, default=str))
        validate_environment(args.gpu, args.target, cfg, report)
    return report.emit()


if __name__ == "__main__":
    raise SystemExit(main())
