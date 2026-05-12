"""
Reference FEM + design-language distance metrics for diffusion outputs.

FEM proxy loss is **not** stored in checkpoints (only hyperparameters like fem_loss_weight).
Recompute proxy on each decoded displacement grid for apples-to-apples comparison with training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iass_fem_diff.datasets.mesh_displacement_rgb import FIXED_PHYS_MAX, FIXED_PHYS_MIN
from iass_fem_diff.physics.fem_proxy import structural_efficiency_loss
from iass_fem_diff.physics.reference_fem_fields import ReferenceFEMConfig, solve_reference_fem_on_displacement_grid


def _decode_png_to_disp(path: Path, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 PNG/JPEG -> physical displacement [H,W,3] (fixed-linear)."""
    from PIL import Image

    tw, th = size
    im = Image.open(path).convert("RGB")
    if im.size != (tw, th):
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        im = im.resize((tw, th), resample)
    arr = np.asarray(im, dtype=np.float64)
    span = FIXED_PHYS_MAX - FIXED_PHYS_MIN
    disp = FIXED_PHYS_MIN + (arr / 255.0) * span
    return disp


def _dataset_image_paths(processed_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(processed_dir.glob(pat))
    out = sorted({p for p in paths if not p.name.endswith(".meta.json")})
    return out


def _mse_vs_dataset(gen_disp: np.ndarray, dataset_disps: list[np.ndarray]) -> tuple[float, int]:
    """Return (min_mean_squared_error, argmin_index)."""
    best = float("inf")
    best_i = -1
    for i, ref in enumerate(dataset_disps):
        mse = float(np.mean((gen_disp - ref) ** 2))
        if mse < best:
            best = mse
            best_i = i
    return best, best_i


@dataclass
class SampleEvalRow:
    stem: str
    fem_valid: bool
    fem_message: str
    mean_delta_m: float
    max_delta_m: float
    max_von_mises_pa: float
    fem_proxy_loss: float
    nearest_dataset_mse: float | None
    nearest_dataset_index: int | None


def evaluate_inference_folder(
    samples_dir: Path,
    *,
    dataset_dir: Path | None = None,
    image_size: tuple[int, int] = (80, 80),
    plane_size: float = 2.0,
    gravity_total: float = 1.0,
    young_modulus: float = 210e9,
    poisson_ratio: float = 0.3,
    curvature_span: int = 1,
    vertical_channel: int = 2,
    dataset_limit: int = 0,
) -> dict[str, Any]:
    """
    Run reference FEM + optional nearest-neighbor MSE to training rasters + FEM proxy per file.

    Args:
        samples_dir: Folder containing generated PNG/JPEG samples (displacement RGB).
        dataset_dir: Training / trig dataset folder for nearest-neighbor MSE (optional).
        dataset_limit: Max training images to load (0 = all).
    """
    samples_dir = samples_dir.resolve()
    patterns = ("*.png", "*.jpg", "*.jpeg")
    gen_paths: list[Path] = []
    for pat in patterns:
        gen_paths.extend(samples_dir.glob(pat))
    gen_paths = sorted({p for p in gen_paths if not p.name.endswith(".meta.json")})
    if not gen_paths:
        raise FileNotFoundError(f"No images under {samples_dir}")

    W, H = image_size[0], image_size[1]
    hx = plane_size / max(W - 1, 1)
    hy = plane_size / max(H - 1, 1)

    ref_cfg = ReferenceFEMConfig(
        plane_size=float(plane_size),
        gravity_total=float(gravity_total),
        young_modulus=float(young_modulus),
        poisson_ratio=float(poisson_ratio),
    )

    dataset_disps: list[np.ndarray] | None = None
    if dataset_dir is not None:
        ddir = dataset_dir.resolve()
        if not ddir.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {ddir}")
        dpaths = _dataset_image_paths(ddir)
        if dataset_limit and dataset_limit > 0:
            dpaths = dpaths[: int(dataset_limit)]
        dataset_disps = [_decode_png_to_disp(p, image_size) for p in dpaths]

    rows: list[SampleEvalRow] = []
    for p in gen_paths:
        disp = _decode_png_to_disp(p, image_size)
        sol = solve_reference_fem_on_displacement_grid(disp, cfg=ref_cfg)
        valid = bool(sol.get("valid", False))
        delta = sol.get("delta")
        sigma = sol.get("sigma")
        if valid and delta is not None:
            mean_d = float(np.mean(delta))
            max_d = float(np.max(delta))
            max_vm = float(np.max(sigma)) if sigma is not None else 0.0
        else:
            mean_d = max_d = max_vm = 0.0

        disp_t = (
            torch.from_numpy(disp).permute(2, 0, 1).unsqueeze(0).float()
        )  # 1,3,H,W — physical units (same as training decode * extent)
        pr = float(
            structural_efficiency_loss(
                disp_t,
                vertical_channel=vertical_channel,
                hx=hx,
                hy=hy,
                span=int(curvature_span),
            ).item()
        )

        nn_mse: float | None = None
        nn_idx: int | None = None
        if dataset_disps is not None and len(dataset_disps) > 0:
            nn_mse, nn_idx = _mse_vs_dataset(disp, dataset_disps)

        rows.append(
            SampleEvalRow(
                stem=p.stem,
                fem_valid=valid,
                fem_message=str(sol.get("message", "")),
                mean_delta_m=mean_d,
                max_delta_m=max_d,
                max_von_mises_pa=max_vm,
                fem_proxy_loss=pr,
                nearest_dataset_mse=nn_mse,
                nearest_dataset_index=nn_idx,
            )
        )

    valid_rows = [r for r in rows if r.fem_valid]
    mean_of_mean = (
        float(np.mean([r.mean_delta_m for r in valid_rows])) if valid_rows else float("nan")
    )
    mean_of_max = (
        float(np.mean([r.max_delta_m for r in valid_rows])) if valid_rows else float("nan")
    )
    global_max_delta = (
        float(max(r.max_delta_m for r in valid_rows)) if valid_rows else float("nan")
    )
    mean_proxy = float(np.mean([r.fem_proxy_loss for r in rows]))
    mean_nn = (
        float(np.mean([r.nearest_dataset_mse for r in rows if r.nearest_dataset_mse is not None]))
        if any(r.nearest_dataset_mse is not None for r in rows)
        else float("nan")
    )

    return {
        "samples_dir": str(samples_dir),
        "dataset_dir": str(dataset_dir) if dataset_dir else None,
        "image_size": list(image_size),
        "reference_fem": {
            "plane_size": plane_size,
            "gravity_total": gravity_total,
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "valid_count": len(valid_rows),
            "total_count": len(rows),
            "mean_fem_deflection_spatial_mean_over_samples_m": mean_of_mean,
            "mean_fem_deflection_spatial_max_over_samples_m": mean_of_max,
            "global_max_fem_deflection_m": global_max_delta,
        },
        "fem_proxy": {
            "note": "Same structural_efficiency_loss as training (Coons sag + Laplacian); not read from checkpoints.",
            "curvature_span": curvature_span,
            "vertical_channel": vertical_channel,
            "mean_proxy_loss": mean_proxy,
        },
        "nearest_dataset_mse": (
            {
                "mean_min_mse_design_language_distance": mean_nn,
                "dataset_images_used": len(dataset_disps) if dataset_disps else 0,
            }
            if dataset_disps is not None
            else None
        ),
        "per_sample": [
            {
                "stem": r.stem,
                "fem_valid": r.fem_valid,
                "fem_message": r.fem_message,
                "mean_delta_m": r.mean_delta_m,
                "max_delta_m": r.max_delta_m,
                "max_von_mises_pa": r.max_von_mises_pa,
                "fem_proxy_loss": r.fem_proxy_loss,
                "nearest_dataset_mse": r.nearest_dataset_mse,
                "nearest_dataset_index": r.nearest_dataset_index,
            }
            for r in rows
        ],
    }


def _fmt_float(x: Any) -> str:
    if isinstance(x, (int, float)) and np.isfinite(x):
        return f"{float(x):.6e}"
    return "n/a"


def summarize_for_table(summary: dict[str, Any]) -> dict[str, str]:
    """Human-readable placeholders for paper tables."""
    rf = summary["reference_fem"]
    fp = summary["fem_proxy"]
    nn = summary.get("nearest_dataset_mse")
    dd = "n/a"
    if isinstance(nn, dict):
        v = nn.get("mean_min_mse_design_language_distance")
        dd = _fmt_float(v) if v is not None else "n/a"
    return {
        "mean_FEM_deflection_m": _fmt_float(rf.get("mean_fem_deflection_spatial_mean_over_samples_m")),
        "max_FEM_deflection_m_global_peak": _fmt_float(rf.get("global_max_fem_deflection_m")),
        "mean_per_sample_max_FEM_deflection_m": _fmt_float(rf.get("mean_fem_deflection_spatial_max_over_samples_m")),
        "design_language_distance_mean_min_mse": dd,
        "mean_FEM_proxy_loss": _fmt_float(fp.get("mean_proxy_loss")),
    }


def write_csv(per_sample: list[dict[str, Any]], path: Path) -> None:
    import csv

    if not per_sample:
        return
    keys = list(per_sample[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_sample:
            clean = {k: ("" if v is None else v) for k, v in row.items()}
            w.writerow(clean)
