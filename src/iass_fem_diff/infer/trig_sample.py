"""
Sampling (inference) for the trig displacement RGB DDPM.

Loads a trained Diffusers `UNet2DModel` checkpoint saved by `train/trig_diffusion.py` and
runs an unconditional DDPM reverse process to generate new 80×80 RGB displacement images.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from iass_fem_diff.datasets.mesh_displacement_rgb import (
    FIXED_PHYS_EXTENT,
    FIXED_PHYS_MAX,
    FIXED_PHYS_MIN,
)
from iass_fem_diff.physics.reference_fem_fields import (
    ReferenceFEMConfig,
    solve_reference_fem_on_displacement_grid,
)
from iass_fem_diff.physics.fem_proxy import structural_efficiency_loss


def _to_uint8_rgb_fixed_linear(x_bchw: torch.Tensor, phys_scale: float) -> np.ndarray:
    """Map model output (displacement / phys_scale) in [-1,1] to bytes via fixed physical range."""
    span = FIXED_PHYS_MAX - FIXED_PHYS_MIN
    x = x_bchw.detach().clamp(-1.0, 1.0) * phys_scale
    b = ((x - FIXED_PHYS_MIN) / span * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    return b.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def sample_from_checkpoint(
    *,
    checkpoint_path: Path,
    out_dir: Path,
    num_samples: int = 16,
    seed: int = 42,
    image_size: tuple[int, int] = (80, 80),
    num_inference_steps: int | None = None,
    device: str = "cuda",
) -> None:
    """
    Generate samples and save them as PNG + `.meta.json` sidecars.

    The sidecar contains a FEM proxy score (lower is smoother under the pinned-edge model).
    """
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as e:
        raise ImportError(
            "Inference requires optional deps: pip install 'iass-fem-diff[train]' diffusers accelerate"
        ) from e

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    train_T = int(ckpt.get("num_train_timesteps", 1000))
    plane = float(ckpt.get("plane_size", 2.0))
    span = int(ckpt.get("curvature_span", 1))
    phys_scale = (
        float(ckpt["vertex_rgb_phys_scale"])
        if "vertex_rgb_phys_scale" in ckpt
        else FIXED_PHYS_EXTENT
    )
    in_ch = int(ckpt.get("in_channels", ckpt.get("vertex_rgb_in_channels", 3)) or 3)

    w, h = int(image_size[0]), int(image_size[1])
    if w < 2 or h < 2:
        raise ValueError("image_size must be >= 2×2")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = UNet2DModel(
        sample_size=h,
        in_channels=in_ch,
        out_channels=in_ch,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=train_T)
    steps = int(num_inference_steps) if num_inference_steps is not None else train_T
    scheduler.set_timesteps(steps, device=dev)

    # Start from pure noise x_T.
    x = torch.randn((num_samples, in_ch, h, w), device=dev, dtype=torch.float32)

    with torch.no_grad():
        for t in scheduler.timesteps:
            out = model(x, t)
            eps = out.sample if hasattr(out, "sample") else out[0]
            x = scheduler.step(eps, t, x).prev_sample

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = _to_uint8_rgb_fixed_linear(x[:, :3], phys_scale)

    hx = plane / max(w - 1, 1)
    hy = plane / max(h - 1, 1)

    disp = x[:, :3].detach().clamp(-1.0, 1.0) * phys_scale
    fem = float(
        structural_efficiency_loss(disp, hx=hx, hy=hy, span=span).detach().cpu().item()
    )

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required to save images. Install with: pip install Pillow") from e

    for i in range(num_samples):
        stem = f"sample_{i:04d}"
        img_path = out_dir / f"{stem}.png"
        meta_path = out_dir / f"{stem}.meta.json"
        Image.fromarray(rgb[i], mode="RGB").save(img_path)

        # Reference FEM (scikit-fem) fields on the deformed grid (base plane + disp).
        ref_cfg = ReferenceFEMConfig(plane_size=plane, gravity_total=1.0)
        ref = solve_reference_fem_on_displacement_grid(
            disp[i].permute(1, 2, 0).contiguous().cpu().numpy(), cfg=ref_cfg
        )
        sigma = ref.get("sigma") if ref.get("valid") else None
        delta = ref.get("delta") if ref.get("valid") else None
        max_stress = float(np.max(sigma)) if sigma is not None else 0.0
        avg_stress = float(np.mean(sigma)) if sigma is not None else 0.0
        max_disp = float(np.max(delta)) if delta is not None else 0.0
        avg_disp = float(np.mean(delta)) if delta is not None else 0.0

        meta = {
            "max_displacement": max_disp,
            "avg_displacement": avg_disp,
            "max_stress": max_stress,
            "avg_stress": avg_stress,
            "extra": {
                "source": "iass_fem_diff.infer.trig_sample",
                "checkpoint": str(checkpoint_path),
                "seed": seed,
                "num_inference_steps": steps,
                "image_size": [w, h],
                "fem_proxy": {
                    "model": "four_edge_coons + laplacian_bending",
                    "plane_size": plane,
                    "curvature_span": span,
                    "score_batch_mean": fem,
                },
                "fem_reference": {
                    "model": "scikit-fem linear elasticity (thin tet layer)",
                    "plane_size": plane,
                    "gravity_total": 1.0,
                    "young_modulus": ref_cfg.young_modulus,
                    "poisson_ratio": ref_cfg.poisson_ratio,
                    "valid": bool(ref.get("valid", False)),
                    "message": str(ref.get("message", "")),
                },
                "fem_stress_grid": (
                    None
                    if sigma is None
                    else {
                        "values": np.asarray(sigma, dtype=np.float64).reshape(-1).tolist(),
                        "width": w,
                        "height": h,
                        "units": "Pa",
                        "field": "von_mises",
                    }
                ),
                "fem_disp_grid": (
                    None
                    if delta is None
                    else {
                        "values": np.asarray(delta, dtype=np.float64).reshape(-1).tolist(),
                        "width": w,
                        "height": h,
                        "units": "m",
                        "field": "displacement_magnitude",
                    }
                ),
                "rgb_mapping": {
                    "encoding": "fixed_linear",
                    "physical_min": FIXED_PHYS_MIN,
                    "physical_max": FIXED_PHYS_MAX,
                    "model_units": "channels are displacement_i / phys_scale in [-1,1]",
                    "phys_scale": phys_scale,
                    "byte_formula": "byte = round(255 * (displacement - physical_min) / (physical_max - physical_min))",
                },
            },
        }
        # Drop None fields for JSON cleanliness
        if meta["extra"].get("fem_stress_grid") is None:
            meta["extra"].pop("fem_stress_grid", None)
        if meta["extra"].get("fem_disp_grid") is None:
            meta["extra"].pop("fem_disp_grid", None)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

