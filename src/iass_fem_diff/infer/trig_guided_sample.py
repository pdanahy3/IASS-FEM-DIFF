"""
Transparent + guided sampling for trig displacement RGB diffusion.

Adds:
- step-by-step dumps (PNG per saved step + metrics CSV/JSON)
- seed image init (img2img-style: start from a forward-diffused seed at a chosen strength)
- goal steering (mix model eps with eps implied by a goal x0 at each timestep)

All images are assumed to use the fixed-linear encoding: physical [-15,15] -> byte [0,255].
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from iass_fem_diff.datasets.mesh_displacement_rgb import FIXED_PHYS_EXTENT, FIXED_PHYS_MAX, FIXED_PHYS_MIN
from iass_fem_diff.physics.fem_proxy import structural_efficiency_loss
from iass_fem_diff.physics.reference_fem_fields import ReferenceFEMConfig, solve_reference_fem_on_displacement_grid


@dataclass
class GuidedRunConfig:
    checkpoint_path: Path
    out_dir: Path
    seed: int = 42
    steps: int = 1000
    device: str = "cuda"
    # img2img seed init
    seed_image: Path | None = None
    strength: float = 0.0  # 0 = ignore seed image; 1 = start from near pure noise
    # goal steering
    goal_image: Path | None = None
    goal_mix: float = 0.0  # 0 = none; 1 = fully match goal eps
    # transparency
    save_every: int = 50
    save_first_last: bool = True


def _decode_rgb_to_model_units(img_rgb_u8: np.ndarray, phys_scale: float) -> torch.Tensor:
    """
    img_rgb_u8: [H,W,3] uint8 with fixed-linear encoding for physical displacement.
    Returns x0: [1,3,H,W] float in [-1,1] where channel values are displacement/phys_scale.
    """
    span = FIXED_PHYS_MAX - FIXED_PHYS_MIN
    phys = FIXED_PHYS_MIN + (img_rgb_u8.astype(np.float32) / 255.0) * span  # H,W,3 in physical units
    x = torch.from_numpy(phys).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1,3,H,W
    x = (x / float(phys_scale)).clamp(-1.0, 1.0)
    return x


def _to_uint8_rgb_fixed_linear(x_bchw: torch.Tensor, phys_scale: float) -> np.ndarray:
    span = FIXED_PHYS_MAX - FIXED_PHYS_MIN
    x = x_bchw.detach().clamp(-1.0, 1.0) * phys_scale
    b = ((x - FIXED_PHYS_MIN) / span * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
    return b.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _maybe_plot_metrics(out_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    ts = [r["t"] for r in rows]
    mse_goal = [r.get("goal_mse_x0", 0.0) for r in rows]
    fem_proxy = [r.get("fem_proxy", 0.0) for r in rows]
    plt.figure(figsize=(9, 4))
    ax1 = plt.gca()
    ax1.plot(ts, mse_goal, label="goal_mse_x0")
    ax1.set_xlabel("timestep (descending)")
    ax1.set_ylabel("goal MSE (x0)")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(ts, fem_proxy, color="orange", alpha=0.8, label="fem_proxy")
    ax2.set_ylabel("proxy (Δw*)² mean")
    plt.title("Guided sampling diagnostics")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=140)
    plt.close()


def run_guided_sampling(cfg: GuidedRunConfig) -> None:
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as e:
        raise ImportError(
            "Inference requires optional deps: pip install 'iass-fem-diff[train]' diffusers accelerate"
        ) from e

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required to save images. Install with: pip install 'iass-fem-diff[train]'") from e

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    train_T = int(ckpt.get("num_train_timesteps", 1000))
    plane = float(ckpt.get("plane_size", 2.0))
    span = int(ckpt.get("curvature_span", 1))
    phys_scale = float(ckpt.get("vertex_rgb_phys_scale", FIXED_PHYS_EXTENT))
    # Prefer inferring channel count from weights (robust to older checkpoints).
    conv_in_w = state.get("conv_in.weight")
    if isinstance(conv_in_w, torch.Tensor) and conv_in_w.ndim == 4:
        in_ch = int(conv_in_w.shape[1])
    else:
        in_ch = int(ckpt.get("in_channels", ckpt.get("vertex_rgb_in_channels", 3)) or 3)

    dev = torch.device(cfg.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")

    torch.manual_seed(int(cfg.seed))
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    model = UNet2DModel(
        sample_size=80,
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
    steps = int(cfg.steps) if cfg.steps is not None else train_T
    scheduler.set_timesteps(steps, device=dev)

    # Prepare optional seed / goal x0 tensors (model units).
    x0_seed = None
    if cfg.seed_image:
        im = Image.open(cfg.seed_image).convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        if arr.shape[0] != 80 or arr.shape[1] != 80:
            im = im.resize((80, 80), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8)
        x0_seed = _decode_rgb_to_model_units(arr, phys_scale).to(dev)

    x0_goal = None
    if cfg.goal_image:
        im = Image.open(cfg.goal_image).convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
        if arr.shape[0] != 80 or arr.shape[1] != 80:
            im = im.resize((80, 80), Image.Resampling.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8)
        x0_goal = _decode_rgb_to_model_units(arr, phys_scale).to(dev)

    # Determine start timestep from strength.
    strength = float(cfg.strength)
    strength = 0.0 if not np.isfinite(strength) else max(0.0, min(1.0, strength))
    start_idx = int(round(strength * (len(scheduler.timesteps) - 1)))
    start_t = scheduler.timesteps[start_idx]

    # Initialize x at start_t.
    if x0_seed is not None and strength > 0:
        noise = torch.randn((1, in_ch, 80, 80), device=dev, dtype=torch.float32)
        x = torch.zeros_like(noise)
        x[:, :3] = x0_seed
        x = scheduler.add_noise(x, noise, start_t)
    else:
        x = torch.randn((1, in_ch, 80, 80), device=dev, dtype=torch.float32)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Save run config for transparency.
    run_json = {
        "checkpoint": str(cfg.checkpoint_path),
        "steps": steps,
        "seed": int(cfg.seed),
        "device": str(cfg.device),
        "seed_image": str(cfg.seed_image) if cfg.seed_image else None,
        "goal_image": str(cfg.goal_image) if cfg.goal_image else None,
        "strength": strength,
        "goal_mix": float(cfg.goal_mix),
        "start_timestep": int(start_t),
        "encoding": {"fixed_linear_range": [FIXED_PHYS_MIN, FIXED_PHYS_MAX], "phys_scale": phys_scale},
        "notes": [
            "Sampling saves intermediate decoded RGB displacements to make the denoising trajectory interpretable.",
            "goal_mix steers by mixing model eps with eps implied by the goal x0 at each timestep.",
        ],
    }
    (cfg.out_dir / "run.json").write_text(json.dumps(run_json, indent=2), encoding="utf-8")

    metrics: list[dict] = []
    hx = plane / 79.0
    hy = plane / 79.0

    goal_mix = float(cfg.goal_mix)
    goal_mix = 0.0 if not np.isfinite(goal_mix) else max(0.0, min(1.0, goal_mix))

    def _should_save(step_i: int, t_int: int) -> bool:
        if cfg.save_first_last and (step_i == 0 or t_int == 0):
            return True
        if cfg.save_every and cfg.save_every > 0:
            return (step_i % int(cfg.save_every)) == 0
        return False

    # Iterate only from start_idx down to the end.
    timesteps = list(scheduler.timesteps[start_idx:])
    with torch.no_grad():
        for step_i, t in enumerate(timesteps):
            out = model(x, t)
            eps = out.sample if hasattr(out, "sample") else out[0]

            # Goal steering via eps_goal mixing.
            goal_mse = 0.0
            if x0_goal is not None and goal_mix > 0:
                a_bar = scheduler.alphas_cumprod[t].to(dev)  # scalar
                sqrt_a = torch.sqrt(a_bar.clamp(min=1e-12))
                sqrt_oma = torch.sqrt((1.0 - a_bar).clamp(min=1e-12))
                x0g = torch.zeros_like(x)
                x0g[:, :3] = x0_goal
                eps_goal = (x - sqrt_a * x0g) / sqrt_oma
                eps = (1.0 - goal_mix) * eps + goal_mix * eps_goal

                # Diagnostic: current predicted x0 mse vs goal.
                x0_hat = (x - sqrt_oma * eps) / sqrt_a
                goal_mse = float(torch.mean((x0_hat[:, :3] - x0_goal) ** 2).detach().cpu().item())

            x = scheduler.step(eps, t, x).prev_sample

            # Proxy metric on current decoded displacement (cheap).
            disp = x[:, :3].detach().clamp(-1.0, 1.0) * phys_scale
            fem_proxy = float(
                structural_efficiency_loss(disp, hx=hx, hy=hy, span=span).detach().cpu().item()
            )
            metrics.append({"step": step_i, "t": int(t), "goal_mse_x0": goal_mse, "fem_proxy": fem_proxy})

            if _should_save(step_i, int(t)):
                rgb = _to_uint8_rgb_fixed_linear(x[:, :3], phys_scale)[0]
                Image.fromarray(rgb, mode="RGB").save(cfg.out_dir / f"step_{step_i:04d}_t{int(t):04d}.png")

    # Final save
    final_rgb = _to_uint8_rgb_fixed_linear(x[:, :3], phys_scale)[0]
    Image.fromarray(final_rgb, mode="RGB").save(cfg.out_dir / "final.png")

    # Final reference FEM fields (slower, but just once).
    ref_cfg = ReferenceFEMConfig(plane_size=plane, gravity_total=1.0)
    disp_final = x[:, :3].detach().clamp(-1.0, 1.0) * phys_scale
    ref = solve_reference_fem_on_displacement_grid(
        disp_final[0].permute(1, 2, 0).contiguous().cpu().numpy(), cfg=ref_cfg
    )
    if ref.get("valid", False):
        (cfg.out_dir / "final_fem.json").write_text(
            json.dumps(
                {
                    "max_stress": float(np.max(ref["sigma"])),
                    "avg_stress": float(np.mean(ref["sigma"])),
                    "max_displacement": float(np.max(ref["delta"])),
                    "avg_displacement": float(np.mean(ref["delta"])),
                    "message": str(ref.get("message", "")),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Write metrics as CSV for easy plotting.
    with (cfg.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "t", "goal_mse_x0", "fem_proxy"])
        w.writeheader()
        for r in metrics:
            w.writerow(r)
    _maybe_plot_metrics(cfg.out_dir, metrics)

