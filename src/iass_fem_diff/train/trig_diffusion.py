"""
DDPM training on trig displacement RGB with an auxiliary FEM proxy loss on the predicted
clean image x0: steers samples toward lower bending (four-edge sag, Laplacian proxy) while
the primary MSE on noise keeps the distribution aligned with training rasters.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from iass_fem_diff.datasets.mesh_displacement_rgb import (
    VertexDisplacementRGBDataset,
    load_config as load_disp_cfg,
)
from iass_fem_diff.physics.fem_proxy import structural_efficiency_loss


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _collate_batch(samples: list[dict]) -> dict[str, torch.Tensor | list]:
    out: dict[str, torch.Tensor | list] = {
        "pixel_values": torch.stack([s["pixel_values"] for s in samples], dim=0),
        "extent": torch.stack([s["extent"] for s in samples], dim=0),
        "path": [str(s["path"]) for s in samples],
    }
    if "fem_stress" in samples[0]:
        out["fem_stress"] = torch.stack([s.get("fem_stress") for s in samples], dim=0)  # type: ignore[arg-type]
    return out


def train_from_config(config_path: Path) -> None:
    try:
        from diffusers import DDPMScheduler, UNet2DModel
    except ImportError as e:
        raise ImportError(
            "Trig diffusion training requires optional deps: "
            "pip install 'iass-fem-diff[train]' diffusers accelerate"
        ) from e

    raw = _load_yaml(config_path)
    repo_root = config_path.resolve().parent.parent
    disp_cfg = load_disp_cfg(config_path)
    if not disp_cfg.processed_dir.is_absolute():
        disp_cfg.processed_dir = (repo_root / disp_cfg.processed_dir).resolve()

    train_cfg = raw.get("training", {})
    diff_cfg = raw.get("diffusion", {})
    fem_cfg = raw.get("fem", {})

    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_epochs = int(train_cfg.get("num_epochs", 100))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints/displacement_vertex_rgb"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = (repo_root / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(diff_cfg.get("num_train_timesteps", 1000))
    fem_weight = float(fem_cfg.get("loss_weight", 0.05))
    plane = float(fem_cfg.get("plane_size", 2.0))
    fem_span = int(fem_cfg.get("curvature_span", 1))
    vert_ch = int(fem_cfg.get("vertical_channel", 2))

    phys_scale = 15.0 if disp_cfg.normalization == "fixed_linear" else 1.0

    torch.manual_seed(seed)

    dev_pref = str(train_cfg.get("device", "auto")).strip().lower()
    if dev_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training.device is 'cuda' but PyTorch was built without CUDA or no GPU driver. "
                "Verify: python -c \"import torch; print(torch.cuda.is_available())\". "
                "Install a CUDA wheel from https://pytorch.org/get-started/locally/ "
                "(pip --index-url https://download.pytorch.org/whl/cu124 or newer cu* for your stack)."
            )
        device = torch.device("cuda")
    elif dev_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        _name = torch.cuda.get_device_name(0)
        _cc = torch.version.cuda or "?"
        print(f"Training on GPU: {_name} (torch CUDA {_cc})")
    else:
        print(
            "Training on CPU — install a CUDA-enabled PyTorch build to use your RTX GPU. "
            "See https://pytorch.org/get-started/locally/"
        )

    ds = VertexDisplacementRGBDataset(disp_cfg)
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No images under {disp_cfg.processed_dir}. "
            "Run: node scripts/generate-trig-surfaces.js --max N"
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_batch,
        drop_last=False,
    )

    iw, ih = disp_cfg.image_size[0], disp_cfg.image_size[1]
    hx = plane / max(iw - 1, 1)
    hy = plane / max(ih - 1, 1)
    h, w = ih, iw
    in_ch = 4 if disp_cfg.include_fem_stress else 3
    model = UNet2DModel(
        sample_size=h,
        in_channels=in_ch,
        out_channels=in_ch,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=num_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_noise = 0.0
        epoch_fem = 0.0
        n_batches = 0

        for batch in loader:
            images = batch["pixel_values"].to(device)
            if in_ch == 4:
                fem_s = batch.get("fem_stress")
                if fem_s is None:
                    raise RuntimeError(
                        "data.include_fem_stress is true but dataset did not provide fem_stress. "
                        "Run `iass-fem-diff precompute-fem-trig ...` first."
                    )
                images = torch.cat([images, fem_s.to(device)], dim=1)
            extent = batch["extent"].to(device)
            b = images.shape[0]
            noise = torch.randn_like(images)
            t = torch.randint(0, num_steps, (b,), device=device, dtype=torch.long)
            noisy = scheduler.add_noise(images, noise, t)

            out = model(noisy, t)
            noise_pred = out.sample if hasattr(out, "sample") else out[0]
            loss_noise = F.mse_loss(noise_pred, noise)

            loss_fem = torch.tensor(0.0, device=device)
            if fem_weight > 0.0:
                a_bar = alphas_cumprod[t].view(b, 1, 1, 1)
                sqrt_a = torch.sqrt(a_bar.clamp(min=1e-8))
                sqrt_oma = torch.sqrt((1.0 - a_bar).clamp(min=1e-8))
                x0_hat = (noisy - sqrt_oma * noise_pred) / sqrt_a
                x0_hat = x0_hat.clamp(-1.0, 1.0)
                disp = x0_hat[:, :3] * extent.view(b, 3, 1, 1)
                loss_fem = structural_efficiency_loss(
                    disp,
                    vertical_channel=vert_ch,
                    hx=hx,
                    hy=hy,
                    span=fem_span,
                )

            loss = loss_noise + fem_weight * loss_fem
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += float(loss.item())
            epoch_noise += float(loss_noise.item())
            epoch_fem += float(loss_fem.item())
            n_batches += 1
            global_step += 1

        avg = epoch_loss / max(n_batches, 1)
        avg_n = epoch_noise / max(n_batches, 1)
        avg_f = epoch_fem / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{num_epochs}  loss={avg:.5f}  noise={avg_n:.5f}  fem={avg_f:.5f}  "
            f"fem_w={fem_weight}"
        )

        ckpt_path = ckpt_dir / f"unet_epoch_{epoch + 1:04d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "config": str(config_path),
                "num_train_timesteps": num_steps,
                "fem_loss_weight": fem_weight,
                "plane_size": plane,
                "curvature_hx": hx,
                "curvature_hy": hy,
                "curvature_span": fem_span,
                "vertex_rgb_phys_scale": phys_scale,
                "vertex_rgb_normalization": disp_cfg.normalization,
            },
            ckpt_path,
        )

    print(f"Checkpoints saved under {ckpt_dir}")
