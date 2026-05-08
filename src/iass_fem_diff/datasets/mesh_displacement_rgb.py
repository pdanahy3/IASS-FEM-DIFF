"""Load trig (or other) displacement RGB rasters + optional .meta.json for extents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# Fixed world-space displacement range encoded linearly into R,G,B bytes 0…255 (see generate-trig-surfaces.js).
FIXED_PHYS_MIN = -15.0
FIXED_PHYS_MAX = 15.0
FIXED_PHYS_EXTENT = max(abs(FIXED_PHYS_MIN), abs(FIXED_PHYS_MAX))

@dataclass
class VertexDisplacementRGBConfig:
    processed_dir: Path
    image_size: tuple[int, int]
    normalization: str = "extent_from_meta"
    include_fem_stress: bool = False
    fem_stress_norm: float = 1.0


def load_config(path: Path) -> VertexDisplacementRGBConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = raw["data"]
    w, h = data["image_size"]
    return VertexDisplacementRGBConfig(
        processed_dir=Path(data["processed_dir"]),
        image_size=(int(w), int(h)),
        normalization=str(data.get("normalization", "extent_from_meta")),
        include_fem_stress=bool(data.get("include_fem_stress", False)),
        fem_stress_norm=float(data.get("fem_stress_norm", 1.0)),
    )


def _default_extent() -> torch.Tensor:
    return torch.ones(3, dtype=torch.float32)


def _fixed_extent_tensor() -> torch.Tensor:
    return torch.full((3,), FIXED_PHYS_EXTENT, dtype=torch.float32)


def _extent_from_meta(meta_path: Path) -> torch.Tensor:
    if not meta_path.is_file():
        return _default_extent()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ch = meta.get("extra", {}).get("channel_min_max_raw")
        if not ch:
            return _default_extent()
        ex = []
        for key in ("x", "y", "z"):
            c = ch.get(key) or {}
            r = float(c.get("extent_R") or 1.0)
            if r <= 0 or not np.isfinite(r):
                r = 1.0
            ex.append(r)
        return torch.tensor(ex, dtype=torch.float32)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _default_extent()


class VertexDisplacementRGBDataset(Dataset):
    """Trig JPEG/PNG rasters.

    - ``normalization: fixed_linear`` — each channel: physical = vmin + (byte/255)*(vmax-vmin) with
      vmin,vmax = ±FIXED_PHYS_EXTENT; training targets ``pixel_values = physical / FIXED_PHYS_EXTENT``
      in [-1, 1]; FEM decode ``disp = pixel_values * extent`` with ``extent = FIXED_PHYS_EXTENT`` per axis.
    - ``extent_from_meta`` — legacy zero-mid JPEGs: ``enc = (byte/127.5) - 1`` divided by per-channel
      ``extent_R`` from the sidecar when present.
    """

    def __init__(self, cfg: VertexDisplacementRGBConfig) -> None:
        self.cfg = cfg
        self._paths: list[Path] = []
        if cfg.processed_dir.is_dir():
            paths: list[Path] = []
            for pat in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
                paths.extend(cfg.processed_dir.glob(pat))
            self._paths = sorted({p for p in paths if not p.name.endswith(".meta.json")})

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self._paths[index]
        from PIL import Image

        im = Image.open(path).convert("RGB")
        tw, th = self.cfg.image_size
        if im.size != (tw, th):
            try:
                resample = Image.Resampling.BICUBIC
            except AttributeError:
                resample = Image.BICUBIC
            im = im.resize((tw, th), resample)

        arr = np.asarray(im, dtype=np.float32)  # H, W, 3
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 3, H, W

        stem = path.stem
        meta_path = path.parent / f"{stem}.meta.json"
        meta = None
        if meta_path.is_file() and (self.cfg.include_fem_stress or self.cfg.normalization != "raw_enc"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = None

        if self.cfg.normalization == "fixed_linear":
            phys = FIXED_PHYS_MIN + (t / 255.0) * (FIXED_PHYS_MAX - FIXED_PHYS_MIN)
            pixel_values = (phys / FIXED_PHYS_EXTENT).clamp(-1.0, 1.0)
            extent = _fixed_extent_tensor()
        else:
            enc = (t / 127.5) - 1.0
            extent = _extent_from_meta(meta_path)
            if self.cfg.normalization == "raw_enc":
                pixel_values = enc.clamp(-10.0, 10.0)
            else:
                pixel_values = (enc / extent.clamp(min=1e-6).view(3, 1, 1)).clamp(-1.0, 1.0)

        out: dict[str, torch.Tensor | str] = {
            "pixel_values": pixel_values,
            "extent": extent,
            "path": str(path),
        }
        if self.cfg.include_fem_stress and meta:
            ex = meta.get("extra", {}) if isinstance(meta, dict) else {}
            grid = ex.get("fem_stress_grid") if isinstance(ex, dict) else None
            vals = grid.get("values") if isinstance(grid, dict) else None
            if isinstance(vals, list) and len(vals) >= tw * th:
                arrs = np.asarray(vals[: tw * th], dtype=np.float32).reshape(th, tw)
                s = torch.from_numpy(arrs).unsqueeze(0)  # 1,H,W
                denom = float(self.cfg.fem_stress_norm)
                if not np.isfinite(denom) or denom <= 0:
                    denom = float(meta.get("max_stress") or 1.0) if isinstance(meta, dict) else 1.0
                if not np.isfinite(denom) or denom <= 0:
                    denom = 1.0
                out["fem_stress"] = (s / denom).clamp(0.0, 10.0)
        return out
