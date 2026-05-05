"""Load trig (or other) displacement RGB rasters + optional .meta.json for extents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class VertexDisplacementRGBConfig:
    processed_dir: Path
    image_size: tuple[int, int]
    normalization: str = "extent_from_meta"


def load_config(path: Path) -> VertexDisplacementRGBConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = raw["data"]
    w, h = data["image_size"]
    return VertexDisplacementRGBConfig(
        processed_dir=Path(data["processed_dir"]),
        image_size=(int(w), int(h)),
        normalization=str(data.get("normalization", "extent_from_meta")),
    )


def _default_extent() -> torch.Tensor:
    return torch.ones(3, dtype=torch.float32)


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
    """Trig JPEG/PNG rasters: ``enc = (byte/127.5) - 1`` per channel, normalized by extent.

    ``pixel_values`` are approximately in [-1, 1]; decode with ``disp = pixel_values * extent``.
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
        enc = (t / 127.5) - 1.0

        stem = path.stem
        meta_path = path.parent / f"{stem}.meta.json"
        extent = _extent_from_meta(meta_path)

        if self.cfg.normalization == "raw_enc":
            pixel_values = enc.clamp(-10.0, 10.0)
        else:
            pixel_values = (enc / extent.clamp(min=1e-6).view(3, 1, 1)).clamp(-1.0, 1.0)

        return {
            "pixel_values": pixel_values,
            "extent": extent,
            "path": str(path),
        }
