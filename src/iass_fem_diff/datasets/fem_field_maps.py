"""Load (input_image, target_field_map) pairs for FEM displacement / stress diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class FEMFieldMapsConfig:
    input_images_dir: Path
    target_images_dir: Path
    image_size: tuple[int, int]
    field: str  # "displacement_magnitude" | "stress"


def load_config(path: Path) -> FEMFieldMapsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = raw["data"]
    w, h = data["image_size"]
    return FEMFieldMapsConfig(
        input_images_dir=Path(data["input_images_dir"]),
        target_images_dir=Path(data["target_images_dir"]),
        image_size=(int(w), int(h)),
        field=str(raw["field"]),
    )


class FEMFieldMapsDataset:
    """Stub: pair inputs and targets by shared stem; metadata via FieldStatsMetadata."""

    def __init__(self, cfg: FEMFieldMapsConfig) -> None:
        self.cfg = cfg
        self._pairs: list[tuple[Path, Path]] = []
        if cfg.input_images_dir.is_dir() and cfg.target_images_dir.is_dir():
            for p in sorted(cfg.input_images_dir.glob("*.png")):
                t = cfg.target_images_dir / p.name
                if t.is_file():
                    self._pairs.append((p, t))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError(
            "Load input/target tensors and optional metadata JSON; return training batch dict."
        )
