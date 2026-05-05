"""Sidecar metadata for raster exports (displacement + stress aggregates)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FieldStatsMetadata:
    """Stored next to each output image (e.g. `case_001.png.meta.json`)."""

    max_displacement: float
    avg_displacement: float
    max_stress: float
    avg_stress: float
    # Optional hooks for Rhino / traceability
    extra: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", None)
        out = {k: v for k, v in d.items() if v is not None}
        if extra:
            out["extra"] = extra
        return out

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> FieldStatsMetadata:
        extra = data.pop("extra", None)
        return cls(extra=extra, **data)


def write_metadata(image_path: Path, meta: FieldStatsMetadata) -> Path:
    """Write `<stem>.meta.json` beside the image (e.g. `case.png` → `case.meta.json`)."""
    out = image_path.with_name(f"{image_path.stem}.meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta.to_json_dict(), indent=2), encoding="utf-8")
    return out


def read_metadata(path: Path) -> FieldStatsMetadata:
    text = path.read_text(encoding="utf-8")
    return FieldStatsMetadata.from_json_dict(json.loads(text))
