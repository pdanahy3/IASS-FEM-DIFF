"""Colormaps for FEM raster exports (training targets and Rhino-aligned inference)."""

from __future__ import annotations

import numpy as np


def displacement_magnitude_to_rgb(
    magnitude: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map scalar displacement magnitude to RGB: white (low) → pink (high).

    `magnitude` is 2D float array. Returns uint8 HxWx3 suitable for PNG.
    """
    m = np.asarray(magnitude, dtype=np.float64)
    lo = float(np.nanmin(m)) if vmin is None else float(vmin)
    hi = float(np.nanmax(m)) if vmax is None else float(vmax)
    if hi <= lo:
        t = np.zeros_like(m, dtype=np.float64)
    else:
        t = (m - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    # White (1,1,1) to pink (1,0.4,0.8) — adjust to taste
    white = np.array([1.0, 1.0, 1.0])
    pink = np.array([1.0, 0.4, 0.8])
    rgb = (1.0 - t[..., None]) * white + t[..., None] * pink
    return (rgb * 255.0).round().astype(np.uint8)


def stress_signed_to_rgb(
    stress: np.ndarray,
    *,
    vlim: float | None = None,
) -> np.ndarray:
    """Map signed stress to RGB: blue (compression) ← white (0) → red (tension).

    `stress` is 2D float; sign convention must match your FEM export.
    """
    s = np.asarray(stress, dtype=np.float64)
    lim = float(np.nanmax(np.abs(s))) if vlim is None else float(vlim)
    if lim <= 0:
        lim = 1.0
    t = np.clip(s / lim, -1.0, 1.0)
    rgb = np.zeros(t.shape + (3,), dtype=np.float64)
    pos = t > 0
    neg = t < 0
    # Positive: white → red
    a = np.clip(t, 0.0, 1.0)
    rgb[pos] = np.stack(
        [np.ones_like(t[pos]), 1.0 - a[pos], 1.0 - a[pos]], axis=-1
    )
    # Negative: white → blue
    b = np.clip(-t, 0.0, 1.0)
    rgb[neg] = np.stack(
        [1.0 - b[neg], 1.0 - b[neg], np.ones_like(t[neg])], axis=-1
    )
    neu = ~(pos | neg)
    rgb[neu] = 1.0
    return (rgb * 255.0).round().astype(np.uint8)
