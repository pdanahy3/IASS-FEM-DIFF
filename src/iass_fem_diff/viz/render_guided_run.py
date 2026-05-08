"""
Render a guided sampling run (step PNGs) as 3D surface images with a fixed camera.

Input: a folder produced by `sample-vertex-rgb-guided` containing `step_*.png` and `run.json`.
Output: a folder of rendered frames (one PNG per step).

Uses matplotlib (optional dependency).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iass_fem_diff.datasets.mesh_displacement_rgb import FIXED_PHYS_MAX, FIXED_PHYS_MIN


@dataclass
class RenderConfig:
    run_dir: Path
    out_dir: Path
    plane_size: float = 2.0
    disp_scale: float = 1.0
    disp_smooth: int = 1  # box blur radius (0 = off)
    color_mode: str = "z"  # z | rgb | fem_stress | fem_disp
    clim: tuple[float, float] | None = None  # color limits for scalar modes (fem_* or z)
    fem_every: int = 1  # when fem_* mode: compute every Nth rendered frame
    elev: float = 28.0
    azim: float = -50.0
    roll: float | None = None
    zlim: tuple[float, float] | None = None
    dpi: int = 160
    every: int = 1  # render every Nth step image


def _box_blur_2d(src: np.ndarray, W: int, H: int, radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return src
    # Separable box blur: horizontal then vertical, edge-clamped.
    a = src.reshape(H, W).astype(np.float32, copy=False)
    tmp = np.empty_like(a)
    out = np.empty_like(a)
    k = 2 * r + 1
    # horizontal
    for j in range(H):
        row = a[j]
        acc = 0.0
        for i in range(-r, r + 1):
            acc += row[min(W - 1, max(0, i))]
        tmp[j, 0] = acc / k
        for i in range(1, W):
            acc -= row[min(W - 1, max(0, i - r - 1))]
            acc += row[min(W - 1, max(0, i + r))]
            tmp[j, i] = acc / k
    # vertical
    for i in range(W):
        col = tmp[:, i]
        acc = 0.0
        for j in range(-r, r + 1):
            acc += col[min(H - 1, max(0, j))]
        out[0, i] = acc / k
        for j in range(1, H):
            acc -= col[min(H - 1, max(0, j - r - 1))]
            acc += col[min(H - 1, max(0, j + r))]
            out[j, i] = acc / k
    return out.reshape(H * W)


def _decode_fixed_linear_rgb_to_disp(rgb_u8: np.ndarray) -> np.ndarray:
    """rgb_u8 [H,W,3] -> disp [H,W,3] in physical units."""
    span = float(FIXED_PHYS_MAX - FIXED_PHYS_MIN)
    disp = float(FIXED_PHYS_MIN) + (rgb_u8.astype(np.float32) / 255.0) * span
    return disp


def _facecolors_white_to_pink_mag(mag_hw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Match viewer displacementMagToLinearRgb: white at min mag, pink at max.
    mag_hw: [H,W] scalar field -> facecolors [H-1,W-1,4] for plot_surface.
    """
    h, w = mag_hw.shape
    if h < 2 or w < 2:
        raise ValueError("mag field must be at least 2×2")
    m = 0.25 * (
        mag_hw[:-1, :-1] + mag_hw[:-1, 1:] + mag_hw[1:, :-1] + mag_hw[1:, 1:]
    )
    if hi <= lo:
        t = np.zeros_like(m, dtype=np.float32)
    else:
        t = ((m - lo) / (hi - lo)).astype(np.float32)
    t = np.clip(t, 0.0, 1.0)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pink = np.array([1.0, 0.4, 0.8], dtype=np.float32)
    rgb = (1.0 - t[..., np.newaxis]) * white + t[..., np.newaxis] * pink
    alpha = np.ones((*t.shape, 1), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=-1)


def render_guided_run(cfg: RenderConfig) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        raise ImportError("Rendering requires matplotlib. Install: pip install matplotlib") from e

    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Rendering requires Pillow. Install: pip install Pillow") from e

    run_dir = cfg.run_dir.resolve()
    out_dir = cfg.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_json_path = run_dir / "run.json"
    if run_json_path.is_file():
        try:
            run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
            # If present, prefer plane_size from the checkpoint metadata.
            # (Guided run stores phys_scale + fixed range; plane_size is not guaranteed here.)
            _ = run_meta
        except Exception:
            pass

    steps = sorted(run_dir.glob("step_*.png"))
    if not steps:
        raise FileNotFoundError(f"No step_*.png under {run_dir}")

    # Determine W,H
    im0 = Image.open(steps[0]).convert("RGB")
    W, H = im0.size
    if W < 2 or H < 2:
        raise ValueError("Step images must be at least 2×2")

    # Base XY grid (reference plane is XY; z=0).
    uu = np.linspace(0.0, 1.0, W, dtype=np.float32)
    vv = np.linspace(0.0, 1.0, H, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)
    BX = (U - 0.5) * float(cfg.plane_size)
    BY = (V - 0.5) * float(cfg.plane_size)

    r = max(0, int(cfg.disp_smooth))
    zmin = None
    zmax = None
    if cfg.zlim is not None:
        zmin, zmax = float(cfg.zlim[0]), float(cfg.zlim[1])

    color_mode = str(cfg.color_mode or "z").strip().lower()
    if color_mode not in {"z", "rgb", "fem_stress", "fem_disp"}:
        raise ValueError("color_mode must be one of: z, rgb, fem_stress, fem_disp")

    ref_cfg = None
    solve_ref = None
    if color_mode in {"fem_stress", "fem_disp"}:
        from iass_fem_diff.physics.reference_fem_fields import (
            ReferenceFEMConfig,
            solve_reference_fem_on_displacement_grid,
        )

        ref_cfg = ReferenceFEMConfig(plane_size=float(cfg.plane_size), gravity_total=1.0)
        solve_ref = solve_reference_fem_on_displacement_grid

    vmin = vmax = None
    if cfg.clim is not None:
        vmin, vmax = float(cfg.clim[0]), float(cfg.clim[1])

    for idx, p in enumerate(steps):
        if cfg.every > 1 and (idx % int(cfg.every)) != 0:
            continue

        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        disp = _decode_fixed_linear_rgb_to_disp(rgb)  # H,W,3 physical

        dx = disp[:, :, 0].reshape(H * W)
        dy = disp[:, :, 1].reshape(H * W)
        dz = disp[:, :, 2].reshape(H * W)
        if r > 0:
            dx = _box_blur_2d(dx, W, H, r)
            dy = _box_blur_2d(dy, W, H, r)
            dz = _box_blur_2d(dz, W, H, r)

        X = BX + float(cfg.disp_scale) * dx.reshape(H, W)
        Y = BY + float(cfg.disp_scale) * dy.reshape(H, W)
        Z = float(cfg.disp_scale) * dz.reshape(H, W)

        fig = plt.figure(figsize=(7.2, 6.0), dpi=int(cfg.dpi))
        ax = fig.add_subplot(111, projection="3d")
        facecolors = None
        cmap = "viridis"
        if color_mode == "rgb":
            # Per-face average of image RGB (plot_surface expects (H-1,W-1,4))
            fc = rgb.astype(np.float32) / 255.0  # H,W,3
            fc_face = 0.25 * (
                fc[:-1, :-1] + fc[:-1, 1:] + fc[1:, :-1] + fc[1:, 1:]
            )
            alpha = np.ones((H - 1, W - 1, 1), dtype=np.float32)
            facecolors = np.concatenate([fc_face, alpha], axis=2)
            cmap = None
        elif color_mode == "z":
            cmap = "viridis"
        else:
            # Reference FEM per frame (slow). Uses disp grid directly (physical units).
            # Cache by only recomputing every fem_every frames.
            if cfg.fem_every < 1:
                cfg.fem_every = 1
            if (idx % int(cfg.fem_every)) == 0:
                assert solve_ref is not None and ref_cfg is not None
                sol = solve_ref(disp, cfg=ref_cfg)
                if sol.get("valid", False):
                    field = sol["sigma"] if color_mode == "fem_stress" else sol["delta"]
                    field = np.asarray(field, dtype=np.float32)
                else:
                    field = np.zeros((H, W), dtype=np.float32)
                render_guided_run._last_fem_field = field  # type: ignore[attr-defined]
            field = getattr(render_guided_run, "_last_fem_field", np.zeros((H, W), dtype=np.float32))
            if color_mode == "fem_disp":
                lo = float(vmin) if vmin is not None else float(np.min(field))
                hi = float(vmax) if vmax is not None else float(np.max(field))
                if hi <= lo:
                    hi = lo + 1e-12
                facecolors = _facecolors_white_to_pink_mag(field, lo, hi)
                cmap = None
            else:
                mcell = 0.25 * (
                    field[:-1, :-1] + field[:-1, 1:] + field[1:, :-1] + field[1:, 1:]
                )
                if vmin is not None and vmax is not None:
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = plt.Normalize(
                        vmin=float(np.min(mcell)), vmax=float(np.max(mcell))
                    )
                facecolors = plt.cm.get_cmap("magma")(norm(mcell))
                cmap = None

        surf_kw: dict = {"linewidth": 0, "antialiased": True}
        # Custom facecolors are built for (H-1)×(W-1) quads; avoid rcount/ccount subsampling mismatch.
        if facecolors is None:
            surf_kw["rcount"] = min(200, H)
            surf_kw["ccount"] = min(200, W)
        ax.plot_surface(X, Y, Z, cmap=cmap, facecolors=facecolors, **surf_kw)
        ax.view_init(elev=float(cfg.elev), azim=float(cfg.azim), roll=cfg.roll)
        ax.set_axis_off()
        if zmin is not None and zmax is not None:
            ax.set_zlim(zmin, zmax)

        out_path = out_dir / p.name.replace(".png", "_3d.png")
        fig.tight_layout(pad=0.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


# Cache slot for FEM coloring (per-process).
render_guided_run._last_fem_field = None  # type: ignore[attr-defined]

