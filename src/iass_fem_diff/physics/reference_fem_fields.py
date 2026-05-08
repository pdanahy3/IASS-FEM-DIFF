"""
Helpers to compute reference FEM fields (von Mises stress + deflection) on the same UV grid
used by the trig viewer.

This uses the vendored scikit-fem solver in :mod:`iass_fem_diff.physics.reference_fem_solver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iass_fem_diff.physics.reference_fem_solver import solve_fem


@dataclass
class ReferenceFEMConfig:
    plane_size: float = 2.0
    young_modulus: float = 210e9
    poisson_ratio: float = 0.3
    gravity_total: float = 1.0
    thickness: float | None = None


def _base_xy_grid(H: int, W: int, plane_size: float) -> np.ndarray:
    """Return undeformed base grid positions [H,W,3] spanning [-plane/2, plane/2] in x/y."""
    if H < 2 or W < 2:
        raise ValueError("Grid must be at least 2×2")
    u = np.linspace(0.0, 1.0, W, dtype=np.float64)
    v = np.linspace(0.0, 1.0, H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    bx = (uu - 0.5) * float(plane_size)
    by = (vv - 0.5) * float(plane_size)
    bz = np.zeros_like(bx)
    return np.stack([bx, by, bz], axis=-1)


def _perimeter_bc_mask(H: int, W: int) -> np.ndarray:
    bc = np.zeros((H, W), dtype=bool)
    bc[0, :] = True
    bc[-1, :] = True
    bc[:, 0] = True
    bc[:, -1] = True
    return bc


def _uniform_gravity_load(H: int, W: int, gravity_total: float) -> np.ndarray:
    """Return load_grid [H,W,3] with total downward load distributed over interior nodes."""
    load = np.zeros((H, W, 3), dtype=np.float64)
    bc = _perimeter_bc_mask(H, W)
    interior = ~bc
    n = int(np.sum(interior))
    if n > 0:
        load[interior, 2] = -float(gravity_total) / n
    return load


def solve_reference_fem_on_displacement_grid(
    disp_grid: np.ndarray,
    *,
    cfg: ReferenceFEMConfig,
) -> dict:
    """
    Args:
        disp_grid: [H,W,3] displacement (dx,dy,dz) added on top of base plane.
        cfg: solver + boundary/load configuration.

    Returns:
        dict with keys compatible with solve_fem plus base_xyz.
    """
    disp = np.asarray(disp_grid, dtype=np.float64)
    if disp.ndim != 3 or disp.shape[2] != 3:
        raise ValueError(f"disp_grid must be [H,W,3], got {disp.shape}")
    H, W, _ = disp.shape
    base = _base_xy_grid(H, W, cfg.plane_size)
    xyz = base + disp
    bc_mask = _perimeter_bc_mask(H, W)
    load_grid = _uniform_gravity_load(H, W, cfg.gravity_total)
    out = solve_fem(
        xyz,
        bc_mask,
        load_grid,
        E=float(cfg.young_modulus),
        nu=float(cfg.poisson_ratio),
        thickness=cfg.thickness,
    )
    out["base_xyz"] = base
    return out

