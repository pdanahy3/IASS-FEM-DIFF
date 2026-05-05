"""
FEM-style proxies aligned with the trig viewer: pinned row 0 and row H−1, chord-relative
vertical sag w*, and a uniform-gravity bending measure ∂²w*/∂y² (row direction).

Used as a differentiable penalty on predicted clean images so diffusion training steers
toward lower bending (structurally lighter shapes) while the noise MSE term keeps samples
close to the data manifold.
"""

from __future__ import annotations

import torch


def chord_relative_sag_z(zz: torch.Tensor) -> torch.Tensor:
    """Chord-relative sag per column; zero on first and last rows.

    Args:
        zz: (B, H, W) vertical displacement component (or any scalar field on the grid).

    Returns:
        w_star with same shape; w_star[:, 0] = w_star[:, -1] = 0 when H >= 2.
    """
    if zz.dim() != 3:
        raise ValueError(f"Expected zz (B,H,W), got {tuple(zz.shape)}")
    _b, h, _w = zz.shape
    if h < 2:
        return torch.zeros_like(zz)
    z0 = zz[:, 0:1, :]
    z1 = zz[:, -1:, :]
    j = torch.arange(h, device=zz.device, dtype=zz.dtype).view(1, h, 1) / (h - 1)
    chord = (1.0 - j) * z0 + j * z1
    return zz - chord


def bending_curvature_y(w_star: torch.Tensor, *, hy: float = 1.0) -> torch.Tensor:
    """Second central difference along row (dim=1); zeros on boundary rows.

    Args:
        w_star: (B, H, W)
        hy: Row spacing in physical units (use 1.0 if displacements are normalized).

    Returns:
        (B, H, W) with interior filled, boundary 0.
    """
    if w_star.dim() != 3:
        raise ValueError(f"Expected w_star (B,H,W), got {tuple(w_star.shape)}")
    _b, h, _w = w_star.shape
    out = torch.zeros_like(w_star)
    if h < 3:
        return out
    hy2 = hy * hy
    if hy2 < 1e-24:
        return out
    w_m = w_star[:, :-2, :]
    w_c = w_star[:, 1:-1, :]
    w_p = w_star[:, 2:, :]
    out[:, 1:-1, :] = (w_p - 2.0 * w_c + w_m) / hy2
    return out


def structural_efficiency_loss(
    disp_bchw: torch.Tensor,
    *,
    vertical_channel: int = 2,
    hy: float = 1.0,
) -> torch.Tensor:
    """Mean squared bending-curvature proxy on chord-relative vertical sag.

    Args:
        disp_bchw: Decoded displacements (B, 3, H, W) — same units as trig RGB decode.
        vertical_channel: Index of out-of-plane direction (B channel = z in trig export).
        hy: Row spacing for curvature (match viewer ``plane / (H-1)`` if using physical units).

    Returns:
        Scalar loss (averaged over batch and grid).
    """
    if disp_bchw.dim() != 4 or disp_bchw.size(1) != 3:
        raise ValueError(f"Expected disp (B,3,H,W), got {tuple(disp_bchw.shape)}")
    zz = disp_bchw[:, vertical_channel, :, :]
    w_star = chord_relative_sag_z(zz)
    kappa = bending_curvature_y(w_star, hy=hy)
    return (kappa ** 2).mean()
