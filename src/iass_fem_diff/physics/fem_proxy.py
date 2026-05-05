"""
FEM-style proxies aligned with the trig viewer: pinned boundaries on all four edges
(first/last row and first/last column in index space), Coons patch reference surface,
and a bending proxy Δw* ≈ ∂²w*/∂x² + ∂²w*/∂y² (central stencil, optional span).

Used as a differentiable penalty on predicted clean images so diffusion training steers
toward lower bending while the noise MSE term keeps samples close to the data manifold.
"""

from __future__ import annotations

import torch


def four_edge_relative_sag_z(zz: torch.Tensor) -> torch.Tensor:
    """Interior sag vs Coons bilinear boundary blend; zero on all four edges.

    Args:
        zz: (B, H, W) vertical displacement (row index ↔ v, column ↔ u).

    Returns:
        w_star with same shape; boundary rows/columns match zero when W,H >= 2.
    """
    if zz.dim() != 3:
        raise ValueError(f"Expected zz (B,H,W), got {tuple(zz.shape)}")
    _b, h, w = zz.shape
    if h < 2 or w < 2:
        return torch.zeros_like(zz)
    v = torch.arange(h, device=zz.device, dtype=zz.dtype).view(1, h, 1) / (h - 1)
    u = torch.arange(w, device=zz.device, dtype=zz.dtype).view(1, 1, w) / (w - 1)
    bottom = zz[:, 0:1, :]
    top = zz[:, -1:, :]
    left = zz[:, :, 0:1]
    right = zz[:, :, -1:]
    z00 = zz[:, 0:1, 0:1]
    z_w0 = zz[:, 0:1, -1:]
    z_0h = zz[:, -1:, 0:1]
    z_wh = zz[:, -1:, -1:]
    surf = (1.0 - v) * bottom + v * top + (1.0 - u) * left + u * right
    surf = (
        surf
        - (1.0 - u) * (1.0 - v) * z00
        - u * (1.0 - v) * z_w0
        - (1.0 - u) * v * z_0h
        - u * v * z_wh
    )
    return zz - surf


def chord_relative_sag_z(zz: torch.Tensor) -> torch.Tensor:
    """Deprecated alias for :func:`four_edge_relative_sag_z`."""
    return four_edge_relative_sag_z(zz)


def bending_laplacian(
    w_star: torch.Tensor,
    *,
    hx: float,
    hy: float,
    span: int = 1,
) -> torch.Tensor:
    """Central Laplacian on interior; zero on boundary band of width ``span``."""
    if w_star.dim() != 3:
        raise ValueError(f"Expected w_star (B,H,W), got {tuple(w_star.shape)}")
    _b, h, w = w_star.shape
    sp = max(1, int(span))
    out = torch.zeros_like(w_star)
    if h < 2 * sp + 1 or w < 2 * sp + 1:
        return out
    hx2 = (sp * hx) ** 2
    hy2 = (sp * hy) ** 2
    if hx2 < 1e-24 or hy2 < 1e-24:
        return out
    c = w_star[:, sp : h - sp, sp : w - sp]
    d2x = (
        w_star[:, sp : h - sp, 2 * sp : w]
        - 2.0 * c
        + w_star[:, sp : h - sp, 0 : w - 2 * sp]
    ) / hx2
    d2y = (
        w_star[:, 2 * sp : h, sp : w - sp]
        - 2.0 * c
        + w_star[:, 0 : h - 2 * sp, sp : w - sp]
    ) / hy2
    out[:, sp : h - sp, sp : w - sp] = d2x + d2y
    return out


def bending_curvature_y(w_star: torch.Tensor, *, hy: float = 1.0) -> torch.Tensor:
    """Deprecated: use :func:`bending_laplacian` with ``span=1`` and set ``hx``."""
    return bending_laplacian(w_star, hx=hy, hy=hy, span=1)


def structural_efficiency_loss(
    disp_bchw: torch.Tensor,
    *,
    vertical_channel: int = 2,
    hx: float = 1.0,
    hy: float = 1.0,
    span: int = 1,
) -> torch.Tensor:
    """Mean squared Laplacian proxy on four-edge-relative sag w*."""
    if disp_bchw.dim() != 4 or disp_bchw.size(1) != 3:
        raise ValueError(f"Expected disp (B,3,H,W), got {tuple(disp_bchw.shape)}")
    zz = disp_bchw[:, vertical_channel, :, :]
    w_star = four_edge_relative_sag_z(zz)
    kappa = bending_laplacian(w_star, hx=hx, hy=hy, span=span)
    return (kappa**2).mean()
