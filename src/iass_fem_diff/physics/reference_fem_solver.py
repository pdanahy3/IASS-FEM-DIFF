"""
Reference FEM (linear elasticity) solver, vendored from the sibling project
`IASS-FEM_Diffusion` to keep this repo self-contained.

This is *not* differentiable and is intended for:
- preprocessing sidecars for visualization
- inference-time scoring / attaching FEM fields
- optional supervised training signals (precomputed)

Public API:
  solve_fem(xyz_grid, bc_mask, load_grid, E, nu, thickness=None) -> dict
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import skfem as fem
    from skfem import Basis, ElementTetP1, ElementVector, condense, solve
    from skfem.helpers import ddot, dot, sym_grad
    from skfem.mesh import MeshTet

    HAS_SKFEM = True
except ImportError:  # pragma: no cover
    HAS_SKFEM = False
    fem = None


def solve_fem(
    xyz_grid: np.ndarray,
    bc_mask: np.ndarray,
    load_grid: np.ndarray,
    *,
    E: float = 210e9,
    nu: float = 0.3,
    thickness: float | None = None,
) -> dict:
    """
    Solve 3D linear elasticity on a thin tet mesh built from a UV grid.

    Args:
        xyz_grid: [H, W, 3] node positions.
        bc_mask: [H, W] bool, True = fixed Dirichlet (all 3 dof).
        load_grid: [H, W, 3] nodal forces (N).
        E, nu: material parameters (Pa, unitless).
        thickness: optional thickness of the second layer offset.

    Returns:
        dict with keys: sigma [H,W], delta [H,W], u_vec [H,W,3], valid bool, message str
    """
    H, W, _ = xyz_grid.shape

    if not HAS_SKFEM:
        logger.warning("scikit-fem not installed — FEM skipped.")
        return _empty_result(H, W, valid=False, message="scikit-fem not installed")

    valid, msg = _check_geometry(xyz_grid)
    if not valid:
        logger.warning("FEM skipped — invalid geometry: %s", msg)
        return _empty_result(H, W, valid=False, message=msg)

    try:
        mesh, n_surf = _build_tet_mesh(xyz_grid, H, W, thickness=thickness)
        basis = Basis(mesh, ElementVector(ElementTetP1(), mesh.dim()))
        lam, mu = _lame(E, nu)

        @fem.BilinearForm
        def elasticity(u, v, _):
            e_u = sym_grad(u)
            e_v = sym_grad(v)
            div_u = e_u[0, 0] + e_u[1, 1] + e_u[2, 2]
            div_v = e_v[0, 0] + e_v[1, 1] + e_v[2, 2]
            return 2.0 * mu * ddot(e_u, e_v) + lam * div_u * div_v

        @fem.LinearForm
        def load(v, w):
            return dot(w["load"], v)

        A = elasticity.assemble(basis)
        load_vec = _nodal_load_to_global(load_grid, H, W, n_surf, basis.N)
        b = load.assemble(basis, load=basis.interpolate(load_vec))

        fixed_dofs = _fixed_dofs_from_bc_mask(basis, bc_mask, H, W, n_surf)
        if fixed_dofs.size == 0:
            return _empty_result(H, W, valid=False, message="no boundary conditions")

        condensed = condense(A, b, D=fixed_dofs)
        # skfem.condense API differs by version:
        # - older: returns (A_c, b_c)
        # - newer: returns (A_c, b_c, x, I) or similar 4-tuple
        if isinstance(condensed, tuple) and len(condensed) >= 2:
            A_c, b_c = condensed[0], condensed[1]
        else:  # pragma: no cover
            A_c, b_c = condensed
        u_global = solve(A_c, b_c)

        u_full = np.zeros(basis.N, dtype=np.float64)
        free = np.setdiff1d(np.arange(basis.N), fixed_dofs)
        u_full[free] = u_global

        u_nodes = u_full.reshape(-1, 3)
        u_surf = u_nodes[:n_surf]
        delta = np.linalg.norm(u_surf, axis=1).reshape(H, W)
        u_vec = u_surf.reshape(H, W, 3)

        sigma_vm = _von_mises_nodal(basis, mesh, u_full, lam, mu, H, W, n_surf)
        return {"sigma": sigma_vm, "delta": delta, "u_vec": u_vec, "valid": True, "message": "ok"}
    except Exception as e:  # pragma: no cover
        logger.exception("FEM solve failed")
        return _empty_result(H, W, valid=False, message=str(e))


def extract_von_mises(solution: dict) -> np.ndarray:
    if "sigma" in solution:
        return solution["sigma"]
    if "von_mises" in solution:
        return solution["von_mises"]
    raise KeyError("solution must contain 'sigma' or 'von_mises'")


def _build_tet_mesh(
    xyz_grid: np.ndarray, H: int, W: int, *, thickness: float | None = None
) -> Tuple["MeshTet", int]:
    points = xyz_grid.reshape(-1, 3).astype(np.float64)
    n_surf = H * W

    normals = np.zeros_like(points)
    for i in range(H):
        for j in range(W):
            k = i * W + j
            if i < H - 1 and j < W - 1:
                e1 = points[(i + 1) * W + j] - points[k]
                e2 = points[i * W + (j + 1)] - points[k]
                n = np.cross(e1, e2)
                ln = np.linalg.norm(n)
                if ln > 1e-14:
                    n /= ln
                normals[k] = n
            elif i > 0 and j > 0:
                normals[k] = normals[(i - 1) * W + (j - 1)]

    nnorm = np.linalg.norm(normals, axis=1, keepdims=True)
    bad = nnorm.ravel() < 1e-10
    if np.any(bad):
        default = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.any(~bad):
            default = np.mean(normals[~bad], axis=0)
            dn = np.linalg.norm(default)
            default = default / dn if dn > 1e-14 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normals[bad] = default

    if thickness is not None and thickness > 0:
        offset = float(thickness)
    else:
        scale = np.max(np.linalg.norm(points, axis=1)) - np.min(np.linalg.norm(points, axis=1)) + 1e-12
        offset = 1e-4 * scale

    points_second = points + offset * normals
    all_points = np.vstack([points, points_second])

    def idx(ii: int, jj: int) -> int:
        return ii * W + jj

    tets: list[list[int]] = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            a2, b2, c2, d2 = a + n_surf, b + n_surf, c + n_surf, d + n_surf
            tets.append([a, b, c, a2])
            tets.append([a2, b, c, b2])
            tets.append([a2, b2, c, c2])
            tets.append([b, d, c, b2])
            tets.append([b2, d, c, d2])
            tets.append([b2, d2, c, c2])
    t = np.array(tets, dtype=np.int32).T
    mesh = MeshTet(all_points.T, t)
    return mesh, n_surf


def _fixed_dofs_from_bc_mask(basis, bc_mask: np.ndarray, H: int, W: int, n_surf: int) -> np.ndarray:
    fixed_flat = np.where(bc_mask.reshape(-1))[0]
    dofs: list[int] = []
    for n in fixed_flat:
        if n < n_surf:
            dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])
    return np.unique(np.array(dofs, dtype=np.int64))


def _nodal_load_to_global(load_grid: np.ndarray, H: int, W: int, n_surf: int, n_dofs: int) -> np.ndarray:
    load_flat = load_grid.reshape(-1, 3).astype(np.float64)
    out = np.zeros(n_dofs, dtype=np.float64)
    n = min(3 * n_surf, n_dofs)
    out[:n] = load_flat.ravel()[:n]
    return out


def _von_mises_nodal(
    basis,
    mesh,
    u_full: np.ndarray,
    lam: float,
    mu: float,
    H: int,
    W: int,
    n_surf: int,
) -> np.ndarray:
    u_vec = u_full.reshape(-1, 3)
    n_el = mesh.t.shape[1]
    vm_el = np.zeros(n_el, dtype=np.float64)

    ref_grad = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    for e in range(n_el):
        verts = mesh.p[:, mesh.t[:, e]].T
        u_el = u_vec[mesh.t[:, e]].ravel()
        J = ref_grad.T @ verts
        detJ = np.linalg.det(J)
        if detJ <= 0:
            vm_el[e] = 0.0
            continue
        invJ = np.linalg.inv(J)
        grad_N = (invJ.T @ ref_grad.T).T

        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            B[0, i * 3] = grad_N[i, 0]
            B[1, i * 3 + 1] = grad_N[i, 1]
            B[2, i * 3 + 2] = grad_N[i, 2]
            B[3, i * 3] = grad_N[i, 1]
            B[3, i * 3 + 1] = grad_N[i, 0]
            B[4, i * 3 + 1] = grad_N[i, 2]
            B[4, i * 3 + 2] = grad_N[i, 1]
            B[5, i * 3] = grad_N[i, 2]
            B[5, i * 3 + 2] = grad_N[i, 0]

        eps = B @ u_el
        s = np.zeros(6, dtype=np.float64)
        tr = eps[0] + eps[1] + eps[2]
        s[0] = lam * tr + 2 * mu * eps[0]
        s[1] = lam * tr + 2 * mu * eps[1]
        s[2] = lam * tr + 2 * mu * eps[2]
        s[3], s[4], s[5] = mu * eps[3], mu * eps[4], mu * eps[5]
        sx, sy, sz = s[0], s[1], s[2]
        txy, tyz, txz = s[3], s[4], s[5]
        vm_el[e] = np.sqrt(sx * sx + sy * sy + sz * sz - sx * sy - sy * sz - sz * sx + 3 * (txy * txy + tyz * tyz + txz * txz))

    sigma_node = np.zeros(n_surf, dtype=np.float64)
    count = np.zeros(n_surf, dtype=np.float64)
    for e in range(n_el):
        for i in range(4):
            n = mesh.t[i, e]
            if n < n_surf:
                sigma_node[n] += vm_el[e]
                count[n] += 1
    count[count == 0] = 1
    sigma_node /= count
    return sigma_node.reshape(H, W)


def _lame(E: float, nu: float) -> Tuple[float, float]:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2.0 * (1 + nu))
    return lam, mu


def _check_geometry(xyz_grid: np.ndarray) -> Tuple[bool, str]:
    if not np.isfinite(xyz_grid).all():
        return False, "Non-finite values in xyz_grid (NaN or Inf)"

    H, W, _ = xyz_grid.shape
    min_area = 1e-12
    for i in range(H - 1):
        for j in range(W - 1):
            p00 = xyz_grid[i, j, :]
            p10 = xyz_grid[i, j + 1, :]
            p11 = xyz_grid[i + 1, j + 1, :]
            p01 = xyz_grid[i + 1, j, :]
            d1 = p11 - p00
            d2 = p10 - p01
            cross = np.cross(d1, d2)
            area = 0.5 * np.linalg.norm(cross)
            if area < min_area:
                return False, f"Degenerate element at UV cell ({i},{j}): area={area:.2e}"
            e1 = p10 - p00
            e2 = p01 - p00
            normal = np.cross(e1, e2)
            if np.linalg.norm(normal) < 1e-15:
                return False, f"Zero normal at UV cell ({i},{j}) — collinear edges"
    return True, "ok"


def _empty_result(H: int, W: int, *, valid: bool, message: str) -> dict:
    return {
        "sigma": np.zeros((H, W), dtype=np.float64),
        "delta": np.zeros((H, W), dtype=np.float64),
        "u_vec": np.zeros((H, W, 3), dtype=np.float64),
        "valid": valid,
        "message": message,
    }

