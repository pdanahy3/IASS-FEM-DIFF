"""Physics utilities: differentiable proxies + optional reference FEM solver."""

from iass_fem_diff.physics.fem_proxy import (
    bending_curvature_y,
    bending_laplacian,
    chord_relative_sag_z,
    four_edge_relative_sag_z,
    structural_efficiency_loss,
)
from iass_fem_diff.physics.reference_fem_solver import solve_fem as solve_reference_fem

__all__ = [
    "bending_curvature_y",
    "bending_laplacian",
    "chord_relative_sag_z",
    "four_edge_relative_sag_z",
    "structural_efficiency_loss",
    "solve_reference_fem",
]
