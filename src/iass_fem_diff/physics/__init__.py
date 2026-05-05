"""Differentiable proxies for structural behavior (training-time steering)."""

from iass_fem_diff.physics.fem_proxy import (
    bending_curvature_y,
    chord_relative_sag_z,
    structural_efficiency_loss,
)

__all__ = [
    "bending_curvature_y",
    "chord_relative_sag_z",
    "structural_efficiency_loss",
]
