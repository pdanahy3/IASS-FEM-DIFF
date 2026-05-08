"""Visualization helpers (offline renders, plots)."""

from iass_fem_diff.viz.render_guided_run import RenderConfig, render_guided_run

__all__ = ["RenderConfig", "render_guided_run"]

from iass_fem_diff.viz.colormaps import displacement_magnitude_to_rgb, stress_signed_to_rgb

__all__ = ["displacement_magnitude_to_rgb", "stress_signed_to_rgb"]
