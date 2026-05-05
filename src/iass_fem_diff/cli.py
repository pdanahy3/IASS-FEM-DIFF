"""CLI for training stubs (`python -m iass_fem_diff.cli ...`)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from iass_fem_diff.datasets.fem_field_maps import FEMFieldMapsDataset, load_config as load_fem_cfg
from iass_fem_diff.datasets.mesh_displacement_rgb import (
    VertexDisplacementRGBDataset,
    load_config as load_disp_cfg,
)

app = typer.Typer(no_args_is_help=True, help="IASS-FEM-DIFF training scaffolding")
console = Console()


@app.command("train-vertex-rgb")
def train_vertex_rgb(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Train DDPM on trig RGB rasters with optional FEM proxy loss (see config ``fem``)."""
    cfg_path = config.resolve()
    repo_root = cfg_path.parent.parent
    disp_cfg = load_disp_cfg(cfg_path)
    proc = disp_cfg.processed_dir
    if not proc.is_absolute():
        proc = (repo_root / proc).resolve()
    disp_cfg = replace(disp_cfg, processed_dir=proc)
    ds = VertexDisplacementRGBDataset(disp_cfg)
    console.print(f"[green]Dataset[/green] {len(ds)} samples under {proc}")
    if len(ds) == 0:
        console.print("[red]No images found.[/red] Run: node scripts/generate-trig-surfaces.js --max N")
        raise typer.Exit(code=1)
    from iass_fem_diff.train.trig_diffusion import train_from_config

    train_from_config(cfg_path)


@app.command("train-fem-field")
def train_fem_field(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Stub: image-conditioned diffusion for FEM displacement or stress maps."""
    cfg = load_fem_cfg(config)
    ds = FEMFieldMapsDataset(cfg)
    console.print(
        f"[green]FEM field[/green] {cfg.field}: {len(ds)} paired samples "
        f"(in={cfg.input_images_dir}, out={cfg.target_images_dir})"
    )
    console.print(
        "[yellow]Stub:[/yellow] use image-conditioned diffusion; "
        "emit targets with viz.colormaps + metadata sidecars in preprocessing."
    )


@app.command("configs")
def list_configs(config_dir: Path = Path("configs")) -> None:
    """Print available YAML configs (default: ./configs)."""
    if not config_dir.is_dir():
        typer.echo(f"No directory: {config_dir}")
        raise typer.Exit(code=1)
    for p in sorted(config_dir.glob("*.yaml")):
        typer.echo(str(p))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
