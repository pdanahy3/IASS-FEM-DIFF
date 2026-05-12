"""CLI for training stubs (`python -m iass_fem_diff.cli ...`)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
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
def train_vertex_rgb(
    config: Path = typer.Argument(..., exists=True, readable=True),
    fem_ridge: bool = typer.Option(
        False,
        "--fem-ridge",
        help="Maximize Laplacian proxy (subtract lambda*proxy from loss); ridge-seeking vs smooth sag. Overrides YAML fem.maximize_laplacian_proxy.",
    ),
) -> None:
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

    train_from_config(cfg_path, maximize_laplacian_proxy=True if fem_ridge else None)


@app.command("sample-vertex-rgb")
def sample_vertex_rgb(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(
        Path("outputs/samples/displacement_vertex_rgb"),
        "--out-dir",
        help="Output directory for PNG + .meta.json samples (relative to repo root by default).",
    ),
    num: int = typer.Option(16, "--num", min=1, help="Number of samples to generate."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    steps: int | None = typer.Option(
        None, "--steps", help="Override DDPM inference steps (default: training timesteps)."
    ),
    device: str = typer.Option(
        "cuda", "--device", help="Device for inference: cuda|cpu"
    ),
) -> None:
    """Sample new trig displacement RGB images from a trained checkpoint."""
    ckpt_path = checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    out = out_dir
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    from iass_fem_diff.infer.trig_sample import sample_from_checkpoint

    sample_from_checkpoint(
        checkpoint_path=ckpt_path,
        out_dir=out,
        num_samples=int(num),
        seed=int(seed),
        image_size=(80, 80),
        num_inference_steps=steps,
        device=device,
    )
    console.print(f"[green]Wrote samples[/green] under {out}")


@app.command("sample-vertex-rgb-guided")
def sample_vertex_rgb_guided(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(
        Path("outputs/samples/displacement_vertex_rgb_guided"),
        "--out-dir",
        help="Output directory for step PNGs + run.json + metrics.*",
    ),
    seed_image: Path | None = typer.Option(
        None,
        "--seed-image",
        help="Optional 80×80 RGB displacement image (same encoding as training) to start from (img2img).",
    ),
    strength: float = typer.Option(
        0.0,
        "--strength",
        help="img2img strength in [0,1]. 0 disables seed init; 1 starts from near pure noise.",
    ),
    goal_image: Path | None = typer.Option(
        None,
        "--goal-image",
        help="Optional 80×80 goal image to steer toward during sampling.",
    ),
    goal_mix: float = typer.Option(
        0.0,
        "--goal-mix",
        help="Steering amount in [0,1]. Mixes model eps with eps implied by goal x0.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    steps: int = typer.Option(1000, "--steps", help="DDPM inference steps."),
    save_every: int = typer.Option(50, "--save-every", help="Save an intermediate PNG every N steps."),
    device: str = typer.Option("cuda", "--device", help="Device: cuda|cpu"),
) -> None:
    """Guided sampling with step-by-step outputs + optional seed/goal images."""
    ckpt_path = checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    out = out_dir if out_dir.is_absolute() else (repo_root / out_dir).resolve()
    from iass_fem_diff.infer.trig_guided_sample import GuidedRunConfig, run_guided_sampling

    run_guided_sampling(
        GuidedRunConfig(
            checkpoint_path=ckpt_path,
            out_dir=out,
            seed=int(seed),
            steps=int(steps),
            device=device,
            seed_image=seed_image.resolve() if seed_image else None,
            strength=float(strength),
            goal_image=goal_image.resolve() if goal_image else None,
            goal_mix=float(goal_mix),
            save_every=int(save_every),
        )
    )
    console.print(f"[green]Wrote guided run[/green] under {out}")


@app.command("render-guided-run-3d")
def render_guided_run_3d(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(
        None,
        "--out-dir",
        help="Output directory for rendered frames (default: <run_dir>/render_3d).",
    ),
    plane_size: float = typer.Option(2.0, "--plane-size", help="Reference plane size (XY)."),
    disp_scale: float = typer.Option(1.0, "--disp-scale", help="Multiplier on decoded displacements."),
    disp_smooth: int = typer.Option(1, "--disp-smooth", help="Box blur radius on decoded displacement (0=off)."),
    color_mode: str = typer.Option(
        "z",
        "--color-mode",
        help="Coloring: z | rgb | fem_stress | fem_disp. fem_* runs reference FEM per frame (slow).",
    ),
    clim_min: float | None = typer.Option(None, "--clim-min", help="Color min for scalar modes."),
    clim_max: float | None = typer.Option(None, "--clim-max", help="Color max for scalar modes."),
    fem_every: int = typer.Option(1, "--fem-every", help="When fem_* mode: compute FEM every Nth rendered frame."),
    elev: float = typer.Option(28.0, "--elev", help="Camera elevation angle."),
    azim: float = typer.Option(-50.0, "--azim", help="Camera azimuth angle."),
    every: int = typer.Option(1, "--every", help="Render every Nth step frame."),
    dpi: int = typer.Option(160, "--dpi", help="Output DPI (affects resolution)."),
) -> None:
    """Render `step_*.png` from a guided run folder as fixed-view 3D surface frames."""
    repo_root = Path(__file__).resolve().parents[2]
    rdir = run_dir.resolve()
    odir = out_dir
    if odir is None:
        odir = rdir / "render_3d"
    if not odir.is_absolute():
        odir = (repo_root / odir).resolve()
    from iass_fem_diff.viz.render_guided_run import RenderConfig, render_guided_run

    render_guided_run(
        RenderConfig(
            run_dir=rdir,
            out_dir=odir,
            plane_size=float(plane_size),
            disp_scale=float(disp_scale),
            disp_smooth=int(disp_smooth),
            color_mode=str(color_mode),
            clim=None if (clim_min is None or clim_max is None) else (float(clim_min), float(clim_max)),
            fem_every=int(fem_every),
            elev=float(elev),
            azim=float(azim),
            every=int(every),
            dpi=int(dpi),
        )
    )
    console.print(f"[green]Wrote 3D frames[/green] under {odir}")


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


@app.command("evaluate-inference-samples")
def evaluate_inference_samples(
    samples_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Folder of generated displacement PNG/JPEG images (e.g. outputs/samples/run1).",
    ),
    dataset_dir: Path | None = typer.Option(
        None,
        "--dataset-dir",
        help="Trig training folder for nearest-neighbor MSE (design-language distance); "
        "defaults to repo data/processed/displacement_rgb/trig when present.",
    ),
    skip_nearest_dataset: bool = typer.Option(
        False,
        "--skip-nearest-dataset",
        help="Do not load training rasters; skip design-language MSE.",
    ),
    out_json: Path | None = typer.Option(
        None,
        "--out-json",
        help="Write full metrics JSON (default: <samples_dir>/eval_fem_metrics.json).",
    ),
    out_csv: Path | None = typer.Option(
        None,
        "--out-csv",
        help="Write per-sample CSV (default: <samples_dir>/eval_fem_metrics.csv).",
    ),
    plane_size: float = typer.Option(2.0, "--plane-size", help="Must match training / inference FEM."),
    gravity_total: float = typer.Option(1.0, "--gravity-total"),
    curvature_span: int = typer.Option(1, "--curvature-span", help="FEM proxy Laplacian span (training fem.curvature_span)."),
    dataset_limit: int = typer.Option(
        0,
        "--dataset-limit",
        help="Load at most N training images for nearest-neighbor (0 = all).",
        min=0,
    ),
    width: int = typer.Option(80, "--width", min=2),
    height: int = typer.Option(80, "--height", min=2),
) -> None:
    """
    Reference FEM (scikit-fem) + FEM proxy loss + optional nearest-dataset MSE on inference outputs.

    Checkpoints do not store historical proxy values; proxy loss is recomputed here on each image.
    """
    repo_root = Path(__file__).resolve().parents[2]
    default_ds = repo_root / "data" / "processed" / "displacement_rgb" / "trig"
    if skip_nearest_dataset:
        ds = None
    elif dataset_dir is not None:
        ds = dataset_dir.resolve() if dataset_dir.is_absolute() else (repo_root / dataset_dir).resolve()
    elif default_ds.is_dir():
        ds = default_ds
    else:
        ds = None

    from iass_fem_diff.eval.inference_eval import evaluate_inference_folder, summarize_for_table, write_csv

    summary = evaluate_inference_folder(
        samples_dir.resolve(),
        dataset_dir=ds,
        image_size=(int(width), int(height)),
        plane_size=float(plane_size),
        gravity_total=float(gravity_total),
        curvature_span=int(curvature_span),
        dataset_limit=int(dataset_limit),
    )

    jpath = out_json or (samples_dir.resolve() / "eval_fem_metrics.json")
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {jpath}")

    cpath = out_csv or (samples_dir.resolve() / "eval_fem_metrics.csv")
    write_csv(summary["per_sample"], cpath)
    console.print(f"[green]Wrote[/green] {cpath}")

    tbl = summarize_for_table(summary)
    console.print("[bold]Summary (paper-friendly)[/bold]")
    for k, v in tbl.items():
        console.print(f"  {k}: {v}")


@app.command("configs")
def list_configs(config_dir: Path = Path("configs")) -> None:
    """Print available YAML configs (default: ./configs)."""
    if not config_dir.is_dir():
        typer.echo(f"No directory: {config_dir}")
        raise typer.Exit(code=1)
    for p in sorted(config_dir.glob("*.yaml")):
        typer.echo(str(p))


@app.command("precompute-fem-trig")
def precompute_fem_trig(
    processed_dir: Path = typer.Option(
        Path("data/processed/displacement_rgb/trig"),
        "--processed-dir",
        help="Directory of trig images (jpg/png) + .meta.json sidecars.",
    ),
    plane_size: float = typer.Option(2.0, "--plane-size", help="Base plane size used by viewer."),
    gravity_total: float = typer.Option(1.0, "--gravity-total", help="Total downward load distributed on interior nodes."),
    young_modulus: float = typer.Option(210e9, "--E", help="Young's modulus (Pa)."),
    poisson_ratio: float = typer.Option(0.3, "--nu", help="Poisson ratio."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing FEM fields in sidecars."),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Process at most N images after sharding/stride filtering (0 = no limit).",
        min=0,
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        help="Only process every k-th image after sorting (useful for quick subsets).",
        min=1,
    ),
    shard: int = typer.Option(
        0,
        "--shard",
        help="Shard index in [0, num_shards-1]. Run multiple shards in parallel terminals.",
        min=0,
    ),
    num_shards: int = typer.Option(
        1,
        "--num-shards",
        help="Number of shards to split the workload into.",
        min=1,
    ),
) -> None:
    """
    Compute reference FEM fields (von Mises + deflection) for trig displacement rasters.

    Writes into `<stem>.meta.json`:
      - max_stress / avg_stress (from von Mises)
      - extra.fem_stress_grid: { values, width, height, units }
      - extra.fem_disp_grid: { values, width, height, units }
    """
    from iass_fem_diff.physics.reference_fem_fields import (
        ReferenceFEMConfig,
        solve_reference_fem_on_displacement_grid,
    )
    from iass_fem_diff.datasets.mesh_displacement_rgb import (
        FIXED_PHYS_EXTENT,
        FIXED_PHYS_MAX,
        FIXED_PHYS_MIN,
    )

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required. Install with: pip install 'iass-fem-diff[train]'") from e

    proc = processed_dir.resolve()
    if not proc.is_dir():
        console.print(f"[red]No directory[/red] {proc}")
        raise typer.Exit(code=1)

    cfg = ReferenceFEMConfig(
        plane_size=float(plane_size),
        gravity_total=float(gravity_total),
        young_modulus=float(young_modulus),
        poisson_ratio=float(poisson_ratio),
    )

    imgs: list[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(proc.glob(pat))
    imgs = [p for p in imgs if not p.name.endswith(".meta.json")]
    if not imgs:
        console.print(f"[red]No images found[/red] under {proc}")
        raise typer.Exit(code=1)
    imgs = sorted(imgs)
    # Shard + stride filtering to keep large folders manageable.
    if num_shards < 1:
        num_shards = 1
    if shard < 0 or shard >= num_shards:
        console.print(f"[red]Invalid shard[/red] shard={shard} must be in [0,{num_shards-1}]")
        raise typer.Exit(code=1)
    if stride < 1:
        stride = 1
    imgs = [p for i, p in enumerate(imgs) if (i % num_shards) == shard]
    if stride > 1:
        imgs = imgs[::stride]
    if limit and limit > 0:
        imgs = imgs[:limit]
    console.print(
        f"Precompute FEM on {len(imgs)} images (shard {shard}/{num_shards}, stride={stride}, limit={limit})."
    )

    span = FIXED_PHYS_MAX - FIXED_PHYS_MIN
    done = 0
    skipped = 0
    for img_path in imgs:
        meta_path = img_path.with_name(f"{img_path.stem}.meta.json")
        if not meta_path.is_file():
            skipped += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ex = meta.get("extra", {}) or {}
        if not overwrite and ("fem_stress_grid" in ex or "fem_disp_grid" in ex):
            skipped += 1
            continue

        im = Image.open(img_path).convert("RGB")
        arr = np.asarray(im, dtype=np.float64)  # H,W,3 bytes
        disp = FIXED_PHYS_MIN + (arr / 255.0) * span  # H,W,3 physical

        sol = solve_reference_fem_on_displacement_grid(disp, cfg=cfg)
        if not sol.get("valid", False):
            skipped += 1
            continue
        sigma = np.asarray(sol["sigma"], dtype=np.float64)
        delta = np.asarray(sol["delta"], dtype=np.float64)
        H, W = sigma.shape

        meta["max_stress"] = float(np.max(sigma)) if sigma.size else 0.0
        meta["avg_stress"] = float(np.mean(sigma)) if sigma.size else 0.0
        meta["max_displacement"] = float(np.max(delta)) if delta.size else 0.0
        meta["avg_displacement"] = float(np.mean(delta)) if delta.size else 0.0

        ex["fem_solver"] = {
            "name": "reference_scfem_linear_elasticity",
            "plane_size": cfg.plane_size,
            "gravity_total": cfg.gravity_total,
            "young_modulus": cfg.young_modulus,
            "poisson_ratio": cfg.poisson_ratio,
        }
        ex["fem_stress_grid"] = {
            "values": sigma.reshape(-1).tolist(),
            "width": int(W),
            "height": int(H),
            "units": "Pa",
            "field": "von_mises",
        }
        ex["fem_disp_grid"] = {
            "values": delta.reshape(-1).tolist(),
            "width": int(W),
            "height": int(H),
            "units": "m",
            "field": "displacement_magnitude",
        }
        meta["extra"] = ex
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        done += 1
        if done % 25 == 0:
            console.print(f"… {done} updated")

    console.print(f"[green]Updated[/green] {done} sidecars  [yellow]skipped[/yellow] {skipped}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
