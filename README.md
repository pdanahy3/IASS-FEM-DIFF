# IASS-FEM-DIFF

Training and inference scaffolding for image-based diffusion on shell **displacement rasters** (RGB = \(d_x,d_y,d_z\)) and optional **FEM-derived fields**, with a browser **3D viewer** and offline **reference FEM** (scikit-fem) for visualization and dataset enrichment.

## Quick start

```powershell
cd IASS-FEM-DIFF
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[train]"
npm install
python -m iass_fem_diff.cli --help
```

- **Trig data:** `node scripts/generate-trig-surfaces.js --max N`
- **Train:** `python -m iass_fem_diff.cli train-vertex-rgb configs/displacement_vertex_rgb.yaml`
- **Sample:** `python -m iass_fem_diff.cli sample-vertex-rgb <checkpoint.pt> --out-dir outputs/samples/run1`

GPU/CUDA notes and troubleshooting remain as in earlier docs; use a CUDA-enabled PyTorch wheel if `torch.cuda.is_available()` is `False`.

---

## Command-line interface

Entry point: **`python -m iass_fem_diff.cli`** (or **`iass-fem-diff`** after `pip install -e .`).

| Command | Purpose |
|--------|---------|
| `train-vertex-rgb` | Train DDPM on trig displacement RGB rasters (optional FEM proxy loss + optional 4th channel for precomputed stress). |
| `sample-vertex-rgb` | Batch sample PNG + `.meta.json` from a checkpoint (includes reference FEM summary on final outputs). |
| `sample-vertex-rgb-guided` | Single-sample guided run: step PNGs, `run.json`, `metrics.csv`, optional seed/goal images. |
| `precompute-fem-trig` | Offline reference linear-elasticity FEM → writes `fem_stress_grid` / `fem_disp_grid` into each `.meta.json`. |
| `render-guided-run-3d` | Turn a guided run’s `step_*.png` into fixed-camera 3D surface PNGs (matplotlib). |
| `train-fem-field` | Stub: lists FEM field map dataset (full training not wired). |
| `configs` | Print `configs/*.yaml` paths. |

### `train-vertex-rgb`

```text
python -m iass_fem_diff.cli train-vertex-rgb <config.yaml>
```

- **Config:** `configs/displacement_vertex_rgb.yaml` — `data.normalization` (`fixed_linear` vs legacy `extent_from_meta`), `data.include_fem_stress`, `data.fem_stress_norm`, `training.*`, `diffusion.*`, `fem.*` (proxy loss weight, plane size, curvature span).
- **Loss:** noise MSE + optional `fem.loss_weight` × differentiable **Coons sag + Laplacian proxy** (`src/iass_fem_diff/physics/fem_proxy.py`).
- **Checkpoints:** `outputs/checkpoints/.../unet_epoch_*.pt` (includes `vertex_rgb_phys_scale`, optional channel count inferred from weights).

### `sample-vertex-rgb`

```text
python -m iass_fem_diff.cli sample-vertex-rgb <checkpoint.pt> [--out-dir DIR] [--num N] [--seed S] [--steps T] [--device cuda|cpu]
```

- Writes `sample_XXXX.png` + `sample_XXXX.meta.json` (fixed-linear RGB encoding, reference FEM fields when solve succeeds).

### `sample-vertex-rgb-guided`

```text
python -m iass_fem_diff.cli sample-vertex-rgb-guided <checkpoint.pt> \
  [--out-dir DIR] [--seed-image PATH] [--strength 0..1] [--goal-image PATH] [--goal-mix 0..1] \
  [--seed S] [--steps T] [--save-every N] [--device cuda|cpu]
```

- **`--seed-image` + `--strength`:** img2img-style start (decode seed → forward noise to a starting timestep, then denoise). **`--strength 0`** disables the seed.
- **`--goal-image` + `--goal-mix`:** mixes the model’s predicted noise with the noise implied by the goal \(x_0\) at each step (steering).
- **Outputs:** `step_*_t*.png`, `final.png`, `run.json`, `metrics.csv`, optional `metrics.png` (if matplotlib available), `final_fem.json`.

### `precompute-fem-trig`

```text
python -m iass_fem_diff.cli precompute-fem-trig \
  [--processed-dir DIR] [--plane-size F] [--gravity-total F] [--E Pa] [--nu μ] [--overwrite] \
  [--limit N] [--stride K] [--shard I] [--num-shards M]
```

- Requires **`scikit-fem`** (included in `pip install -e ".[train]"`).
- Updates each `<stem>.meta.json` with `extra.fem_stress_grid`, `extra.fem_disp_grid`, and aggregate stress/displacement stats.
- **Sharding:** run the same command in parallel with `--num-shards M --shard 0..M-1` to saturate CPU cores on large folders.

### `render-guided-run-3d`

```text
python -m iass_fem_diff.cli render-guided-run-3d <run_dir> \
  [--out-dir DIR] [--plane-size F] [--disp-scale F] [--disp-smooth R] \
  [--color-mode z|rgb|fem_stress|fem_disp] [--clim-min A --clim-max B] [--fem-every N] \
  [--elev °] [--azim °] [--every N] [--dpi D]
```

- **`--color-mode`:** height (`z`), image RGB (`rgb`), reference von Mises (`fem_stress`), reference |disp| with **white→pink** (`fem_disp`). `fem_*` modes solve FEM per frame (slow); use `--fem-every` to skip frames.
- Default output: `<run_dir>/render_3d` if `--out-dir` omitted.

### `configs`

```text
python -m iass_fem_diff.cli configs [--config-dir path]
```

### `train-fem-field`

Stub only — prints dataset size and reminds to wire image-conditioned diffusion for FEM maps.

---

## Node.js scripts (`package.json`)

| Script | Command | Role |
|--------|---------|------|
| Generate trig JPEGs + meta | `npm run generate-trig-surfaces` or `node scripts/generate-trig-surfaces.js` | Synthetic \(80\times80\) surfaces; **fixed linear** RGB map \([-15,15]\) per channel → bytes \([0,255]\). |
| Build 3D viewer HTML | `npm run view-trig-surface` or `node scripts/view-trig-surface.js` | Writes `scripts/trig-viewer.html` (Three.js). |

- **Viewer:** serve repo root (`npx serve .`) and open `/scripts/trig-viewer.html`; supports fixed-linear decode, optional sidecar FEM stress/deflection grids, disp smoothing, gravity arrows (Z-up), anchor polylines.

---

## Python package layout & features

```
src/iass_fem_diff/
  cli.py                      # Typer CLI (all commands above)
  datasets/
    mesh_displacement_rgb.py  # Trig rasters + optional fem_stress channel from meta
    fem_field_maps.py         # FEM map dataset stub
  infer/
    trig_sample.py            # Standard batch sampling
    trig_guided_sample.py     # Guided / transparent sampling
  train/
    trig_diffusion.py         # DDPM training loop
  physics/
    fem_proxy.py              # Differentiable Coons + Laplacian proxy (training steering)
    reference_fem_solver.py # Vendored scikit-fem linear elasticity
    reference_fem_fields.py   # Perimeter BC + gravity load; solve on displacement grid
  viz/
    render_guided_run.py      # Matplotlib 3D frames from guided runs
    colormaps.py              # FEM colormap conventions
  io/metadata.py              # Sidecar JSON helpers
```

- **Encoding:** displacement channels use a **global** physical range (default ±15) mapped linearly to RGB bytes; see `data/README.md` and `configs/displacement_vertex_rgb.yaml`.
- **FEM:** (1) **Proxy** for training gradients; (2) **Reference solve** for precompute, inference meta, and offline renders — not the same as backprop through a full FE system.

---

## Scope and roadmap

Rhino / Grasshopper integration is **out of scope** for this repo’s current code; see `docs/RHINO_ROADMAP.md` if present.

## License

Add a `LICENSE` when you publish.
