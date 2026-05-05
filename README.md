# IASS-FEM-DIFF

Training and checkpoint layout for image-based diffusion models on shell displacement and FEM-derived fields, with a future path to Rhino 3D inference and multi-model agent guidance.

## Scope (current)

This repository is set up for **training only**: producing weights/checkpoints you can later load for inference. **Rhino integration** (Python GH component or C# plugin, competing diffusion priors over a designer-defined u,v surface) is **out of scope** for now; see [docs/RHINO_ROADMAP.md](docs/RHINO_ROADMAP.md).

## Model families

### 1. Vertex displacement as RGB (unsupervised, unstructured mesh)

- **Idea:** Each training sample is a raster image whose **R, G, B** channels encode **X, Y, Z** vertex displacement (after a fixed normalization), for meshes that share the same **u,v** topology as your training corpus.
- **Unstructured data:** Meshes are not on a regular grid; you **rasterize** (or otherwise map) per-vertex displacements onto a 2D image (e.g. canonical UV layout, or a consistent orthographic view) so a 2D diffusion model can operate on fixed-resolution tensors.
- **Unsupervised training:** Learn the distribution of these RGB displacement images (e.g. DDPM / latent diffusion) **without** paired “target” labels beyond the samples themselves.
- **Trig surfaces + FEM steering:** Synthetic trig JPEGs from `scripts/generate-trig-surfaces.js` can train a DDPM via `train-vertex-rgb`. The loss is **noise prediction MSE** (standard diffusion) plus **`fem.loss_weight` × structural proxy** on the predicted clean image: chord-relative sag between **pinned rows 0 and H−1** and mean **(∂²w*/∂y²)²** on that sag (same idealization as `scripts/view-trig-surface.js`). That biases the sampler toward **lower bending** while still matching the data distribution. Tune `fem.loss_weight` and `fem.curvature_hy` in `configs/displacement_vertex_rgb.yaml`.

### 2. FEM-informed maps (two checkpoints)

Separate trained models (separate checkpoints) for:

| Checkpoint | Input | Output image semantics |
|------------|--------|-------------------------|
| **FEM displacement** | Conditioning image(s) from `data/` (see config) | Scalar displacement field encoded **white → pink** (monotone colormap). |
| **FEM stress** | Same idea | Signed or von Mises–style stress encoded **blue (compression) → white (neutral) → red (tension)**. Exact sign convention should match your solver exports; adjust in `viz/colormaps.py` when you wire real FEM data. |

Each generated or exported training/output image should have a **sidecar metadata** JSON with at least:

- `max_displacement`, `avg_displacement`
- `max_stress`, `avg_stress`

(Units and coordinate system should be documented per dataset in `data/README.md`.)

## Repository layout

```
configs/          # YAML training configs (two families + FEM sub-checkpoints)
data/             # Place raw/processed inputs here (not committed)
docs/             # Design notes and Rhino roadmap
outputs/          # Checkpoints and samples (not committed)
src/iass_fem_diff/
  cli.py          # Typer entrypoint
  datasets/       # Trig RGB dataset + FEM map stubs
  io/metadata.py # Metadata schema + read/write
  physics/        # Differentiable FEM proxy (bending on chord-relative sag)
  train/          # DDPM on trig RGB (`trig_diffusion.py`)
  viz/colormaps.py # FEM visualization conventions
```

## Quick start (after you add dependencies)

The library lives under `src/`. You must either **install it in editable mode** (recommended) or set `PYTHONPATH=src` before running modules.

```powershell
cd IASS-FEM-DIFF
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[train]"
python -m iass_fem_diff.cli --help
```

Use the **same** `python` / `pip` pair (e.g. if you use `C:\Python314\python.exe`, run `C:\Python314\python.exe -m pip install -e ".[train]"` from the repo root).

**Without installing:** from the repo root, `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`), then `python -m iass_fem_diff.cli ...`.

Train on trig JPEGs:

```bash
node scripts/generate-trig-surfaces.js --max 2000
python -m iass_fem_diff.cli train-vertex-rgb configs/displacement_vertex_rgb.yaml
```

Other training commands (`train-fem-field`) remain stubs until you wire paired FEM map data.

## License

Add a `LICENSE` when you publish.
