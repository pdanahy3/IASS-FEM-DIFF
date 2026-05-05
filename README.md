# IASS-FEM-DIFF

Training and checkpoint layout for image-based diffusion models on shell displacement and FEM-derived fields, with a future path to Rhino 3D inference and multi-model agent guidance.

## Scope (current)

This repository is set up for **training only**: producing weights/checkpoints you can later load for inference. **Rhino integration** (Python GH component or C# plugin, competing diffusion priors over a designer-defined u,v surface) is **out of scope** for now; see [docs/RHINO_ROADMAP.md](docs/RHINO_ROADMAP.md).

## Model families

### 1. Vertex displacement as RGB (unsupervised, unstructured mesh)

- **Idea:** Each training sample is a raster image whose **R, G, B** channels encode **X, Y, Z** vertex displacement (after a fixed normalization), for meshes that share the same **u,v** topology as your training corpus.
- **Unstructured data:** Meshes are not on a regular grid; you **rasterize** (or otherwise map) per-vertex displacements onto a 2D image (e.g. canonical UV layout, or a consistent orthographic view) so a 2D diffusion model can operate on fixed-resolution tensors.
- **Unsupervised training:** Learn the distribution of these RGB displacement images (e.g. DDPM / latent diffusion) **without** paired “target” labels beyond the samples themselves.
- **Trig surfaces + FEM steering:** Synthetic trig JPEGs from `scripts/generate-trig-surfaces.js` can train a DDPM via `train-vertex-rgb`. The loss is **noise prediction MSE** plus **`fem.loss_weight` × structural proxy** on the predicted clean image: **four-edge** pinned reference (Coons sag (w*)) and mean **Laplacian (Δw*)²**, matching `scripts/view-trig-surface.js`. Tune `fem.loss_weight`, `fem.plane_size`, and `fem.curvature_span` in `configs/displacement_vertex_rgb.yaml`.

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
  physics/        # Differentiable FEM proxy (four-edge sag, Laplacian)
  train/          # DDPM on trig RGB (`trig_diffusion.py`)
  viz/colormaps.py # FEM visualization conventions
```

## Quick start (after you add dependencies)

On **Windows 11**, use **PowerShell** (or Command Prompt) from the repo root. The library lives under `src/`. You must either **install it in editable mode** (recommended) or set `PYTHONPATH=src` before running modules.

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

```powershell
node scripts/generate-trig-surfaces.js --max 2000
python -m iass_fem_diff.cli train-vertex-rgb configs/displacement_vertex_rgb.yaml
```

**GPU (Windows 11 + NVIDIA):** The training loop uses `.to(device)` with `device: cuda` in `configs/displacement_vertex_rgb.yaml` (or `auto` to pick CUDA when available).

If `torch.cuda.is_available()` is **False** even though `nvidia-smi` works, your PyTorch build is almost certainly **CPU-only**. Check the version string: `2.11.0+cpu` means no CUDA in that install (bundled CUDA in PyTorch is separate from the NVIDIA driver). Replace it with a CUDA wheel, e.g.:

```powershell
python -m pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu128
```

Use the **same** `python` you use for training (`where python` — Windows may list several). After installing, you should see `+cu128` (or similar), not `+cpu`:

```text
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

You want `torch.cuda.is_available()` → `True` and a non-empty `torch.version.cuda`. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) if you need a different CUDA flavor.

Other training commands (`train-fem-field`) remain stubs until you wire paired FEM map data.

## License

Add a `LICENSE` when you publish.
