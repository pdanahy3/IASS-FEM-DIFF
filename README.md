# IASS-FEM-DIFF

Training and inference for **DDPM** on shell **displacement rasters** (RGB = \(d_x,d_y,d_z\) on an \(80\times80\) grid), optional **FEM-style proxy loss** during training, **reference linear elasticity** (scikit-fem) for precompute and validation, a **Three.js** viewer, and **evaluation** scripts for paper-style metrics.

---

## Contents

1. [Setup](#1-setup)  
2. [Dependencies](#2-dependencies)  
3. [Hardware requirements](#3-hardware-requirements)  
4. [Typical workflow](#4-typical-workflow)  
5. [Training](#5-training)  
6. [Inference](#6-inference)  
7. [Testing and smoke checks](#7-testing-and-smoke-checks)  
8. [Evaluation](#8-evaluation)  
9. [CLI reference (all commands and options)](#9-cli-reference-all-commands-and-options)  
10. [Node.js, viewer, and optional demos](#10-nodejs-viewer-and-optional-demos)  
11. [Configuration YAML](#11-configuration-yaml)  
12. [Further reading](#12-further-reading)

---

## 1. Setup

From the repository root:

```powershell
cd IASS-FEM-DIFF
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[train]"
npm install
python -m iass_fem_diff.cli --help
```

| Step | Purpose |
|------|---------|
| `python -m venv .venv` | Isolated Python environment (use `source .venv/bin/activate` on Linux/macOS). |
| `pip install -e ".[train]"` | Installs the package in editable mode with **training + FEM + inference** extras. |
| `npm install` | Installs **jpeg-js** for `scripts/generate-trig-surfaces.js`. |

**CUDA:** Install a **GPU-enabled** PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) if `python -c "import torch; print(torch.cuda.is_available())"` prints `False` after installing `[train]`.

---

## 2. Dependencies

### Python (`pip install -e ".[train]"`)

| Package | Role |
|---------|------|
| `torch`, `torchvision` | Training and inference tensors, UNet, CUDA. |
| `diffusers` | `DDPMScheduler`, `UNet2DModel`. |
| `accelerate` | Declared for compatibility; training loop does not require an Accelerate wrapper. |
| `numpy`, `PyYAML`, `Pillow` | Data I/O, configs, images. |
| `typer`, `rich` | CLI and terminal output. |
| `scikit-fem` | Reference 3D linear elasticity on the displacement grid. |
| `matplotlib` | Optional plots (`metrics.png` in guided runs) and `render-guided-run-3d`. |

**Core install** (`pip install -e .`) has **no** runtime dependencies; you need **`[train]`** for almost everything in this README.

### Node.js

| Need | Version hint |
|------|----------------|
| Node.js | LTS recommended for `generate-trig-surfaces.js` and `view-trig-surface.js`. |

---

## 3. Hardware requirements

| Workload | Recommendation |
|----------|------------------|
| **Training** (`train-vertex-rgb`) | **NVIDIA GPU** with sufficient VRAM for `batch_size` and `80×80` U-Net (CUDA strongly recommended). Set `training.device: cuda` in YAML or rely on `auto`. |
| **Inference** (`sample-*`) | GPU preferred; **`--device cpu`** works but is slower. |
| **`precompute-fem-trig`** | **CPU-bound** (scikit-fem); use `--shard` / `--num-shards` for parallel machines. |
| **Disk** | Checkpoints per epoch; JPEG dataset; optional large `.meta.json` with FEM grids. |

---

## 4. Typical workflow

1. **Generate data:** `node scripts/generate-trig-surfaces.js --max 500`  
2. **(Optional)** **Precompute FEM** for 4th channel or analysis: `precompute-fem-trig`  
3. **Train:** `train-vertex-rgb configs/displacement_vertex_rgb.yaml`  
4. **Sample:** `sample-vertex-rgb <checkpoint.pt> --num 100 --out-dir outputs/samples/run1`  
5. **Evaluate:** `evaluate-inference-samples outputs/samples/run1`  

---

## 5. Training

**Command**

```text
python -m iass_fem_diff.cli train-vertex-rgb <config.yaml> [--fem-ridge]
```

| Item | Description |
|------|-------------|
| **Config** | Primary control surface: `configs/displacement_vertex_rgb.yaml`. |
| **Objective** | `loss = loss_noise + sign * fem.loss_weight * fem_proxy`, with `loss_noise = MSE(noise_pred, noise)` and `fem_proxy = mean((Δw*)^2)` on Coons-relative sag (`fem_proxy.py`). |
| **`sign`** | `+1` default (smooth bias). **`--fem-ridge`** or `fem.maximize_laplacian_proxy: true` sets `sign = -1` (ridge-seeking / maximize proxy contribution to loss gradient). |
| **Checkpoints** | `training.checkpoint_dir` / `unet_epoch_XXXX.pt` — stores weights, `num_train_timesteps`, `fem_loss_weight`, `plane_size`, `curvature_span`, `vertex_rgb_phys_scale`, `fem_maximize_laplacian_proxy`, normalization mode. |

**YAML blocks** (see [§11](#11-configuration-yaml)): `data`, `training`, `diffusion`, `fem`.

---

## 6. Inference

| Mode | Command | Output |
|------|---------|--------|
| **Batch unconditional** | `sample-vertex-rgb <ckpt.pt> [--out-dir] [--num] [--seed] [--steps] [--device]` | `sample_XXXX.png` + `.meta.json` per sample (reference FEM + proxy score in meta when valid). |
| **Guided (single chain)** | `sample-vertex-rgb-guided <ckpt.pt> …` | `step_*.png`, `final.png`, `run.json`, `metrics.csv`, optional `metrics.png`, `final_fem.json`. |

**Encoding:** outputs use the same **fixed-linear** physical range as training (see `data/README.md`).

---

## 7. Testing and smoke checks

These are **manual** sanity checks (there is no bundled `pytest` suite for the full pipeline).

| Check | Command |
|--------|---------|
| CLI loads | `python -m iass_fem_diff.cli --help` |
| Dataset non-empty | `python -m iass_fem_diff.cli train-vertex-rgb configs/displacement_vertex_rgb.yaml` (fails fast with a clear message if zero images). |
| FEM precompute on a few files | `python -m iass_fem_diff.cli precompute-fem-trig --limit 3` |
| Evaluation on a small folder | `python -m iass_fem_diff.cli evaluate-inference-samples <dir> --dataset-limit 50` |

---

## 8. Evaluation

**Command**

```text
python -m iass_fem_diff.cli evaluate-inference-samples <samples_dir> [OPTIONS]
```

Computes **per-image** and **aggregate** metrics:

- **Reference FEM** (scikit-fem): spatial mean and max of \(\|\mathbf{u}\|\), global max over all samples; von Mises peak per sample.  
- **FEM proxy loss:** recomputed with `structural_efficiency_loss` (same as training proxy) — **not** read from checkpoints.  
- **Design-language distance:** mean over samples of **minimum MSE** to rasters in the dataset directory (fixed-linear decode, same grid size).

**Default outputs:** `<samples_dir>/eval_fem_metrics.json` and `eval_fem_metrics.csv`.

---

## 9. CLI reference (all commands and options)

**Entry points:** `python -m iass_fem_diff.cli` or `iass-fem-diff` (after `pip install -e .`).

### Command summary

| Command | Purpose |
|---------|---------|
| `train-vertex-rgb` | Train DDPM + optional FEM proxy (+ optional 4ch stress). |
| `sample-vertex-rgb` | Batch unconditional sampling. |
| `sample-vertex-rgb-guided` | Guided run with optional seed/goal images. |
| `precompute-fem-trig` | Write reference FEM fields into `.meta.json`. |
| `evaluate-inference-samples` | Reference FEM + proxy + nearest-dataset MSE on a folder. |
| `render-guided-run-3d` | Matplotlib 3D frames from a guided run folder. |
| `configs` | List `*.yaml` in a config directory. |
| `train-fem-field` | Stub (dataset listing only). |

---

### `train-vertex-rgb`

```text
python -m iass_fem_diff.cli train-vertex-rgb <CONFIG.yaml> [--fem-ridge]
```

| Argument / option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `CONFIG` | path (required) | — | Training YAML (e.g. `configs/displacement_vertex_rgb.yaml`). |
| `--fem-ridge` | flag | off | If set: **maximize** Laplacian proxy term (`loss_noise - λ * proxy`). Overrides `fem.maximize_laplacian_proxy` in YAML. |

---

### `sample-vertex-rgb`

```text
python -m iass_fem_diff.cli sample-vertex-rgb <CHECKPOINT.pt> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CHECKPOINT` | path (required) | — | Saved `unet_epoch_*.pt`. |
| `--out-dir` | path | `outputs/samples/displacement_vertex_rgb` | Output folder (relative paths resolved from repo root). |
| `--num` | int ≥ 1 | `16` | Number of samples. |
| `--seed` | int | `42` | RNG seed. |
| `--steps` | int or omit | training \(T\) | DDPM inference step count. |
| `--device` | str | `cuda` | `cuda` or `cpu`. |

---

### `sample-vertex-rgb-guided`

```text
python -m iass_fem_diff.cli sample-vertex-rgb-guided <CHECKPOINT.pt> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CHECKPOINT` | path (required) | — | Saved checkpoint. |
| `--out-dir` | path | `outputs/samples/displacement_vertex_rgb_guided` | Run output directory. |
| `--seed-image` | path or omit | — | Optional 80×80 RGB displacement image (img2img seed). |
| `--strength` | float [0,1] | `0.0` | img2img strength; `0` disables seed path. |
| `--goal-image` | path or omit | — | Optional 80×80 goal image for steering. |
| `--goal-mix` | float [0,1] | `0.0` | Mix toward goal-implied noise each step. |
| `--seed` | int | `42` | RNG seed. |
| `--steps` | int | `1000` | Inference diffusion steps. |
| `--save-every` | int | `50` | Save intermediate `step_*.png` every N steps (plus first/last). |
| `--device` | str | `cuda` | `cuda` or `cpu`. |

---

### `precompute-fem-trig`

```text
python -m iass_fem_diff.cli precompute-fem-trig [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--processed-dir` | path | `data/processed/displacement_rgb/trig` | Images + existing `.meta.json` sidecars. |
| `--plane-size` | float | `2.0` | Base plane size (meters scale; consistent with viewer). |
| `--gravity-total` | float | `1.0` | Total downward load on interior nodes (N). |
| `--E` | float | `210e9` | Young's modulus (Pa). |
| `--nu` | float | `0.3` | Poisson's ratio. |
| `--overwrite` | flag | off | Replace existing `fem_stress_grid` / `fem_disp_grid` in meta. |
| `--limit` | int ≥ 0 | `0` | Max images after shard/stride (`0` = no cap). |
| `--stride` | int ≥ 1 | `1` | Process every k-th sorted image. |
| `--shard` | int ≥ 0 | `0` | Shard index in `[0, num_shards-1]`. |
| `--num-shards` | int ≥ 1 | `1` | Parallel sharding factor. |

---

### `evaluate-inference-samples`

```text
python -m iass_fem_diff.cli evaluate-inference-samples <SAMPLES_DIR> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `SAMPLES_DIR` | path (required) | — | Folder of generated `.png`/`.jpg` (not `*.meta.json` globs as inputs). |
| `--dataset-dir` | path or omit | auto | Trig training folder for nearest-neighbor MSE; if omitted and repo `data/processed/displacement_rgb/trig` exists, that is used. |
| `--skip-nearest-dataset` | flag | off | Skip loading dataset; no design-language MSE. |
| `--out-json` | path or omit | `<samples>/eval_fem_metrics.json` | Full metrics JSON. |
| `--out-csv` | path or omit | `<samples>/eval_fem_metrics.csv` | Per-sample CSV. |
| `--plane-size` | float | `2.0` | Must match FEM setup used for training/inference. |
| `--gravity-total` | float | `1.0` | Reference FEM load. |
| `--curvature-span` | int | `1` | Proxy Laplacian span (match `fem.curvature_span`). |
| `--dataset-limit` | int ≥ 0 | `0` | Cap training images for NN search (`0` = all). |
| `--width` | int ≥ 2 | `80` | Raster width. |
| `--height` | int ≥ 2 | `80` | Raster height. |

---

### `render-guided-run-3d`

```text
python -m iass_fem_diff.cli render-guided-run-3d <RUN_DIR> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `RUN_DIR` | path (required) | — | Guided output folder with `step_*.png`. |
| `--out-dir` | path or omit | `<RUN_DIR>/render_3d` | Output frames directory. |
| `--plane-size` | float | `2.0` | XY plane extent. |
| `--disp-scale` | float | `1.0` | Displacement scale on mesh. |
| `--disp-smooth` | int | `1` | Box blur radius on decoded disp (`0` = off). |
| `--color-mode` | str | `z` | `z`, `rgb`, `fem_stress`, or `fem_disp`. |
| `--clim-min`, `--clim-max` | float | omit | Scalar color limits (both required to apply). |
| `--fem-every` | int | `1` | In `fem_*` modes, solve reference FEM every N frames. |
| `--elev` | float | `28.0` | Camera elevation (deg). |
| `--azim` | float | `-50.0` | Camera azimuth (deg). |
| `--every` | int | `1` | Render every Nth step PNG. |
| `--dpi` | int | `160` | Figure DPI. |

---

### `configs`

```text
python -m iass_fem_diff.cli configs [--config-dir PATH]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config-dir` | path | `configs` | Directory containing `*.yaml`. |

---

### `train-fem-field`

```text
python -m iass_fem_diff.cli train-fem-field <CONFIG.yaml>
```

| Argument | Description |
|----------|-------------|
| `CONFIG` | FEM field-map YAML (stub only — prints dataset size). |

---

## 10. Node.js, viewer, and optional demos

### `package.json` scripts

| Script | Command |
|--------|---------|
| Generate trig data | `npm run generate-trig-surfaces` or `node scripts/generate-trig-surfaces.js [OPTIONS]` |
| Build viewer HTML | `npm run view-trig-surface` or `node scripts/view-trig-surface.js` |

**Common `generate-trig-surfaces.js` flags:** `--out`, `--size`, `--coeff-min`, `--coeff-max`, `--coeff-step`, `--jpeg-quality`, `--max`, `--force` (see file header).

**Viewer:** from repo root run `npx serve .` and open **`/scripts/trig-viewer.html`** over **http** (ES modules). Supports fixed-linear decode, FEM sidecars, smoothing, Z-up gravity arrows.

### Optional: diffusion mesh demo + HTTP streamer

| Asset | Purpose |
|-------|---------|
| `scripts/diffusion_mesh_infer_server.py` | Streams DDPM steps to a browser for the multi-agent mesh demo. |
| `scripts/multi-agent-diffusion-mesh.html` (+ `.js`) | 80×80 agents + boids + optional streamed diffusion steering. |

---

## 11. Configuration YAML

Primary file: **`configs/displacement_vertex_rgb.yaml`**.

| Section | Keys (summary) |
|---------|----------------|
| **`data`** | `processed_dir`, `image_size`, `normalization` (`fixed_linear` recommended), `include_fem_stress`, `fem_stress_norm`. |
| **`training`** | `seed`, `batch_size`, `num_epochs`, `learning_rate`, `checkpoint_dir`, `device` (`auto` / `cuda` / `cpu`). |
| **`diffusion`** | `num_train_timesteps` (default 1000). |
| **`fem`** | `loss_weight`, `maximize_laplacian_proxy`, `plane_size`, `curvature_span`, `vertical_channel`. |

Paths under `data` / `outputs` are resolved relative to the **repository root** when not absolute.

---

## 12. Further reading

| Document | Content |
|----------|---------|
| `data/README.md` | Dataset layout and RGB encoding conventions. |
| `docs/RHINO_ROADMAP.md` | Out-of-scope roadmap (if present). |

---

## License

Add a `LICENSE` file when you publish.
