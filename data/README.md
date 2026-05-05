# Data layout

Place dataset-specific files here. Nothing under `data/` is tracked by git except this README.

## Suggested structure

```
data/
  raw/
    meshes/           # Source meshes (format TBD: OBJ, 3DM export, etc.)
    fem/              # Solver outputs (CSV, VTK, JSON per case)
  processed/
    displacement_rgb/ # Rasterized R,G,B = dx,dy,dz (PNG/JPEG + sidecar JSON)
      trig/           # Synthetic 80×80 JPEGs from scripts/generate-trig-surfaces.js
    fem_maps/         # Input conditioning images + target stress/disp maps
```

### Synthetic trig surfaces (Node.js)

From the repo root (Node 18+):

```bash
npm install
node scripts/generate-trig-surfaces.js --max 500
```

The generator enumerates **225** u,v domains (every span `[min,max]` on each axis with 90° knots 0…360 and min≤max), multiplied by 10³ formula triples and coefficient triples (default **0.2–6**, step **0.2** → 30 values per axis → **225 × 27,000,000** full combinations). **`--max N`** samples **N** surfaces uniformly across that full space. Use `--max 0 --force` for a full run. Filenames include `d###` (domain index) and `u…_v…` bounds.

**3D viewer:** `node scripts/view-trig-surface.js` writes `scripts/trig-viewer.html`. **HTTP:** serve repo root (`npx --yes serve . -p 8765`), open `/scripts/trig-viewer.html`, pick the JPEG — sidecar JSON is **fetched** from `data/processed/.../trig/`. **file://:** browsers block that fetch — in one dialog **Ctrl+select** `stem.jpg` and `stem.meta.json`, or use HTTP.

## Metadata sidecar (JSON)

Each raster export should ship with a `.meta.json` (or embedded in a manifest) using the schema in `src/iass_fem_diff/io/metadata.py`.

Document here for your project:

- Displacement normalization (global max per dataset vs per sample).
- Stress definition (von Mises vs principal vs shell membrane/bending).
- u,v orientation and seam placement for rasterization.
