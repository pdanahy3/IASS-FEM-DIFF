# Rhino inference and competing diffusion priors (later)

## Goal

A designer defines a **mathematical surface** with the same **u,v** structure as the training data. That surface (or its rasterized u,v image) becomes the **base conditioning** for inference. At each agent step:

1. **Displacement RGB diffusion** steers toward a geometric prior (how surfaces “typically” displace in u,v space).
2. **FEM displacement** and **FEM stress** diffusion models (or their latents) provide **structural** signals: prefer states with less extreme pink (displacement magnitude), less red/blue (stress extremes), subject to your chosen balance.

“Fighting” or competing influence is **not implemented here**; in Rhino you would combine scores or gradients from multiple denoising steps or from distilled guidance networks.

## Integration options

- **Grasshopper Python:** Fastest iteration; load Torch + checkpoints; run denoising on GPU if available; write meshes or textures back to Rhino.
- **C# Rhino plugin:** Better packaging and performance integration; host ONNX or a small native inference server if you export models.

## What this repo must deliver first

- Checkpoints for:
  - Vertex displacement RGB diffusion
  - FEM displacement map diffusion
  - FEM stress map diffusion
- Agreed **image resolution**, **normalization** for RGB displacements, and **colormap** definitions matching training exports.

Then Rhino-side code only needs: rasterize current design → normalize → run `pipeline` → map outputs back to mesh or agent state.
