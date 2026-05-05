/**
 * Writes a self-contained HTML 3D viewer (Three.js) with image upload.
 * Sidecar `<stem>.meta.json` is fetched automatically (same stem as the image) from
 * ../data/processed/displacement_rgb/trig/ or ./ relative to the HTML URL.
 * Without a successful fetch, decoding uses auto scale per channel.
 *
 * Usage (from repo root):
 *   node scripts/view-trig-surface.js
 *   node scripts/view-trig-surface.js --out path/to/viewer.html
 *
 * Open the HTML in a browser. Uses Three.js r160 (ES modules + import map). If the
 * page is blank, open via a local server so modules load.
 *
 * Sidecar metadata:
 *  - Same dialog: enable multiple and Ctrl+select <stem>.jpg + <stem>.meta.json (needed for file://).
 *  - http(s): also tries ../data/processed/displacement_rgb/trig/<stem>.meta.json relative to this page.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const opts = { out: path.join(__dirname, "trig-viewer.html") };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") opts.out = path.resolve(argv[++i] ?? "");
    else if (a === "--help" || a === "-h") {
      console.log(`Usage: node scripts/view-trig-surface.js [--out viewer.html]`);
      process.exit(0);
    }
  }
  return opts;
}

function buildViewerHtml() {
  const THREE_VER = "0.160.0";
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trig shell viewer</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; overflow: hidden; font-family: system-ui, sans-serif; background: #111; }
    #panel {
      position: fixed; top: 10px; left: 10px; z-index: 20;
      background: rgba(35, 35, 42, 0.95); color: #e8e8ec;
      padding: 14px 16px; border-radius: 10px; max-width: 380px;
      font-size: 13px; box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    #panel h1 { margin: 0 0 10px; font-size: 15px; font-weight: 600; }
    #panel label { display: block; margin-top: 10px; cursor: pointer; }
    #panel input[type="file"] { display: block; margin-top: 4px; max-width: 100%; font-size: 12px; }
    #panel .row { display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; align-items: center; }
    #panel .row label { margin-top: 0; display: flex; align-items: center; gap: 6px; }
    #panel input[type="number"] { width: 72px; padding: 4px 6px; border-radius: 4px; border: 1px solid #555; background: #1e1e24; color: #fff; }
    #btnLoad {
      margin-top: 12px; padding: 8px 14px; border: none; border-radius: 6px;
      background: #3d6ad6; color: #fff; font-size: 13px; cursor: pointer; font-weight: 500;
    }
    #btnLoad:hover { background: #5080f0; }
    #btnLoad:disabled { opacity: 0.5; cursor: not-allowed; }
    #status { margin-top: 10px; font-size: 12px; line-height: 1.45; color: #b0b0b8; white-space: pre-wrap; }
    #hint {
      position: fixed; left: 10px; bottom: 10px; z-index: 10;
      color: #888; font-size: 11px; background: rgba(0,0,0,0.45);
      padding: 8px 10px; border-radius: 6px; max-width: 90vw; pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="panel">
    <h1>Trig displacement shell</h1>
    <label>Image (+ optional sidecar in the same picker)
      <input type="file" id="imgFile" accept="image/jpeg,image/png,.jpg,.jpeg,.png,.json,application/json" multiple />
    </label>
    <p style="margin:10px 0 0;font-size:12px;color:#a8a8b0;line-height:1.4">
      <strong>file://</strong>: Browsers block loading sibling JSON via URL — in this dialog,
      <strong>Ctrl+select</strong> both <code>stem.jpg</code> and <code>stem.meta.json</code>.
      <strong>http</strong> (e.g. <code>npx serve .</code>): you can pick only the image; sidecar is fetched from
      <code>../data/processed/displacement_rgb/trig/</code> automatically.
    </p>
    <div class="row">
      <label>Plane size <input type="number" id="planeSize" value="2" min="0.1" step="0.1" /></label>
      <label>Disp scale <input type="number" id="dispScale" value="1" min="0" step="0.05" /></label>
    </div>
    <div class="row">
      <label><input type="checkbox" id="showBase" checked /> Show flat reference grid</label>
    </div>
    <div class="row" style="margin-top:12px;flex-direction:column;align-items:flex-start;gap:6px">
      <span style="font-size:12px;color:#a8a8b0">FEM-style colors (see <code>viz/colormaps.py</code>). Model: <strong>row 0 &amp; row H−1</strong> = pinned anchors (orange), <strong>uniform gravity</strong> (−Z) on all nodes; deflection is sag relative to the chord between anchors; stress proxy ≈ ∂²w/∂y² on that sag.</span>
      <label><input type="checkbox" id="femDispColors" /> Deflection · white→pink from |sag| (chord between anchor rows)</label>
      <label><input type="checkbox" id="femStressColors" /> Stress · blue←white→red from bending (∂²w/∂y²) or sidecar <code>extra.fem_stress_*</code></label>
      <span style="font-size:11px;color:#888;line-height:1.35">Sidecar stress overrides the bending proxy when present. If both FEM boxes are on, stress colormap wins.</span>
    </div>
    <button type="button" id="btnLoad">Build / refresh mesh</button>
    <div id="status">Select an image, then click <strong>Build / refresh mesh</strong>.</div>
  </div>
  <div id="hint">Orbit: left drag · Zoom: scroll · Pan: right drag · Three r${THREE_VER} (ESM)</div>

  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@${THREE_VER}/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@${THREE_VER}/examples/jsm/"
    }
  }
  </script>
  <script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const imgFile = document.getElementById('imgFile');
const planeSizeEl = document.getElementById('planeSize');
const dispScaleEl = document.getElementById('dispScale');
const showBaseEl = document.getElementById('showBase');
const femDispColorsEl = document.getElementById('femDispColors');
const femStressColorsEl = document.getElementById('femStressColors');
const btnLoad = document.getElementById('btnLoad');
const statusEl = document.getElementById('status');

/** Last successful decode; toggles rebuild without re-picking files. */
let loadCache = {
  rgba: null,
  W: 0,
  H: 0,
  meta: null,
  metaHint: '',
};

function setStatus(msg) {
  statusEl.textContent = msg;
}

function decodeChannelMeta(byte, ch) {
  if (!ch) return byte / 127.5 - 1;
  if (ch.spatially_constant) return ch.min;
  const R = ch.extent_R;
  if (R == null || R <= 0 || !isFinite(R)) return 0;
  return R * (byte / 127.5 - 1);
}

function maxAbsT(rgba, W, H, comp) {
  let T = 0;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const b = rgba[(j * W + i) * 4 + comp];
      const t = Math.abs(b / 127.5 - 1);
      if (t > T) T = t;
    }
  }
  return T;
}

function decodeChannelAuto(byte, T) {
  if (T < 1e-7) return 0;
  return (byte / 127.5 - 1) / T;
}

function getChannels(meta) {
  if (!meta || !meta.extra || !meta.extra.channel_min_max_raw) return null;
  const c = meta.extra.channel_min_max_raw;
  if (!c.x || !c.y || !c.z) return null;
  return { x: c.x, y: c.y, z: c.z };
}

/** Match Python displacement_magnitude_to_rgb (white → pink). Returns RGB in 0..1. */
function displacementMagToLinearRgb(mag, lo, hi) {
  let t = 0;
  if (hi > lo) t = (mag - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  const w = [1, 1, 1];
  const pink = [1, 0.4, 0.8];
  return [
    (1 - t) * w[0] + t * pink[0],
    (1 - t) * w[1] + t * pink[1],
    (1 - t) * w[2] + t * pink[2],
  ];
}

/** Match Python stress_signed_to_rgb. Returns RGB in 0..1. */
function stressSignedToLinearRgb(s, lim) {
  let L = lim;
  if (L == null || L <= 0 || !isFinite(L)) L = 1;
  const t = Math.max(-1, Math.min(1, s / L));
  if (t > 0) {
    const a = t;
    return [1, 1 - a, 1 - a];
  }
  if (t < 0) {
    const b = -t;
    return [1 - b, 1 - b, 1];
  }
  return [1, 1, 1];
}

/**
 * Per-vertex stress samples aligned with image pixels (row-major). Optional in sidecar.
 * Supports extra.fem_stress_flat: number[] or extra.fem_stress_grid: { values, width?, height? }.
 */
function parseStressFlat(meta, W, H) {
  const n = W * H;
  const ex = meta && meta.extra;
  if (!ex) return null;
  let raw = ex.fem_stress_flat;
  if (raw == null && ex.fem_stress_grid && Array.isArray(ex.fem_stress_grid.values)) {
    raw = ex.fem_stress_grid.values;
    const gw = ex.fem_stress_grid.width;
    const gh = ex.fem_stress_grid.height;
    if (gw != null && gh != null && (gw !== W || gh !== H)) {
      console.warn('fem_stress_grid size mismatch vs image', gw, gh, W, H);
    }
  }
  if (!Array.isArray(raw) || raw.length < n) return null;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Number(raw[i]);
  return out;
}

/**
 * Vertical position z per node (row-major). Subtract straight chord along each column
 * between row 0 and row H−1 so pinned anchor rows are the reference (w* = 0 on edges).
 */
function chordRelativeSagZ(zz, W, H) {
  const n = W * H;
  const wStar = new Float32Array(n);
  if (H < 2) return wStar;
  const inv = 1 / (H - 1);
  for (let i = 0; i < W; i++) {
    const z0 = zz[i];
    const z1 = zz[(H - 1) * W + i];
    for (let j = 0; j < H; j++) {
      const k = j * W + i;
      const alpha = j * inv;
      wStar[k] = zz[k] - (1 - alpha) * z0 - alpha * z1;
    }
  }
  return wStar;
}

/**
 * Uniform-gravity bending proxy: d²(w*) / dy² in world units (y = row direction, spacing plane/(H−1)).
 * Anchor rows and dangling rows set to 0.
 */
function bendingCurvatureY(wStar, W, H, plane) {
  const n = W * H;
  const out = new Float32Array(n);
  if (H < 3) return out;
  const hy = plane / (H - 1);
  const hy2 = hy * hy;
  if (hy2 < 1e-24) return out;
  for (let j = 1; j < H - 1; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const km = (j - 1) * W + i;
      const kp = (j + 1) * W + i;
      out[k] = (wStar[kp] - 2 * wStar[k] + wStar[km]) / hy2;
    }
  }
  return out;
}

function stemFromImageFileName(name) {
  return name.replace(/\\.(jpe?g|png)$/i, '');
}

/** Try to load sidecar via HTTP (fails on file:// — use same-picker .meta.json). */
async function fetchMetaForImageStem(stem) {
  if (location.protocol === 'file:') {
    return { meta: null, url: null };
  }
  const paths = [
    \`../data/processed/displacement_rgb/trig/\${stem}.meta.json\`,
    \`./\${stem}.meta.json\`,
  ];
  for (const p of paths) {
    const url = new URL(p, document.baseURI).href;
    try {
      const res = await fetch(url, { cache: 'no-store' });
      if (res.ok) {
        const meta = await res.json();
        return { meta, url };
      }
    } catch {
      /* ignore network / CORS */
    }
  }
  return { meta: null, url: null };
}

function findPrimaryImageFile(files) {
  const list = Array.from(files || []);
  return list.find((f) => /\\.(jpe?g|png)$/i.test(f.name)) || null;
}

function findSidecarMetaFile(files, stem) {
  const want = stem + '.meta.json';
  return Array.from(files || []).find((f) => f.name === want) || null;
}

function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      try {
        resolve(JSON.parse(r.result));
      } catch (e) {
        reject(e);
      }
    };
    r.onerror = () => reject(new Error('read failed'));
    r.readAsText(file);
  });
}

/** Copy index array (separate GPU buffers for main mesh vs wireframe). */
function copyIndices(arr) {
  if (arr instanceof Uint32Array) return new Uint32Array(arr);
  return new Uint16Array(arr);
}

/**
 * Three r160 setIndex() only wraps plain Arrays; raw TypedArrays break WebGL (no .array).
 */
function indexBufferAttribute(arr) {
  if (arr instanceof Uint32Array) {
    return new THREE.Uint32BufferAttribute(arr, 1);
  }
  return new THREE.Uint16BufferAttribute(arr, 1);
}

function buildBuffers(
  rgba,
  W,
  H,
  meta,
  plane,
  dispScale,
  showBase,
  femDeflectionColors,
  femStressColors
) {
  const ch = getChannels(meta);
  const Tr = ch ? null : maxAbsT(rgba, W, H, 0);
  const Tg = ch ? null : maxAbsT(rgba, W, H, 1);
  const Tb = ch ? null : maxAbsT(rgba, W, H, 2);

  const n = W * H;
  const stressFlat = parseStressFlat(meta, W, H);

  const positions = new Float32Array(n * 3);

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const p = k * 4;
      const r = rgba[p];
      const gch = rgba[p + 1];
      const bch = rgba[p + 2];

      const dx = ch ? decodeChannelMeta(r, ch.x) : decodeChannelAuto(r, Tr);
      const dy = ch ? decodeChannelMeta(gch, ch.y) : decodeChannelAuto(gch, Tg);
      const dz = ch ? decodeChannelMeta(bch, ch.z) : decodeChannelAuto(bch, Tb);

      const u = i / (W - 1);
      const v = j / (H - 1);
      const bx = (u - 0.5) * plane;
      const by = (v - 0.5) * plane;

      positions[k * 3] = bx + dispScale * dx;
      positions[k * 3 + 1] = by + dispScale * dy;
      positions[k * 3 + 2] = dispScale * dz;
    }
  }

  let wStar = null;
  if (
    femDeflectionColors ||
    (femStressColors && !stressFlat)
  ) {
    const zz = new Float32Array(n);
    for (let ii = 0; ii < n; ii++) zz[ii] = positions[ii * 3 + 2];
    wStar = chordRelativeSagZ(zz, W, H);
  }

  let stressSamples = stressFlat;
  if (femStressColors && !stressSamples && wStar) {
    stressSamples = bendingCurvatureY(wStar, W, H, plane);
  }

  let stressLim =
    meta && typeof meta.max_stress === 'number' && meta.max_stress > 0
      ? meta.max_stress
      : null;
  if (femStressColors && stressSamples) {
    let mx = 0;
    for (let i = 0; i < n; i++) {
      const a = Math.abs(stressSamples[i]);
      if (a > mx) mx = a;
    }
    if (stressLim == null || stressLim <= 0) stressLim = mx > 0 ? mx : 1;
  } else if (femStressColors && stressLim == null) {
    stressLim = 1;
  }

  let magLo = 0;
  let magHi = 1;
  if (femDeflectionColors && wStar) {
    magLo = Infinity;
    magHi = -Infinity;
    for (let i = 0; i < n; i++) {
      const m = Math.abs(wStar[i]);
      if (m < magLo) magLo = m;
      if (m > magHi) magHi = m;
    }
    if (!isFinite(magLo) || !isFinite(magHi) || magHi <= magLo) {
      magLo = 0;
      magHi = 1;
    }
  }

  const colors = new Float32Array(n * 3);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const p = k * 4;
      const r = rgba[p];
      const gch = rgba[p + 1];
      const bch = rgba[p + 2];

      let cr = r / 255;
      let cg = gch / 255;
      let cb = bch / 255;

      if (femStressColors) {
        const s = stressSamples ? stressSamples[k] : 0;
        const rgb = stressSignedToLinearRgb(s, stressLim);
        cr = rgb[0];
        cg = rgb[1];
        cb = rgb[2];
      } else if (femDeflectionColors && wStar) {
        const rgb = displacementMagToLinearRgb(
          Math.abs(wStar[k]),
          magLo,
          magHi
        );
        cr = rgb[0];
        cg = rgb[1];
        cb = rgb[2];
      }

      colors[k * 3] = cr;
      colors[k * 3 + 1] = cg;
      colors[k * 3 + 2] = cb;
    }
  }

  const indices = [];
  for (let j2 = 0; j2 < H - 1; j2++) {
    for (let i2 = 0; i2 < W - 1; i2++) {
      const a = j2 * W + i2;
      const b1 = j2 * W + i2 + 1;
      const c = (j2 + 1) * W + i2 + 1;
      const d = (j2 + 1) * W + i2;
      indices.push(a, b1, d, b1, c, d);
    }
  }

  let basePositions = null;
  if (showBase) {
    basePositions = new Float32Array(n * 3);
    for (let j3 = 0; j3 < H; j3++) {
      for (let i3 = 0; i3 < W; i3++) {
        const k3 = j3 * W + i3;
        const u3 = i3 / (W - 1);
        const v3 = j3 / (H - 1);
        basePositions[k3 * 3] = (u3 - 0.5) * plane;
        basePositions[k3 * 3 + 1] = (v3 - 0.5) * plane;
        basePositions[k3 * 3 + 2] = 0;
      }
    }
  }

  const indexTyped =
    n < 65536 ? new Uint16Array(indices) : new Uint32Array(indices);
  return { positions, colors, indices: indexTyped, basePositions };
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1e);

let mesh = null;
let wire = null;
let anchorGroup = null;

function disposeAnchorLines() {
  if (!anchorGroup) return;
  anchorGroup.traverse((obj) => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  });
  scene.remove(anchorGroup);
  anchorGroup = null;
}

/** Pinned supports: polylines along row 0 and row H−1 on the deformed mesh. */
function makeAnchorSupportLines(positions, W, H) {
  const g = new THREE.Group();
  const rowLine = (jRow) => {
    const verts = new Float32Array(W * 3);
    for (let i = 0; i < W; i++) {
      const k = jRow * W + i;
      verts[i * 3] = positions[k * 3];
      verts[i * 3 + 1] = positions[k * 3 + 1];
      verts[i * 3 + 2] = positions[k * 3 + 2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    return new THREE.Line(
      geo,
      new THREE.LineBasicMaterial({ color: 0xff9933 })
    );
  };
  g.add(rowLine(0));
  g.add(rowLine(H - 1));
  return g;
}

scene.add(new THREE.AmbientLight(0x606060));
const dl = new THREE.DirectionalLight(0xffffff, 0.85);
dl.position.set(1.2, 1.5, 2);
scene.add(dl);
const dl2 = new THREE.DirectionalLight(0xaaccff, 0.35);
dl2.position.set(-2, -1, 0.5);
scene.add(dl2);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.01,
  1000
);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

function disposeMesh(m) {
  if (!m) return;
  if (m.geometry) m.geometry.dispose();
  const mat = m.material;
  if (mat) {
    if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
    else mat.dispose();
  }
  scene.remove(m);
}

function fitCamera(geo) {
  geo.computeBoundingSphere();
  const c = geo.boundingSphere.center;
  const r = Math.max(geo.boundingSphere.radius, 0.001);
  controls.target.copy(c);
  camera.near = Math.max(0.001, r / 2000);
  camera.far = Math.max(500, r * 50);
  camera.updateProjectionMatrix();
  camera.position.set(c.x + r * 1.15, c.y - r * 1.15, c.z + r * 1.35);
  camera.lookAt(c);
  controls.update();
}

function rebuildFromRgba(rgba, W, H, meta, metaHint) {
  if (W < 2 || H < 2) {
    setStatus('Image must be at least 2×2 pixels.');
    return;
  }
  loadCache.rgba = new Uint8ClampedArray(rgba);
  loadCache.W = W;
  loadCache.H = H;
  loadCache.meta = meta;
  loadCache.metaHint = metaHint || '';

  const plane = parseFloat(planeSizeEl.value, 10) || 2;
  let dispScale = parseFloat(dispScaleEl.value, 10);
  if (!isFinite(dispScale) || dispScale < 0) dispScale = 1;
  const showBase = showBaseEl.checked;
  const femDeflectionColors = femDispColorsEl.checked;
  const femStressColors = femStressColorsEl.checked;

  const ch = getChannels(meta);
  if (meta && meta.extra && meta.extra.grid) {
    const gw = meta.extra.grid.width;
    const gh = meta.extra.grid.height;
    if (gw !== W || gh !== H) {
      setStatus(
        'Warning: image is ' +
          W +
          '×' +
          H +
          ' but meta.grid is ' +
          gw +
          '×' +
          gh +
          '. Using image size.'
      );
    }
  }

  const buf = buildBuffers(
    rgba,
    W,
    H,
    meta,
    plane,
    dispScale,
    showBase,
    femDeflectionColors,
    femStressColors
  );

  disposeMesh(mesh);
  disposeMesh(wire);
  disposeAnchorLines();

  const pos = new Float32Array(buf.positions);
  const col = new Float32Array(buf.colors);
  const idxMain = copyIndices(buf.indices);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(indexBufferAttribute(idxMain));
  geo.computeVertexNormals();

  mesh = new THREE.Mesh(
    geo,
    new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 35,
      specular: new THREE.Color(0x222222),
    })
  );
  scene.add(mesh);

  if (showBase && buf.basePositions) {
    const baseGeo = new THREE.BufferGeometry();
    baseGeo.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(buf.basePositions), 3)
    );
    baseGeo.setIndex(indexBufferAttribute(copyIndices(buf.indices)));
    wire = new THREE.Mesh(
      baseGeo,
      new THREE.MeshBasicMaterial({
        color: 0x4488ff,
        wireframe: true,
        opacity: 0.35,
        transparent: true,
      })
    );
    wire.position.z = -0.001;
    scene.add(wire);
  }

  if (femDeflectionColors || femStressColors) {
    anchorGroup = makeAnchorSupportLines(buf.positions, W, H);
    scene.add(anchorGroup);
  }

  fitCamera(geo);

  const mode = ch ? 'metadata decode' : 'auto scale (per-channel mid-gray)';
  const stressFieldArr = parseStressFlat(meta, W, H);
  const hasStressField = stressFieldArr != null;
  let colorMode = 'vertex colors: image RGB';
  if (femStressColors) {
    colorMode = hasStressField
      ? 'vertex colors: FEM stress (sidecar field)'
      : 'vertex colors: FEM stress (∂²w/∂y² proxy, uniform −Z)';
  } else if (femDeflectionColors) {
    colorMode =
      'vertex colors: FEM |sag| white→pink (chord from anchor rows)';
  }
  let msg =
    'Mesh: ' +
    W +
    '×' +
    H +
    ' · ' +
    mode +
    ' · ' +
    colorMode +
    ' · plane=' +
    plane +
    ' dispScale=' +
    dispScale;
  if (femStressColors && femDeflectionColors) {
    msg += '\\nBoth FEM toggles on — stress colormap is shown.';
  }
  if (metaHint) msg += '\\n' + metaHint;
  setStatus(msg);
}

async function loadImageAndMeta() {
  const fImg = findPrimaryImageFile(imgFile.files);
  if (!fImg) {
    setStatus('Choose at least one JPEG/PNG (Ctrl+click to add .meta.json on file://).');
    return;
  }

  btnLoad.disabled = true;
  const stem = stemFromImageFileName(fImg.name);
  let meta = null;
  let metaHint = '';

  const metaPick = findSidecarMetaFile(imgFile.files, stem);
  try {
    if (metaPick) {
      meta = await readJsonFile(metaPick);
      metaHint = 'Sidecar: ' + metaPick.name + ' (same file dialog)';
    } else {
      const got = await fetchMetaForImageStem(stem);
      meta = got.meta;
      if (got.meta && got.url) {
        metaHint = 'Sidecar: ' + got.url;
      } else if (location.protocol === 'file:') {
        metaHint =
          'file://: add ' +
          stem +
          '.meta.json in the same selection (Ctrl+click), or use npx serve . for auto-fetch.';
      } else {
        metaHint =
          'No sidecar in picker or at ../data/.../trig/' +
          stem +
          '.meta.json — auto decode.';
      }
    }
  } catch (e) {
    metaHint = 'Meta: ' + e.message;
    meta = null;
  }

  const rImg = new FileReader();
  rImg.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const W = img.width;
      const H = img.height;
      const canvas = document.createElement('canvas');
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const id = ctx.getImageData(0, 0, W, H);
      try {
        rebuildFromRgba(id.data, W, H, meta, metaHint);
      } catch (e) {
        setStatus('Error: ' + e.message);
      }
      btnLoad.disabled = false;
    };
    img.onerror = () => {
      setStatus('Could not decode image.');
      btnLoad.disabled = false;
    };
    img.src = ev.target.result;
  };
  rImg.onerror = () => {
    setStatus('Could not read image file.');
    btnLoad.disabled = false;
  };
  rImg.readAsDataURL(fImg);
}

function rebuildFromCache() {
  if (!loadCache.rgba || loadCache.W < 2) return;
  rebuildFromRgba(
    loadCache.rgba,
    loadCache.W,
    loadCache.H,
    loadCache.meta,
    loadCache.metaHint
  );
}

femDispColorsEl.addEventListener('change', rebuildFromCache);
femStressColorsEl.addEventListener('change', rebuildFromCache);

btnLoad.addEventListener('click', () => {
  loadImageAndMeta().catch((e) => {
    setStatus('Error: ' + e.message);
    btnLoad.disabled = false;
  });
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  renderer.render(scene, camera);
}
tick();
  </script>
</body>
</html>
`;
}

function main() {
  const opts = parseArgs(process.argv);
  const html = buildViewerHtml();
  fs.mkdirSync(path.dirname(opts.out), { recursive: true });
  fs.writeFileSync(opts.out, html, "utf8");
  console.log(`Wrote ${opts.out}`);
  console.log(
    "Open trig-viewer.html: use http + serve repo root for auto meta, or file:// with Ctrl+select jpg+json."
  );
}

main();
