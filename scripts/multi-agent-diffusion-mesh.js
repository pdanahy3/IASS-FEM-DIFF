/**
 * 80×80 vertex agents on a plane: cohesion / separation / alignment + diffusion steering
 * from streamed DDPM steps (see diffusion_mesh_infer_server.py) or offline guide image.
 *
 * Serve from repo root: npx serve .  → open /scripts/multi-agent-diffusion-mesh.html
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const GRID = 80;
const FIXED_PHYS_MIN = -15;
const FIXED_PHYS_MAX = 15;
const FIXED_SPAN = FIXED_PHYS_MAX - FIXED_PHYS_MIN;

/** @param {Uint8Array} u8 length W*H*3 row-major RGB */
function decodeRgbGridToDisp(u8, W, H) {
  const n = W * H;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const j = i * 3;
    const r = u8[j];
    const g = u8[j + 1];
    const b = u8[j + 2];
    out[j] = FIXED_PHYS_MIN + (r / 255) * FIXED_SPAN;
    out[j + 1] = FIXED_PHYS_MIN + (g / 255) * FIXED_SPAN;
    out[j + 2] = FIXED_PHYS_MIN + (b / 255) * FIXED_SPAN;
  }
  return out;
}

function buildQuadIndices(W, H) {
  const quads = (W - 1) * (H - 1);
  const idx = new Uint32Array(quads * 6);
  let o = 0;
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const a = j * W + i;
      const b = a + 1;
      const c = a + W;
      const d = c + 1;
      idx[o++] = a;
      idx[o++] = c;
      idx[o++] = b;
      idx[o++] = b;
      idx[o++] = c;
      idx[o++] = d;
    }
  }
  return idx;
}

function fillBasePositions(base, plane, W, H) {
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const u = i / (W - 1);
      const v = j / (H - 1);
      base[k * 3] = (u - 0.5) * plane;
      base[k * 3 + 1] = (v - 0.5) * plane;
      base[k * 3 + 2] = 0;
    }
  }
}

function dispToColor(dx, dy, dz) {
  const m = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const t = Math.min(1, m / 15);
  return new THREE.Color().setHSL(0.55 - 0.45 * t, 0.75, 0.45 + 0.25 * t);
}

/**
 * @param {Float32Array} pos
 * @param {Float32Array} diffDisp
 * @param {number} W
 * @param {number} H
 */
function updateVertexColors(posAttr, colorAttr, diffDisp, W, H) {
  const n = W * H;
  const col = colorAttr.array;
  for (let k = 0; k < n; k++) {
    const d0 = diffDisp[k * 3];
    const d1 = diffDisp[k * 3 + 1];
    const d2 = diffDisp[k * 3 + 2];
    const c = dispToColor(d0, d1, d2);
    col[k * 3] = c.r;
    col[k * 3 + 1] = c.g;
    col[k * 3 + 2] = c.b;
  }
  colorAttr.needsUpdate = true;
}

/**
 * @param {object} p
 */
function simulateAgents(p) {
  const {
    W,
    H,
    pos,
    vel,
    basePos,
    diffDisp,
    dispScale,
    cohesion,
    separation,
    alignment,
    diffusion,
    dt,
    maxSpeed,
    sepRadius,
  } = p;
  const n = W * H;
  const fx = new Float32Array(n);
  const fy = new Float32Array(n);
  const fz = new Float32Array(n);

  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      let cx = 0;
      let cy = 0;
      let cz = 0;
      let nc = 0;
      let sx = 0;
      let sy = 0;
      let sz = 0;
      let ax = 0;
      let ay = 0;
      let az = 0;
      let na = 0;
      const px = pos[k * 3];
      const py = pos[k * 3 + 1];
      const pz = pos[k * 3 + 2];
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dj === 0) continue;
          const ni = i + di;
          const nj = j + dj;
          if (ni < 0 || ni >= W || nj < 0 || nj >= H) continue;
          const nk = nj * W + ni;
          const ox = pos[nk * 3];
          const oy = pos[nk * 3 + 1];
          const oz = pos[nk * 3 + 2];
          const dx = px - ox;
          const dy = py - oy;
          const dz = pz - oz;
          const dist = Math.hypot(dx, dy, dz) + 1e-7;
          cx += ox;
          cy += oy;
          cz += oz;
          nc++;
          if (dist < sepRadius) {
            const s = (sepRadius - dist) / (dist * sepRadius);
            sx += dx * s;
            sy += dy * s;
            sz += dz * s;
          }
          ax += vel[nk * 3];
          ay += vel[nk * 3 + 1];
          az += vel[nk * 3 + 2];
          na++;
        }
      }
      let fcx = 0;
      let fcy = 0;
      let fcz = 0;
      if (nc > 0) {
        cx /= nc;
        cy /= nc;
        cz /= nc;
        fcx = (cx - px) * cohesion;
        fcy = (cy - py) * cohesion;
        fcz = (cz - pz) * cohesion;
      }
      const fsx = sx * separation;
      const fsy = sy * separation;
      const fsz = sz * separation;
      let fax = 0;
      let fay = 0;
      let faz = 0;
      if (na > 0) {
        ax /= na;
        ay /= na;
        az /= na;
        fax = (ax - vel[k * 3]) * alignment;
        fay = (ay - vel[k * 3 + 1]) * alignment;
        faz = (az - vel[k * 3 + 2]) * alignment;
      }
      const d0 = diffDisp[k * 3];
      const d1 = diffDisp[k * 3 + 1];
      const d2 = diffDisp[k * 3 + 2];
      const tx = basePos[k * 3] + dispScale * d0;
      const ty = basePos[k * 3 + 1] + dispScale * d1;
      const tz = basePos[k * 3 + 2] + dispScale * d2;
      const fdx = (tx - px) * diffusion;
      const fdy = (ty - py) * diffusion;
      const fdz = (tz - pz) * diffusion;
      fx[k] = fcx + fsx + fax + fdx;
      fy[k] = fcy + fsy + fay + fdy;
      fz[k] = fcz + fsz + faz + fdz;
    }
  }

  const damp = 0.988;
  for (let k = 0; k < n; k++) {
    vel[k * 3] = (vel[k * 3] + fx[k] * dt) * damp;
    vel[k * 3 + 1] = (vel[k * 3 + 1] + fy[k] * dt) * damp;
    vel[k * 3 + 2] = (vel[k * 3 + 2] + fz[k] * dt) * damp;
    const sp = Math.hypot(vel[k * 3], vel[k * 3 + 1], vel[k * 3 + 2]);
    if (sp > maxSpeed) {
      const sc = maxSpeed / sp;
      vel[k * 3] *= sc;
      vel[k * 3 + 1] *= sc;
      vel[k * 3 + 2] *= sc;
    }
    pos[k * 3] += vel[k * 3] * dt;
    pos[k * 3 + 1] += vel[k * 3 + 1] * dt;
    pos[k * 3 + 2] += vel[k * 3 + 2] * dt;
  }
}

function b64ToUint8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = /** @type {string} */ (r.result);
      const i = s.indexOf(',');
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/** Draw image to canvas 80×80 and return ImageData RGBA */
function imageFileToGridRGBA(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const c = document.createElement('canvas');
      c.width = GRID;
      c.height = GRID;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0, GRID, GRID);
      const im = ctx.getImageData(0, 0, GRID, GRID);
      resolve(im.data);
    };
    img.onerror = reject;
    img.src = url;
  });
}

function rgbaToRgbUint8(rgba) {
  const n = GRID * GRID;
  const u8 = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    u8[i * 3] = rgba[i * 4];
    u8[i * 3 + 1] = rgba[i * 4 + 1];
    u8[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return u8;
}

// ——— scene ———
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x141418);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(2.2, 2.0, 2.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.2);
controls.update();

scene.add(new THREE.AmbientLight(0x9090a0, 0.9));
const dl = new THREE.DirectionalLight(0xffffff, 0.55);
dl.position.set(2, 4, 3);
scene.add(dl);

const W = GRID;
const H = GRID;
const indices = buildQuadIndices(W, H);

const basePos = new Float32Array(W * H * 3);
fillBasePositions(basePos, 2.0, W, H);
const pos = new Float32Array(basePos.length);
const vel = new Float32Array(basePos.length);
const diffDisp = new Float32Array(basePos.length);
for (let i = 0; i < diffDisp.length; i++) diffDisp[i] = 0;

function jitterReset(amt) {
  for (let k = 0; k < W * H; k++) {
    pos[k * 3] = basePos[k * 3] + (Math.random() - 0.5) * amt;
    pos[k * 3 + 1] = basePos[k * 3 + 1] + (Math.random() - 0.5) * amt;
    pos[k * 3 + 2] = basePos[k * 3 + 2] + (Math.random() - 0.5) * amt * 0.5;
    vel[k * 3] = 0;
    vel[k * 3 + 1] = 0;
    vel[k * 3 + 2] = 0;
  }
}
jitterReset(0.04);

const geom = new THREE.BufferGeometry();
geom.setIndex(new THREE.BufferAttribute(indices, 1));
geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
const colors = new Float32Array(W * H * 3);
geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
updateVertexColors(null, geom.attributes.color, diffDisp, W, H);
geom.computeVertexNormals();

const mat = new THREE.MeshPhongMaterial({
  vertexColors: true,
  side: THREE.DoubleSide,
  flatShading: false,
  shininess: 22,
});
const mesh = new THREE.Mesh(geom, mat);
scene.add(mesh);

let pollAbort = false;

function readSlider(id, def) {
  const el = document.getElementById(id);
  if (!el) return def;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : def;
}

function animate() {
  requestAnimationFrame(animate);
  const plane = readSlider('planeSize', 2);
  fillBasePositions(basePos, plane, W, H);
  const dispScale = readSlider('dispScale', 1);
  const cohesion = readSlider('cohesion', 0.8);
  const separation = readSlider('separation', 2.5);
  const alignment = readSlider('alignment', 0.6);
  const diffusion = readSlider('diffusion', 1.2);
  const maxSpeed = readSlider('maxSpeed', 1.8);
  const sepRadius = readSlider('sepRadius', 0.12);

  simulateAgents({
    W,
    H,
    pos,
    vel,
    basePos,
    diffDisp,
    dispScale,
    cohesion,
    separation,
    alignment,
    diffusion,
    dt: 0.018,
    maxSpeed,
    sepRadius,
  });

  const pAttr = geom.attributes.position;
  pAttr.needsUpdate = true;
  geom.computeVertexNormals();

  updateVertexColors(null, geom.attributes.color, diffDisp, W, H);

  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

document.getElementById('btnReset')?.addEventListener('click', () => {
  const plane = readSlider('planeSize', 2);
  fillBasePositions(basePos, plane, W, H);
  jitterReset(0.04);
  setStatus('Mesh reset with jitter.');
});

document.getElementById('btnStop')?.addEventListener('click', () => {
  pollAbort = true;
  setStatus('Stopped polling server.');
});

document.getElementById('btnOfflineGuide')?.addEventListener('click', async () => {
  const f = document.getElementById('guideFile')?.files?.[0];
  if (!f) {
    setStatus('Pick a guide/seed image first.');
    return;
  }
  try {
    const rgba = await imageFileToGridRGBA(f);
    const rgb = rgbaToRgbUint8(rgba);
    const d = decodeRgbGridToDisp(rgb, W, H);
    diffDisp.set(d);
    document.getElementById('offlineMode').checked = true;
    setStatus('Offline guide loaded (80×80 decode).');
  } catch (e) {
    setStatus(`Decode error: ${e}`);
  }
});

document.getElementById('btnStartServer')?.addEventListener('click', async () => {
  pollAbort = false;
  const baseUrl = document.getElementById('serverUrl')?.value?.trim() || 'http://127.0.0.1:8765';
  const checkpoint = document.getElementById('checkpointPath')?.value?.trim();
  if (!checkpoint) {
    setStatus('Set checkpoint path (relative to repo root on server).');
    return;
  }
  const seed = parseInt(document.getElementById('inferSeed')?.value || '42', 10);
  const steps = parseInt(document.getElementById('inferSteps')?.value || '200', 10);
  const strength = parseFloat(document.getElementById('inferStrength')?.value || '0.35');
  const goalMix = parseFloat(document.getElementById('inferGoalMix')?.value || '0');
  const device = document.getElementById('inferDevice')?.value || 'cuda';

  const body = {
    checkpoint,
    seed,
    steps,
    strength,
    goal_mix: goalMix,
    device,
  };

  const seedF = document.getElementById('seedFile')?.files?.[0];
  const goalF = document.getElementById('goalFile')?.files?.[0];
  if (seedF) body.seed_image_png_base64 = await fileToBase64(seedF);
  if (goalF) body.goal_image_png_base64 = await fileToBase64(goalF);

  setStatus('Starting session…');
  let res;
  try {
    res = await fetch(`${baseUrl}/api/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    setStatus(`Network error (is the server running?): ${e}`);
    return;
  }
  const j = await res.json().catch(() => ({}));
  if (!res.ok || j.error) {
    setStatus(`Start failed: ${j.error || res.status}`);
    return;
  }
  const sid = j.session;
  setStatus(`Session ${sid}; streaming steps…`);
  document.getElementById('offlineMode').checked = false;

  (async () => {
    while (!pollAbort) {
      let r;
      try {
        r = await fetch(`${baseUrl}/api/session/${sid}/next`);
      } catch (e) {
        setStatus(`Poll error: ${e}`);
        break;
      }
      const item = await r.json().catch(() => ({}));
      if (item.pending) continue;
      if (item.error) {
        setStatus(`Run error: ${item.error}`);
        break;
      }
      if (item.done) {
        setStatus('Diffusion finished (last field held).');
        break;
      }
      if (document.getElementById('offlineMode')?.checked) break;
      const u8 = b64ToUint8(item.rgb_b64);
      const d = decodeRgbGridToDisp(u8, item.w, item.h);
      diffDisp.set(d);
      setStatus(`Step ${item.step}  t=${item.t}`);
    }
  })();
});
