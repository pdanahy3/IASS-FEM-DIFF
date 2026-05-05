/**
 * Generate 80×80 RGB images from parametric (u,v) trigonometric surfaces.
 * u and v sample degree ranges from the full permutation set: each axis uses a span
 * [min,max] with min,max ∈ {0,90,180,270,360} and min≤max (15 u-spans × 15 v-spans = 225
 * domains). That Cartesian product extends the total count with formula × coefficient combos.
 * Per pixel:
 * formula triple; R=x, G=y, B=z with symmetric encoding: 0 → mid gray (127.5),
 * −R_ext → 0, +R_ext → 255, where R_ext = max(|min|, |max|) on that channel.
 * If a channel is spatially constant (no u,v variation), it is encoded as mid
 * gray (128) so solid “1×coeff” axes do not wash out the image as all white.
 * Writes JPEG (RGB via jpeg-js). Run `npm install` once in the repo root.
 *
 * Usage (from repo root):
 *   npm install
 *   node scripts/generate-trig-surfaces.js
 *   node scripts/generate-trig-surfaces.js --max 0 --force
 *
 * Options:
 *   --out <dir>        Output directory (default: data/processed/displacement_rgb/trig)
 *   --size <n>         Grid size (default: 80)
 *   --coeff-min <n>    Coefficient lower bound (default: 0.2)
 *   --coeff-max <n>    Coefficient upper bound (default: 6)
 *   --coeff-step <n>   Step for coefficient sweep (default: 0.2)
 *   --jpeg-quality <n> JPEG quality 1–100 (default: 95)
 *   --max <n>          Sample N surfaces uniformly across the full Cartesian product of
 *                      formulas × coefficients × UV domains (default: 500).
 *                      Use 0 to write every combination (requires --force if huge).
 *   --force            Allow full enumeration when total combinations is large
 */

import fs from "fs";
import { createRequire } from "module";
import path from "path";
import { fileURLToPath } from "url";

const require = createRequire(import.meta.url);
const jpeg = require("jpeg-js");

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..");

/** Degrees → radians */
const DEG2RAD = Math.PI / 180;

/** Knots for 90° steps from 0° to 360° */
const DEG_KNOTS = [0, 90, 180, 270, 360];

/**
 * All [min,max] spans with min,max ∈ DEG_KNOTS and min ≤ max (15 pairs).
 * @returns {Array<[number, number]>}
 */
function buildDegSpanPairs() {
  const pairs = [];
  for (let i = 0; i < DEG_KNOTS.length; i++) {
    for (let j = i; j < DEG_KNOTS.length; j++) {
      pairs.push([DEG_KNOTS[i], DEG_KNOTS[j]]);
    }
  }
  return pairs;
}

/**
 * Full set of u,v domains: 15×15 = 225 quadruples.
 * @returns {Array<{ uMinDeg: number, uMaxDeg: number, vMinDeg: number, vMaxDeg: number }>}
 */
function buildUvDomainQuads() {
  const spans = buildDegSpanPairs();
  const quads = [];
  for (const [uMinDeg, uMaxDeg] of spans) {
    for (const [vMinDeg, vMaxDeg] of spans) {
      quads.push({ uMinDeg, uMaxDeg, vMinDeg, vMaxDeg });
    }
  }
  return quads;
}

/**
 * @param {Float32Array} uArr
 * @param {Float32Array} vArr
 * @param {number} W
 * @param {number} H
 * @param {{ uMinDeg: number, uMaxDeg: number, vMinDeg: number, vMaxDeg: number }} dom
 */
function fillUvArrays(uArr, vArr, W, H, dom) {
  const uDen = W - 1;
  const vDen = H - 1;
  const uSpan = dom.uMaxDeg - dom.uMinDeg;
  const vSpan = dom.vMaxDeg - dom.vMinDeg;
  for (let j = 0; j < H; j++) {
    const vDeg = dom.vMinDeg + (j / vDen) * vSpan;
    const v = vDeg * DEG2RAD;
    for (let i = 0; i < W; i++) {
      const uDeg = dom.uMinDeg + (i / uDen) * uSpan;
      const u = uDeg * DEG2RAD;
      const k = j * W + i;
      uArr[k] = u;
      vArr[k] = v;
    }
  }
}

/** @type {Array<(u: number, v: number) => number>} */
const BASIS = [
  (u, v) => 1,
  (u, v) => Math.cos(u),
  (u, v) => Math.sin(u),
  (u, v) => Math.cos(v),
  (u, v) => Math.sin(v),
  (u, v) => Math.cos(u) * Math.sin(u),
  (u, v) => Math.cos(u) + Math.sin(u),
  (u, v) => Math.cos(u) * Math.sin(v),
  (u, v) => Math.cos(u) + Math.sin(v),
  (u, v) => Math.cos(u + Math.sin(v)) + Math.cos(v),
];

/**
 * Flatten order: ix, iy, iz, icx, icy, icz, idom (idom innermost = fastest-changing).
 * @param {number} linear
 * @param {number} nF
 * @param {number} nC
 * @param {number} nDomain
 * @param {number[]} coeffs
 * @param {ReturnType<typeof buildUvDomainQuads>} domainQuads
 */
function decodeLinear(linear, nF, nC, nDomain, coeffs, domainQuads) {
  let t = linear;
  const idom = t % nDomain;
  t = (t - idom) / nDomain;
  const icz = t % nC;
  t = (t - icz) / nC;
  const icy = t % nC;
  t = (t - icy) / nC;
  const icx = t % nC;
  t = (t - icx) / nC;
  const iz = t % nF;
  t = (t - iz) / nF;
  const iy = t % nF;
  t = (t - iy) / nF;
  const ix = t;
  const domain = domainQuads[idom];
  return {
    ix,
    iy,
    iz,
    cx: coeffs[icx],
    cy: coeffs[icy],
    cz: coeffs[icz],
    idom,
    domain,
  };
}

/**
 * Evenly spaced indices in [0, total-1], deduped.
 * If sampleCount <= 0 or sampleCount >= total, returns every index (full set).
 * @param {number} sampleCount
 * @param {number} total
 */
function sampleLinearIndices(sampleCount, total) {
  if (total <= 0) return [];
  if (sampleCount <= 0 || sampleCount >= total) {
    return Array.from({ length: total }, (_, i) => i);
  }
  const n = Math.min(sampleCount, total);
  const set = new Set();
  if (n === 1) {
    set.add(Math.floor((total - 1) / 2));
  } else {
    for (let s = 0; s < n; s++) {
      const lin = Math.round((s / (n - 1)) * (total - 1));
      set.add(lin);
    }
  }
  return [...set].sort((a, b) => a - b);
}

const BASIS_LABELS = [
  "1",
  "cos(u)",
  "sin(u)",
  "cos(v)",
  "sin(v)",
  "cos(u)*sin(u)",
  "cos(u)+sin(u)",
  "cos(u)*sin(v)",
  "cos(u)+sin(v)",
  "cos(u+sin(v))+cos(v)",
];

/**
 * @param {string} filePath
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array} rgba length width*height*4, row-major top-down
 * @param {number} quality 1–100
 */
function writeJpeg(filePath, width, height, rgba, quality) {
  const raw = Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength);
  const encoded = jpeg.encode(
    { data: raw, width, height },
    Math.min(100, Math.max(1, Math.round(quality)))
  );
  fs.writeFileSync(filePath, encoded.data);
}

function parseArgs(argv) {
  const opts = {
    out: path.join(REPO_ROOT, "data", "processed", "displacement_rgb", "trig"),
    size: 80,
    coeffMin: 0.2,
    coeffMax: 6,
    coeffStep: 0.2,
    jpegQuality: 95,
    max: 500,
    force: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--out") opts.out = path.resolve(argv[++i] ?? "");
    else if (a === "--size") opts.size = Number(argv[++i]);
    else if (a === "--coeff-min") opts.coeffMin = Number(argv[++i]);
    else if (a === "--coeff-max") opts.coeffMax = Number(argv[++i]);
    else if (a === "--coeff-step") opts.coeffStep = Number(argv[++i]);
    else if (a === "--jpeg-quality") opts.jpegQuality = Number(argv[++i]);
    else if (a === "--max") opts.max = Number(argv[++i]);
    else if (a === "--force") opts.force = true;
    else if (a === "--help" || a === "-h") {
      console.log(
        "generate-trig-surfaces.js — see file header for options."
      );
      process.exit(0);
    }
  }
  if (!opts.out || Number.isNaN(opts.size) || opts.size < 2) {
    console.error("Invalid --out or --size");
    process.exit(1);
  }
  return opts;
}

/** @param {number} min @param {number} max @param {number} step */
function coeffValues(min, max, step) {
  const out = [];
  if (step <= 0) {
    console.error("coeff-step must be positive");
    process.exit(1);
  }
  for (let c = min; c <= max + 1e-9; c += step) {
    out.push(Math.round(c * 1e9) / 1e9);
  }
  return out;
}

/**
 * @param {Float32Array} xs
 * @param {Float32Array} ys
 * @param {Float32Array} zs
 */
function displacementStats(xs, ys, zs) {
  let maxDisp = 0;
  let sumDisp = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) {
    const d = Math.hypot(xs[i], ys[i], zs[i]);
    if (d > maxDisp) maxDisp = d;
    sumDisp += d;
  }
  return { maxDisp, avgDisp: sumDisp / n };
}

/**
 * Min/max and symmetric extent R = max(|min|, |max|) for zero-mid encoding.
 * @param {Float32Array} arr
 */
function channelMinMaxExtent(arr) {
  let mn = Infinity;
  let mx = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (!Number.isFinite(mn)) mn = 0;
  if (!Number.isFinite(mx)) mx = 0;
  const R = Math.max(Math.abs(mn), Math.abs(mx), 1e-12);
  return { mn, mx, R };
}

const CHANNEL_CONST_EPS = 1e-7;

/**
 * R ← x, G ← y, B ← z. value 0 → 127.5; −R → 0; +R → 255 (clamped).
 * Spatially flat channels → 128 (avoids all-white when x,y,z are constants).
 * @param {Float32Array} xs
 * @param {Float32Array} ys
 * @param {Float32Array} zs
 */
function zeroMidChannelsToRGBA(xs, ys, zs, width, height) {
  const X = channelMinMaxExtent(xs);
  const Y = channelMinMaxExtent(ys);
  const Z = channelMinMaxExtent(zs);
  const xConst = X.mx - X.mn <= CHANNEL_CONST_EPS;
  const yConst = Y.mx - Y.mn <= CHANNEL_CONST_EPS;
  const zConst = Z.mx - Z.mn <= CHANNEL_CONST_EPS;
  const rgba = new Uint8Array(width * height * 4);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const k = j * width + i;
      const encVar = (v, R) =>
        Math.round(Math.min(255, Math.max(0, 127.5 * (1 + v / R))));
      const p = k * 4;
      rgba[p] = xConst ? 128 : encVar(xs[k], X.R);
      rgba[p + 1] = yConst ? 128 : encVar(ys[k], Y.R);
      rgba[p + 2] = zConst ? 128 : encVar(zs[k], Z.R);
      rgba[p + 3] = 255;
    }
  }
  return {
    rgba,
    norm: {
      x: {
        min: X.mn,
        max: X.mx,
        extent_R: xConst ? null : X.R,
        spatially_constant: xConst,
      },
      y: {
        min: Y.mn,
        max: Y.mx,
        extent_R: yConst ? null : Y.R,
        spatially_constant: yConst,
      },
      z: {
        min: Z.mn,
        max: Z.mx,
        extent_R: zConst ? null : Z.R,
        spatially_constant: zConst,
      },
    },
  };
}

function main() {
  const opts = parseArgs(process.argv);
  const coeffs = coeffValues(opts.coeffMin, opts.coeffMax, opts.coeffStep);
  const domainQuads = buildUvDomainQuads();
  const nDomain = domainQuads.length;
  const nF = BASIS.length;
  const nC = coeffs.length;
  const totalCombos = nF * nF * nF * nC * nC * nC * nDomain;

  console.log(
    `UV domain permutations: ${nDomain} (90° spans on u and v, min≤max; extends formula×coeff space).`
  );

  const wantAll = opts.max === 0;
  if (totalCombos > 50_000 && wantAll && !opts.force) {
    console.error(
      `Refusing to write all ${totalCombos} images without --force. ` +
        `Use --max N to sample N surfaces across the full space, or add --force.`
    );
    process.exit(1);
  }

  const sampleCount = wantAll ? totalCombos : opts.max;
  const linearIndices = sampleLinearIndices(sampleCount, totalCombos);

  fs.mkdirSync(opts.out, { recursive: true });

  const W = opts.size;
  const H = opts.size;
  const N = W * H;
  const uArr = new Float32Array(N);
  const vArr = new Float32Array(N);

  let written = 0;
  if (!wantAll) {
    console.log(
      `Sampling ${linearIndices.length} of ${totalCombos} combinations (uniform linear index spacing).`
    );
  }

  for (const linear of linearIndices) {
    const { ix, iy, iz, cx, cy, cz, idom, domain } = decodeLinear(
      linear,
      nF,
      nC,
      nDomain,
      coeffs,
      domainQuads
    );
    fillUvArrays(uArr, vArr, W, H, domain);
    const fx = BASIS[ix];
    const fy = BASIS[iy];
    const fz = BASIS[iz];

    const xs = new Float32Array(N);
    const ys = new Float32Array(N);
    const zs = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      const u = uArr[k];
      const v = vArr[k];
      xs[k] = cx * fx(u, v);
      ys[k] = cy * fy(u, v);
      zs[k] = cz * fz(u, v);
    }

    const { maxDisp, avgDisp } = displacementStats(xs, ys, zs);
    const { rgba, norm } = zeroMidChannelsToRGBA(xs, ys, zs, W, H);

    const domTag = [
      `d${String(idom).padStart(3, "0")}`,
      `u${domain.uMinDeg}_${domain.uMaxDeg}`,
      `v${domain.vMinDeg}_${domain.vMaxDeg}`,
    ].join("_");
    const tag = [
      domTag,
      String(ix).padStart(2, "0"),
      String(iy).padStart(2, "0"),
      String(iz).padStart(2, "0"),
      String(Math.round(cx * 1000)).padStart(5, "0"),
      String(Math.round(cy * 1000)).padStart(5, "0"),
      String(Math.round(cz * 1000)).padStart(5, "0"),
    ].join("_");
    const base = `trig_${tag}`;
    const imgPath = path.join(opts.out, `${base}.jpg`);
    const metaPath = path.join(opts.out, `${base}.meta.json`);

    writeJpeg(imgPath, W, H, rgba, opts.jpegQuality);

    const meta = {
      max_displacement: maxDisp,
      avg_displacement: avgDisp,
      max_stress: 0,
      avg_stress: 0,
      extra: {
        source: "scripts/generate-trig-surfaces.js",
        format: "jpeg",
        jpeg_quality: opts.jpegQuality,
        grid: { width: W, height: H },
        linear_index: linear,
        sampling: wantAll
          ? { mode: "full_enumeration", total_combinations: totalCombos }
          : {
              mode: "uniform_sample",
              requested_samples: opts.max,
              written: linearIndices.length,
              total_combinations: totalCombos,
            },
        uv_domain_index: idom,
        uv_domain_degrees: {
          u: [domain.uMinDeg, domain.uMaxDeg],
          v: [domain.vMinDeg, domain.vMaxDeg],
        },
        rgb_mapping: {
          R: "x",
          G: "y",
          B: "z",
          zero_to_byte: 127.5,
          extent:
            "per_channel max(|min|,|max|); spatially flat channel → byte 128",
        },
        formula_indices: { x: ix, y: iy, z: iz },
        formula_labels: {
          x: BASIS_LABELS[ix],
          y: BASIS_LABELS[iy],
          z: BASIS_LABELS[iz],
        },
        coefficients: { x: cx, y: cy, z: cz },
        channel_min_max_raw: {
          x: {
            min: norm.x.min,
            max: norm.x.max,
            extent_R: norm.x.extent_R,
            spatially_constant: norm.x.spatially_constant,
          },
          y: {
            min: norm.y.min,
            max: norm.y.max,
            extent_R: norm.y.extent_R,
            spatially_constant: norm.y.spatially_constant,
          },
          z: {
            min: norm.z.min,
            max: norm.z.max,
            extent_R: norm.z.extent_R,
            spatially_constant: norm.z.spatially_constant,
          },
        },
      },
    };
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), "utf8");

    written++;
    if (written % 100 === 0) {
      console.log(`… ${written} images`);
    }
  }

  console.log(`Wrote ${written} JPEG + meta pairs under ${opts.out}`);
  if (!wantAll && written < totalCombos) {
    console.log(
      `Sampled ${written} of ${totalCombos} possible surfaces (see extra.sampling in each meta.json).`
    );
  }
}

main();
