// Procedural New York: geography (land mask), districts, ~60k building tiers, landmarks, bridges,
// lights. Everything is generated on the CPU once at start-up into typed arrays.
//
// Coordinates: world X = east, Z = south (-Z = north), Y up, metres. The Manhattan grid frame is
// (s, a): s runs along the streets (ESE, +s = East River side), a runs along the avenues
// (NNE, +a = uptown). Origin = the Battery. Grid rotation is 29 degrees.
import { rng } from '../../engine/math.js';

export const GRID_DEG = 29;
export const GC = Math.cos(GRID_DEG * Math.PI / 180);
export const GS = Math.sin(GRID_DEG * Math.PI / 180);
export const GRID_ROT = GRID_DEG * Math.PI / 180;
export const gridToWorld = (s, a) => [s * GC + a * GS, s * GS - a * GC];
export const worldToGrid = (x, z) => [x * GC + z * GS, x * GS - z * GC];

// block / street layout (metres)
export const DS = 275, DA = 80, AVE_W = 28, ST_W = 18;

// ---- land mask (signed distance, + inside land) ------------------------------------------
export const LAND = { s0: -8000, a0: -8000, cell: 10, w: 1600, h: 2400 };
// ---- height map for sun shadows (max building height per 10 m cell, with max-pooled mips) ---
export const HMAP = { s0: -5500, a0: -3500, cell: 10, w: 1100, h: 1900, levels: 5 };

const interp = (tbl, x) => {
  if (x <= tbl[0][0]) return tbl[0][1];
  for (let i = 1; i < tbl.length; i++) {
    if (x <= tbl[i][0]) { const t = (x - tbl[i - 1][0]) / (tbl[i][0] - tbl[i - 1][0]); return tbl[i - 1][1] + t * (tbl[i][1] - tbl[i - 1][1]); }
  }
  return tbl[tbl.length - 1][1];
};
// shore tables, km: [a, s]
const MAN_W = [[-0.15, 0], [-0.05, -0.18], [0.15, -0.42], [0.45, -0.62], [0.9, -0.72], [1.3, -0.95], [1.8, -1.18], [2.4, -1.40], [3.1, -1.60], [4.0, -1.82], [5.0, -1.92], [6.5, -1.95], [8.0, -1.92], [9.5, -1.95], [12, -1.9], [20, -1.9]];
const MAN_E = [[-0.15, 0], [-0.05, 0.2], [0.1, 0.40], [0.35, 0.58], [0.9, 0.78], [1.5, 0.88], [2.1, 1.02], [2.7, 1.18], [3.3, 1.38], [4.0, 1.55], [5.0, 1.62], [6.5, 1.62], [8.0, 1.6], [9.5, 1.55], [12, 1.5], [20, 1.45]];
const NJ = [[-8, -3.2], [-4, -2.9], [-2.4, -2.45], [-1.0, -2.3], [0.5, -2.45], [1.5, -2.8], [2.5, -3.0], [4, -3.4], [5.5, -3.55], [7, -3.62], [9, -3.7], [12, -3.8], [20, -3.8]];
const BK = [[-8, 3.6], [-4, 2.9], [-2.2, 1.75], [-1.0, 1.38], [0, 1.2], [1.0, 1.18], [1.5, 1.40], [2.2, 1.72], [3.5, 2.4], [5.0, 2.55], [6.5, 2.62], [7.6, 2.7], [9, 2.75], [12, 2.85], [20, 3.0]];

const manW = (a) => interp(MAN_W, a / 1000) * 1000;
const manE = (a) => interp(MAN_E, a / 1000) * 1000;
const PIERS = [];
for (let k = 0; k < 19; k++) {
  const a = 950 + k * 340;
  PIERS.push({ a, hw: 22, len: 90 + ((k * 53) % 11) * 14, east: false });
}
for (let k = 0; k < 7; k++) PIERS.push({ a: 5200 + k * 520, hw: 20, len: 110, east: true });
for (const a of [3900, 4250, 4700, 5350, 5750, 6150, 7050]) PIERS.push({ a, hw: 32, len: 240 + (a % 7) * 12, nj: true });
const ISLANDS = [
  { s: -1350, a: -3000, rs: 300, ra: 300 },   // Liberty Island
  { s: -1550, a: -1800, rs: 220, ra: 200 },   // Ellis Island
  { s: 1870, a: 7900, rs: 90, ra: 750 },      // Roosevelt Island
];
export const LIBERTY = { s: -1350, a: -3000 };

function insideLand(s, a) {
  // s, a in metres
  if (s < interp(NJ, a / 1000) * 1000) return true;
  if (s > interp(BK, a / 1000) * 1000) return true;
  if (a > -150 && s >= manW(a) && s <= manE(a)) return true;
  if (a < -6000 && s > -3200 && s < 800) return true;
  for (const I of ISLANDS) { const ds = (s - I.s) / I.rs, da = (a - I.a) / I.ra; if (ds * ds + da * da < 1) return true; }
  for (const P of PIERS) {
    if (Math.abs(a - P.a) < P.hw) {
      if (P.nj) { const e = interp(NJ, P.a / 1000) * 1000; if (s > e - 10 && s < e + P.len) return true; continue; }
      if (!P.east && s < manW(P.a) && s > manW(P.a) - P.len) return true;
      if (P.east && s > manE(P.a) && s < manE(P.a) + P.len) return true;
    }
  }
  return false;
}

export function buildLandMask() {
  const { w, h, cell, s0, a0 } = LAND;
  const mask = new Uint8Array(w * h);
  for (let j = 0; j < h; j++) {
    const a = a0 + (j + 0.5) * cell;
    for (let i = 0; i < w; i++) mask[j * w + i] = insideLand(s0 + (i + 0.5) * cell, a) ? 1 : 0;
  }
  // two chamfer passes -> distance to the opposite class, in metres
  const d = new Float32Array(w * h);
  const INF = 1e9;
  const A = 1, B = 1.4142;
  const dt = (target) => {
    for (let k = 0; k < w * h; k++) d[k] = mask[k] === target ? INF : 0;
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const k = j * w + i; let v = d[k];
      if (i > 0) v = Math.min(v, d[k - 1] + A);
      if (j > 0) { v = Math.min(v, d[k - w] + A); if (i > 0) v = Math.min(v, d[k - w - 1] + B); if (i < w - 1) v = Math.min(v, d[k - w + 1] + B); }
      d[k] = v;
    }
    for (let j = h - 1; j >= 0; j--) for (let i = w - 1; i >= 0; i--) {
      const k = j * w + i; let v = d[k];
      if (i < w - 1) v = Math.min(v, d[k + 1] + A);
      if (j < h - 1) { v = Math.min(v, d[k + w] + A); if (i < w - 1) v = Math.min(v, d[k + w + 1] + B); if (i > 0) v = Math.min(v, d[k + w - 1] + B); }
      d[k] = v;
    }
  };
  const depth = new Float32Array(w * h);        // metres inside land (>0) / outside (<0)
  dt(1);   // land cells get distance to water
  for (let k = 0; k < w * h; k++) depth[k] = mask[k] ? (d[k] - 0.5) * cell : 0;
  dt(0);   // water cells get distance to land
  for (let k = 0; k < w * h; k++) if (!mask[k]) depth[k] = -(d[k] - 0.5) * cell;
  const tex = new Uint8Array(w * h);
  for (let k = 0; k < w * h; k++) tex[k] = Math.max(0, Math.min(255, Math.round((depth[k] / 320 + 0.5) * 255)));
  return { depth, tex };
}

// ---- float -> half for the r16float height map -------------------------------------------
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toHalf(v) {
  f32[0] = v; const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const e = ((x >>> 23) & 0xff) - 127 + 15;
  if (e <= 0) return sign;
  if (e >= 31) return sign | 0x7bff;
  return sign | (e << 10) | ((x & 0x7fffff) >>> 13);
}

// ---- builder -----------------------------------------------------------------------------
class Buf {
  constructor(stride) { this.stride = stride; this.a = new Float32Array(stride * 4096); this.n = 0; }
  push(...v) {
    if ((this.n + 1) * this.stride > this.a.length) { const b = new Float32Array(this.a.length * 2); b.set(this.a); this.a = b; }
    this.a.set(v, this.n * this.stride); this.n++;
  }
  done() { return this.a.slice(0, this.n * this.stride); }
}

// materials (match the building shader table)
export const MAT = { BRICK: 0, LIME: 1, GLASS_B: 2, GLASS_D: 3, STEEL: 4, WHITE: 5, MECH: 6, WOOD: 7, GRANITE: 8, COPPER: 9, BOAT: 10, DARK: 11, DECK: 12, ORANGE: 13, SAND: 14, BLACKGLASS: 15 };
// window styles
export const STY = { NONE: 0, PUNCH: 1, RIBBON: 2, GRID: 3, CURTAIN: 4, ARCH: 5 };

export function buildCity(log = () => {}) {
  const t0 = performance.now();
  const R = rng(20240917);
  const land = buildLandMask();
  const inst = new Buf(12);       // a: x z y0 h | b: hw hd seed mat | c: taper rot wash style
  const beams = new Buf(8);       // p0 xyz w | p1 xyz kind
  const sprites = new Buf(8);     // pos size | rgb*I rate
  const hm = new Float32Array(HMAP.w * HMAP.h);
  const reserved = [];            // [s0, s1, a0, a1] in metres

  const depthAt = (s, a) => {
    const i = Math.floor((s - LAND.s0) / LAND.cell), j = Math.floor((a - LAND.a0) / LAND.cell);
    if (i < 0 || j < 0 || i >= LAND.w || j >= LAND.h) return 50;
    return land.depth[j * LAND.w + i];
  };
  const raster = (sc, ac, hws, hda, top) => {
    const i0 = Math.max(0, Math.floor((sc - hws - HMAP.s0) / HMAP.cell)), i1 = Math.min(HMAP.w - 1, Math.floor((sc + hws - HMAP.s0) / HMAP.cell));
    const j0 = Math.max(0, Math.floor((ac - hda - HMAP.a0) / HMAP.cell)), j1 = Math.min(HMAP.h - 1, Math.floor((ac + hda - HMAP.a0) / HMAP.cell));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) { const k = j * HMAP.w + i; if (top > hm[k]) hm[k] = top; }
  };
  let seedCtr = 1;
  // one box/frustum tier; (sc, ac) in grid metres
  const box = (sc, ac, y0, h, hw, hd, mat, sty, o = {}) => {
    const [x, z] = gridToWorld(sc, ac);
    inst.push(x, z, y0, h, hw, hd, o.seed ?? (seedCtr++ * 7.13 % 977), mat, o.taper ?? 1, o.rot ?? GRID_ROT, o.wash ?? 0, sty);
    if (!o.noShadow) raster(sc, ac, hw * ((o.taper ?? 1) > 0.5 ? 1 : 0.7) + 1, hd * ((o.taper ?? 1) > 0.5 ? 1 : 0.7) + 1, y0 + h);
  };
  const beam = (p0, p1, w, kind) => beams.push(p0[0], p0[1], p0[2], w, p1[0], p1[1], p1[2], kind);
  const gb = (s, a, y) => { const [x, z] = gridToWorld(s, a); return [x, y, z]; };   // grid point -> world xyz
  const light = (p, size, r, g, b, rate = 0) => sprites.push(p[0], p[1], p[2], size, r, g, b, rate);
  const reserve = (s, a, hs, ha) => reserved.push([s - hs, s + hs, a - ha, a + ha]);
  const isReserved = (s0, s1, a0, a1) => reserved.some((r) => s1 > r[0] && s0 < r[1] && a1 > r[2] && a0 < r[3]);

  // ========================================================================================
  // Landmarks (hand built, grid metres)
  // ========================================================================================
  const lm = (s, a, hs, ha) => reserve(s, a, hs, ha);

  // Empire State Building, 34th St & 5th Ave
  {
    const s = -60, a = 6450; lm(s, a, 90, 45);
    box(s, a, 0, 23, 65, 29, MAT.LIME, STY.PUNCH, { wash: 0.5 });
    box(s, a, 23, 38, 58, 25, MAT.LIME, STY.PUNCH, { wash: 0.5 });
    box(s, a, 61, 154, 31, 21, MAT.LIME, STY.PUNCH, { wash: 0.5 });
    box(s, a, 215, 55, 26, 17, MAT.LIME, STY.PUNCH, { wash: 1.0 });
    box(s, a, 270, 50, 20, 14, MAT.LIME, STY.PUNCH, { wash: 1.4 });
    box(s, a, 320, 25, 14, 9, MAT.LIME, STY.PUNCH, { wash: 1.6 });
    box(s, a, 345, 36, 11, 7.5, MAT.LIME, STY.NONE, { taper: 0.55, wash: 1.8 });
    box(s, a, 381, 5, 5.2, 5.2, MAT.LIME, STY.NONE, { wash: 1.0 });
    box(s, a, 386, 40, 3.6, 3.6, MAT.STEEL, STY.NONE, { taper: 0.18, wash: 0.9 });
    beam(gb(s, a, 424), gb(s, a, 443), 0.5, 1);
    light(gb(s, a, 443), 5, 8, 8, 8, -1.2);
    light(gb(s, a, 443), 14, 0.8, 0.1, 0.06, 0.8);
    // art-deco corner lights halfway up the crown
    light(gb(s, a, 400), 6, 3, 2.4, 1.6, 0);
  }
  // Chrysler Building, 42nd & Lexington
  {
    const s = 520, a = 7040; lm(s, a, 55, 55);
    box(s, a, 0, 70, 31, 31, MAT.BRICK, STY.PUNCH, { wash: 0.2 });
    box(s, a, 70, 80, 26, 26, MAT.WHITE, STY.PUNCH, { wash: 0.25 });
    box(s, a, 150, 36, 21, 21, MAT.WHITE, STY.PUNCH, { wash: 0.3 });
    box(s, a, 186, 28, 17, 17, MAT.WHITE, STY.PUNCH, { wash: 0.3 });
    let y = 214, r = 14.5;
    for (let k = 0; k < 7; k++) { const hh = 11 - k * 0.7; box(s, a, y, hh, r, r, MAT.STEEL, STY.NONE, { taper: 0.82, wash: 1.0, rot: GRID_ROT + (k % 2) * 0.785 }); y += hh * 0.95; r *= 0.84; }
    box(s, a, y, 12, 1.8, 1.8, MAT.STEEL, STY.NONE, { taper: 0.1, wash: 1.0 });
    beam(gb(s, a, y), gb(s, a, 319), 0.45, 1);
    light(gb(s, a, 319), 5, 3, 3, 3, 0);
  }
  // Rockefeller Center (30 Rock), MetLife, Citigroup Center, Time Warner Center, United Nations
  {
    // 30 Rockefeller Plaza: limestone slab stepping back in four tiers
    let s = -250, a = 7690; lm(s, a, 80, 60);
    box(s, a, 0, 60, 52, 38, MAT.LIME, STY.PUNCH, { wash: 0.5 });
    box(s, a, 60, 80, 40, 30, MAT.LIME, STY.PUNCH, { wash: 0.55 });
    box(s, a, 140, 70, 32, 24, MAT.LIME, STY.PUNCH, { wash: 0.6 });
    box(s, a, 210, 40, 25, 18, MAT.LIME, STY.PUNCH, { wash: 0.8 });
    box(s, a, 250, 9, 16, 11, MAT.LIME, STY.NONE, { taper: 0.7, wash: 1.0 });
    light(gb(s, a, 260), 4, 3, 0.4, 0.3, 0.8);
    // MetLife (Helmsley) building straddling Park Avenue
    s = 245; a = 7200; lm(s, a, 70, 60);
    box(s, a, 0, 40, 62, 50, MAT.GRANITE, STY.PUNCH, { wash: 0.3 });
    box(s, a, 40, 120, 52, 42, MAT.GRANITE, STY.PUNCH, { wash: 0.3 });
    box(s, a, 160, 70, 40, 32, MAT.GRANITE, STY.PUNCH, { taper: 0.9, wash: 0.5 });
    box(s, a, 230, 16, 24, 20, MAT.GRANITE, STY.NONE, { taper: 0.5, wash: 0.7 });
    light(gb(s, a, 248), 4, 3, 0.4, 0.3, 0.8);
    // Citigroup Center: aluminium shaft with the famous 45 degree crown
    s = 640; a = 7925; lm(s, a, 55, 55);
    box(s, a, 0, 24, 40, 40, MAT.STEEL, STY.NONE);
    box(s, a, 24, 226, 24, 24, MAT.STEEL, STY.CURTAIN, { wash: 0.3 });
    box(s, a, 250, 16, 24, 24, MAT.STEEL, STY.CURTAIN, { taper: 0.8, wash: 0.4 });
    box(s, a, 266, 14, 18, 18, MAT.STEEL, STY.NONE, { taper: 0.35, wash: 0.5 });
    light(gb(s, a, 281), 4, 3, 0.4, 0.3, 0.8);
    // Time Warner Center twin towers at Columbus Circle
    s = -420; a = 8215; lm(s, a, 50, 60);
    box(s, a, 0, 70, 38, 62, MAT.GLASS_B, STY.CURTAIN);
    for (const sg of [-1, 1]) {
      box(s, a + sg * 22, 70, 160, 17, 20, MAT.GLASS_B, STY.CURTAIN, { wash: 0.1 });
      light(gb(s, a + sg * 22, 232), 4, 3, 0.4, 0.3, 0.8);
    }
    // United Nations: green glass Secretariat slab and the low Assembly hall on the East River
    s = 1470; a = 7340; lm(s, a, 190, 110);
    box(s, a, 0, 154, 11.5, 44, MAT.GLASS_B, STY.CURTAIN, { wash: 0.25 });
    box(s - 12, a, 120, 34, 0.5, 44, MAT.WHITE, STY.NONE, { noShadow: true });
    box(s, a - 44.5, 0, 154, 11.5, 0.8, MAT.WHITE, STY.NONE);
    box(s, a + 44.5, 0, 154, 11.5, 0.8, MAT.WHITE, STY.NONE);
    box(s + 90, a - 40, 0, 22, 58, 46, MAT.WHITE, STY.RIBBON, { wash: 0.1 });
    box(s + 90, a - 40, 22, 9, 36, 28, MAT.WHITE, STY.NONE, { taper: 0.6 });
  }
  // One World Trade Center and neighbours
  {
    const s = -520, a = 1480; lm(s, a, 200, 200);
    box(s, a, 0, 18, 33, 33, MAT.GLASS_D, STY.NONE);
    box(s, a, 18, 400, 30.5, 30.5, MAT.GLASS_B, STY.CURTAIN, { taper: 0.64, wash: 0.05 });
    box(s, a, 70, 347, 23, 23, MAT.GLASS_B, STY.CURTAIN, { taper: 0.78, rot: GRID_ROT + 0.785, wash: 0.05 });
    box(s, a, 417, 6, 20, 20, MAT.STEEL, STY.NONE, { wash: 0.5 });
    beam(gb(s, a, 423), gb(s, a, 541), 1.6, 1);
    light(gb(s, a, 541), 7, 9, 9, 9, 0);
    light(gb(s, a, 541), 16, 0.9, 0.12, 0.08, 0.7);
    // 3 and 4 WTC, 7 WTC
    box(-390, a + 70, 0, 329, 24, 24, MAT.GLASS_B, STY.CURTAIN, { taper: 0.9 });
    box(-440, a - 110, 0, 298, 22, 22, MAT.BLACKGLASS, STY.CURTAIN, { taper: 0.95 });
    box(-300, a - 230, 0, 226, 21, 21, MAT.GLASS_B, STY.CURTAIN, {});
    light(gb(-390, a + 70, 331), 6, 3, 0.4, 0.3, 0.8);
    light(gb(-440, a - 110, 300), 6, 3, 0.4, 0.3, 0.8);
    light(gb(-300, a - 230, 228), 6, 3, 0.4, 0.3, 0.8);
  }
  // Woolworth Building
  {
    const s = 0, a = 1760; lm(s, a, 50, 40);
    box(s, a, 0, 120, 32, 30, MAT.LIME, STY.PUNCH, { wash: 0.4 });
    box(s, a, 120, 62, 19, 19, MAT.LIME, STY.PUNCH, { wash: 0.5 });
    box(s, a, 182, 14, 12, 12, MAT.COPPER, STY.NONE, { taper: 0.8, wash: 0.6 });
    box(s, a, 196, 45, 10, 10, MAT.COPPER, STY.NONE, { taper: 0.05, wash: 0.8 });
    light(gb(s, a, 243), 4, 3, 2.5, 2, 0);
  }
  // Bank of America Tower, One Vanderbilt, Billionaires' Row, Hudson Yards
  {
    lm(-460, 7040, 60, 50); box(-460, 7040, 0, 280, 40, 36, MAT.GLASS_B, STY.CURTAIN, { taper: 0.82 });
    box(-460, 7040, 280, 60, 22, 20, MAT.GLASS_B, STY.CURTAIN, { taper: 0.3, wash: 0.3 });
    beam(gb(-460, 7040, 340), gb(-460, 7040, 366), 0.6, 1); light(gb(-460, 7040, 366), 5, 3, 3, 3, -0.9);
    lm(330, 7010, 50, 40); box(330, 7010, 0, 427, 30, 36, MAT.GLASS_B, STY.CURTAIN, { taper: 0.4, wash: 0.2 });
    light(gb(330, 7010, 428), 5, 3, 0.4, 0.3, 0.8);
    // 432 Park, Central Park Tower, 111 W 57th, One57
    lm(210, 8240, 28, 28); box(210, 8240, 0, 426, 14, 14, MAT.WHITE, STY.GRID, { wash: 0.2 }); light(gb(210, 8240, 428), 5, 3, 0.4, 0.3, 0.8);
    lm(-280, 8250, 40, 40); box(-280, 8250, 0, 472, 15, 22, MAT.GLASS_B, STY.CURTAIN, { wash: 0.2 });
    box(-280, 8250, 472, 20, 10, 12, MAT.STEEL, STY.NONE, { taper: 0.4, wash: 0.6 }); light(gb(-280, 8250, 492), 5, 3, 0.4, 0.3, 0.8);
    lm(-420, 8236, 30, 40); box(-420, 8236, 0, 435, 8.5, 21, MAT.LIME, STY.GRID, { wash: 0.3 });
    light(gb(-420, 8236, 437), 5, 3, 0.4, 0.3, 0.8);
    lm(-120, 8225, 40, 40); box(-120, 8225, 0, 306, 25, 25, MAT.GLASS_B, STY.CURTAIN, { taper: 0.55 });
    // Hudson Yards
    lm(-1480, 6560, 60, 90); box(-1480, 6560, 0, 387, 28, 38, MAT.GLASS_B, STY.CURTAIN, { taper: 0.55 }); light(gb(-1480, 6560, 389), 5, 3, 0.4, 0.3, 0.8);
    lm(-1520, 6790, 40, 40); box(-1520, 6790, 0, 330, 24, 24, MAT.GLASS_B, STY.CURTAIN, { taper: 0.5 });
    lm(-1380, 6330, 50, 40); box(-1380, 6330, 0, 268, 30, 25, MAT.GLASS_D, STY.CURTAIN, { taper: 0.7 });
  }
  // Statue of Liberty
  {
    const s = LIBERTY.s, a = LIBERTY.a;
    box(s, a, 0, 3, 90, 90, MAT.GRANITE, STY.NONE, { taper: 0.7, noShadow: true });
    box(s, a, 3, 28, 15, 15, MAT.GRANITE, STY.PUNCH, { taper: 0.8 });
    box(s, a, 31, 44, 6.5, 6.5, MAT.COPPER, STY.NONE, { taper: 0.4, wash: 0.5 });
    box(s, a, 75, 11, 3, 3, MAT.COPPER, STY.NONE, { taper: 0.9, wash: 0.5 });
    beam(gb(s + 4, a, 66), gb(s + 7, a, 92), 2.0, 2);
    light(gb(s + 7, a, 94), 10, 6, 4.2, 1.8, 0);
  }

  // ========================================================================================
  // Bridges
  // ========================================================================================
  function suspensionBridge(o) {
    const { a, sW, sE, towerH, deckY, stone, anchorW, anchorE, cableOff, stays, width } = o;
    const L = sE - sW;
    const deckAt = (s) => {
      if (s >= sW && s <= sE) return deckY;
      const d = s < sW ? sW - s : s - sE; const lim = s < sW ? sW - anchorW : anchorE - sE;
      return deckY - (deckY - 24) * Math.min(1, d / lim);
    };
    const cableY = (s) => { const t = (s - (sW + sE) / 2) / (L / 2); return deckY + 3.5 + (towerH - deckY - 3.5) * t * t; };
    reserve((sW + sE) / 2, a, L / 2 + 400, 60);
    // towers
    for (const ts of [sW, sE]) {
      if (stone) {
        box(ts, a, 0, towerH, 8.5, 23, MAT.GRANITE, STY.ARCH, { seed: ts % 77 });
        box(ts, a, towerH, 3, 7.5, 21.5, MAT.GRANITE, STY.NONE);
        box(ts, a, 0, 24, 11, 25, MAT.GRANITE, STY.NONE);
      } else {
        for (const sg of [-1, 1]) {
          box(ts, a + sg * 13, 0, towerH, 4, 4, MAT.STEEL, STY.NONE, { seed: 3 });
          box(ts, a + sg * 13, 0, 22, 6, 6, MAT.GRANITE, STY.NONE);
        }
        for (let k = 0; k < 4; k++) box(ts, a, deckY + 8 + k * ((towerH - deckY - 16) / 3), 3.6, 3, 11.5, MAT.STEEL, STY.NONE);
        light(gb(ts, a + 13, towerH + 1), 4, 3, 0.4, 0.3, 0.9); light(gb(ts, a - 13, towerH + 1), 4, 3, 0.4, 0.3, 0.9);
      }
    }
    // deck in 20 m segments
    const s1 = sE + anchorE, s0 = sW - anchorW;
    for (let s = s0; s < s1; s += 20) {
      const y = deckAt(s + 10);
      box(s + 10, a, y - 4, 4.5, 10.2, width / 2, stone ? MAT.DECK : MAT.STEEL, STY.NONE, { noShadow: true, seed: 5 });
      if (((s - s0) / 20) % 1 === 0) {
        for (const sg of [-1, 1]) light(gb(s + 10, a + sg * (width / 2 + 0.5), y + 5), 3.2, 2.6, 2.0, 1.2, 0);
      }
    }
    // main + back cables and suspenders
    for (const co of cableOff) for (const sg of [-1, 1]) {
      const ao = a + sg * co;
      const pts = [];
      const nseg = 60;
      for (let k = 0; k <= nseg; k++) { const s = sW + (L * k) / nseg; pts.push([s, cableY(s)]); }
      for (let k = 0; k < nseg; k++) beam(gb(pts[k][0], ao, pts[k][1]), gb(pts[k + 1][0], ao, pts[k + 1][1]), 0.55, 3);
      const back = (ts, dir, len) => {
        const sa = ts + dir * len;
        for (let k = 0; k < 16; k++) {
          const t0 = k / 16, t1 = (k + 1) / 16;
          const y = (t) => towerH + (deckAt(sa) + 2 - towerH) * t - Math.sin(t * Math.PI) * 4;
          beam(gb(ts + dir * len * t0, ao, y(t0)), gb(ts + dir * len * t1, ao, y(t1)), 0.55, 3);
        }
      };
      back(sW, -1, anchorW); back(sE, 1, anchorE);
      for (let s = sW + 12; s < sE - 6; s += 6) beam(gb(s, ao, cableY(s)), gb(s, ao, deckAt(s) + 1), 0.09, 3);
    }
    // diagonal stays (the Brooklyn Bridge web)
    if (stays) {
      for (const ts of [sW, sE]) for (const dir of [-1, 1]) for (const sg of [-1, 1]) for (const co of [cableOff[0]]) {
        for (let k = 0; k < 28; k++) {
          const sd = ts + dir * (14 + k * 5.2);
          if (sd < s0 || sd > s1) continue;
          beam(gb(ts + dir * 3, a + sg * co, towerH - 6 - (k % 4) * 0.5), gb(sd, a + sg * co, deckAt(sd) + 1), 0.07, 3);
        }
      }
    }
  }
  suspensionBridge({ a: 1790, sW: 930, sE: 1425, towerH: 84, deckY: 41, stone: true, anchorW: 300, anchorE: 330, cableOff: [13.4, 14.4], stays: true, width: 26 });
  suspensionBridge({ a: 2520, sW: 1130, sE: 1790, towerH: 102, deckY: 56, stone: false, anchorW: 330, anchorE: 330, cableOff: [14.5], stays: false, width: 32 });

  // ========================================================================================
  // Districts
  // ========================================================================================
  const gauss = (x, c, sg) => Math.exp(-((x - c) * (x - c)) / (2 * sg * sg));
  const bump = (s, a, cs, ca, ss, sa, peak) => peak * gauss(s, cs, ss) * gauss(a, ca, sa);
  function zone(s, a) {
    // returns { base, tall, dense } ; heights metres
    const out = { base: 20, tall: 0, kind: 'low' };
    const man = a > -150 && s >= manW(a) - 40 && s <= manE(a) + 40;
    if (man) {
      out.base = 16 + 10 * gauss(a, 3000, 800);
      if (a > 3600 && a < 6300) out.base = 20 + 10 * gauss(a, 5000, 1000);       // Chelsea / Flatiron
      if (a > 8400) out.base = 34;                                              // Upper East/West
      out.tall = bump(s, a, -30, 800, 520, 560, 190)                            // Financial District
        + bump(s, a, -500, 1500, 420, 360, 150)                                 // WTC / Tribeca edge
        + bump(s, a, 100, 1850, 500, 260, 40)                                   // Civic Center
        + bump(s, a, -150, 7200, 950, 820, 205)                                 // Midtown
        + bump(s, a, -150, 6350, 700, 400, 125)                                 // Penn / Herald
        + bump(s, a, -1500, 6600, 320, 380, 220)                                // Hudson Yards
        + bump(s, a, -120, 8300, 650, 260, 105)                                 // 57th St
        + bump(s, a, -100, 5000, 500, 300, 25);                                 // Union Sq
      out.kind = 'man';
    } else if (s < interp(NJ, a / 1000) * 1000) {
      out.base = 14;
      out.tall = bump(s, a, -2750, 800, 330, 380, 190) + bump(s, a, -3050, 1800, 330, 350, 120) + bump(s, a, -3780, 8000, 250, 420, 80) + bump(s, a, -3500, 4300, 320, 400, 45);
      out.kind = 'nj';
    } else {
      out.base = 12;
      out.tall = bump(s, a, 2100, 1500, 420, 360, 125) + bump(s, a, 2850, 7500, 350, 420, 170) + bump(s, a, 1500, 650, 300, 300, 45);
      out.kind = 'bk';
    }
    return out;
  }
  const pick = (w) => { let r = R() * w.reduce((x, y) => x + y[1], 0); for (const [v, p] of w) { r -= p; if (r <= 0) return v; } return w[0][0]; };
  function material(h, kind) {
    if (h < 22) return pick([[MAT.BRICK, 6], [MAT.LIME, 1.2], [MAT.SAND, 1.2], [MAT.WHITE, 1.0], [MAT.DARK, 0.5]]);
    if (h < 80) return pick([[MAT.BRICK, 5], [MAT.LIME, 2], [MAT.SAND, 1.0], [MAT.WHITE, 1.2], [MAT.GLASS_B, 1.6], [MAT.GLASS_D, 1.0]]);
    if (h < 200) return pick([[MAT.GLASS_B, 4.5], [MAT.GLASS_D, 3.2], [MAT.LIME, 1.3], [MAT.BRICK, 0.7], [MAT.WHITE, 0.9], [MAT.STEEL, 0.9], [MAT.SAND, 0.3], [MAT.BLACKGLASS, 1.0]]);
    return pick([[MAT.GLASS_B, 6], [MAT.GLASS_D, 3.5], [MAT.STEEL, 1.1], [MAT.WHITE, 0.8], [MAT.LIME, 0.9], [MAT.BLACKGLASS, 1.2]]);
  }
  const styleFor = (mat) => {
    if (mat === MAT.GLASS_B || mat === MAT.GLASS_D || mat === MAT.STEEL || mat === MAT.BLACKGLASS) return R() < 0.82 ? STY.CURTAIN : STY.RIBBON;
    if (mat === MAT.WHITE) return pick([[STY.GRID, 3], [STY.RIBBON, 2], [STY.PUNCH, 2]]);
    return R() < 0.82 ? STY.PUNCH : STY.RIBBON;
  };

  function addBuilding(sc, ac, lw, ld, z) {
    // total height
    let h = z.base * (0.6 + 0.9 * R() ** 1.4);
    if (z.tall > 8) {
      // dense districts: most lots carry a real tower, a few supertall spikes poke out
      const dens = Math.min(1, z.tall / 120);
      const frac = (1 - dens) * (R() ** 2.1) + dens * (0.28 + 0.78 * (R() ** 1.25));
      h += z.tall * 1.05 * frac * (R() < 0.1 + 0.06 * dens ? 1.5 : 1);
    }
    const dE = Math.hypot(sc + 60, ac - 6450), dC = Math.hypot(sc - 520, ac - 7040), dW = Math.hypot(sc + 520, ac - 1480);
    if (dE < 520 || dC < 420) h = Math.min(h, 150 + 30 * R());
    if (dW < 380) h = Math.min(h, 170 + 40 * R());
    h = Math.max(h, 9 + 6 * R());
    h = Math.min(h, 330);
    const mat = material(h, z.kind), sty = styleFor(mat), seed = seedCtr++ * 3.77 % 991;
    const hw = lw / 2, hd = ld / 2;
    let top;
    const rooftop = (sc2, ac2, ty, hw2, hd2) => {
      for (let c = 0; c < (ty > 40 ? 3 : 2); c++) if (R() < 0.8) {
        const bw = Math.min(hw2 * 0.55, 3 + 5 * R()), bd = Math.min(hd2 * 0.5, 3 + 4 * R());
        box(sc2 + (R() - 0.5) * hw2 * 1.2, ac2 + (R() - 0.5) * hd2 * 1.2, ty, 2 + 6 * R(), bw * (0.5 + 0.7 * R()), bd * (0.5 + 0.7 * R()), MAT.MECH, STY.NONE, { noShadow: true });
      }
      if (ty < 90 && ty > 14 && R() < 0.5) {
        const wx = sc2 + (R() - 0.5) * hw2 * 1.2, wz = ac2 + (R() - 0.5) * hd2 * 1.2;
        box(wx, wz, ty, 3.2, 1.0, 1.0, MAT.DARK, STY.NONE, { noShadow: true, rot: 0 });
        box(wx, wz, ty + 3.2, 4.2, 2.1, 2.1, MAT.WOOD, STY.NONE, { noShadow: true });
        box(wx, wz, ty + 7.4, 1.6, 2.2, 2.2, MAT.WOOD, STY.NONE, { noShadow: true, taper: 0.05 });
      }
    };
    if (h < 55 || R() < 0.15) {
      let tp = 1, hh = h;
      if (h > 22 && h < 120 && R() < 0.22 && (mat === MAT.LIME || mat === MAT.BRICK || mat === MAT.SAND)) { hh = h - 14; tp = 1; }
      box(sc, ac, 0, hh, hw, hd, mat, sty, { seed, taper: tp });
      top = hh;
      if (hh < h) { box(sc, ac, hh, h - hh, hw * 0.8, hd * 0.8, MAT.COPPER, STY.NONE, { taper: 0.1, seed }); top = h; }
      rooftop(sc, ac, hh, hw, hd);
    } else {
      // stepped tower: 2-3 tiers
      const classic = h > 90 && (mat === MAT.LIME || mat === MAT.BRICK || mat === MAT.SAND || mat === MAT.WHITE);
      const n = classic ? (h > 150 ? 4 : 3) : (h > 170 && R() < 0.6 ? 3 : 2);
      let y = 0, cs = sc, ca = ac, chw = hw, chd = hd;
      const cuts = n === 4 ? [0.22 + 0.06 * R(), 0.46 + 0.08 * R(), 0.72 + 0.08 * R(), 1] : n === 3 ? [0.30 + 0.12 * R(), 0.62 + 0.12 * R(), 1] : [0.38 + 0.2 * R(), 1];
      for (let t = 0; t < n; t++) {
        const ty = h * cuts[t];
        const last = t === n - 1;
        const taper = last && h > 150 && R() < 0.5 ? 0.55 + 0.35 * R() : 1;
        box(cs, ca, y, ty - y, chw, chd, mat, sty, { seed: seed + t * 17, taper });
        y = ty;
        if (!last) {
          const k = 0.62 + 0.25 * R();
          chw *= k; chd *= R() < 0.6 ? k : 0.95;
          cs += (R() - 0.5) * (hw - chw) * 0.8; ca += (R() - 0.5) * (hd - chd) * 0.8;
        }
      }
      top = h;
      if (classic && R() < 0.5) { const ph = 10 + 22 * R(); box(cs, ca, h, ph, chw * 0.92, chd * 0.92, R() < 0.6 ? MAT.COPPER : mat, STY.NONE, { taper: R() < 0.5 ? 0.04 : 0.35, seed }); top = h + ph; }
      else rooftop(cs, ca, h, chw, chd);
      if (h > 110 && R() < 0.45) beam(gb(cs, ca, h), gb(cs, ca, h + 14 + 40 * R()), 0.3, R() < 0.5 ? 1 : 0);
      if (h > 135) light(gb(cs, ca, h + 3), 4.5, 3, 0.4, 0.3, 0.55 + R() * 0.4);
    }
  }

  // piers: reserve their land and build sheds + lamp rows
  for (const P of PIERS) {
    if (P.nj) {
      const e = interp(NJ, P.a / 1000) * 1000;
      reserve(e + P.len * 0.5, P.a, P.len * 0.5 + 12, P.hw + 6);
      if (R() < 0.7) box(e + P.len * 0.45, P.a, 0, 9, P.len * 0.38, P.hw * 0.55, MAT.WHITE, STY.NONE, { noShadow: true });
      for (let u = 14; u < P.len - 6; u += 22) for (const sg of [-1, 1]) light(gb(e + u, P.a + sg * (P.hw - 3), 5.5), 2.2, 2.6, 1.8, 0.9, 0);
    } else if (!P.east) {
      const e = manW(P.a);
      reserve(e - P.len * 0.5, P.a, P.len * 0.5 + 12, P.hw + 6);
      if (R() < 0.5) box(e - P.len * 0.5 - 6, P.a, 0, 7, P.len * 0.36, P.hw * 0.6, MAT.WHITE, STY.NONE, { noShadow: true });
      for (let u = 12; u < P.len - 6; u += 26) for (const sg of [-1, 1]) light(gb(e - u, P.a + sg * (P.hw - 3), 5.0), 2.0, 2.4, 1.7, 0.9, 0);
    }
  }
  // blocks
  const S0 = -5400, S1 = 5400, A0 = -3400, A1 = 15400;
  let nLots = 0;
  const i0 = Math.floor(S0 / DS), i1 = Math.ceil(S1 / DS), j0 = Math.floor(A0 / DA), j1 = Math.ceil(A1 / DA);
  const PARK = [-480, 330, 8330, 12400];
  for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) {
    const bs0 = i * DS + AVE_W / 2, bs1 = (i + 1) * DS - AVE_W / 2, ba0 = j * DA + ST_W / 2, ba1 = (j + 1) * DA - ST_W / 2;
    const cs = (bs0 + bs1) / 2, ca = (ba0 + ba1) / 2;
    if (bs1 > PARK[0] && bs0 < PARK[1] && ba1 > PARK[2] && ba0 < PARK[3]) continue;
    const z = zone(cs, ca);
    const outer = z.kind !== 'man';
    const dNear = Math.hypot(cs + 300, ca - 4000);
    if (outer && R() < 0.22 + Math.max(0, dNear - 6000) * 0.00004) continue;       // gaps / sparser far away
    const tallZone = z.tall > 60 || z.base > 40;
    // split the block along s into lots
    let sCur = bs0;
    while (sCur < bs1 - 8) {
      let lw = tallZone && z.tall > 110 ? 24 + 36 * R() : outer ? 40 + 50 * R() : 16 + 30 * R();
      if (sCur + lw > bs1 - 12) lw = bs1 - sCur;
      const s0 = sCur, s1 = sCur + lw; sCur = s1;
      // one or two rows along a
      const rows = !tallZone && !outer && R() < 0.5 ? 2 : 1;
      for (let r = 0; r < rows; r++) {
        const la0 = rows === 1 ? ba0 : (r === 0 ? ba0 : (ba0 + ba1) / 2 + 0.5);
        const la1 = rows === 1 ? ba1 : (r === 0 ? (ba0 + ba1) / 2 - 0.5 : ba1);
        if (R() < 0.05) continue;
        const ls = (s0 + s1) / 2, lad = (la0 + la1) / 2;
        const gap = 0.6 + R() * 1.5;
        if (isReserved(s0, s1, la0, la1)) continue;
        if (depthAt(s0, la0) < 14 || depthAt(s1, la0) < 14 || depthAt(s0, la1) < 14 || depthAt(s1, la1) < 14 || depthAt(ls, lad) < 14) continue;
        addBuilding(ls, lad, s1 - s0 - gap, la1 - la0 - gap, zone(ls, lad));
        nLots++;
      }
    }
  }

  // street lights aren't instances; they are a ground-shader effect.

  // ---- height map mips (max pooled), converted to half -----------------------------------
  const levels = [];
  let cur = hm, cw = HMAP.w, ch = HMAP.h;
  for (let l = 0; l < HMAP.levels; l++) {
    const half = new Uint16Array(cw * ch);
    for (let k = 0; k < cw * ch; k++) half[k] = toHalf(cur[k]);
    levels.push({ data: half, w: cw, h: ch });
    const nw = Math.max(1, cw >> 1), nh = Math.max(1, ch >> 1);
    const nxt = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      let m = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const xx = x * 2 + dx, yy = y * 2 + dy;
        if (xx < cw && yy < ch) m = Math.max(m, cur[yy * cw + xx]);
      }
      nxt[y * nw + x] = m;
    }
    cur = nxt; cw = nw; ch = nh;
  }

  log(`city: ${inst.n} instances, ${nLots} lots, ${beams.n} beams, ${sprites.n} lights, ${(performance.now() - t0).toFixed(0)} ms`);
  return {
    inst: inst.done(), instCount: inst.n,
    beams: beams.done(), beamCount: beams.n,
    sprites: sprites.done(), spriteCount: sprites.n,
    landTex: land.tex, hmapLevels: levels,
  };
}

// light sprites for tall rooftops are added after generation by main.js? no: done inline above.
