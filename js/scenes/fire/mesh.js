// CPU-built geometry for the fire pit: leaning sticks, big logs, kindling and a ring of stones.
// Also owns groundHeight(), which must match groundH() in the WGSL.
import { rng } from '../../engine/math.js';

export function groundHeight(x, z) {
  const r = Math.hypot(x, z);
  const h = 0.5 * (Math.sin(x * 0.07 + 1.3) * Math.cos(z * 0.055) + 0.5 * Math.sin(x * 0.13 + z * 0.11 + 2.0));
  const t = Math.min(1, Math.max(0, (r - 3) / 7));
  return h * t * t * (3 - 2 * t);
}

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export const NBLK = 28;

export function buildSolids() {
  const rnd = rng(20261017);
  const V = [];       // pos3 nrm3 uv2 mat seed
  const I = [];
  const blockers = new Float32Array(NBLK * 4);
  let nb = 0;
  const addBlocker = (x, y, z, r) => { if (nb < NBLK) { blockers.set([x, y, z, r], nb * 4); nb++; } };

  function vert(p, n, u, v, mat, seed) { V.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, mat, seed); }

  function cyl(a, b, r0, r1, seed, bend = 0) {
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const len = Math.hypot(d[0], d[1], d[2]);
    const dn = [d[0] / len, d[1] / len, d[2] / len];
    const ref = Math.abs(dn[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = norm(cross(ref, dn));
    const w = cross(dn, u);
    const rings = 22, segs = 20;
    const base = V.length / 10;
    for (let i = 0; i <= rings; i++) {
      const s = i / rings;
      const bo = bend * Math.sin(s * Math.PI);
      const c = [a[0] + d[0] * s + w[0] * bo, a[1] + d[1] * s + w[1] * bo, a[2] + d[2] * s + w[2] * bo];
      for (let j = 0; j <= segs; j++) {
        const ang = (j / segs) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const knot = Math.exp(-Math.pow((s - ((seed * 0.37) % 1)) * 7, 2)) * Math.max(0, Math.sin(ang * 1 + seed)) * 0.22;
        const rad = (r0 + (r1 - r0) * s) * (1 + 0.06 * Math.sin(s * len * 9 + seed * 3 + ang * 2) + 0.05 * Math.sin(ang * 3 + seed * 7 + s * len * 4) + 0.03 * Math.sin(ang * 7 + s * len * 17 + seed) + knot);
        const nx = [u[0] * ca + w[0] * sa, u[1] * ca + w[1] * sa, u[2] * ca + w[2] * sa];
        vert([c[0] + nx[0] * rad, c[1] + nx[1] * rad, c[2] + nx[2] * rad], nx, ang, s * len, 0, seed);
      }
    }
    for (let i = 0; i < rings; i++) {
      for (let j = 0; j < segs; j++) {
        const p0 = base + i * (segs + 1) + j, p1 = p0 + 1, p2 = p0 + segs + 1, p3 = p2 + 1;
        I.push(p0, p2, p1, p1, p2, p3);
      }
    }
    for (let e = 0; e < 2; e++) {
      const c = e ? b : a, rad = e ? r1 : r0, sgn = e ? 1 : -1;
      const nrm = [dn[0] * sgn, dn[1] * sgn, dn[2] * sgn];
      const cb = V.length / 10;
      vert(c, nrm, 0, 0, 2, seed);
      for (let j = 0; j <= segs; j++) {
        const ang = (j / segs) * Math.PI * 2, ca = Math.cos(ang), sa = Math.sin(ang);
        const nx = [u[0] * ca + w[0] * sa, u[1] * ca + w[1] * sa, u[2] * ca + w[2] * sa];
        vert([c[0] + nx[0] * rad, c[1] + nx[1] * rad, c[2] + nx[2] * rad], nrm, ca, sa, 2, seed);
      }
      for (let j = 0; j < segs; j++) { if (e) I.push(cb, cb + 1 + j, cb + 2 + j); else I.push(cb, cb + 2 + j, cb + 1 + j); }
    }
  }

  function stone(cx, cz, r, seed) {
    const sy = 0.62;
    const rings = 12, segs = 18;
    const base = V.length / 10;
    const cy = r * sy * 0.45;
    for (let i = 0; i <= rings; i++) {
      const th = (i / rings) * Math.PI;
      for (let j = 0; j <= segs; j++) {
        const ph = (j / segs) * Math.PI * 2;
        // poles point sideways (+-x) so the pinch never sits on the visible top of the rock
        let ny = Math.sin(th) * Math.sin(ph), nx = Math.cos(th), nz = Math.sin(th) * Math.cos(ph);
        // squarish, faceted boulder: push the sphere towards a rounded box and add lumps
        const q = (v) => Math.sign(v) * Math.pow(Math.abs(v), 0.72);
        const sx = q(nx), sy2 = q(ny), sz = q(nz);
        const bump = 1 + 0.12 * Math.sin(ph * 2 + seed * 5 + th * 3) + 0.08 * Math.sin(ph * 5 + seed * 9) * Math.sin(th * 2) + 0.05 * Math.sin(ph * 9 + th * 7 + seed * 3);
        vert([cx + sx * r * bump * (1 + 0.15 * Math.sin(seed)), cy + sy2 * r * sy * bump, cz + sz * r * bump * (1 + 0.15 * Math.cos(seed * 1.7))], norm([Math.sign(sx) * Math.pow(Math.abs(sx), 0.39), Math.sign(sy2) * Math.pow(Math.abs(sy2), 0.39) / sy, Math.sign(sz) * Math.pow(Math.abs(sz), 0.39)]), ph, th, 1, seed);
      }
    }
    for (let i = 0; i < rings; i++) {
      for (let j = 0; j < segs; j++) {
        const p0 = base + i * (segs + 1) + j, p1 = p0 + 1, p2 = p0 + segs + 1, p3 = p2 + 1;
        I.push(p0, p1, p2, p1, p3, p2);
      }
    }
    addBlocker(cx, cy + r * sy * 0.2, cz, r * 0.82);
  }

  // five big logs lying like a star, inner ends raised and burning
  const NL = 5;
  for (let k = 0; k < NL; k++) {
    const ang = (k / NL) * Math.PI * 2 + 0.35 + (rnd() - 0.5) * 0.28;
    const dx = Math.cos(ang), dz = Math.sin(ang);
    const rr = 0.13 + rnd() * 0.025;
    const a = [dx * 0.10, 0.30 + rnd() * 0.05, dz * 0.10], b = [dx * (0.98 + rnd() * 0.18), 0.14, dz * (0.98 + rnd() * 0.18)];
    cyl(a, b, rr * 0.95, rr, k + 1, (rnd() - 0.5) * 0.06);
    for (const t of [0.42, 0.85]) addBlocker(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, rr * 0.95);
  }
  // three cross logs resting on the star
  for (let k = 0; k < 3; k++) {
    const ang = 0.9 + k * 2.1 + rnd() * 0.3;
    const dx = Math.cos(ang), dz = Math.sin(ang);
    const off = 0.13 + rnd() * 0.08;
    const px = -dz * off, pz = dx * off;
    const a = [-dx * 0.46 + px, 0.37 + rnd() * 0.04, -dz * 0.46 + pz], b = [dx * 0.46 + px, 0.34 + rnd() * 0.04, dz * 0.46 + pz];
    cyl(a, b, 0.088, 0.078, 20 + k, (rnd() - 0.5) * 0.05);
    for (const t of [0.25, 0.75]) addBlocker(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, 0.085);
  }
  // teepee sticks
  const NS = 9;
  for (let k = 0; k < NS; k++) {
    const ang = (k / NS) * Math.PI * 2 + rnd() * 0.5;
    const rb = 0.35 + rnd() * 0.22;
    const top = 0.62 + rnd() * 0.3;
    const a = [Math.cos(ang) * rb, 0.06, Math.sin(ang) * rb];
    const ta = ang + (rnd() - 0.5) * 0.6;
    const b = [Math.cos(ta) * 0.07, top, Math.sin(ta) * 0.07];
    cyl(a, b, 0.05 + rnd() * 0.014, 0.025 + rnd() * 0.01, 40 + k, (rnd() - 0.5) * 0.04);
  }
  // kindling and broken branches
  for (let k = 0; k < 20; k++) {
    const ang = rnd() * Math.PI * 2;
    const rb = 0.18 + rnd() * 0.5;
    const top = 0.18 + rnd() * 0.3;
    const a = [Math.cos(ang) * rb, 0.05, Math.sin(ang) * rb];
    const ta = ang + (rnd() - 0.5) * 1.6;
    const b = [Math.cos(ta) * (rb * 0.5), top, Math.sin(ta) * (rb * 0.5)];
    cyl(a, b, 0.014 + rnd() * 0.012, 0.008 + rnd() * 0.004, 80 + k, (rnd() - 0.5) * 0.03);
  }
  // stones
  const NSTONE = 12;
  for (let k = 0; k < NSTONE; k++) {
    const ang = (k / NSTONE) * Math.PI * 2 + (rnd() - 0.5) * 0.18;
    const rr = 1.08 + (rnd() - 0.5) * 0.14;
    stone(Math.cos(ang) * rr, Math.sin(ang) * rr, 0.17 + rnd() * 0.09, 100 + k);
  }

  return { vertices: new Float32Array(V), indices: new Uint32Array(I), blockers };
}

// Static blade placement: x, z, yaw, seed. Denser near the fire, none inside the stone ring.
export function buildGrass(count) {
  const rnd = rng(777);
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    let x, z;
    for (;;) {
      const r = 52 * Math.pow(rnd(), 0.8);
      const a = rnd() * Math.PI * 2;
      x = Math.cos(a) * r; z = Math.sin(a) * r;
      if (Math.hypot(x, z) > 1.35) break;
    }
    out[i * 4] = x; out[i * 4 + 1] = z; out[i * 4 + 2] = rnd() * Math.PI * 2; out[i * 4 + 3] = rnd();
  }
  return out;
}
