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

export function buildSolids() {
  const rnd = rng(20261017);
  const V = [];       // pos3 nrm3 uv2 mat seed
  const I = [];
  const blockers = new Float32Array(24 * 4);
  let nb = 0;
  const addBlocker = (x, y, z, r) => { if (nb < 24) { blockers.set([x, y, z, r], nb * 4); nb++; } };

  function vert(p, n, u, v, mat, seed) { V.push(p[0], p[1], p[2], n[0], n[1], n[2], u, v, mat, seed); }

  function cyl(a, b, r0, r1, seed) {
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const len = Math.hypot(d[0], d[1], d[2]);
    const dn = [d[0] / len, d[1] / len, d[2] / len];
    const ref = Math.abs(dn[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = norm(cross(ref, dn));
    const w = cross(dn, u);
    const rings = 14, segs = 16;
    const base = V.length / 10;
    for (let i = 0; i <= rings; i++) {
      const s = i / rings;
      const c = [a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s];
      for (let j = 0; j <= segs; j++) {
        const ang = (j / segs) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const rad = (r0 + (r1 - r0) * s) * (1 + 0.07 * Math.sin(s * 11 + seed * 3 + ang * 2) + 0.05 * Math.sin(ang * 3 + seed * 7 + s * 5));
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
        const nx = Math.sin(th) * Math.cos(ph), ny = Math.cos(th), nz = Math.sin(th) * Math.sin(ph);
        const bump = 1 + 0.16 * Math.sin(ph * 2 + seed * 5 + th * 3) + 0.1 * Math.sin(ph * 5 + seed * 9) * Math.sin(th * 2);
        vert([cx + nx * r * bump, cy + ny * r * sy * bump, cz + nz * r * bump], [nx, ny / sy, nz], ph, th, 1, seed);
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

  // three big logs lying like a star, inner ends raised
  for (let k = 0; k < 3; k++) {
    const ang = (k / 3) * Math.PI * 2 + 0.5 + rnd() * 0.2;
    const dx = Math.cos(ang), dz = Math.sin(ang);
    const a = [dx * 0.08, 0.24, dz * 0.08], b = [dx * 1.0, 0.12, dz * 1.0];
    cyl(a, b, 0.115, 0.125, k + 1);
    for (const t of [0.35, 0.65, 0.92]) addBlocker(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, 0.12);
  }
  // two cross logs
  for (let k = 0; k < 2; k++) {
    const ang = 1.0 + k * 2.6 + rnd() * 0.3;
    const dx = Math.cos(ang), dz = Math.sin(ang);
    cyl([-dx * 0.5, 0.33, -dz * 0.5], [dx * 0.5, 0.31, dz * 0.5], 0.075, 0.07, 20 + k);
  }
  // teepee sticks
  const NS = 11;
  for (let k = 0; k < NS; k++) {
    const ang = (k / NS) * Math.PI * 2 + rnd() * 0.4;
    const rb = 0.4 + rnd() * 0.22;
    const top = 0.72 + rnd() * 0.25;
    const a = [Math.cos(ang) * rb, 0.02, Math.sin(ang) * rb];
    const ta = ang + (rnd() - 0.5) * 0.6;
    const b = [Math.cos(ta) * 0.07, top, Math.sin(ta) * 0.07];
    cyl(a, b, 0.06 + rnd() * 0.014, 0.03 + rnd() * 0.01, 40 + k);
  }
  // kindling
  for (let k = 0; k < 14; k++) {
    const ang = rnd() * Math.PI * 2;
    const rb = 0.22 + rnd() * 0.25;
    const top = 0.35 + rnd() * 0.25;
    const a = [Math.cos(ang) * rb, 0.04, Math.sin(ang) * rb];
    const ta = ang + (rnd() - 0.5) * 1.2;
    const b = [Math.cos(ta) * 0.1, top, Math.sin(ta) * 0.1];
    cyl(a, b, 0.016 + rnd() * 0.01, 0.01, 80 + k);
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
