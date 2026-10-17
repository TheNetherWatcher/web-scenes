// Column-major 4x4 matrix helpers (WebGPU / WGSL convention) writing into preallocated arrays.
export const D2R = Math.PI / 180;

export function mat4Mul(out, a, b) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}
export function mat4Identity(out) {
  out.fill(0); out[0] = out[5] = out[10] = out[15] = 1; return out;
}
export function mat4Translate(out, x, y, z) {
  mat4Identity(out); out[12] = x; out[13] = y; out[14] = z; return out;
}
export function mat4Scale(out, x, y, z) {
  mat4Identity(out); out[0] = x; out[5] = y; out[10] = z; return out;
}
export function mat4RotY(out, a) {
  mat4Identity(out); const c = Math.cos(a), s = Math.sin(a);
  out[0] = c; out[2] = -s; out[8] = s; out[10] = c; return out;
}
export function mat4RotX(out, a) {
  mat4Identity(out); const c = Math.cos(a), s = Math.sin(a);
  out[5] = c; out[6] = s; out[9] = -s; out[10] = c; return out;
}
export function mat4RotZ(out, a) {
  mat4Identity(out); const c = Math.cos(a), s = Math.sin(a);
  out[0] = c; out[1] = s; out[4] = -s; out[5] = c; return out;
}
export function mat4Copy(out, a) { out.set(a); return out; }

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// Small seeded PRNG (mulberry32) so procedural content is identical on every load.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
