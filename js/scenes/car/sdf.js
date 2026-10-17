// Signed-distance models of the car. They are evaluated once at start-up in a compute shader on a
// regular grid (see mesher.js) and turned into triangle meshes, so the runtime is plain raster.
// Car space: +x forward, +y up, z lateral (the model is symmetric in z), metres, ground at y = 0.

export const SDF_COMMON = /* wgsl */ `
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
struct Grid { origin: vec4f, dims: vec4u, h: vec4f }   // h: cell, band, style, unused
struct Attrs { m1: vec4f, m2: vec4f }
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> M1: array<u32>;
@group(0) @binding(3) var<storage, read_write> M2: array<u32>;

fn smin(a: f32, b: f32, k: f32) -> f32 { let h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }
fn sdBox(p: vec3f, b: vec3f) -> f32 { let q = abs(p) - b; return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0); }
fn sdRBox(p: vec3f, b: vec3f, r: f32) -> f32 { return sdBox(p, b - vec3f(r)) - r; }
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 { let k0 = length(p / r); let k1 = length(p / (r * r)); return k0 * (k0 - 1.0) / max(k1, 1e-6); }
fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a; let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}
fn rotXZ(v: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z); }
`;

// compute entry: distance, material channels and baked ambient occlusion on a grid
export const FIELD_MAIN = /* wgsl */ `
@compute @workgroup_size(8, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= G.dims.x || id.y >= G.dims.y || id.z >= G.dims.z) { return; }
  let idx = id.x + G.dims.x * (id.y + G.dims.y * id.z);
  let p = G.origin.xyz + vec3f(id) * G.h.x;
  let d = dist(p);
  F[idx] = d;
  var a = Attrs(vec4f(0.0), vec4f(0.0));
  if (abs(d) < G.h.y) {
    a = attrs(p);
    let e = G.h.x * 0.8;
    let k = vec2f(1.0, -1.0);
    let n = normalize(k.xyy * dist(p + k.xyy * e) + k.yyx * dist(p + k.yyx * e) + k.yxy * dist(p + k.yxy * e) + k.xxx * dist(p + k.xxx * e));
    var occ = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 5; i++) {
      let t = 0.012 + 0.07 * f32(i);
      occ += (t - dist(p + n * t)) * w;
      w *= 0.62;
    }
    a.m2.x = clamp(1.0 - occ * 2.2, 0.0, 1.0);
  }
  M1[idx] = pack4x8unorm(a.m1);
  M2[idx] = pack4x8unorm(a.m2);
}
`;

// Catmull-Rom curve over uniform knots x = -2.4 .. 2.4 (dx 0.2), generated as a WGSL function.
function curve(name, vals) {
  const n = vals.length;
  const list = vals.map((v) => Number(v).toFixed(4)).join(', ');
  return `
fn ${name}(x: f32) -> f32 {
  var a = array<f32, ${n}>(${list});
  let t = clamp((x + 2.4) / 0.2, 0.0, ${(n - 1.001).toFixed(3)});
  let i = i32(floor(t));
  let f = t - f32(i);
  let p0 = a[max(i - 1, 0)]; let p1 = a[i]; let p2 = a[i + 1]; let p3 = a[min(i + 2, ${n - 1})];
  return 0.5 * ((2.0 * p1) + (-p0 + p2) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f * f * f);
}`;
}

//           x:  -2.4  -2.2  -2.0  -1.8  -1.6  -1.4  -1.2  -1.0  -0.8  -0.6  -0.4  -0.2   0.0   0.2   0.4   0.6   0.8   1.0   1.2   1.4   1.6   1.8   2.0   2.2   2.4
const W_ = [0.60, 0.74, 0.86, 0.94, 0.985, 1.00, 0.995, 0.975, 0.955, 0.94, 0.935, 0.935, 0.935, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.975, 0.95, 0.90, 0.84, 0.68, 0.50];
const T_ = [0.60, 0.70, 0.80, 0.86, 0.90, 0.93, 0.94, 0.93, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.75, 0.73, 0.69, 0.60, 0.50, 0.385, 0.29, 0.21];
const B_ = [0.40, 0.34, 0.27, 0.19, 0.13, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.085, 0.08, 0.075, 0.08, 0.10];
const WC = [0.40, 0.40, 0.40, 0.45, 0.50, 0.56, 0.62, 0.66, 0.69, 0.71, 0.72, 0.72, 0.72, 0.72, 0.71, 0.69, 0.65, 0.59, 0.50, 0.44, 0.40, 0.40, 0.40, 0.40, 0.40];
const TC = [0.80, 0.80, 0.82, 0.86, 0.88, 0.90, 0.925, 0.95, 0.975, 0.995, 1.005, 1.010, 1.010, 1.000, 0.975, 0.930, 0.870, 0.810, 0.760, 0.720, 0.700, 0.700, 0.700, 0.700, 0.700];

export const BODY_PART = curve('cW', W_) + curve('cT', T_) + curve('cB', B_) + curve('cWc', WC) + curve('cTc', TC) + /* wgsl */ `
struct Parts { d: f32, rest: f32, can: f32, cdark: f32, clamp_: f32 }

fn loftMain(q: vec3f) -> f32 {
  let W = cW(q.x); let T = cT(q.x); let B = cB(q.x);
  let ys = B + (T - B) * 0.46;
  let u = max(q.z / W, 1e-5);
  var v: f32; var n: f32;
  if (q.y > ys) { v = (q.y - ys) / (T - ys); n = 4.2; } else { v = (ys - q.y) / (ys - B); n = 4.5; }
  v = max(v, 1e-5);
  let r = pow(pow(u, n) + pow(v, n), 1.0 / n);
  var f = (r - 1.0) * min(W, 0.5 * (T - B)) * 0.75;
  f = smax(f, q.x - 2.30, 0.05);
  f = smax(f, -2.26 - q.x, 0.08);
  return f;
}

fn loftCan(q: vec3f) -> f32 {
  let W = cWc(q.x); let T = cTc(q.x);
  let yb = 0.70;
  let h = max(T - yb, 0.05);
  let u = max(q.z / W, 1e-5);
  var v = (q.y - yb) / h;
  if (v < 0.0) { v = -v * h / 0.30; }
  v = max(v, 1e-5);
  let n = 3.0;
  let r = pow(pow(u, n) + pow(v, n), 1.0 / n);
  var f = (r - 1.0) * min(W, h) * 0.7;
  f = smax(f, q.x - 1.16, 0.06);
  f = smax(f, -1.44 - q.x, 0.14);
  return f;
}

fn parts(p: vec3f) -> Parts {
  let q = vec3f(p.x, p.y, abs(p.z));
  let fend = sdRBox(q - vec3f(1.34, 0.54, 0.70), vec3f(0.46, 0.24, 0.30), 0.13);
  var rest = smin(loftMain(q), fend, 0.13);
  // concave door panel
  rest = smax(rest, -sdEllipsoid(q - vec3f(0.12, 0.50, 1.20), vec3f(0.90, 0.17, 0.34)), 0.015);
  let can = loftCan(q);
  var d = smin(rest, can, 0.07);

  // cavities: wheel arches, front mouth + brake ducts, side air intake, engine-cover vent, exhausts
  let archF = max(length(vec2f(q.x - 1.40, q.y - 0.359)) - 0.389, 0.50 - q.z);
  let archR = max(length(vec2f(q.x + 1.32, q.y - 0.371)) - 0.401, 0.45 - q.z);
  let mouth = sdRBox(q - vec3f(2.10, 0.165, 0.0), vec3f(0.20, 0.052, 0.50), 0.03);
  let nost = sdRBox(q - vec3f(1.90, 0.18, 0.70), vec3f(0.15, 0.045, 0.12), 0.025);
  let sidei = min(sdRBox(rotXZ(q - vec3f(-0.85, 0.52, 0.97), 0.22), vec3f(0.50, 0.14, 0.17), 0.04), sdRBox(rotXZ(q - vec3f(-0.62, 0.86, 0.66), 0.35), vec3f(0.26, 0.035, 0.11), 0.02));
  let vent = sdRBox(q - vec3f(-1.30, 1.00, 0.0), vec3f(0.30, 0.05, 0.36), 0.03);
  let exh = max(length(vec2f(q.y - 0.47, q.z - 0.16)) - 0.055, q.x + 2.10);
  let cdark = min(min(min(archF, archR), min(mouth, nost)), min(min(sidei, vent), exh));
  d = smax(d, -cdark, 0.012);

  // light units: slim head blades and a full-width tail bar
  let hl = sdRBox(rotXZ(q - vec3f(1.90, 0.365, 0.47), 0.55), vec3f(0.05, 0.034, 0.22), 0.014);
  let tl = sdRBox(q - vec3f(-2.26, 0.56, 0.0), vec3f(0.05, 0.022, 0.72), 0.01);
  let drl = sdRBox(rotXZ(q - vec3f(2.10, 0.215, 0.30), -0.35), vec3f(0.03, 0.014, 0.17), 0.007);
  let clampL = min(min(hl, tl), drl);
  d = smax(d, -clampL, 0.006);

  // mirrors
  let mir = sdEllipsoid(q - vec3f(0.74, 0.885, 0.88), vec3f(0.12, 0.05, 0.075));
  let stalk = sdCapsule(q, vec3f(0.82, 0.80, 0.74), vec3f(0.76, 0.87, 0.86), 0.02);
  d = smin(d, smin(mir, stalk, 0.03), 0.03);

  // rear wing: pitched blade, end plates and swan-neck pylons
  let wl = q - vec3f(-2.08, 1.19, 0.0);
  let c = cos(0.12); let s = sin(0.12);
  let wr = vec3f(c * wl.x - s * wl.y, s * wl.x + c * wl.y, wl.z);
  let wing = sdRBox(wr, vec3f(0.19, 0.024, 0.98), 0.014);
  let endp = sdRBox(q - vec3f(-2.08, 1.17, 0.98), vec3f(0.20, 0.07, 0.012), 0.006);
  var pq = q - vec3f(-1.98, 0.96, 0.50);
  pq.x += pq.y * 0.35;
  let pyl = sdRBox(pq, vec3f(0.09, 0.27, 0.03), 0.02);
  d = smin(d, min(min(wing, endp), pyl), 0.02);
  let split = sdRBox(q - vec3f(2.12, 0.088, 0.0), vec3f(0.26, 0.013, 0.66), 0.006);
  d = smin(d, split, 0.02);
  var fin = 1e3;
  for (var i = 0; i < 5; i++) {
    let fp = q - vec3f(-2.05, 0.255, 0.04 + 0.19 * f32(i));
    let fr = vec3f(0.9428 * fp.x - 0.3335 * fp.y, 0.3335 * fp.x + 0.9428 * fp.y, fp.z);
    fin = min(fin, sdRBox(fr, vec3f(0.30, 0.045, 0.013), 0.004));
  }
  d = smin(d, fin, 0.01);

  return Parts(d, rest, can, cdark, clampL);
}

fn dist(p: vec3f) -> f32 { return parts(p).d; }

fn winMask(q: vec3f, uc: f32, yh: f32, phi: f32, m: f32) -> f32 {
  let ws = smoothstep(0.30 - m, 0.34 - m, q.x) * smoothstep(1.06 + m, 1.02 + m, q.x) * (1.0 - smoothstep(0.78 + m, 0.84 + m, uc));
  let xf = 0.74 - 0.55 * clamp(yh, 0.0, 1.0) + m;
  let sg = smoothstep(-0.80 - m, -0.74 - m, q.x) * smoothstep(xf + 0.04, xf - 0.02, q.x)
         * smoothstep(1.10 + m, 0.96 + m, phi) * smoothstep(0.80 - m, 0.83 - m, q.y);
  return max(ws, sg);
}

fn attrs(p: vec3f) -> Attrs {
  let P = parts(p);
  let q = vec3f(p.x, p.y, abs(p.z));
  let onCan = clamp(0.5 + (P.rest - P.can) / 0.03, 0.0, 1.0);
  let Wc = cWc(q.x); let Tc = cTc(q.x);
  let h = max(Tc - 0.70, 0.05);
  let uc = q.z / Wc;
  let yh = (q.y - 0.70) / h;
  let phi = atan2(max(yh, 0.0), max(uc, 1e-3));
  let g0 = winMask(q, uc, yh, phi, 0.0);
  let glass = onCan * g0;
  let trim = onCan * clamp(winMask(q, uc, yh, phi, 0.03) - g0, 0.0, 1.0);
  let band = smoothstep(0.215, 0.17, q.y);
  let tail = smoothstep(-1.85, -1.95, q.x) * smoothstep(0.58, 0.52, q.y);
  let wingC = smoothstep(1.00, 1.04, q.y) * smoothstep(-1.70, -1.78, q.x);
  let roofC = onCan * smoothstep(1.05, 1.18, phi) * smoothstep(0.34, 0.28, q.x);
  let carbon = max(max(band, tail), max(wingC, roofC));
  let cav = smoothstep(0.02, 0.0, P.cdark);
  let lamp = smoothstep(0.012, 0.0, P.clamp_);
  return Attrs(vec4f(glass, carbon, cav, lamp), vec4f(0.0, trim, 0.0, 0.0));
}
`;

// ---- wheel: tyre + rim (barrel, spokes, hub) + brake disc, axle along z, outer face at +z ----------
export const WHEEL_PART = /* wgsl */ `
fn tireD(p: vec3f) -> f32 {
  let r = length(p.xy);
  let tt = (r - 0.2975) / 0.0475;
  let hz = 0.150 + 0.012 * (1.0 - clamp(tt * tt, 0.0, 1.0));
  let dq = abs(vec2f(r - 0.2975, abs(p.z))) - vec2f(0.0475 - 0.035, hz - 0.035);
  return length(max(dq, vec2f(0.0))) + min(max(dq.x, dq.y), 0.0) - 0.035;
}
fn rimD(p: vec3f) -> f32 {
  let r = length(p.xy);
  let style = i32(G.h.z + 0.5);
  let barrel = max(max(r - 0.252, 0.208 - r), abs(p.z) - 0.152);
  var n = 5.0;
  if (style == 1) { n = 10.0; }
  if (style == 2) { n = 7.0; }
  let sector = TAU / n;
  let a = atan2(p.y, p.x);
  let aa = (fract(a / sector + 0.5) - 0.5) * sector;
  let sp = vec2f(r * cos(aa), r * sin(aa));
  let tt = clamp((sp.x - 0.06) / 0.17, 0.0, 1.0);
  var d2: f32;
  if (style == 0) {
    d2 = max(abs(abs(sp.y) - 0.026) - mix(0.020, 0.013, tt), abs(sp.x - 0.135) - 0.105);
  } else if (style == 1) {
    d2 = max(abs(sp.y) - mix(0.016, 0.010, tt), abs(sp.x - 0.135) - 0.105);
  } else {
    d2 = max(abs(sp.y) - mix(0.040, 0.024, tt), abs(sp.x - 0.135) - 0.105);
  }
  let zc = 0.060 + 0.050 * (r * r / 0.0625);
  let dz = abs(p.z - zc) - 0.018;
  let spoke = length(max(vec2f(d2, dz), vec2f(0.0))) + min(max(d2, dz), 0.0) - 0.002;
  let hub = max(r - 0.060, abs(p.z - 0.075) - 0.045);
  let cap = max(r - 0.032, abs(p.z - 0.118) - 0.012);
  return smin(min(min(barrel, spoke), hub), cap, 0.004);
}
fn discD(p: vec3f) -> f32 {
  let r = length(p.xy);
  let ring = max(abs(r - 0.152) - 0.036, abs(p.z - 0.028) - 0.016);
  let hat = max(r - 0.118, abs(p.z - 0.05) - 0.024);
  return min(ring, hat);
}
fn dist(p: vec3f) -> f32 { return min(smin(tireD(p), rimD(p), 0.004), discD(p)); }
fn attrs(p: vec3f) -> Attrs {
  let a = tireD(p); let b = rimD(p); let c = discD(p);
  let dm = min(a, min(b, c));
  let t = clamp(1.0 - (a - dm) / 0.010, 0.0, 1.0);
  let r = clamp(1.0 - (b - dm) / 0.010, 0.0, 1.0);
  let s = clamp(1.0 - (c - dm) / 0.010, 0.0, 1.0);
  return Attrs(vec4f(r, t, s, 0.0), vec4f(0.0));
}
`;

// ---- brake caliper (static, does not spin with the wheel) ------------------------------------------
export const CALIPER_PART = /* wgsl */ `
fn dist(p: vec3f) -> f32 {
  let r = length(p.xy);
  let a = atan2(p.y, p.x);
  let dr = abs(r - 0.163) - 0.042;
  let dl = abs(a - 1.85) * r - 0.085;
  let dz = abs(p.z - 0.028) - 0.036;
  let k = 0.010;
  return length(max(vec3f(dr + k, dl + k, dz + k), vec3f(0.0))) + min(max(dr, max(dl, dz)), 0.0) - k;
}
fn attrs(p: vec3f) -> Attrs { return Attrs(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(0.0)); }
`;
