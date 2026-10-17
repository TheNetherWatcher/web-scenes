// Signed-distance models of the car. They are evaluated once at start-up in a compute shader on a
// regular grid (see mesher.js) and turned into triangle meshes, so the runtime is plain raster.
// Car space: +x forward, +y up, z lateral (the model is symmetric in z), metres, ground at y = 0.

export const SDF_FUNCS = /* wgsl */ `
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
struct Attrs { m1: vec4f, m2: vec4f }

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

export const SDF_COMMON = SDF_FUNCS + /* wgsl */ `
struct Grid { origin: vec4f, dims: vec4u, h: vec4f }   // h: cell, band, style, unused
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read_write> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> M1: array<u32>;
@group(0) @binding(3) var<storage, read_write> M2: array<u32>;
`;

// coarse distance field of the whole car (body + wheel cylinders) in a 3D texture, sampled by the
// raster shaders for sun shadows on the body and on the road. wheels: [[x, z, scale], ...]
export function shadowFieldMain(wheels, wheelR) {
  let wsrc = '';
  for (const w of wheels) {
    wsrc += '  d = min(d, cylZ(p, vec3f(' + w.x.toFixed(3) + ', ' + (wheelR * w.s).toFixed(3) + ', ' + w.z.toFixed(3) + '), ' + (0.345 * w.s).toFixed(3) + ', 0.15));\n';
  }
  return SHADOW_HEAD + wsrc + SHADOW_TAIL;
}
const SHADOW_HEAD = /* wgsl */ `
struct SF { origin: vec4f, cell: vec4f, dims: vec4u }
@group(0) @binding(0) var<uniform> S: SF;
@group(0) @binding(1) var sdfOut: texture_storage_3d<rgba16float, write>;
fn cylZ(p: vec3f, c: vec3f, r: f32, hw: f32) -> f32 {
  let q = p - c;
  let d = vec2f(length(q.xy) - r, abs(q.z) - hw);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= S.dims.x || id.y >= S.dims.y || id.z >= S.dims.z) { return; }
  let p = S.origin.xyz + (vec3f(id) + 0.5) * S.cell.x;
  var d = dist(p);
`;
const SHADOW_TAIL = /* wgsl */ `
  textureStore(sdfOut, id, vec4f(clamp(d, -0.25, 0.9), 0.0, 0.0, 0.0));
}
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
const W_ = [0.6, 0.74, 0.86, 0.94, 0.985, 1.0, 0.995, 0.975, 0.955, 0.94, 0.935, 0.935, 0.935, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.985, 0.985, 0.975, 0.95, 0.9, 0.74];
const T_ = [0.6, 0.7, 0.8, 0.86, 0.9, 0.93, 0.94, 0.93, 0.92, 0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.77, 0.755, 0.73, 0.7, 0.63, 0.54, 0.44, 0.35, 0.27];
const B_ = [0.40, 0.34, 0.27, 0.19, 0.13, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.09, 0.09, 0.085, 0.08, 0.075, 0.08, 0.10];
const WC = [0.40, 0.40, 0.40, 0.45, 0.50, 0.56, 0.62, 0.66, 0.69, 0.71, 0.72, 0.72, 0.72, 0.72, 0.71, 0.69, 0.65, 0.59, 0.50, 0.44, 0.40, 0.40, 0.40, 0.40, 0.40];
const TC = [0.80, 0.80, 0.82, 0.86, 0.88, 0.90, 0.925, 0.95, 0.975, 0.995, 1.005, 1.010, 1.010, 1.000, 0.975, 0.930, 0.870, 0.810, 0.760, 0.720, 0.700, 0.700, 0.700, 0.700, 0.700];

// plan-view (x, |z|) shapes of the headlamp: housing pocket, DRL blade and projector. The housing is carved into
// the mesh; the blade and projector are painted crisply in the fragment shader.
export const HEAD_SHAPES = /* wgsl */ `
fn seg2(p: vec2f, a: vec2f, b: vec2f) -> f32 { let pa = p - a; let ba = b - a; let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
fn headShapes(pxz: vec2f) -> vec3f {
  return vec3f(seg2(pxz, vec2f(2.10, 0.44), vec2f(1.90, 0.80)) - 0.052,
               seg2(pxz, vec2f(2.15, 0.40), vec2f(1.86, 0.86)) - 0.011,
               length(pxz - vec2f(2.015, 0.62)) - 0.030);
}
`;

export const BODY_PART = HEAD_SHAPES + curve('cW', W_) + curve('cT', T_) + curve('cB', B_) + curve('cWc', WC) + curve('cTc', TC) + /* wgsl */ `
struct Parts { d: f32, rest: f32, can: f32, cdark: f32, clamp_: f32, lens: f32, tube: f32, wingD: f32 }

// cut a groove / pocket of plan-view shape g (negative inside) a fixed depth into any surface above ylo
fn carve(d: f32, g: f32, depth: f32, ylo: f32, y: f32) -> f32 { return smax(d, -max(max(g, -d - depth), ylo - y), 0.012); }

fn bump(x: f32, c: f32, wf: f32, wb: f32) -> f32 { let t = (x - c) / select(wb, wf, x > c); return exp(-t * t); }

// polygon distance for the half section (u = |z|, y); the closing edge on the symmetry plane only
// contributes to the sign so the field stays correct next to the centre line
fn sdSection(p: vec2f, v: array<vec2f, 8>) -> f32 {
  var vv = v;
  var d = 1e6;
  var s = 1.0;
  var j = 7;
  for (var i = 0; i < 8; i++) {
    let e = vv[j] - vv[i];
    let w = p - vv[i];
    let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
    if (j != 7 || i != 0) { d = min(d, dot(b, b)); }
    let c0 = p.y >= vv[i].y; let c1 = p.y < vv[j].y; let c2 = e.x * w.y > e.y * w.x;
    if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) { s = -s; }
    j = i;
  }
  return s * sqrt(d);
}

fn loftMain(q: vec3f) -> f32 {
  let x = q.x;
  let W = cW(x); let T = cT(x); let B = cB(x);
  let r = 0.022;
  let ysh = B + (T - B) * 0.50;
  // fender crowns rise above the hood between the nose and the windscreen, and over the rear haunches
  let tf = T + 0.125 * bump(x, 1.34, 0.50, 0.75) + 0.050 * bump(x, -1.32, 0.55, 0.55);
  let Wr = W - r;
  let v = array<vec2f, 8>(
    vec2f(0.0, B + r),
    vec2f(Wr * 0.72, B + r),
    vec2f(Wr * 0.94, B + 0.085),
    vec2f(Wr, B + (T - B) * 0.40),
    vec2f(Wr * 0.965, ysh + 0.01),
    vec2f(Wr * 0.76, tf - r),
    vec2f(Wr * 0.30, T - r - 0.022),
    vec2f(0.0, T - r + 0.004));
  var f = sdSection(vec2f(q.z, q.y), v) - r;
  f = smax(f, (q.x - 2.30) * 0.970 + (q.y - 0.09) * 0.242, 0.045);
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
  var rest = loftMain(q);
  // concave door panel
  rest = smax(rest, -sdEllipsoid(q - vec3f(0.05, 0.575, 1.21), vec3f(1.08, 0.085, 0.31)), 0.02);
  rest = smax(rest, -sdEllipsoid(q - vec3f(0.10, 0.40, 1.19), vec3f(0.80, 0.045, 0.24)), 0.02);
  let can = loftCan(q);
  var d = smin(rest, can, 0.07);

  // front: hood channels, headlamp housings with DRL blades, louvres
  let pxz = q.xz;
  d = smax(d, -max(max(seg2(pxz, vec2f(2.02, 0.31), vec2f(1.26, 0.255)) - 0.052, -d - 0.016), 0.28 - q.y), 0.03);
  let hs = headShapes(pxz);
  d = carve(d, hs.x, 0.026, 0.28, q.y);
  let lens = hs.x;

  // cavities: wheel arches, front mouth + corner ducts, side air intake, engine-cover vent, exhausts
  let archF = max(length(vec2f(q.x - 1.40, q.y - 0.338)) - 0.368, 0.50 - q.z);
  let archR = max(length(vec2f(q.x + 1.32, q.y - 0.369)) - 0.398, 0.45 - q.z);
  let mouth = sdRBox(q - vec3f(2.12, 0.185, 0.0), vec3f(0.22, 0.062, 0.56), 0.03);
  let nost = sdRBox(q - vec3f(2.02, 0.20, 0.78), vec3f(0.16, 0.055, 0.075), 0.025);
  let hvent = sdRBox(q - vec3f(1.50, 0.66, 0.43), vec3f(0.15, 0.03, 0.085), 0.012);
  let sidei = min(sdRBox(rotXZ(q - vec3f(-0.85, 0.52, 0.97), 0.22), vec3f(0.50, 0.14, 0.17), 0.04), sdRBox(rotXZ(q - vec3f(-0.62, 0.86, 0.66), 0.35), vec3f(0.26, 0.035, 0.11), 0.02));
  let vent = sdRBox(q - vec3f(-1.30, 1.00, 0.0), vec3f(0.30, 0.05, 0.36), 0.03);
  let exh = max(length(vec2f(q.y - 0.47, q.z - 0.16)) - 0.050, q.x + 2.12);
  let cdark = min(min(min(archF, archR), min(mouth, nost)), min(min(sidei, vent), min(exh, hvent)));
  d = smax(d, -min(min(mouth, nost), min(min(sidei, vent), min(exh, hvent))), 0.012);
  d = smax(d, -min(archF, archR), 0.026);

  // tail light bar
  let tl = sdRBox(q - vec3f(-2.26, 0.56, 0.0), vec3f(0.05, 0.022, 0.72), 0.01);
  let clampL = tl;
  d = smax(d, -tl, 0.006);

  // mirrors
  let mir = sdEllipsoid(q - vec3f(0.74, 0.885, 0.88), vec3f(0.12, 0.05, 0.075));
  let stalk = sdCapsule(q, vec3f(0.82, 0.80, 0.74), vec3f(0.76, 0.87, 0.86), 0.02);
  d = smin(d, smin(mir, stalk, 0.03), 0.03);

  // rear wing: cambered aerofoil section on swan-neck pylons, end plates with louvres, gurney flap
  let wl = q - vec3f(-2.08, 1.19, 0.0);
  let c = cos(0.12); let s = sin(0.12);
  let wr = vec3f(c * wl.x - s * wl.y, s * wl.x + c * wl.y, wl.z);
  let wt = clamp((wr.x + 0.20) / 0.40, 0.0, 1.0);                 // 0 trailing edge .. 1 leading edge
  let wh = mix(0.0045, 0.027, pow(sin(clamp(wt * 1.15, 0.0, 1.0) * 1.5708), 0.55));
  let wdx = abs(wr.x) - 0.20;
  let wdy = abs(wr.y - 0.014 * (1.0 - wt) * wt * 4.0) - wh;
  let wing2 = min(max(wdx, wdy), 0.0) + length(max(vec2f(wdx, wdy), vec2f(0.0))) - 0.002;
  let wing = max(wing2, abs(wr.z) - 0.98);
  let gurney = sdBox(wr - vec3f(-0.192, 0.012, 0.0), vec3f(0.004, 0.022, 0.98));
  let eq = q - vec3f(-2.08, 1.165, 0.985);
  var endp = sdRBox(eq, vec3f(0.225, 0.082, 0.009), 0.007);
  var pp = q;
  pp.z = 0.50 + (q.z - 0.50) * 2.6;
  let pyl = smin(sdCapsule(pp, vec3f(-1.90, 0.90, 0.50), vec3f(-2.00, 1.07, 0.50), 0.052), sdCapsule(pp, vec3f(-2.00, 1.07, 0.50), vec3f(-2.07, 1.17, 0.50), 0.040), 0.03);
  let wingD = min(min(wing, endp), min(gurney, pyl));
  d = smin(d, wingD, 0.014);
  let split = sdRBox(q - vec3f(2.12, 0.088, 0.0), vec3f(0.26, 0.013, 0.66), 0.006);
  d = smin(d, split, 0.02);

  // exhaust tips: titanium tubes proud of the tail panel
  let exv = vec2f(q.y - 0.47, q.z - 0.16);
  let tube = max(abs(length(exv) - 0.060) - 0.010, abs(q.x + 2.34) - 0.075);
  d = smin(d, tube, 0.006);
  let tubeM = tube;

  return Parts(d, rest, can, cdark, clampL, lens, tubeM, wingD);
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
  let wingC = smoothstep(0.014, 0.0, P.wingD);
  let roofC = onCan * smoothstep(1.05, 1.18, phi) * smoothstep(0.34, 0.28, q.x);
  let carbon = max(max(band, tail), max(wingC, roofC));
  let cav = smoothstep(0.02, 0.0, P.cdark);
  let lamp = smoothstep(0.012, 0.0, P.clamp_);
  let lens = smoothstep(0.012, 0.0, P.lens) * smoothstep(0.26, 0.30, q.y);
  let tubeM = smoothstep(0.012, 0.0, P.tube) * smoothstep(-2.27, -2.29, q.x);
  return Attrs(vec4f(glass, carbon, cav, lamp), vec4f(0.0, trim, lens, tubeM));
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
  let barrel = max(max(r - 0.248, 0.212 - r), abs(p.z) - 0.148) - 0.004;
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
