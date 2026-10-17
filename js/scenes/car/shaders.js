// Car, road and sky shaders.
//
// Lighting model: one procedural dusk environment (SKY_WGSL) is baked into a prefiltered cube map
// that the body, glass and wheels reflect. The sun is kept analytic (sunLobe) so it can be shadowed
// by a coarse signed-distance texture of the car (sun shadows on the body and on the road). The wet
// road reflects the car through a mirrored offscreen render that is mip-blurred by roughness and by
// height above the road (contact hardening), and reflects the sky and street lamps analytically.
import { CAM_WGSL, NOISE_WGSL, FULLSCREEN_VS } from '../../engine/wgsl.js';
import { HEAD_SHAPES } from './sdf.js';

// ---- environment: sky, lamps, ground, sun ------------------------------------------------------------
export const SKY_WGSL = /* wgsl */ `
${NOISE_WGSL}
struct Env {
  sunDir: vec4f,   // xyz direction to the sun
  sunCol: vec4f,   // rgb sun tint, w = sun elevation (rad)
  zen: vec4f,      // zenith rgb
  hor: vec4f,      // horizon rgb, w = day amount
  paint: vec4f,    // rgb body paint (linear), w = flake amount
  opts: vec4f,     // x headlights, y floor reflectivity, z rim style, w mirror pass
  opts2: vec4f,    // x road scroll (m), y time, z lamps, w wetness
  sdfO: vec4f,     // distance texture origin (m)
  sdfI: vec4f,     // 1 / distance texture extent (1/m)
  sc: vec4f,       // x orange peel, y shadow sharpness, z reflection blur, w spare
}
@group(0) @binding(1) var<uniform> E: Env;
const SUN_E: f32 = 6.0;

fn fres(ndv: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0); }
fn hillH(az: f32) -> f32 { return 0.020 + 0.014 * sin(az * 2.0 + 0.7) + 0.010 * sin(az * 5.0 + 2.0) + 0.005 * sin(az * 13.0 + 1.0); }

// street lamps along the road: 20 heads at 7.6 m, both sides, 38 m .. 162 m away
fn lampPos(i: i32) -> vec3f {
  let k = f32(i / 4);
  let sx = select(-1.0, 1.0, (i & 1) == 0);
  let sz = select(-1.0, 1.0, (i & 2) == 0);
  return vec3f(sx * (54.0 + 32.0 * k) + sz * 12.0, 7.6, sz * 7.7);
}
const LAMP_COL = vec3f(1.0, 0.56, 0.20);
// lobe of every lamp seen (or reflected, ys = -1) from pos along d; rough widens the lobe
fn lampGlow(pos: vec3f, d: vec3f, rough: f32, ys: f32) -> vec3f {
  if (E.opts2.z <= 0.001 || abs(d.y) > 0.6) { return vec3f(0.0); }
  var acc = 0.0;
  let w = 0.00003 + rough * rough * 0.25;
  for (var i = 0; i < 20; i++) {
    var lp = lampPos(i);
    lp.y *= ys;
    let v = lp - pos;
    let dl = length(v);
    let c = dot(d, v / dl);
    let el = 0.152 / (dl * dl) * 260.0;
    acc += el / (TAU * w) * exp((c - 1.0) / w);
  }
  return LAMP_COL * acc * E.opts2.z;
}

fn skyBase(d: vec3f, rough: f32) -> vec3f {
  let L = E.sunDir.xyz;
  let sd = dot(d, L);
  let t = clamp(d.y, 0.0, 1.0);
  var c = mix(E.hor.rgb, E.zen.rgb, 1.0 - exp(-t * 6.0));
  let sdp = max(sd, 0.0);
  c += E.sunCol.rgb * (pow(sdp, 4.0) * 0.10 + pow(sdp, 24.0) * 0.30 + pow(sdp, 260.0) * 0.6) * (1.0 - 0.7 * t);
  c += E.hor.rgb * 0.35 * exp(-t * 18.0);
  let az = atan2(d.z, d.x);
  if (rough < 0.75) {
    // clouds on a high plane (perspective compression toward the horizon), lit from below by the sun
    let cp = d.xz / (max(d.y, 0.0) + 0.06);
    let cs = vec2f(L.x, L.z) / max(length(vec2f(L.x, L.z)), 1e-3);
    let n0 = fbm2(cp * 0.55 + vec2f(3.0, 1.0), 5);
    let n1 = fbm2((cp + cs * 0.35) * 0.55 + vec2f(3.0, 1.0), 5);
    let dens = smoothstep(0.50, 0.70, n0);
    let lit = clamp((n0 - n1) * 5.0 + 0.45, 0.0, 1.0);
    let fadeH = smoothstep(0.012, 0.10, d.y) * (1.0 - smoothstep(0.35, 0.85, d.y));
    let side = 0.35 + 0.65 * pow(sdp, 1.5);
    let under = mix(E.zen.rgb * 0.55 + E.hor.rgb * 0.10, E.hor.rgb * 0.70 + E.sunCol.rgb * 0.35, lit * side);
    let rim = E.sunCol.rgb * 0.9 * pow(sdp, 4.0) * lit;
    c = mix(c, under + rim * 0.5, dens * fadeH * (1.0 - rough) * 0.9);
    // thin high streaks
    let cs2 = fbm2(vec2f(cp.x * 0.25, cp.y * 1.4) + vec2f(7.0, 2.0), 4);
    c += E.hor.rgb * 0.25 * smoothstep(0.55, 0.85, cs2) * fadeH * (0.4 + 0.6 * sdp) * (1.0 - rough);
    // low hazy ridge line
    let bandM = exp(-pow((d.y - 0.075) / 0.018, 2.0)) * smoothstep(0.35, 0.65, fbm3(vec3f(d.x * 4.0, d.y * 30.0, d.z * 4.0), 3)) * (1.0 - rough);
    c += E.hor.rgb * 1.4 * bandM * (0.4 + 0.6 * pow(sdp, 2.0));
  }
  // low hills on the horizon
  let hh = hillH(az);
  let blur = 0.004 + 0.25 * rough;
  let hm = smoothstep(hh + blur, hh - blur, d.y);
  let hillCol = mix(E.hor.rgb * 0.22, vec3f(0.010, 0.014, 0.024), smoothstep(0.0, 0.05, hh - d.y));
  c = mix(c, hillCol, hm);
  // tree line along the road: jagged, with gaps
  let tgap = smoothstep(0.30, 0.52, vnoise2(vec2f(az * 7.0, 4.0)));
  let treeH = (0.005 + 0.009 * vnoise2(vec2f(az * 38.0, 1.0)) + 0.0025 * vnoise2(vec2f(az * 150.0, 3.0))) * tgap;
  let tm = smoothstep(treeH + blur * 0.15 + 0.0004, treeH - blur * 0.15, d.y) * smoothstep(-0.003, 0.0015, d.y);
  c = mix(c, E.hor.rgb * 0.045 + vec3f(0.003, 0.004, 0.005), tm * 0.92);
  return c;
}

// the sun as a lobe around the mirror direction; rough widens it (energy preserving)
fn sunLobe(d: vec3f, rough: f32) -> vec3f {
  let sd = dot(d, E.sunDir.xyz);
  let az = atan2(d.z, d.x);
  let hh = hillH(az);
  let blur = 0.004 + 0.25 * rough;
  let hm = smoothstep(hh + blur, hh - blur, d.y);
  let w = 0.00004 + rough * rough * 0.45;
  let amp = SUN_E / (TAU * w);
  let vis = (1.0 - hm) * smoothstep(-0.01, 0.01, d.y);
  return E.sunCol.rgb * amp * exp(-(1.0 - sd) / w) * vis;
}

// wet road seen from the car: dark asphalt plus a Fresnel mirror of the sky (and lamps) at grazing angles
fn groundEnv(d: vec3f, rough: f32) -> vec3f {
  let m = vec3f(d.x, -d.y, d.z);
  let F = clamp(fres(clamp(m.y, 0.0, 1.0), 0.03) * 1.1, 0.0, 1.0);
  let refl = skyBase(m, 0.12 + rough) + lampGlow(vec3f(0.0, 0.6, 0.0), d, 0.03 + rough, -1.0) * 0.6;
  let dark = E.hor.rgb * 0.020 + vec3f(0.004);
  return mix(dark, refl, F);
}
fn envFull(d: vec3f, rough: f32) -> vec3f {
  let sky = skyBase(d, rough) + lampGlow(vec3f(0.0, 0.6, 0.0), d, rough, 1.0);
  let s = smoothstep(-0.004, 0.004, d.y);
  if (s >= 1.0) { return sky; }
  return mix(groundEnv(d, rough), sky, s);
}
`;

// ---- cube map bake (6 faces) and mip filtering ------------------------------------------------------------
const CUBE_FACE = /* wgsl */ `
struct BV { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) face: u32 }
@vertex fn bvs(@builtin(vertex_index) i: u32, @builtin(instance_index) f: u32) -> BV {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: BV;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  o.face = f;
  return o;
}
fn faceDir(f: u32, u: f32, v: f32) -> vec3f {
  switch (f) {
    case 0u: { return vec3f(1.0, -v, -u); }
    case 1u: { return vec3f(-1.0, -v, u); }
    case 2u: { return vec3f(u, 1.0, v); }
    case 3u: { return vec3f(u, -1.0, -v); }
    case 4u: { return vec3f(u, -v, 1.0); }
    default: { return vec3f(-u, -v, -1.0); }
  }
}
`;
export const ENV_BAKE = SKY_WGSL + CUBE_FACE + /* wgsl */ `
@fragment fn bfs(in: BV) -> @location(0) vec4f {
  let uv = in.uv * 2.0 - 1.0;
  return vec4f(min(envFull(normalize(faceDir(in.face, uv.x, uv.y)), 0.0), vec3f(6.0e3)), 1.0);
}
`;
export const ENV_MIP = CUBE_FACE + /* wgsl */ `
@group(0) @binding(0) var src: texture_cube<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> U: vec4f;   // x = 1 / destination size
@fragment fn bfs(in: BV) -> @location(0) vec4f {
  let uv = in.uv * 2.0 - 1.0;
  let h = U.x * 0.5;
  var c = vec3f(0.0);
  for (var j = 0; j < 4; j++) {
    let o = vec2f(select(-h, h, (j & 1) == 1), select(-h, h, (j & 2) == 2));
    c += textureSampleLevel(src, smp, normalize(faceDir(in.face, uv.x + o.x, uv.y + o.y)), 0.0).rgb;
  }
  return vec4f(c * 0.25, 1.0);
}
`;
// 2D mip chain for the planar reflection targets
export const MIP2D = FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@fragment fn fs(in: VO) -> @location(0) vec4f { return textureSampleLevel(src, smp, in.uv, 0.0); }
`;

// ---- shared runtime bindings and helpers ----------------------------------------------------------------
const RT_COMMON = /* wgsl */ `
${CAM_WGSL}
${SKY_WGSL}
${HEAD_SHAPES}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(2) var sdfTex: texture_3d<f32>;
@group(0) @binding(3) var sampL: sampler;
@group(0) @binding(4) var reflA: texture_2d<f32>;   // mirrored scene: premultiplied colour, coverage
@group(0) @binding(5) var reflB: texture_2d<f32>;   // coverage * height above the road
@group(0) @binding(6) var envTex: texture_cube<f32>;

fn rlod(rough: f32) -> f32 { return clamp(log2(max(rough * 200.0, 1.0)), 0.0, 8.0); }
fn envC(d: vec3f, rough: f32) -> vec3f { return textureSampleLevel(envTex, sampL, d, rlod(rough)).rgb; }
fn irr(n: vec3f) -> vec3f { return textureSampleLevel(envTex, sampL, n, 7.0).rgb; }

// ---- car distance field: sun shadow and occlusion ----
fn sdfAt(p: vec3f) -> f32 {
  let q = (p - E.sdfO.xyz) * E.sdfI.xyz;
  let qc = clamp(q, vec3f(0.0), vec3f(1.0));
  let out = length((q - qc) / E.sdfI.xyz);
  return textureSampleLevel(sdfTex, sampL, qc, 0.0).r + out;
}
fn carShadow(p: vec3f, L: vec3f, k: f32, t0: f32) -> f32 {
  let bmin = E.sdfO.xyz;
  let bmax = bmin + 1.0 / E.sdfI.xyz;
  let inv = 1.0 / (L + select(vec3f(1e-6), vec3f(-1e-6), L < vec3f(0.0)));
  let ta = (bmin - p) * inv;
  let tb = (bmax - p) * inv;
  let tmn = min(ta, tb);
  let tmx = max(ta, tb);
  var t = max(max(tmn.x, max(tmn.y, tmn.z)), t0);
  let te = min(tmx.x, min(tmx.y, tmx.z));
  if (te <= t) { return 1.0; }
  var sh = 1.0;
  for (var i = 0; i < 44; i++) {
    let d = sdfAt(p + L * t);
    sh = min(sh, k * d / t);
    t += clamp(d * 0.85, 0.010, 0.30);
    if (t > te || sh < 0.003) { break; }
  }
  let s = clamp(sh, 0.0, 1.0);
  return s * s * (3.0 - 2.0 * s);
}
fn carAO(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var w = 1.0;
  for (var i = 1; i <= 5; i++) {
    let t = 0.04 * f32(i) * f32(i) * 0.5 + 0.02;
    occ += (t - sdfAt(p + n * t)) * w;
    w *= 0.6;
  }
  return clamp(1.0 - occ * 1.6, 0.0, 1.0);
}

// ---- paint: clear coat over a metallic flake layer over a pigmented base ----
fn paintShade(base0: vec3f, flakeAmt0: f32, N0: vec3f, V: vec3f, ao: f32, lp: vec3f, wp: vec3f, fp: f32, sh: f32) -> vec3f {
  let L = E.sunDir.xyz;
  // road grime on the lowest panels
  let base = base0 * (1.0 - 0.45 * smoothstep(0.30, 0.10, wp.y));
  // orange peel: low-amplitude waviness of the coat that breaks up sharp reflections
  var N = N0;
  let pk = E.sc.x * (1.0 - smoothstep(0.0012, 0.0040, fp));
  if (pk > 0.001) {
    let q = lp * 150.0;
    let g = vec3f(vnoise3(q), vnoise3(q + vec3f(19.1, 7.3, 3.7)), vnoise3(q + vec3f(41.7, 11.9, 29.3))) - 0.5
          + (vec3f(vnoise3(q * 2.9 + 5.0), vnoise3(q * 2.9 + 23.0), vnoise3(q * 2.9 + 61.0)) - 0.5) * 0.6;
    N = normalize(N0 + (g - N0 * dot(g, N0)) * 0.07 * pk);
  }
  // rain beads on upward-facing paint: small lenses that bend the reflection (only resolvable close up)
  let dfade = E.sc.w * (1.0 - smoothstep(0.0012, 0.0032, fp)) * smoothstep(0.05, 0.6, N0.y);
  if (dfade > 0.01) {
    let pr = projOf(lp, N0);
    let uvd = pr.uv * 26.0;
    let cid = floor(uvd);
    let cf = fract(uvd) - 0.5;
    let rn = hash22(cid);
    let off = cf - (rn - 0.5) * 0.45;
    let rad = mix(0.05, 0.18, rn.x * rn.y);
    let dd = length(off) / rad;
    if (dd < 1.0 && hash12(cid + 7.0) < 0.38 * dfade) {
      let bulge = (1.0 - dd * dd);
      N = normalize(N + (pr.a * off.x + pr.b * off.y) / rad * 0.55 * bulge);
    }
  }
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  let fc = fres(ndv, 0.04);
  let so = mix(1.0, ao, 0.75);
  // coat reflection: crisp environment plus the sun, which the car's own shape can shadow
  let coat = (envC(R, 0.025) + sunLobe(R, 0.025) * sh) * fc * so;
  // metallic flakes: random micro-facets inside the base coat
  let cell = floor(wp * 1500.0);
  let hv = hash33(cell);
  let nf = normalize(N + (hv - 0.5) * 0.9);
  let flakeAmt = flakeAmt0 * (1.0 - smoothstep(0.0008, 0.0030, fp));
  let flake = (envC(reflect(-V, nf), 0.10) + sunLobe(reflect(-V, nf), 0.10) * sh) * pow(hv.x, 3.0) * flakeAmt * 0.5;
  let Nb = normalize(N0 + (N - N0) * 0.35);
  let Rb = reflect(-V, Nb);
  let mf = 0.55;                                   // share of the base coat that is metallic flake
  let f0m = base * 0.9 + vec3f(0.03);
  let metal = f0m * (envC(Rb, 0.22) + sunLobe(Rb, 0.22) * sh) * mf;
  let ndl = sat(dot(N, L));
  let diff = base * (irr(N) * ao + E.sunCol.rgb * SUN_E * ndl * sh / PI) * (1.0 - mf);
  // grazing sheen of the pigmented layer
  let sheen = base * envC(R, 0.6) * pow(1.0 - ndv, 3.0) * 0.4;
  return coat + (1.0 - fc) * (diff + (metal + base * flake) * so + sheen * so);
}

// projection frame for procedural surface patterns, by dominant normal axis
struct Proj { uv: vec2f, a: vec3f, b: vec3f }
fn projOf(lp: vec3f, N: vec3f) -> Proj {
  let an = abs(N);
  if (an.y > an.x && an.y > an.z) { return Proj(lp.xz, vec3f(1, 0, 0), vec3f(0, 0, 1)); }
  if (an.x > an.z) { return Proj(lp.zy, vec3f(0, 0, 1), vec3f(0, 1, 0)); }
  return Proj(lp.xy, vec3f(1, 0, 0), vec3f(0, 1, 0));
}

fn carbonShade(N0: vec3f, V: vec3f, ao: f32, lp: vec3f, fp: f32, sh: f32) -> vec3f {
  let pr = projOf(lp, N0);
  let tow = 0.0042;
  let g = pr.uv / tow;
  let ci = floor(g); let f = fract(g);
  let k = (i32(ci.x) + i32(ci.y)) & 3;
  let vert = k < 2;
  let s = select(f.y, f.x, vert);
  let lod = 1.0 - smoothstep(0.0008, 0.0035, fp);
  let tilt = cos(PI * s) * 0.55 * lod;
  let Nw = normalize(N0 + select(pr.b, pr.a, vert) * tilt);
  let ndv = max(dot(N0, V), 1e-3);
  let R = reflect(-V, N0);
  let Rw = reflect(-V, Nw);
  let fc = fres(ndv, 0.04);
  let so = mix(1.0, ao, 0.75);
  let fibre = mix(1.0, 0.55 + 0.45 * sin(PI * s), lod);
  let L = E.sunDir.xyz;
  let diff = vec3f(0.012) * fibre * (irr(N0) * ao + E.sunCol.rgb * SUN_E * sat(dot(N0, L)) * sh / PI);
  let sheen = (envC(Rw, 0.42) + sunLobe(Rw, 0.42) * sh) * 0.14 * fibre;
  return (envC(R, 0.03) + sunLobe(R, 0.03) * sh) * fc * so + (1.0 - fc) * (diff + sheen * so);
}

// ---- cabin seen through the glass: a handful of boxes in car space, ray-cast from the eye ----
fn boxHit(ro: vec3f, rd: vec3f, c: vec3f, h: vec3f) -> vec4f {
  let m = 1.0 / (rd + select(vec3f(1e-5), vec3f(-1e-5), rd < vec3f(0.0)));
  let n = m * (ro - c);
  let k = abs(m) * h;
  let t1 = -n - k;
  let t2 = -n + k;
  let tN = max(max(t1.x, t1.y), t1.z);
  let tF = min(min(t2.x, t2.y), t2.z);
  if (tN > tF || tF < 0.0) { return vec4f(-1.0); }
  let nor = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
  return vec4f(max(tN, 0.0), nor);
}
fn cabin(ro: vec3f, rd: vec3f) -> vec3f {
  var best = 1e3;
  var nrm = vec3f(0.0, 1.0, 0.0);
  var mat = 0.0;
  // dashboard, binnacle, tunnel, seats (cushion, back, head restraint), door cards, bulkhead
  var hs = array<vec4f, 12>(
    vec4f(0.80, 0.66, 0.0, 0.0), vec4f(0.62, 0.77, 0.30, 0.0), vec4f(0.05, 0.40, 0.0, 0.0),
    vec4f(-0.06, 0.38, 0.30, 0.0), vec4f(-0.06, 0.38, -0.30, 0.0),
    vec4f(-0.30, 0.64, 0.30, 0.0), vec4f(-0.30, 0.64, -0.30, 0.0),
    vec4f(-0.37, 0.95, 0.30, 0.0), vec4f(-0.37, 0.95, -0.30, 0.0),
    vec4f(0.0, 0.46, 0.80, 0.0), vec4f(0.0, 0.46, -0.80, 0.0), vec4f(-0.58, 0.70, 0.0, 0.0));
  var hh = array<vec3f, 12>(
    vec3f(0.17, 0.10, 0.70), vec3f(0.09, 0.05, 0.20), vec3f(0.60, 0.18, 0.075),
    vec3f(0.25, 0.07, 0.19), vec3f(0.25, 0.07, 0.19),
    vec3f(0.065, 0.31, 0.19), vec3f(0.065, 0.31, 0.19),
    vec3f(0.055, 0.10, 0.12), vec3f(0.055, 0.10, 0.12),
    vec3f(0.95, 0.30, 0.025), vec3f(0.95, 0.30, 0.025), vec3f(0.03, 0.50, 0.75));
  for (var i = 0; i < 12; i++) {
    let h = boxHit(ro, rd, vec3f(hs[i].x, hs[i].y, hs[i].z), hh[i]);
    if (h.x > 0.0 && h.x < best) { best = h.x; nrm = h.yzw; mat = f32(i); }
  }
  // headliner and floor
  if (rd.y > 1e-3) { let t = (0.95 - 0.12 * clamp(ro.x, 0.0, 1.2) - ro.y) / rd.y; if (t > 0.0 && t < best) { best = t; nrm = vec3f(0.0, -1.0, 0.0); mat = 20.0; } }
  if (rd.y < -1e-3) { let t = (0.22 - ro.y) / rd.y; if (t > 0.0 && t < best) { best = t; nrm = vec3f(0.0, 1.0, 0.0); mat = 21.0; } }
  if (best > 50.0) { return vec3f(0.002); }
  let p = ro + rd * best;
  var alb = vec3f(0.060, 0.034, 0.022);            // tan leather
  if (mat == 0.0 || mat == 1.0) { alb = vec3f(0.012, 0.012, 0.014); }
  if (mat >= 3.0 && mat <= 8.0) { alb = vec3f(0.085, 0.045, 0.028) * (0.85 + 0.3 * step(0.5, fract(p.y * 22.0))); }   // quilted seats
  if (mat == 20.0) { alb = vec3f(0.020); }
  if (mat == 21.0) { alb = vec3f(0.008); }
  let amb = irr(nrm) * 0.9 + E.hor.rgb * 0.12;
  var c = alb * amb * (0.4 + 0.6 * smoothstep(-0.6, 0.8, nrm.y));
  // instrument cluster and ambient strip glow
  let cl = step(abs(p.x - 0.66), 0.06) * step(abs(p.y - 0.80), 0.030) * step(abs(abs(p.z) - 0.30), 0.12) * step(mat, 1.5);
  c += vec3f(1.0, 0.35, 0.10) * cl * 0.0;
  let strip = step(abs(p.y - 0.735), 0.004) * step(mat, 0.5);
  c += vec3f(1.0, 0.2, 0.08) * strip * 0.6;
  return c;
}

fn glassShade(N: vec3f, V: vec3f, ao: f32, sh: f32, wp: vec3f) -> vec3f {
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  let fg = fres(ndv, 0.07);
  let tint = vec3f(0.34, 0.38, 0.42);
  let inside = cabin(wp - V * 0.004, -V) * tint + E.hor.rgb * 0.004 * (1.0 - ndv);
  return (envC(R, 0.012) + sunLobe(R, 0.012) * sh) * fg + (1.0 - fg) * inside * mix(0.6, 1.0, ao);
}

fn cavShade(N: vec3f, V: vec3f, ao: f32, lp: vec3f) -> vec3f {
  let pr = projOf(lp, N);
  let q = pr.uv * (TAU / 0.011);
  let f = cos(q.x) + cos(dot(q, vec2f(-0.5, 0.866))) + cos(dot(q, vec2f(-0.5, -0.866)));
  let hole = smoothstep(1.2, 1.9, f);
  let mesh = 1.0 - hole;
  let R = reflect(-V, N);
  let rib = 0.55 + 0.45 * sin(lp.y * 150.0);
  return vec3f(0.004) * (0.3 + hole) + mesh * (envC(R, 0.5) * 0.03 + vec3f(0.014) * irr(N)) * ao * rib;
}

fn gapLines(lp: vec3f, N: vec3f, fp: f32) -> f32 {
  var d = 1e3;
  let sideF = smoothstep(0.4, 0.7, abs(N.z));
  let topF = smoothstep(0.4, 0.7, N.y);
  // door outline
  let q2 = vec2f(lp.x - 0.19, lp.y - 0.55);
  let dd = abs(q2) - vec2f(0.58, 0.245);
  let sdDoor = length(max(dd, vec2f(0.0))) + min(max(dd.x, dd.y), 0.0);
  d = min(d, mix(1e3, abs(sdDoor), sideF * smoothstep(0.7, 0.8, abs(lp.z))));
  // hood outline (plan polyline) and engine cover
  let hp = vec2f(lp.x, abs(lp.z));
  var hl = seg2(hp, vec2f(1.14, 0.0), vec2f(1.14, 0.60));
  hl = min(hl, seg2(hp, vec2f(1.14, 0.60), vec2f(1.72, 0.58)));
  hl = min(hl, seg2(hp, vec2f(1.72, 0.58), vec2f(1.98, 0.36)));
  hl = min(hl, seg2(hp, vec2f(1.98, 0.36), vec2f(2.21, 0.32)));
  hl = min(hl, seg2(hp, vec2f(2.21, 0.32), vec2f(2.23, 0.0)));
  d = min(d, mix(1e3, hl, topF));
  let eq = abs(vec2f(lp.x + 1.40, abs(lp.z))) - vec2f(0.46, 0.60);
  d = min(d, mix(1e3, abs(length(max(eq, vec2f(0.0))) + min(max(eq.x, eq.y), 0.0)), topF));
  return 1.0 - smoothstep(0.0007, 0.0007 + fp * 1.1, d);
}

struct VIn {
  @location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) m1: vec4f, @location(3) m2: vec4f,
  @location(4) i0: vec4f, @location(5) i1: vec4f, @location(6) i2: vec4f, @location(7) i3: vec4f,
}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) m1: vec4f, @location(3) m2: vec4f,
  @location(4) lp: vec3f, @location(5) ln: vec3f,
}
@vertex fn cvs(v: VIn) -> VOut {
  let M = mat4x4f(v.i0, v.i1, v.i2, v.i3);
  let w = (M * vec4f(v.pos, 1.0)).xyz;
  var wm = w;
  if (E.opts.w > 0.5) { wm.y = -wm.y; }
  var o: VOut;
  o.clip = cam.viewProj * vec4f(wm - cam.pos.xyz, 1.0);
  o.wp = w;
  o.n = normalize((M * vec4f(v.nrm, 0.0)).xyz);
  o.m1 = v.m1; o.m2 = v.m2;
  o.lp = v.pos; o.ln = v.nrm;
  return o;
}
fn viewVec(wp: vec3f) -> vec3f {
  var cp = cam.pos.xyz;
  if (E.opts.w > 0.5) { cp.y = -cp.y; }
  return normalize(cp - wp);
}

fn bodyColor(in: VOut) -> vec3f {
  let V = viewVec(in.wp);
  var N = normalize(in.n);
  if (dot(N, V) < 0.0) { N = -N; }
  let fp = length(fwidth(in.wp));
  let glass = smoothstep(0.4, 0.6, in.m1.x);
  let carbon = smoothstep(0.4, 0.6, in.m1.y);
  let cav = smoothstep(0.4, 0.6, in.m1.z);
  let lamp = smoothstep(0.4, 0.6, in.m1.w);
  let trim = smoothstep(0.4, 0.6, in.m2.y);
  let tubeM = smoothstep(0.4, 0.6, in.m2.w);
  let L = E.sunDir.xyz;
  var ao = in.m2.x * mix(0.35, 1.0, smoothstep(0.02, 0.32, in.wp.y));
  var sh = 0.0;
  if (dot(N, L) > -0.1 && L.y > -0.02) { sh = carShadow(in.wp + N * 0.035, L, E.sc.y, 0.0); }
  let gap = gapLines(in.lp, N, fp);
  var col = paintShade(E.paint.rgb, E.paint.w, N, V, ao, in.lp, in.wp, fp, sh);
  col = mix(col, carbonShade(N, V, ao, in.lp, fp, sh), carbon);
  col *= mix(1.0, 0.30, gap * (1.0 - glass));
  if (trim > 0.01) { col = mix(col, paintShade(vec3f(0.006), 0.0, N, V, ao, in.lp, in.wp, fp, sh), trim); }
  if (tubeM > 0.01) {
    // titanium exhaust tips: heat-tinted towards the outlet, sooty inside
    let heat = smoothstep(-2.36, -2.26, in.lp.x);
    let ti = mix(vec3f(0.30, 0.28, 0.27), vec3f(0.32, 0.22, 0.45), heat);
    let rr = (envC(reflect(-V, N), 0.18) + sunLobe(reflect(-V, N), 0.18) * sh);
    col = mix(col, ti * rr * 1.2 * ao + ti * irr(N) * 0.1, tubeM);
  }
  if (glass > 0.01) { col = mix(col, glassShade(N, V, ao, sh, in.wp), glass); }
  if (cav > 0.01) { col = mix(col, cavShade(N, V, ao, in.lp), cav); }
  // headlamps: dark glossy housing, crisp DRL blade and projector painted from their plan-view shapes
  let hs = headShapes(vec2f(in.lp.x, abs(in.lp.z)));
  let topF = smoothstep(0.15, 0.45, in.ln.y) * step(0.28, in.lp.y) * step(1.7, in.lp.x);
  let aa = max(fp * 1.1, 0.0005);
  let hous = (1.0 - smoothstep(-aa, aa, hs.x)) * topF;
  if (hous > 0.01) {
    let ndvh = max(dot(N, V), 1e-3);
    let hg = (envC(reflect(-V, N), 0.02) + sunLobe(reflect(-V, N), 0.02) * sh) * fres(ndvh, 0.06) + vec3f(0.003, 0.004, 0.006);
    col = mix(col, hg, hous);
    let drl = (1.0 - smoothstep(0.0, aa * 1.5, hs.y)) * topF;
    let prj = (1.0 - smoothstep(0.0, aa * 1.5, hs.z)) * topF;
    let glow = vec3f(0.85, 0.93, 1.0) * (0.3 + 14.0 * E.opts.x);
    col = mix(col, glow, drl);
    col = mix(col, glow * 1.4 + vec3f(0.1), prj * 0.85);
  }
  if (lamp > 0.01) {
    let ndv = max(dot(N, V), 1e-3);
    let lens = envC(reflect(-V, N), 0.02) * fres(ndv, 0.05);
    var em: vec3f;
    if (in.lp.x > 0.0) { em = vec3f(0.85, 0.93, 1.0) * (0.25 + 12.0 * E.opts.x); }
    else { em = vec3f(1.0, 0.025, 0.012) * (1.2 + 9.0 * (0.35 + 0.65 * E.opts.x)); }
    col = mix(col, em + lens, lamp);
  }
  return col;
}

// pseudo lettering on the tyre sidewall: blocks of a 3x5 grid, switched on by a hash
fn sidewallText(r: f32, ang: f32) -> f32 {
  let v = (r - 0.268) / 0.017;
  if (v < 0.0 || v > 1.0) { return 0.0; }
  let arc = ang * r;
  let id = floor(arc / 0.0100);
  let cu = fract(arc / 0.0100);
  let gx = floor(cu * 3.0);
  let gy = floor(v * 5.0);
  let on = step(0.50, hash12(vec2f(id * 1.37, gx + 3.0 * gy + 11.0 * floor(id * 0.37)))) * step(0.12, cu) * step(cu, 0.9) * step(0.10, fract(v * 5.0));
  let word = step(abs(ang - 0.45), 0.5) + step(abs(ang + 2.15), 0.42);
  return on * clamp(word, 0.0, 1.0);
}

fn wheelColor(in: VOut) -> vec3f {
  let V = viewVec(in.wp);
  var N = normalize(in.n);
  if (dot(N, V) < 0.0) { N = -N; }
  let fp = length(fwidth(in.wp));
  let rim = smoothstep(0.4, 0.6, in.m1.x);
  let tire = smoothstep(0.4, 0.6, in.m1.y);
  let disc = smoothstep(0.4, 0.6, in.m1.z);
  let cal = smoothstep(0.4, 0.6, in.m1.w);
  let lp = in.lp;
  let r = length(lp.xy);
  let ang = atan2(lp.y, lp.x);
  var ao = in.m2.x * mix(0.18, 1.0, smoothstep(0.62, 0.20, in.wp.y)) * mix(0.55, 1.0, smoothstep(0.0, 0.2, in.wp.y));
  let L = E.sunDir.xyz;
  var sh = 0.0;
  if (dot(N, L) > -0.1 && L.y > -0.02) { sh = carShadow(in.wp + N * 0.02, L, E.sc.y, 0.0); }
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  var col: vec3f;
  if (tire > 0.5 && rim < 0.5) {
    let onTread = smoothstep(0.65, 0.35, abs(in.ln.z));
    let lod = 1.0 - smoothstep(0.002, 0.008, fp);
    let gz = abs(abs(lp.z) - 0.045);
    let g1 = 1.0 - smoothstep(0.004, 0.006, gz);
    let g2 = 1.0 - smoothstep(0.004, 0.006, abs(abs(lp.z) - 0.105));
    let sipe = smoothstep(0.42, 0.49, abs(fract(ang * 56.0 / TAU + lp.z * 6.0) - 0.5));
    let groove = max(max(g1, g2) * onTread, sipe * onTread * 0.55) * lod;
    let rings = 0.5 + 0.5 * sin(r * 520.0);
    let side = (1.0 - onTread) * lod;
    var alb = vec3f(0.020) * (1.0 - 0.7 * groove) * (1.0 - 0.18 * rings * side);
    // lettering and a moulded rim-protector ridge on the outer sidewall
    let outer = step(0.05, lp.z) * (1.0 - onTread);
    let txt = sidewallText(r, ang) * outer * lod;
    let ridge = exp(-pow((r - 0.258) / 0.004, 2.0)) * outer;
    alb = alb * (1.0 - 0.0) + vec3f(0.07, 0.07, 0.065) * txt + vec3f(0.03) * ridge;
    // road dust / brake dust, heavier low and near the rim
    let dust = smoothstep(0.30, 0.25, r) * 0.012 * outer;
    alb += vec3f(dust, dust * 0.9, dust * 0.8);
    col = alb * (irr(N) * ao + E.sunCol.rgb * SUN_E * sat(dot(N, L)) * sh / PI) + (envC(R, 0.58) + sunLobe(R, 0.58) * sh) * fres(ndv, 0.04) * 0.3 * mix(1.0, ao, 0.8) * (1.0 - 0.8 * groove);
  } else if (rim > 0.5 || (tire <= 0.5 && disc <= 0.5 && cal <= 0.5)) {
    // wheel: satin graphite face with a diamond-cut polished lip and a machined spoke edge
    let lip = smoothstep(0.236, 0.246, r) * smoothstep(0.06, 0.13, lp.z);
    let brush = 0.85 + 0.15 * hash11(floor(r * 900.0)) + 0.1 * hash11(floor(ang * 160.0));
    let graphite = vec3f(0.055, 0.056, 0.060) * brush;
    let alu = vec3f(0.82, 0.80, 0.77);
    let f0 = mix(graphite, alu, lip);
    let rough = mix(0.30, 0.10, lip);
    let spec = (envC(reflect(-V, N), rough) + sunLobe(reflect(-V, N), rough) * sh);
    col = f0 * spec * mix(1.0, ao, 0.85) + graphite * irr(N) * ao * 0.35 * (1.0 - lip)
        + (envC(R, 0.03) + sunLobe(R, 0.03) * sh) * fres(ndv, 0.04) * 0.6 * ao;
    // spoke sides and hub are lit like dark anodised metal; centre cap badge ring
    let cap = smoothstep(0.034, 0.030, r) * step(0.10, lp.z);
    col = mix(col, vec3f(0.5, 0.05, 0.04) * spec * 0.5, cap * smoothstep(0.014, 0.010, abs(r - 0.022)));
  } else if (disc > 0.5) {
    let brushd = 0.80 + 0.2 * hash11(floor(r * 1400.0));
    var bc = vec3f(0.30, 0.29, 0.28) * brushd;
    if (r < 0.118) { bc = vec3f(0.035); }
    col = bc * (envC(R, 0.32) + sunLobe(R, 0.32) * sh) * mix(1.0, ao, 0.8) * 1.0 + bc * irr(N) * ao * 0.12;
    // heat-darkened outer edge
    col *= mix(1.0, 0.7, smoothstep(0.17, 0.188, r));
  } else {
    // caliper: gloss red enamel with a lettering strip
    var c0 = paintShade(vec3f(0.60, 0.010, 0.008), 0.0, N, V, ao, in.lp, in.wp, fp, sh);
    let strip = step(abs(lp.x + 0.012), 0.016) * step(0.0, lp.z);
    col = c0;
  }
  return col;
}

fn wheelDiscard(in: VOut) {
  let disc = smoothstep(0.4, 0.6, in.m1.z);
  let lp = in.lp;
  let r = length(lp.xy);
  let ang = atan2(lp.y, lp.x);
  if (disc > 0.5 && r > 0.12 && r < 0.186) {
    let stp = TAU / 28.0;
    let row = select(0.136, 0.168, (i32(floor(ang / stp)) & 1) == 0);
    let a2 = (fract(ang / stp) - 0.5) * stp * r;
    if (length(vec2f(a2, r - row)) < 0.0056) { discard; }
  }
}

@fragment fn fs_body(in: VOut) -> @location(0) vec4f { return vec4f(bodyColor(in), 1.0); }
@fragment fn fs_wheel(in: VOut) -> @location(0) vec4f { wheelDiscard(in); return vec4f(wheelColor(in), 1.0); }
struct MOut { @location(0) a: vec4f, @location(1) b: vec4f }
@fragment fn fs_body_m(in: VOut) -> MOut { return MOut(vec4f(bodyColor(in), 1.0), vec4f(in.wp.y, 0.0, 0.0, 1.0)); }
@fragment fn fs_wheel_m(in: VOut) -> MOut { wheelDiscard(in); return MOut(vec4f(wheelColor(in), 1.0), vec4f(in.wp.y, 0.0, 0.0, 1.0)); }

// ---- sky (analytic, full resolution) with distant street lamps ----
${FULLSCREEN_VS}
// closest distance between a ray (ro, unit rd) and a segment a..b; y = ray parameter at the closest point
fn raySeg(ro: vec3f, rd: vec3f, a: vec3f, b: vec3f) -> vec2f {
  let v = b - a;
  let w0 = ro - a;
  let bb = dot(rd, v); let cc = dot(v, v); let dd = dot(rd, w0); let ee = dot(v, w0);
  let den = max(cc - bb * bb, 1e-6);
  var s = clamp((ee - bb * dd) / den, 0.0, 1.0);
  let t = max((bb * s - dd), 0.0);
  return vec2f(length(ro + rd * t - (a + v * s)), t);
}
@fragment fn fs_sky(in: VO) -> @location(0) vec4f {
  let d = camRay(cam, in.pos.xy);
  var c = skyBase(d, 0.0) + sunLobe(d, 0.0);
  if (E.opts2.z > 0.001 && d.y < 0.5) {
    for (var i = 0; i < 20; i++) {
      let lp = lampPos(i);
      let v = lp - cam.pos.xyz;
      let dl = length(v);
      let cs = dot(d, v / dl);
      let ang = acos(clamp(cs, -1.0, 1.0));
      let head = 0.20 / dl;
      c += LAMP_COL * (smoothstep(head * 1.4, head * 0.6, ang) * 140.0 + exp(-ang * ang / (head * head * 40.0)) * 3.0 + exp(-ang * ang / (head * head * 1400.0)) * 0.45) * E.opts2.z;
      // pole and arm: distance between the view ray and each segment
      let base = vec3f(lp.x, 0.0, lp.z + sign(lp.z) * 1.6);
      let top = vec3f(base.x, 8.0, base.z);
      let pxA = max(dl * length(cam.up.xyz) * 2.0 * cam.res.w, 1e-4);
      let c1 = raySeg(cam.pos.xyz, d, base, top);
      let c2 = raySeg(cam.pos.xyz, d, top, lp + vec3f(0.0, 0.05, 0.0));
      let cov = max(1.0 - smoothstep(0.07, 0.07 + pxA * 1.2, c1.x), 1.0 - smoothstep(0.035, 0.035 + pxA * 1.2, c2.x));
      c = mix(c, vec3f(0.004, 0.004, 0.006), cov * step(0.0, d.y + 0.02) * step(0.0, c1.y));
    }
  }
  return vec4f(c, 1.0);
}

// ---- wet road ----
struct GV { @builtin(position) clip: vec4f, @location(0) wp: vec3f }
@vertex fn gvs(@builtin(vertex_index) i: u32) -> GV {
  var p = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let w = vec3f(cam.pos.x + p[i].x * 4000.0, 0.0, cam.pos.z + p[i].y * 4000.0);
  var o: GV;
  o.clip = cam.viewProj * vec4f(w - cam.pos.xyz, 1.0);
  o.wp = w;
  return o;
}
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
fn screenUV(q: vec3f) -> vec2f {
  let c = cam.viewProj * vec4f(q - cam.pos.xyz, 1.0);
  let n = c.xy / max(c.w, 1e-4);
  return vec2f(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5);
}

// distance-to-border of a cellular pattern: sealed cracks and expansion seams in the asphalt
fn cellEdge(p: vec2f) -> f32 {
  let ip = floor(p); let f = fract(p);
  var d1 = 8.0; var d2 = 8.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let g = vec2f(f32(x), f32(y));
      let r = g + hash22(ip + g) - f;
      let d = dot(r, r);
      if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
  }
  return sqrt(d2) - sqrt(d1);
}

@fragment fn gfs(in: GV) -> @location(0) vec4f {
  let p = in.wp;
  let toCam = cam.pos.xyz - p;
  let dcam = length(toCam);
  let V = toCam / dcam;
  let L = E.sunDir.xyz;
  let fpx = length(fwidth(p.xz));
  let sc = vec2f(p.x + E.opts2.x, p.z);
  // asphalt: aggregate, patches, sealed cracks
  let agg = vnoise2(sc * 220.0);
  let agg2 = vnoise2(sc * 61.0);
  let stains = fbm2(sc * 0.7, 4);
  var alb = vec3f(0.052, 0.050, 0.049) * (0.50 + 0.95 * stains) * mix(0.75, 1.25, agg) * mix(0.9, 1.1, agg2);
  let speck = step(0.93, hash12(floor(sc * 160.0))) * (1.0 - smoothstep(0.004, 0.02, fpx));
  alb += vec3f(0.06) * speck;
  let cw = cellEdge(sc * 0.42 + (vec2f(vnoise2(sc * 3.1), vnoise2(sc * 3.1 + 9.0)) - 0.5) * 0.5);
  let crackOn = step(0.74, hash12(floor(sc * 0.42) + 3.0));
  let crack = (1.0 - smoothstep(0.0, 0.006 + fpx * 0.6, cw)) * crackOn;
  let patchM = smoothstep(0.55, 0.60, fbm2(sc * 0.18 + vec2f(40.0, 5.0), 3));
  alb *= 1.0 - 0.55 * crack;
  alb *= mix(1.0, 0.55, patchM);
  // lane paint: dashed centre lines at +-3.6 m, solid edge lines at +-7.2 m
  let dash = step(fract(sc.x / 9.0), 0.33);
  let ln1 = (1.0 - smoothstep(0.06, 0.06 + fpx * 1.5, abs(abs(p.z) - 3.6))) * dash;
  let ln2 = 1.0 - smoothstep(0.08, 0.08 + fpx * 1.5, abs(abs(p.z) - 7.2));
  let paintM = max(ln1, ln2) * (0.45 + 0.4 * stains) * (0.55 + 0.45 * agg) * (1.0 - 0.5 * crack);
  alb = mix(alb, vec3f(0.55, 0.53, 0.48), paintM);
  // shoulders: wet grass and gravel outside the lane edge lines
  let shoulder = smoothstep(8.2, 8.9, abs(p.z));
  let gr = fbm2(sc * 3.0, 4);
  let fieldB = smoothstep(0.45, 0.55, fbm2(vec2f(sc.x * 0.012, p.z * 0.03), 3));
  let grass = mix(vec3f(0.010, 0.022, 0.006), vec3f(0.024, 0.026, 0.008), fieldB) * (0.4 + 1.3 * gr) * mix(0.35, 1.7, vnoise2(sc * 40.0)) * mix(0.55, 1.45, vnoise2(sc * 5.0)) * mix(0.7, 1.3, vnoise2(sc * 0.9)) * mix(0.75, 1.15, 0.5 + 0.5 * sin(p.z * 2.2 + 3.0 * vnoise2(sc * 0.05))) * mix(1.0, 0.8, fieldB);
  alb = mix(alb, grass, shoulder);
  // wetness: puddles in the low spots, a damp film everywhere else, cracks hold water
  let wetN = fbm2(sc * 0.32 + vec2f(3.1, 7.7), 5);
  let puddle = max(smoothstep(0.47, 0.55, wetN), crack * 0.35) * (1.0 - shoulder);
  let wet = clamp(puddle * 0.9 + 0.40, 0.0, 1.0) * (1.0 - 0.9 * shoulder);
  let wetAlb = alb * mix(1.0, 0.42, wet);
  // surface normal: aggregate micro-bumps (damp asphalt) vs. almost flat water, plus a gentle warp
  let rough0 = mix(mix(0.42, 0.85, shoulder), 0.045, puddle);
  let mf = 1.0 - smoothstep(0.002, 0.014, fpx);
  let e = 0.0025;
  let h0 = vnoise2(sc * 220.0);
  let hx = vnoise2((sc + vec2f(e, 0.0)) * 220.0);
  let hz = vnoise2((sc + vec2f(0.0, e)) * 220.0);
  let bump = vec2f(h0 - hx, h0 - hz) * (1.0 - puddle) * 0.32 * mf;
  let warp = vec2f(vnoise2(sc * 2.3 + 11.0), vnoise2(sc * 2.3 + 71.0)) - 0.5;
  let warp2 = vec2f(vnoise2(sc * 9.0 + 3.0), vnoise2(sc * 9.0 + 33.0)) - 0.5;
  let nlow = (warp * 0.006 + warp2 * 0.003) * mix(0.5, 1.0, puddle);
  let N = normalize(vec3f(bump.x + nlow.x, 1.0, bump.y + nlow.y));
  let rough = clamp(rough0 + (1.0 - mf) * 0.15, 0.03, 0.9);
  let ndv = clamp(dot(N, V), 1e-3, 1.0);
  let F = fres(ndv, mix(0.04, 0.02, wet)) * E.opts.y * (1.0 - 0.96 * shoulder);
  // contact occlusion and sun shadow from the car's distance field
  let ao = carAO(vec3f(p.x, 0.0, p.z), vec3f(0.0, 1.0, 0.0));
  var sh = 1.0;
  if (L.y > 0.0) { sh = carShadow(p + vec3f(0.0, 0.01, 0.0), L, E.sc.y, 0.05); }
  let sunD = E.sunCol.rgb * SUN_E * max(L.y, 0.0) * sh / PI;
  let diff = wetAlb * (irr(vec3f(0.0, 1.0, 0.0)) * ao + sunD);
  // reflected environment: sky, lamps and sun analytically (lobes widen with roughness)
  let R = reflect(-V, N);
  var Rg = R;
  Rg.y = max(Rg.y, 0.0005);
  let skyR = skyBase(Rg, rough * 0.8) + lampGlow(p, Rg, rough * 0.16, 1.0) + sunLobe(Rg, rough * 0.55);
  // mirrored car: contact-hardening blur, rotated taps through the mip chain
  let uv0 = vec2f(in.clip.x * cam.res.z, in.clip.y * cam.res.w);
  let base0 = textureSampleLevel(reflB, sampL, uv0, 0.0);
  let pxPerM = cam.res.y / (2.0 * length(cam.up.xyz) * dcam);
  let reach = base0.r * 1.3 + 0.06;
  let q = p + vec3f(N.x, 0.0, N.z) * reach * 2.2;
  let uv = screenUV(q);
  let blurPx = (rough * rough * 2.2 + 0.004) * (0.15 + reach * 0.9) * E.sc.z * pxPerM;
  var acc = vec4f(0.0);
  if (blurPx < 0.8) {
    acc = textureSampleLevel(reflA, sampL, uv, 0.0);
  } else {
    let lodT = clamp(log2(max(blurPx * 0.4, 1.0)), 0.0, 6.0);
    let rot = ign(in.clip.xy) * TAU;
    for (var i = 0; i < 12; i++) {
      let rr = sqrt((f32(i) + 0.5) / 12.0) * blurPx;
      let ang = f32(i) * 2.39996 + rot;
      acc += textureSampleLevel(reflA, sampL, uv + vec2f(cos(ang), sin(ang)) * rr * cam.res.zw, lodT);
    }
    acc /= 12.0;
  }
  let cov = clamp(acc.a, 0.0, 1.0);
  let carR = acc.rgb / max(cov, 1e-3);
  let refl = mix(skyR, carR, smoothstep(0.0, 1.0, cov));
  var col = diff * (1.0 - F) + refl * F;
  // distance haze toward the horizon
  let far = smoothstep(300.0, 2500.0, length(p.xz - cam.pos.xz));
  col = mix(col, skyBase(normalize(vec3f(-V.x, 0.002, -V.z)), 0.3), far);
  return vec4f(col, 1.0);
}
`;

export const CAR_SHADER = RT_COMMON;
