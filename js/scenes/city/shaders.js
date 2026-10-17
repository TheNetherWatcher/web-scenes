// WGSL for the city scene: shared preamble (sky, fog, grid helpers, sun shadows from the height
// map), instanced buildings, ribbon beams (cables, masts), light sprites, ground + river, sky pass.
import { CAM_WGSL, NOISE_WGSL } from '../../engine/wgsl.js';
import { GC, GS, HMAP, LAND } from './gen.js';

const F = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

export const COMMON = /* wgsl */ `
diagnostic(off, derivative_uniformity);
${CAM_WGSL}
${NOISE_WGSL}
struct Scn {
  sunDir: vec4f,     // xyz toward sun, w = night factor
  sunRad: vec4f,     // rgb sunlight on surfaces, w = disc multiplier
  zen: vec4f,        // rgb zenith, w = haze beta
  hor: vec4f,        // rgb horizon away from sun, w = lit window amount
  horSun: vec4f,     // rgb horizon toward sun, w = cloud cover
  moon: vec4f,       // xyz dir, w = moon intensity
  ambUp: vec4f,      // rgb, w = water roughness
  ambSide: vec4f,    // rgb, w = time
  ambDown: vec4f,    // rgb, w = reflection pass flag
  cloudRad: vec4f,   // rgb light on clouds
  boats: array<vec4f, 8>,   // x, z, dirX, dirZ
  boatP: array<vec4f, 8>,   // length, speed
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<uniform> scn: Scn;
@group(0) @binding(2) var hmap: texture_2d<f32>;

const GC: f32 = ${F(GC)};
const GS: f32 = ${F(GS)};
const HM0: vec2f = vec2f(${F(HMAP.s0)}, ${F(HMAP.a0)});
const HCELL: f32 = ${F(HMAP.cell)};

fn toGrid(p: vec2f) -> vec2f { return vec2f(p.x * GC + p.y * GS, p.x * GS - p.y * GC); }
fn viewerPos() -> vec3f {
  let c = cam.pos.xyz;
  return select(c, vec3f(c.x, -c.y, c.z), scn.ambDown.w > 0.5);
}
fn night() -> f32 { return scn.sunDir.w; }

// ---------------------------------------------------------------- sky
fn horizonMix(rd: vec3f) -> vec3f {
  let sh = normalize(scn.sunDir.xz + vec2f(1e-5));
  let rh = normalize(rd.xz + vec2f(1e-5));
  let az = dot(sh, rh) * 0.5 + 0.5;
  return mix(scn.hor.rgb, scn.horSun.rgb, pow(az, 3.0));
}
fn skyBase(rd: vec3f) -> vec3f {
  let e = clamp(rd.y, 0.0, 1.0);
  let hm = horizonMix(rd);
  var col = mix(scn.zen.rgb, hm, pow(1.0 - e, 4.5));
  let cs = max(dot(rd, scn.sunDir.xyz), 0.0);
  let glow = pow(cs, 24.0) * 0.55 + pow(cs, 6.0) * 0.14 + pow(cs, 700.0) * 1.6;
  col += scn.sunRad.rgb * glow * 0.35;
  return col;
}
fn skyClouds(rd: vec3f, base: vec3f) -> vec4f {
  // returns rgb cloud colour, a = coverage
  let cover = scn.horSun.w;
  if (rd.y < 0.012 || cover < 0.01) { return vec4f(0.0); }
  let p = rd.xz / (rd.y + 0.05) * 3300.0 + vec2f(scn.ambSide.w * 7.0, scn.ambSide.w * 3.0);
  let q = p * 0.00032;
  let d = fbm2(q, 5) + 0.12 * fbm2(q * 4.0 + 7.0, 3);
  let thr = mix(0.80, 0.40, cover);
  let dens = smoothstep(thr, thr + 0.2, d);
  let sh = normalize(scn.sunDir.xz + vec2f(1e-5));
  let dl = fbm2(q + sh * 0.05, 4) + 0.12 * fbm2((q + sh * 0.05) * 4.0 + 7.0, 3);
  let thick = max(dl - thr, 0.0);
  let lit = exp(-thick * 4.5);
  let amb = mix(scn.zen.rgb, scn.hor.rgb, 0.45) * 0.9 + scn.ambUp.rgb * 0.15;
  var col = scn.cloudRad.rgb * (0.12 + 0.95 * lit) * (1.25 - 0.5 * dens) + amb * (0.55 - 0.15 * dens);
  col *= 1.0 - 0.35 * smoothstep(0.5, 1.0, dens) * (1.0 - lit);
  let a = dens * smoothstep(0.012, 0.16, rd.y);
  return vec4f(col, a * 0.94);
}
// thin high cloud: stays sunlit long after the ground is in shadow, streaked along the wind
fn cirrus(rd: vec3f, base: vec3f) -> vec3f {
  let cover = scn.horSun.w;
  if (rd.y < 0.015) { return base; }
  let p = rd.xz / (rd.y + 0.03) * 7000.0 + vec2f(scn.ambSide.w * 11.0, scn.ambSide.w * 4.0);
  let c = 0.5; let s = 0.866;
  let q = vec2f(p.x * c - p.y * s, p.x * s + p.y * c) * vec2f(0.00011, 0.00046);
  let w = vnoise2(q * 0.7 + 3.0);
  let d = fbm2(q + vec2f(0.0, w * 0.9), 5);
  let dens = smoothstep(0.5, 0.86, d) * smoothstep(0.015, 0.2, rd.y) * (0.25 + 0.75 * cover);
  // light from a sun about 8 degrees "higher" than the one at the ground
  let hi = smoothstep(-9.0, 4.0, degrees(asin(clamp(scn.sunDir.y, -1.0, 1.0))));
  let amb = base * 0.55 + vec3f(0.02, 0.025, 0.04);
  var lit = scn.cloudRad.rgb * 1.6 * mix(0.35, 1.0, hi) + amb;
  let toSun = max(dot(rd, scn.sunDir.xyz), 0.0);
  lit *= 1.0 + 1.6 * pow(toSun, 8.0);
  return mix(base, lit, dens * 0.68);
}
fn starField(rd: vec3f) -> vec3f {
  let n = night();
  if (n < 0.04 || rd.y < 0.0) { return vec3f(0.0); }
  let id = floor(rd * 260.0);
  let h = hash13(id);
  if (h < 0.9955) { return vec3f(0.0); }
  let c = hash33(id + 3.1) * 0.6 + 0.2;
  let d = length(fract(rd * 260.0) - c);
  let b = (h - 0.9955) / 0.0045;
  let tw = 0.75 + 0.25 * sin(scn.ambSide.w * (2.0 + b * 3.0) + h * 400.0);
  return mix(vec3f(0.7, 0.8, 1.0), vec3f(1.0, 0.85, 0.7), hash11(h * 91.0)) * smoothstep(0.35, 0.0, d) * (0.3 + b) * tw * n * smoothstep(0.0, 0.25, rd.y) * 0.9;
}
fn skyFull(rd0: vec3f, withSun: bool) -> vec3f {
  let rd = vec3f(rd0.x, max(rd0.y, 0.0), rd0.z);
  var col = skyBase(rd);
  col += starField(rd);
  let md = dot(rd, normalize(scn.moon.xyz));
  if (scn.moon.w > 0.0 && scn.moon.y > 0.0) {
    col += vec3f(0.85, 0.92, 1.1) * scn.moon.w * 3.5 * smoothstep(0.99989, 0.99996, md) * (0.75 + 0.25 * vnoise3(rd * 90.0));
    col += vec3f(0.5, 0.6, 0.9) * scn.moon.w * pow(max(md, 0.0), 120.0) * 0.12;
  }
  col = cirrus(rd, col);
  var sunV = 1.0;
  let cl = skyClouds(rd, col);
  sunV = 1.0 - cl.a;
  col = mix(col, cl.rgb, cl.a);
  if (withSun) {
    let cs = dot(rd, scn.sunDir.xyz);
    col += scn.sunRad.rgb * 38.0 * smoothstep(0.999962, 0.999985, cs) * sunV * select(0.0, 1.0, rd0.y > -0.02);
  }
  return col;
}
fn skyLite(rd: vec3f) -> vec3f {
  let r = vec3f(rd.x, max(rd.y, 0.0), rd.z);
  var col = skyBase(r);
  col *= 1.0 + 0.1 * smoothstep(0.1, 0.5, r.y) ;
  return col;
}
fn fogColor(rd: vec3f) -> vec3f {
  // horizon sky without the solar aureole (the Mie lobe in fogTerms supplies that), so haze
  // lifts silhouettes toward the sun by a physical amount instead of by the whole glow
  let sh = normalize(scn.sunDir.xz + vec2f(1e-5));
  let rh = normalize(rd.xz + vec2f(1e-5));
  let az = dot(sh, rh) * 0.5 + 0.5;
  let c = mix(scn.hor.rgb, scn.horSun.rgb, pow(az, 3.0) * 0.4);
  return mix(c, vec3f(luma(c)) * vec3f(0.86, 0.96, 1.16), 0.4) * mix(1.0, 0.72, night());
}
fn hgPhase(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (12.566 * pow(max(1.0 + g2 - 2.0 * g * c, 1e-3), 1.5));
}
// mean of exp(-y/H) along a ray going from height y0 to y1
fn meanDens(y0: f32, y1: f32, H: f32) -> f32 {
  let dy = y1 - y0;
  if (abs(dy) < 0.5) { return exp(-y0 / H); }
  return (exp(-y0 / H) - exp(-y1 / H)) * H / dy;
}
// Is a point in the air lit by the sun, or inside the shadow of a tower? Used to march the haze.
fn sunVisAt(pos: vec3f, L: vec3f) -> f32 {
  if (pos.y > 620.0) { return 1.0; }
  let sa = toGrid(pos.xz);
  let lh = normalize(toGrid(L.xz) + vec2f(1e-6));
  let tanE = L.y / max(length(L.xz), 1e-4);
  var v = 1.0;
  var k = 70.0;
  for (var i = 0; i < 4; i++) {
    let h = hmapLoad(sa + lh * k, i + 1);
    v = min(v, smoothstep(-5.0, 14.0, pos.y + k * tanE - h));
    k *= 3.2;
  }
  return v;
}
// Marches the line of sight through the haze: x = sunlit in-scatter weight (tower shadows cut
// light shafts into it), y = the same weight with no shadows (= 1 - transmittance).
fn marchSun(vp: vec3f, rd: vec3f, dist: f32, px: vec2f) -> vec2f {
  let L = scn.sunDir.xyz;
  let beta0 = scn.zen.w;
  let jit = fract(52.9829189 * fract(dot(px, vec2f(0.06711056, 0.00583715))));
  var sum = 0.0;
  var hom = 0.0;
  var tau = 0.0;
  for (var i = 0; i < 14; i++) {
    let u = (f32(i) + jit) / 14.0;
    let t = dist * u * u;
    let dt = dist * 2.0 * u / 14.0;
    var pos = vp + rd * t;
    pos.y = abs(pos.y);
    let b = beta0 * (0.30 + 1.15 * exp(-pos.y / 1800.0) + 0.9 * exp(-pos.y / 32.0));
    tau += b * dt;
    let w = b * exp(-tau) * dt;
    hom += w;
    sum += w * sunVisAt(pos, L);
  }
  return vec2f(sum, hom);
}
// aerial perspective: height-stratified extinction (slightly stronger in blue), in-scattered
// sky light plus a forward-peaked Mie lobe of sunlight (marched, so towers cast shadows and
// shafts through the haze), and a thin low mist layer that softens the water line.
struct FogCore { T: vec3f, rd: vec3f, dist: f32 }
fn fogCore(P: vec3f) -> FogCore {
  let vp = viewerPos();
  let d = P - vp;
  var o: FogCore;
  o.dist = length(d);
  o.rd = d / max(o.dist, 1e-3);
  let y0 = abs(vp.y); let y1 = max(P.y, 0.0);
  let tau = o.dist * scn.zen.w * (0.30 + 1.15 * meanDens(y0, y1, 1800.0) + 0.9 * meanDens(y0, y1, 32.0));
  o.T = exp(-tau * vec3f(0.92, 1.0, 1.14));
  return o;
}
fn sunPhase(rd: vec3f) -> f32 {
  return (hgPhase(dot(rd, scn.sunDir.xyz), 0.74) * 0.16 + 0.01) * smoothstep(-0.12, 0.08, scn.sunDir.y);
}
// cheap homogeneous haze (no shadow march): cables, far lights
fn applyFog(col: vec3f, P: vec3f) -> vec3f {
  let f = fogCore(P);
  return col * f.T + (fogColor(f.rd) + scn.sunRad.rgb * sunPhase(f.rd)) * (1.0 - f.T);
}
// full haze: the sunlit part is marched so towers cast shadows and light shafts through it
fn applyFogPx(col: vec3f, P: vec3f, px: vec2f) -> vec3f {
  let f = fogCore(P);
  var sunW = 1.0 - f.T.y;
  if (scn.sunRad.w > 0.5 && scn.sunDir.y > 0.003 && f.dist > 150.0) {
    let m = marchSun(viewerPos(), f.rd, f.dist, px);
    sunW = m.x * (1.0 - f.T.y) / max(m.y, 1e-5);
  }
  return col * f.T + fogColor(f.rd) * (1.0 - f.T) + scn.sunRad.rgb * sunPhase(f.rd) * sunW;
}

// ---------------------------------------------------------------- sun shadows from the height map
fn hmapLoad(sa: vec2f, lod: i32) -> f32 {
  let sz = vec2i(textureDimensions(hmap, lod));
  let uv = (sa - HM0) / (HCELL * f32(1 << u32(lod)));
  let ip = vec2i(floor(uv));
  if (ip.x < 0 || ip.y < 0 || ip.x >= sz.x || ip.y >= sz.y) { return 0.0; }
  return textureLoad(hmap, ip, lod).r;
}
fn sunShadow(P: vec3f, L: vec3f) -> f32 {
  if (L.y < 0.004) { return 0.0; }
  let sa = toGrid(P.xz);
  let lh = normalize(toGrid(L.xz) + vec2f(1e-6));
  let tanE = L.y / max(length(L.xz), 1e-4);
  var t = 5.0;
  var lit = 1.0;
  for (var i = 0; i < 56; i++) {
    let ry = P.y + t * tanE;
    if (ry > 560.0 || lit < 0.02) { break; }
    let step = max(7.0, t * 0.075);
    let lod = i32(clamp(floor(log2(step / 9.0)), 0.0, 4.0));
    let h = hmapLoad(sa + lh * t, lod);
    let m = (ry - h) / (t * 0.035 + 2.5);
    lit = min(lit, clamp(m * 0.5 + 0.5, 0.0, 1.0));
    t += step;
  }
  return lit;
}
fn localHeight(P: vec3f) -> f32 { return hmapLoad(toGrid(P.xz), 4); }
fn ambient(n: vec3f) -> vec3f {
  let h = n.y;
  // with the sun near the horizon, faces looking toward it catch the glowing sky and faces
  // looking away only see the dim anti-solar sky
  let sh = normalize(scn.sunDir.xz + vec2f(1e-5));
  let nh = normalize(n.xz + vec2f(1e-5));
  let tw = clamp(dot(sh, nh) * 0.5 + 0.5, 0.0, 1.0);
  let low = 1.0 - smoothstep(0.0, 0.6, scn.sunDir.y);
  let dirF = mix(vec3f(1.0), mix(vec3f(0.5, 0.55, 0.72), vec3f(1.7, 1.4, 1.05), tw), low * 0.85);
  let side = scn.ambSide.rgb * dirF;
  return select(mix(side, scn.ambDown.rgb, -h), mix(side, scn.ambUp.rgb, h), h >= 0.0);
}
`;


// ===================================================================================
// Buildings
// ===================================================================================
export const BUILDINGS = COMMON + /* wgsl */ `
// glass seen in glass: other towers in the mirror direction (dark / sunlit faces looking back at
// us), and the street when the reflection ray points down.
fn nbReflect(P: vec3f, R: vec3f, sky: vec3f, L: vec3f) -> vec3f {
  let rh = normalize(R.xz + vec2f(1e-4));
  let slope = R.y / max(length(R.xz), 1e-3);
  var occ = 0.0;
  var dd = 400.0;
  for (var i = 0; i < 3; i++) {
    let k = 55.0 + 150.0 * f32(i * i + i) * 0.5;
    let h = hmapLoad(toGrid(P.xz + rh * k), 1);
    let o = smoothstep(-3.0, 8.0, h - (P.y + slope * k));
    if (o > occ) { dd = select(dd, k, occ < 0.02); occ = o; }
  }
  let face = vec3f(-rh.x, 0.0, -rh.y);
  let lit = max(dot(face, L), 0.0) * select(0.0, 1.0, L.y > 0.0);
  let nbC = vec3f(0.17, 0.165, 0.16) * (scn.sunRad.rgb * lit * 0.55 + ambient(face) * 0.9);
  let T = exp(-dd * scn.zen.w * 1.5);
  var col = mix(sky, mix(fogColor(R), nbC, T), occ);
  let gr = smoothstep(0.02, -0.28, R.y);
  let nt = night();
  let street = (scn.ambUp.rgb * 0.5 + scn.sunRad.rgb * max(L.y, 0.0) * 0.18) * 0.1 + vec3f(1.0, 0.55, 0.22) * 0.09 * nt * clamp(scn.hor.w, 0.0, 1.4);
  col = mix(col, street, gr);
  return col;
}
struct BIn {
  @builtin(vertex_index) vi: u32,
  @location(0) a: vec4f,   // x z y0 h
  @location(1) b: vec4f,   // hw hd seed mat
  @location(2) c: vec4f,   // taper rot wash style
}
struct BOut {
  @builtin(position) pos: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) uvf: vec3f,
  @location(3) @interpolate(flat) inst: vec4f,   // seed mat style wash
  @location(4) @interpolate(flat) geo: vec4f,    // hw hd y0 h
}
@vertex fn vs(in: BIn) -> BOut {
  let f = in.vi / 6u;
  let k = in.vi % 6u;
  let qk = select(select(0u, k - 2u, k > 3u), k, k < 3u);
  var ring: u32;
  var top: bool;
  if (f < 4u) {
    ring = (f + select(0u, 1u, qk == 1u || qk == 2u)) % 4u;
    top = qk >= 2u;
  } else {
    ring = qk;
    top = true;
  }
  let hw = in.b.x; let hd = in.b.y;
  let taper = in.c.x;
  let sc = select(1.0, taper, top);
  let rx = select(-1.0, 1.0, ring == 1u || ring == 2u);
  let rz = select(-1.0, 1.0, ring >= 2u);
  let lx = rx * hw * sc;
  let lz = rz * hd * sc;
  let y = in.a.z + select(0.0, in.a.w, top);
  let cr = cos(in.c.y); let sr = sin(in.c.y);
  let wp = vec3f(in.a.x + lx * cr - lz * sr, y, in.a.y + lx * sr + lz * cr);
  // normal
  var nl = vec3f(0.0, 1.0, 0.0);
  var wid = 0.0;
  if (f == 0u) { nl = vec3f(0.0, 0.0, -1.0); wid = 2.0 * hw; }
  if (f == 1u) { nl = vec3f(1.0, 0.0, 0.0); wid = 2.0 * hd; }
  if (f == 2u) { nl = vec3f(0.0, 0.0, 1.0); wid = 2.0 * hw; }
  if (f == 3u) { nl = vec3f(-1.0, 0.0, 0.0); wid = 2.0 * hd; }
  if (f < 4u) {
    let ex = select(hd, hw, f == 1u || f == 3u);
    nl.y = ex * (1.0 - taper) / max(in.a.w, 0.1);
    nl = normalize(nl);
  }
  let n = vec3f(nl.x * cr - nl.z * sr, nl.y, nl.x * sr + nl.z * cr);
  var uv = vec3f(0.0);
  if (f < 4u) {
    uv = vec3f(select(0.0, wid, qk == 1u || qk == 2u) + hash11(in.b.z + f2(f)) * 40.0, y, f32(f));
  } else {
    uv = vec3f(lx + in.a.x * 0.0, lz, 4.0);
  }
  let flip = scn.ambDown.w > 0.5;
  let wy = select(wp.y, -wp.y, flip);
  var o: BOut;
  o.pos = cam.viewProj * vec4f(wp.x - cam.pos.x, wy - cam.pos.y, wp.z - cam.pos.z, 1.0);
  o.wp = wp;
  o.n = n;
  o.uvf = uv;
  o.inst = vec4f(in.b.z, in.b.w, in.c.w, in.c.z);
  o.geo = vec4f(hw, hd, in.a.z, in.a.w);
  return o;
}
fn f2(x: u32) -> f32 { return f32(x) * 13.7; }

fn matAlbedo(m: i32) -> vec3f {
  var t = array<vec3f, 16>(
    vec3f(0.17, 0.105, 0.088), vec3f(0.30, 0.285, 0.255), vec3f(0.04, 0.055, 0.07), vec3f(0.04, 0.038, 0.038),
    vec3f(0.42, 0.44, 0.47), vec3f(0.33, 0.33, 0.325), vec3f(0.16, 0.16, 0.17), vec3f(0.14, 0.09, 0.06),
    vec3f(0.20, 0.195, 0.19), vec3f(0.12, 0.28, 0.24), vec3f(0.6, 0.6, 0.62), vec3f(0.04, 0.04, 0.045),
    vec3f(0.06, 0.06, 0.065), vec3f(0.55, 0.20, 0.05), vec3f(0.26, 0.235, 0.19), vec3f(0.02, 0.02, 0.025));
  return t[m];
}
fn glassTint(m: i32) -> vec3f {
  if (m == 2) { return vec3f(0.55, 0.72, 0.86); }
  if (m == 3) { return vec3f(0.72, 0.64, 0.56); }
  if (m == 4) { return vec3f(0.95, 0.97, 1.0); }
  if (m == 15) { return vec3f(0.5, 0.58, 0.7); }
  if (m == 9) { return vec3f(0.55, 0.9, 0.75); }
  return vec3f(0.7, 0.8, 0.9);
}

fn coverage(x: f32, lo: f32, hi: f32, fp: f32) -> f32 {
  let e = min(x - lo, hi - x);
  return clamp(e / max(fp, 1e-4) + 0.5, 0.0, 1.0);
}

fn shadeBuilding(in: BOut) -> vec4f {
  let P = in.wp;
  let N0 = normalize(in.n);
  let vp = viewerPos();
  let V = normalize(vp - P);
  var N = N0;
  if (dot(N, V) < 0.0 && N.y < 0.5) { N = -N; }
  let seed = in.inst.x;
  let mat = i32(in.inst.y + 0.5);
  let sty = i32(in.inst.z + 0.5);
  let wash = in.inst.w;
  let nt = night();
  let L = scn.sunDir.xyz;
  let isSide = N.y < 0.6;
  var albedo = matAlbedo(mat);
  // per building tint
  let th = hash11(seed * 1.7);
  albedo *= vec3f(0.84 + 0.32 * th) * vec3f(1.0, 0.97 + 0.06 * hash11(seed * 3.1), 0.95 + 0.1 * hash11(seed * 5.9));
  albedo = mix(vec3f(luma(albedo)), albedo, 0.8);

  // derivatives first (uniform control flow)
  let uvfx = in.uvf;
  var cellW = 3.1; var cellH = 3.9;
  var lo = vec2f(0.3, 0.2); var hi = vec2f(0.7, 0.78);
  if (sty == 2) { cellW = 2.4; cellH = 3.7; lo = vec2f(0.03, 0.2); hi = vec2f(0.97, 0.84); }
  if (sty == 3) { cellW = 3.05; cellH = 3.05; lo = vec2f(0.1, 0.1); hi = vec2f(0.9, 0.9); }
  if (sty == 4) { cellW = 1.5; cellH = 4.0; lo = vec2f(0.04, 0.14); hi = vec2f(0.96, 0.9); }
  cellW *= 0.85 + 0.3 * hash11(seed * 2.3);
  cellH *= 0.9 + 0.2 * hash11(seed * 4.1);
  let cuv = vec2f(uvfx.x / cellW, uvfx.y / cellH);
  let fw = fwidth(cuv);
  let fp = max(fw.x, fw.y);

  var openC = 0.0;
  var emis = vec3f(0.0);
  var glass = vec3f(0.0);
  var wall = albedo;
  var sillMul = 1.0;
  if (isSide && sty != 0 && sty != 5) {
    let cid = floor(cuv);
    let fr = fract(cuv);
    var cx = coverage(fr.x, lo.x, hi.x, fw.x);
    var cy = coverage(fr.y, lo.y, hi.y, fw.y);
    if (sty == 4 && (i32(cid.x) & 3) == 0) { cx = cx * 0.55; }
    openC = cx * cy;
    if (i32(floor(cid.y)) % 23 == 0 && sty != 3) { openC = 0.0; }
    let avg = (hi.x - lo.x) * (hi.y - lo.y);
    openC = mix(openC, avg, smoothstep(0.4, 1.0, fp));
    // per-cell lit state
    let fl = hash12(vec2f(cid.y, seed * 3.7 + uvfx.z));
    let ch = hash12(cid + vec2f(seed * 5.3 + uvfx.z * 17.0, 1.7));
    let fl2 = hash12(vec2f(floor(cid.y / 5.0), seed * 9.1));
    var litP = scn.hor.w * 0.7 * mix(0.1, 0.85, fl2) * mix(0.35, 1.25, fl);
    litP = clamp(litP, 0.0, 1.0);
    let isLit = step(ch, litP);
    let warm = mix(vec3f(1.0, 0.72, 0.46), vec3f(0.80, 0.90, 1.0), step(0.68, hash12(cid * 1.9 + seed)));
    let inten = 0.55 + 1.3 * hash12(cid + 91.7);
    var lc = warm * inten * 1.9 * isLit;
    // far away the windows average out
    let band = hash12(vec2f(floor(uvfx.y / 7.0), seed * 1.3 + uvfx.z * 5.1));
    let zoneN = vnoise2(vec2f(uvfx.y / 55.0 + seed * 3.0, uvfx.z * 2.0 + uvfx.x / 70.0));
    let fg = 0.6 * hash12(vec2f(floor(uvfx.y / (cellH * 3.0)), seed * 9.1 + uvfx.z)) + 0.4 * hash12(vec2f(floor(uvfx.y / cellH), seed * 2.9 + uvfx.z * 3.0));
    let avgLit = scn.hor.w * 0.076 * vec3f(1.0, 0.72, 0.42) * mix(0.25, 1.25, hash11(seed * 4.3)) * (0.05 + 1.7 * fg * fg * (0.4 + 1.2 * zoneN)) * mix(0.7, 1.3, band);
    lc = mix(lc, avgLit, smoothstep(0.45, 1.0, fp));
    // street-level storefronts: wide shop glass, warm interiors at night
    let sf = (1.0 - smoothstep(4.6, 5.4, P.y)) * select(1.0, 0.0, sty == 4);
    if (sf > 0.0) {
      let sx = uvfx.x / 5.5;
      let scx = coverage(fract(sx), 0.07, 0.93, fwidth(sx));
      let scy = smoothstep(0.3, 0.9, P.y) * (1.0 - smoothstep(3.6, 4.2, P.y));
      openC = mix(openC, scx * scy, sf);
      let shopLit = step(hash12(vec2f(floor(sx), seed * 7.1 + uvfx.z)), clamp(scn.hor.w * 0.85, 0.0, 1.0));
      lc = mix(lc, vec3f(1.0, 0.76, 0.48) * (1.2 + 1.6 * hash12(vec2f(floor(sx) * 1.7, seed))) * shopLit, sf * (1.0 - smoothstep(0.5, 1.0, fp)));
    }
    emis = lc * openC;
    // glass reflectance
    let Rv = reflect(-V, N);
    let F0 = select(0.07, 0.24, sty == 4 || sty == 2 || sty == 3);
    var F = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    if (mat == 4) { F = 0.55; }
    var sk = skyLite(Rv);
    let sunSpec = pow(max(dot(Rv, L), 0.0), 500.0) * 7.0;
    let sh0 = select(1.0, sunShadow(P + N * 1.5, L), L.y > 0.0 && dot(N, L) > 0.0);
    sk += scn.sunRad.rgb * sunSpec * sh0;
    sk = nbReflect(P, Rv, sk, L);
    let deep = vec3f(0.012, 0.02, 0.028) * (0.6 + th);
    glass = mix(deep, sk * glassTint(mat), F);
    // window reveal: shadow under the lintel, darker jamb, lighter sill below the opening
    let wy = (fr.y - lo.y) / max(hi.y - lo.y, 1e-3);
    let wx = (fr.x - lo.x) / max(hi.x - lo.x, 1e-3);
    let nearK = 1.0 - smoothstep(0.25, 0.8, fp);
    if (sty == 1 || sty == 3) {
      glass *= 1.0 - nearK * (0.5 * smoothstep(0.8, 1.0, wy) + 0.28 * smoothstep(0.14, 0.0, wx));
      sillMul = 1.0 + nearK * 0.45 * smoothstep(lo.y - 0.09, lo.y - 0.03, fr.y) * (1.0 - smoothstep(lo.y - 0.03, lo.y, fr.y));
    }
    // bay-scale patchwork (panel batches, tinted glass runs, repaired masonry) that survives
    // when single windows are too small to see
    let bayVar = 0.8 + 0.4 * hash12(vec2f(floor(uvfx.x / (cellW * 3.0)) + seed * 1.3, floor(uvfx.y / (cellH * 2.5)) + uvfx.z * 7.0));
    let bayVar2 = 0.9 + 0.2 * vnoise2(vec2f(uvfx.x * 0.11 + seed * 9.0, uvfx.y * 0.05));
    glass *= mix(1.0, bayVar, 0.8) * bayVar2;
    let blind = hash12(cid * 1.37 + seed);
    glass *= 0.62 + 0.76 * hash12(cid + vec2f(seed, 3.3));
    if (blind > 0.9) { glass = mix(glass, albedo * 0.9 + vec3f(0.02), 0.7); }
    if (sty == 1) { glass *= 1.0 - 0.55 * smoothstep(0.5, 0.78, fr.y); }
    glass *= 1.0 - 0.18 * smoothstep(0.6, 0.0, fr.x) * select(0.0, 1.0, sty != 4);
    let gT = glassTint(mat);
    glass += gT * (scn.sunRad.rgb * max(dot(N, L), 0.0) * sh0 * 0.16 + ambient(N) * 0.22 * mix(1.0, 0.2, nt)) * select(0.5, 1.0, sty == 4 || sty == 2 || sty == 3);
    // wall between windows
    if (sty == 4) { wall = mix(albedo * 0.5 + vec3f(0.02), glass * 0.8, 0.55); }
    else {
      wall *= sillMul * mix(1.0, 0.8 + 0.4 * hash12(vec2f(floor(uvfx.x / (cellW * 3.0)) + seed * 1.3, floor(uvfx.y / (cellH * 2.5)) + uvfx.z * 7.0)), 0.75);
      let grime = vnoise2(vec2f(uvfx.x * 0.9 + seed * 5.0, uvfx.y * 0.035)) * 0.6 + vnoise2(vec2f(uvfx.x * 3.1, uvfx.y * 0.3 + seed)) * 0.4;
      wall *= (0.78 + 0.4 * grime) * mix(0.62, 1.0, smoothstep(0.0, 22.0, P.y));
      // cornice band every few floors on masonry
      let fl4 = fract(uvfx.y / (cellH * 6.0));
      wall *= 1.0 - 0.14 * smoothstep(0.93, 0.985, fl4);
    }
  }

  // bridge towers: pointed arches cut through the faces that look along the roadway
  if (sty == 5 && isSide) {
    let fid = u32(in.uvf.z + 0.5);
    if (fid == 1u || fid == 3u) {
      let off = hash11(seed + f2(fid)) * 40.0;
      let uc = uvfx.x - off - in.geo.y;
      let t = clamp((uvfx.y - 36.0) / 37.0, 0.0, 1.0);
      let half = 4.4 * (1.0 - pow(t, 1.9));
      if (uvfx.y > 36.0 && uvfx.y < 73.0 && min(abs(uc - 11.0), abs(uc + 11.0)) < half) { discard; }
    }
    albedo *= 0.86 + 0.26 * hash12(vec2f(floor(uvfx.x / 1.4), floor(uvfx.y / 0.8)));
    wall = albedo;
  }

  let ao = 1.0;
  var lightRes = vec3f(0.0);
  // direct sun
  let ndl = max(dot(N, L), 0.0);
  var shadow = 0.0;
  if (ndl > 0.0 && L.y > 0.0) { shadow = sunShadow(P + N * 1.5 + vec3f(0.0, 0.5, 0.0), L); }
  let hLoc = localHeight(P);
  let dense = clamp(hLoc / 140.0, 0.0, 1.0);
  let canyon = mix(1.0 - 0.5 * dense, 1.0, smoothstep(0.0, max(hLoc, 20.0) * 0.8, P.y));
  let Nm = scn.moon.xyz;
  let moonL = max(dot(N, normalize(Nm)), 0.0) * scn.moon.w;
  lightRes = scn.sunRad.rgb * ndl * shadow + ambient(N) * canyon * mix(1.0, 0.1, clamp(nt * 1.7, 0.0, 1.0)) + vec3f(0.42, 0.5, 0.75) * moonL * 0.12;
  let rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
  lightRes += scn.sunRad.rgb * ndl * shadow * rim * 0.35;
  var col: vec3f;
  if (isSide && sty != 0 && sty != 5) {
    let wallLit = wall * lightRes;
    col = mix(wallLit, glass, openC) + emis;
  } else {
    var a = albedo;
    if (!isSide) { a = mix(vec3f(0.09, 0.09, 0.095), albedo * 0.35, 0.3) * (0.7 + 0.6 * vnoise2(P.xz * 0.7)); }
    col = a * lightRes;
    if (isSide && mat == 4) { col += skyLite(reflect(-V, N)) * 0.4; }
    if (isSide && mat == 9) { col += skyLite(reflect(-V, N)) * 0.12; }
  }
  if (isSide && uvfx.z < 3.5) {
    let fid = u32(uvfx.z + 0.5);
    let off = hash11(seed + f2(fid)) * 40.0;
    let wid = select(2.0 * in.geo.x, 2.0 * in.geo.y, (fid & 1u) == 1u);
    let along = uvfx.x - off;
    col *= mix(0.80, 1.0, smoothstep(0.0, 2.4, min(along, wid - along)));
  }
  // landmark up-lighting at night
  if (wash > 0.0) {
    let tint = select(select(vec3f(1.0, 0.82, 0.55), vec3f(0.75, 0.9, 1.0), mat == 4), vec3f(0.55, 1.0, 0.8), mat == 9);
    let g = 0.55 + 0.45 * sin(uvfx.y * 0.35 + scn.ambSide.w * 0.4);
    col += tint * wash * clamp(scn.hor.w, 0.0, 1.3) * select(0.6, 1.0, isSide) * 0.8 * (0.6 + 0.4 * g) * smoothstep(40.0, 200.0, in.geo.z + in.geo.w);
  }
  // street-glow bounce on the lowest floors at night
  col += vec3f(1.0, 0.55, 0.22) * 0.035 * nt * smoothstep(60.0, 0.0, P.y) * select(0.3, 1.0, isSide);
  col = applyFogPx(col, P, in.pos.xy);
  return vec4f(col, 1.0);
}
@fragment fn fs(in: BOut) -> @location(0) vec4f { return shadeBuilding(in); }
`;
