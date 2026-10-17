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
  let c = skyBase(vec3f(rd.x, clamp(rd.y, 0.0, 0.12) * 0.5, rd.z));
  return mix(c, vec3f(luma(c)) * vec3f(0.86, 0.96, 1.16), 0.45) * mix(1.0, 0.72, night());
}
fn applyFog(col: vec3f, P: vec3f) -> vec3f {
  let vp = viewerPos();
  let d = P - vp;
  let dist = length(d);
  let rd = d / max(dist, 1e-3);
  let ym = max((P.y + vp.y) * 0.5, 0.0);
  let beta = scn.zen.w * (0.35 + 1.2 * exp(-ym / 1800.0));
  let T = exp(-dist * beta);
  return mix(fogColor(rd), col, T);
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
  return select(mix(scn.ambSide.rgb, scn.ambDown.rgb, -h), mix(scn.ambSide.rgb, scn.ambUp.rgb, h), h >= 0.0);
}
`;

// ===================================================================================
// Buildings
// ===================================================================================
export const BUILDINGS = COMMON + /* wgsl */ `
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
    vec3f(0.21, 0.115, 0.09), vec3f(0.36, 0.33, 0.28), vec3f(0.04, 0.06, 0.08), vec3f(0.045, 0.04, 0.04),
    vec3f(0.45, 0.47, 0.50), vec3f(0.40, 0.40, 0.39), vec3f(0.17, 0.17, 0.18), vec3f(0.14, 0.09, 0.06),
    vec3f(0.22, 0.21, 0.20), vec3f(0.12, 0.28, 0.24), vec3f(0.6, 0.6, 0.62), vec3f(0.04, 0.04, 0.045),
    vec3f(0.06, 0.06, 0.065), vec3f(0.55, 0.20, 0.05), vec3f(0.32, 0.27, 0.20), vec3f(0.02, 0.02, 0.025));
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
    let avgLit = scn.hor.w * 0.2 * 3.4 * vec3f(1.0, 0.72, 0.42) * (0.1 + 1.9 * zoneN * zoneN) * mix(0.6, 1.4, band);
    lc = mix(lc, avgLit, smoothstep(0.45, 1.0, fp));
    emis = lc * openC;
    // glass reflectance
    let Rv = reflect(-V, N);
    let F0 = select(0.07, 0.38, sty == 4 || sty == 2 || sty == 3);
    var F = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    if (mat == 4) { F = 0.55; }
    var sk = skyLite(Rv);
    let sunSpec = pow(max(dot(Rv, L), 0.0), 500.0) * 7.0;
    let sh0 = select(1.0, sunShadow(P + N * 1.5, L), L.y > 0.0 && dot(N, L) > 0.0);
    sk += scn.sunRad.rgb * sunSpec * sh0;
    let deep = vec3f(0.012, 0.02, 0.028) * (0.6 + th);
    glass = mix(deep, sk * glassTint(mat), F);
    let blind = hash12(cid * 1.37 + seed);
    glass *= 0.62 + 0.76 * hash12(cid + vec2f(seed, 3.3));
    if (blind > 0.9) { glass = mix(glass, albedo * 0.9 + vec3f(0.02), 0.7); }
    if (sty == 1) { glass *= 1.0 - 0.55 * smoothstep(0.5, 0.78, fr.y); }
    glass *= 1.0 - 0.18 * smoothstep(0.6, 0.0, fr.x) * select(0.0, 1.0, sty != 4);
    let gT = glassTint(mat);
    glass += gT * (scn.sunRad.rgb * max(dot(N, L), 0.0) * sh0 * 0.16 + ambient(N) * 0.22 * mix(1.0, 0.2, nt)) * select(0.5, 1.0, sty == 4 || sty == 2 || sty == 3);
    // wall between windows
    if (sty == 4) { wall = mix(albedo * 0.5 + vec3f(0.02), glass * 0.8, 0.55); }
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
  // landmark up-lighting at night
  if (wash > 0.0) {
    let tint = select(select(vec3f(1.0, 0.82, 0.55), vec3f(0.75, 0.9, 1.0), mat == 4), vec3f(0.55, 1.0, 0.8), mat == 9);
    let g = 0.55 + 0.45 * sin(uvfx.y * 0.35 + scn.ambSide.w * 0.4);
    col += tint * wash * clamp(scn.hor.w, 0.0, 1.3) * select(0.6, 1.0, isSide) * 1.1 * (0.6 + 0.4 * g) * smoothstep(40.0, 200.0, in.geo.z + in.geo.w);
  }
  // street-glow bounce on the lowest floors at night
  col += vec3f(1.0, 0.55, 0.22) * 0.035 * nt * smoothstep(60.0, 0.0, P.y) * select(0.3, 1.0, isSide);
  col = applyFog(col, P);
  return vec4f(col, 1.0);
}
@fragment fn fs(in: BOut) -> @location(0) vec4f { return shadeBuilding(in); }
`;
