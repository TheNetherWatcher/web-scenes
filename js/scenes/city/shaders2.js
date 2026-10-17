// Beams (cables, masts), light sprites, ground + river, sky pass.
import { COMMON } from './shaders.js';
import { LAND } from './gen.js';

const F = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

// ===================================================================================
// Beams: camera-facing ribbons between two points (cables, suspenders, antennas)
// ===================================================================================
export const BEAMS = COMMON + /* wgsl */ `
struct BmIn { @builtin(vertex_index) vi: u32, @location(0) a: vec4f, @location(1) b: vec4f }
struct BmOut {
  @builtin(position) pos: vec4f,
  @location(0) wp: vec3f,
  @location(1) @interpolate(flat) info: vec4f,   // kind fade len -
  @location(2) t: f32,
}
@vertex fn vs(in: BmIn) -> BmOut {
  let k = in.vi;
  let qk = select(select(0u, k - 2u, k > 3u), k, k < 3u);
  let t = select(0.0, 1.0, qk >= 2u);
  let sg = select(-1.0, 1.0, qk == 1u || qk == 2u);
  var p0 = in.a.xyz; var p1 = in.b.xyz;
  let real = mix(p0, p1, t);
  if (scn.ambDown.w > 0.5) { p0.y = -p0.y; p1.y = -p1.y; }
  let q0 = p0 - cam.pos.xyz; let q1 = p1 - cam.pos.xyz;
  let dirv = q1 - q0; let mid = (q0 + q1) * 0.5;
  let side = normalize(cross(dirv, mid) + vec3f(1e-6));
  let pw = max(length(mid), 1.0) * 2.0 * length(cam.up.xyz) / cam.res.y;
  let w = max(in.a.w, pw * 1.1);
  let fade = clamp(in.a.w / (pw * 1.1), 0.0, 1.0);
  let pos = mix(q0, q1, t) + side * sg * w * 0.5;
  var o: BmOut;
  o.pos = cam.viewProj * vec4f(pos, 1.0);
  o.wp = real;
  o.info = vec4f(in.b.w, fade, length(in.b.xyz - in.a.xyz), 0.0);
  o.t = t;
  return o;
}
@fragment fn fs(in: BmOut) -> @location(0) vec4f {
  let kind = i32(in.info.x + 0.5);
  var alb = vec3f(0.12);
  var emis = vec3f(0.0);
  if (kind == 1) {
    let band = floor(in.t * in.info.z / 3.5);
    alb = select(vec3f(0.75), vec3f(0.8, 0.07, 0.04), (i32(band) & 1) == 0);
  }
  if (kind == 2) { alb = vec3f(0.18, 0.38, 0.3); }
  if (kind == 3) { alb = vec3f(0.2, 0.17, 0.14); emis = vec3f(1.0, 0.6, 0.25) * 0.28 * night(); }
  let L = scn.sunDir.xyz;
  var col = alb * (scn.ambUp.rgb * 0.8 + scn.sunRad.rgb * max(L.y, 0.0) * 0.55) + emis;
  col = mix(fogColor(normalize(in.wp - viewerPos())), col, in.info.y);
  col = applyFog(col, in.wp);
  return vec4f(col, 1.0);
}
`;

// ===================================================================================
// Light sprites (additive billboards)
// ===================================================================================
export const SPRITES = COMMON + /* wgsl */ `
struct SIn { @builtin(vertex_index) vi: u32, @location(0) a: vec4f, @location(1) b: vec4f }
struct SOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) col: vec3f,
  @location(2) T: f32,
}
@vertex fn vs(in: SIn) -> SOut {
  let k = in.vi;
  let qk = select(select(0u, k - 2u, k > 3u), k, k < 3u);
  let cx = select(-1.0, 1.0, qk == 1u || qk == 2u);
  let cy = select(-1.0, 1.0, qk >= 2u);
  var p = in.a.xyz;
  let real = p;
  if (scn.ambDown.w > 0.5) { p.y = -p.y; }
  let q = p - cam.pos.xyz;
  let dm = max(length(q), 1.0);
  let pw = dm * 2.0 * length(cam.up.xyz) / cam.res.y;
  let size = in.a.w;
  let sz = max(size, pw * 2.4);
  let inten = clamp(size / sz, 0.0, 1.0);
  let right = normalize(cam.right.xyz); let up = normalize(cam.up.xyz);
  let off = (right * cx + up * cy) * sz;
  var o: SOut;
  o.pos = cam.viewProj * vec4f(q + off, 1.0);
  o.uv = vec2f(cx, cy);
  let rate = in.b.w;
  let ph = hash13(real * 0.37);
  let t = scn.ambSide.w;
  var blink = 1.0;
  if (rate > 0.0) { blink = pow(0.5 + 0.5 * sin(t * 6.2832 * rate + ph * 6.2832), 2.0); }
  if (rate < 0.0) { blink = select(0.0, 1.0, fract(t * -rate + ph) < 0.07); }
  let nt = night();
  let dayAmt = select(0.03, 0.5, rate != 0.0);
  o.col = in.b.rgb * blink * mix(dayAmt, 1.0, nt) * (0.25 + 0.75 * inten);
  let vp = viewerPos();
  let dist = length(real - vp);
  o.T = exp(-dist * scn.zen.w * 0.8);
  return o;
}
@fragment fn fs(in: SOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  let g = exp(-r2 * 5.0) + 0.18 * exp(-r2 * 1.4);
  let m = g * smoothstep(1.0, 0.75, sqrt(r2));
  return vec4f(in.col * m * in.T, 1.0);
}
`;

// ===================================================================================
// Ground (streets, parks) + river
// ===================================================================================
export const GROUND = COMMON + /* wgsl */ `
@group(0) @binding(3) var landTex: texture_2d<f32>;
@group(0) @binding(4) var landSamp: sampler;
@group(0) @binding(5) var reflTex: texture_2d<f32>;
@group(0) @binding(6) var reflSamp: sampler;

const LS0: f32 = ${F(LAND.s0)};
const LA0: f32 = ${F(LAND.a0)};
const LSPAN: vec2f = vec2f(${F(LAND.w * LAND.cell)}, ${F(LAND.h * LAND.cell)});

struct GOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> GOut {
  var p = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let rel = vec3f(p[i].x * 90000.0, -cam.pos.y, p[i].y * 90000.0);
  var o: GOut;
  o.pos = cam.viewProj * vec4f(rel, 1.0);
  o.wp = vec3f(rel.x + cam.pos.x, 0.0, rel.z + cam.pos.z);
  return o;
}

fn landDepth(sa: vec2f) -> f32 {
  let uv = (sa - vec2f(LS0, LA0)) / LSPAN;
  return (textureSampleLevel(landTex, landSamp, uv, 0.0).r - 0.5) * 320.0;
}

fn carDots(lat: f32, along: f32, key: f32, t: f32, pf: f32) -> vec3f {
  let lane = floor((lat + 13.0) / 3.2);
  let fwdDir = lat > 0.0;
  let sp = 5.0 + 5.0 * hash11(key * 7.3 + lane);
  let x = along * select(-1.0, 1.0, fwdDir) + t * sp + hash11(lane + key * 3.7) * 200.0;
  let cell = floor(x / 13.0);
  let fx = (x - cell * 13.0) - 6.5;
  let on = step(0.5, hash11(cell * 1.31 + key * 17.0 + lane * 5.0));
  let latC = (lat + 13.0) - (lane + 0.5) * 3.2;
  let d = exp(-(fx * fx * 0.5 + latC * latC * 1.2));
  let c = select(vec3f(1.0, 0.06, 0.03), vec3f(1.0, 0.88, 0.65), fwdDir);
  let detail = c * d * on * 2.2;
  return mix(detail, vec3f(0.55, 0.38, 0.25) * 0.12, smoothstep(1.5, 5.0, pf));
}

fn landColor(P: vec3f, sa: vec2f, pf: f32) -> vec3f {
  let ps = sa.x - floor(sa.x / 275.0) * 275.0;
  let pa = sa.y - floor(sa.y / 80.0) * 80.0;
  let dAve = min(ps, 275.0 - ps);
  let dSt = min(pa, 80.0 - pa);
  let ave = 1.0 - smoothstep(14.0 - pf * 0.6, 14.0 + pf * 0.6, dAve);
  let st = 1.0 - smoothstep(9.0 - pf * 0.6, 9.0 + pf * 0.6, dSt);
  var street = max(ave, st);
  street = mix(street, 0.31, smoothstep(10.0, 45.0, pf));
  let blk = vec3f(0.16, 0.155, 0.145) * (0.75 + 0.5 * vnoise2(sa * 0.07));
  let asph = vec3f(0.048, 0.048, 0.052);
  // Central Park
  let pk = smoothstep(0.0, 18.0, min(min(sa.x + 480.0, 330.0 - sa.x), min(sa.y - 8330.0, 12400.0 - sa.y)));
  let canopy = vec3f(0.032, 0.058, 0.022) * (0.45 + 1.1 * fbm2(sa * 0.025, 4)) * (0.8 + 0.4 * vnoise2(sa * 0.6));
  var alb = mix(blk, asph, street);
  alb = mix(alb, canopy, pk);
  let L = scn.sunDir.xyz;
  let sh = select(0.0, sunShadow(P + vec3f(0.0, 0.6, 0.0), L), L.y > 0.0);
  let hLoc = localHeight(P);
  let canyon = 1.0 - 0.55 * clamp(hLoc / 120.0, 0.0, 1.0);
  let moonL = max(normalize(scn.moon.xyz).y, 0.0) * scn.moon.w;
  var col = alb * (scn.sunRad.rgb * max(L.y, 0.0) * sh + scn.ambUp.rgb * canyon + vec3f(0.42, 0.5, 0.75) * moonL * 0.3);
  // night street lights + traffic
  let nt = night();
  if (nt > 0.02) {
    let key = floor(sa.x / 275.0) * 31.0 + floor(sa.y / 80.0);
    let blkVar = 0.6 + 0.8 * hash11(key);
    let sodium = vec3f(1.0, 0.52, 0.2) * street * (1.0 - pk) * 0.5 * blkVar;
    var cars = vec3f(0.0);
    if (pf < 6.0) {
      let latA = ps - select(0.0, 275.0, ps > 137.5);
      let latS = pa - select(0.0, 80.0, pa > 40.0);
      cars += carDots(latA, sa.y, floor(sa.x / 275.0 + 0.5), scn.ambSide.w, pf) * ave;
      cars += carDots(latS, sa.x, floor(sa.y / 80.0 + 0.5) + 100.0, scn.ambSide.w, pf) * st * (1.0 - ave);
    } else {
      cars = vec3f(0.55, 0.38, 0.25) * 0.12 * street;
    }
    col += (sodium + cars) * nt * clamp(scn.hor.w, 0.0, 1.5);
  }
  return col;
}

fn waveSlope(p: vec2f, t: f32, pf: f32, rough: f32) -> vec2f {
  var g = vec2f(0.0);
  for (var i = 0; i < 6; i++) {
    let fi = f32(i);
    let ang = 0.55 + fi * 1.07;
    let d = vec2f(cos(ang), sin(ang));
    let lam = 3.2 * pow(2.0, fi);
    let k = 6.2832 / lam;
    let w = sqrt(9.81 * k);
    let sl = rough * 0.06 * pow(lam / 10.0, -0.25);
    let att = 1.0 - smoothstep(lam * 0.08, lam * 0.3, pf);
    let wob = 5.0 * (vnoise2(p * 0.021 + fi * 7.1) - 0.5) + 2.5 * (vnoise2(p * 0.07 + fi * 3.3) - 0.5);
    let gust = 0.55 + 0.9 * vnoise2(p * 0.004 + fi * 1.7 + t * 0.02);
    g += d * sl * att * gust * cos(k * dot(d, p) - w * t + fi * 2.1 + wob);
  }
  // small wind ripples
  let att2 = 1.0 - smoothstep(0.12, 0.6, pf);
  if (att2 > 0.0) {
    let e = 0.15;
    let q = p * 1.3 + vec2f(t * 0.5, t * 0.3);
    let n0 = vnoise2(q);
    g += rough * 0.5 * att2 * vec2f(vnoise2(q + vec2f(e, 0.0)) - n0, vnoise2(q + vec2f(0.0, e)) - n0) / e * 0.18;
  }
  return g;
}

fn wakeAt(p: vec2f, t: f32) -> vec3f {
  // returns (foam, slopeX, slopeZ)
  var foam = 0.0;
  var sl = vec2f(0.0);
  for (var j = 0; j < 8; j++) {
    let B = scn.boats[j];
    let len = scn.boatP[j].x;
    if (len <= 0.0) { continue; }
    let rel = p - B.xy;
    let dir = B.zw;
    let u = dot(rel, dir);
    let v = dot(rel, vec2f(-dir.y, dir.x));
    let k = len * 0.5 - u;
    if (k < 0.0 || k > 700.0) { continue; }
    let front = smoothstep(0.0, 14.0, k);
    let spread = 0.36 * k + len * 0.12;
    let edge = abs(abs(v) - spread);
    let arms = exp(-edge * edge / (2.0 + 0.03 * k)) * (1.0 - smoothstep(60.0, 650.0, k));
    let trail = exp(-v * v / (7.0 + 0.04 * k * k * 0.05 + 0.1 * k)) * (1.0 - smoothstep(30.0, 380.0, k));
    foam += (arms * 0.55 + trail * 0.85) * front;
    let ph = sin(k * 0.45 - t * 3.0) * arms * front * 0.07;
    sl += vec2f(-dir.y, dir.x) * ph * sign(v) + dir * ph * 0.5;
  }
  return vec3f(clamp(foam, 0.0, 1.0), sl);
}

fn sampleRefl(uv: vec2f, blur: f32) -> vec4f {
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var i = -3; i <= 3; i++) {
    let fi = f32(i);
    let w = exp(-fi * fi * 0.18);
    acc += textureSampleLevel(reflTex, reflSamp, uv + vec2f(0.0, fi * blur), 0.0) * w;
    wsum += w;
  }
  return acc / wsum;
}

@fragment fn fs(in: GOut) -> @location(0) vec4f {
  let P = in.wp;
  let vp = cam.pos.xyz;
  let V = normalize(vp - P);
  let sa = toGrid(P.xz);
  let ld = landDepth(sa);
  let pfv = fwidth(sa);
  let pf = max(pfv.x, pfv.y);
  let aa = max(length(vec2f(dpdx(ld), dpdy(ld))), 0.6);
  let landW = smoothstep(-aa, aa, ld);
  var col = vec3f(0.0);
  let nt = night();
  let L = scn.sunDir.xyz;
  if (landW > 0.001) { col += landColor(P, sa, pf) * landW; }
  if (landW < 0.999) {
    let t = scn.ambSide.w;
    let rough = scn.ambUp.w;
    var g = waveSlope(P.xz, t, pf, rough);
    let wk = wakeAt(P.xz, t);
    g += wk.yz;
    let N = normalize(vec3f(-g.x, 1.0, -g.y));
    let nv = max(dot(N, V), 0.001);
    let F = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
    var R = reflect(-V, N);
    R.y = max(R.y, 0.015);
    let skyR = skyFull(R, false);
    // planar reflection of the skyline (rendered mirrored, half resolution)
    let suv = in.pos.xy * cam.res.zw;
    let fwdH = normalize(cam.fwd.xz + vec2f(1e-5));
    let rgtH = vec2f(-fwdH.y, fwdH.x) ;
    let gv = vec2f(dot(g, rgtH), dot(g, fwdH));
    let duv = vec2f(gv.x * 0.10, -gv.y * 0.22) * (0.5 + rough);
    let blur = 0.0030 + 0.012 * smoothstep(0.02, 0.3, rough) + 0.00005 * min(pf, 60.0);
    let rt = sampleRefl(clamp(suv + duv, vec2f(0.001), vec2f(0.999)), blur);
    var refl = mix(skyR, rt.rgb, clamp(rt.a, 0.0, 1.0));
    // sun glitter (analytic GGX lobe, roughness grows with distance)
    let H = normalize(L + V);
    let ndh = max(dot(N, H), 0.0);
    let al = clamp(0.045 + rough * 0.22 + 0.2 * smoothstep(8.0, 250.0, pf), 0.03, 0.5);
    let a2 = al * al;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159 * dd * dd);
    let sh = select(0.0, sunShadow(P + vec3f(0.0, 0.3, 0.0), L), L.y > 0.0);
    let vdh = max(dot(V, H), 0.0);
    let Fs = 0.02 + 0.98 * pow(1.0 - vdh, 5.0);
    let spec = scn.sunRad.rgb * D * Fs * 0.30 * sh * smoothstep(0.0, 0.08, L.y);
    // body colour
    let shallow = smoothstep(-70.0, 0.0, ld);
    let lightIn = scn.ambUp.rgb * 0.9 + scn.sunRad.rgb * max(L.y, 0.0) * sh * 0.25;
    var body = mix(vec3f(0.018, 0.04, 0.045), vec3f(0.05, 0.075, 0.06), shallow) * lightIn;
    // foam: boat wakes + a little along the seawall
    let wall = (1.0 - smoothstep(0.0, 5.0, -ld)) * (0.4 + 0.6 * vnoise2(P.xz * 0.6 + scn.ambSide.w * 0.2));
    let foam = clamp(wk.x + wall * 0.35, 0.0, 1.0) * smoothstep(0.0, 1.0, 1.0 - clamp(pf * 0.03, 0.0, 1.0));
    let foamC = vec3f(0.75, 0.8, 0.82) * (scn.ambUp.rgb * 0.9 + scn.sunRad.rgb * max(L.y, 0.0) * sh * 0.35 + vec3f(0.04));
    var wcol = refl * F + body * (1.0 - F) + spec;
    wcol = mix(wcol, foamC, foam * 0.7);
    col += wcol * (1.0 - landW);
  }
  col = applyFog(col, P);
  return vec4f(col, 1.0);
}
`;

// ===================================================================================
// Sky pass
// ===================================================================================
export const SKYPASS = COMMON + /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let rd = camRay(cam, in.pos.xy);
  return vec4f(skyFull(rd, true), 1.0);
}
`;
