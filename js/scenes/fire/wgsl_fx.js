// Fire effects: tileable noise volume, raymarched flame, GPU particle sim and particle renderers.
import { FIRE_COMMON } from './wgsl_common.js';

// 128^3 RGBA8 tileable noise: rgb = three independent periodic Perlin fbm (0..1), a = 1 - Worley F1
export const NOISE_GEN = /* wgsl */ `
@group(0) @binding(0) var outT: texture_storage_3d<rgba8unorm, write>;

fn pcg3(v0: vec3u) -> vec3u {
  var v = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
fn wrap3(c: vec3i, per: i32) -> vec3u {
  return vec3u(vec3i(((c.x % per) + per) % per, ((c.y % per) + per) % per, ((c.z % per) + per) % per));
}
fn grad(c: vec3i, per: i32, seed: u32) -> vec3f {
  let h = pcg3(wrap3(c, per) + vec3u(seed * 911u, seed * 577u, seed * 313u + 7u));
  return normalize(vec3f(h) / 2147483647.5 - 1.0 + vec3f(1e-4));
}
fn pnoise(p: vec3f, per: i32, seed: u32) -> f32 {
  let i = vec3i(floor(p));
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(grad(i, per, seed), f);
  let b = dot(grad(i + vec3i(1, 0, 0), per, seed), f - vec3f(1, 0, 0));
  let c = dot(grad(i + vec3i(0, 1, 0), per, seed), f - vec3f(0, 1, 0));
  let d = dot(grad(i + vec3i(1, 1, 0), per, seed), f - vec3f(1, 1, 0));
  let e = dot(grad(i + vec3i(0, 0, 1), per, seed), f - vec3f(0, 0, 1));
  let g = dot(grad(i + vec3i(1, 0, 1), per, seed), f - vec3f(1, 0, 1));
  let h = dot(grad(i + vec3i(0, 1, 1), per, seed), f - vec3f(0, 1, 1));
  let k = dot(grad(i + vec3i(1, 1, 1), per, seed), f - vec3f(1, 1, 1));
  return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(h, k, u.x), u.y), u.z);
}
fn pfbm(p: vec3f, seed: u32) -> f32 {
  var s = 0.0; var a = 1.0; var per = 4; var q = p * 4.0;
  for (var o = 0u; o < 4u; o++) {
    s += a * pnoise(q, per, seed + o * 17u);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return clamp(0.5 + s * 0.52, 0.0, 1.0);
}
fn worley(p: vec3f, per: i32, seed: u32) -> f32 {
  let q = p * f32(per);
  let i = vec3i(floor(q));
  let f = fract(q);
  var m = 8.0;
  for (var z = -1; z <= 1; z++) { for (var y = -1; y <= 1; y++) { for (var x = -1; x <= 1; x++) {
    let o = vec3i(x, y, z);
    let h = vec3f(pcg3(wrap3(i + o, per) + vec3u(seed, seed * 3u, seed * 7u))) / 4294967295.0;
    let d = vec3f(o) + h - f;
    m = min(m, dot(d, d));
  } } }
  return sqrt(m);
}
@compute @workgroup_size(4, 4, 4) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = (vec3f(gid) + 0.5) / 128.0;
  let r = pfbm(p, 1u);
  let g = pfbm(p, 101u);
  let b = pfbm(p, 202u);
  let a = 1.0 - clamp(worley(p, 6, 303u) * 1.15, 0.0, 1.0);
  textureStore(outT, gid, vec4f(r, g, b, a));
}
`;

export const FLAME = /* wgsl */ `
${FIRE_COMMON}
struct FO { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> FO {
  var q = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let S = scn.lightCol.w;
  let c = vec3f(scn.fire.x * 0.3 * S, 1.05 * S, scn.fire.y * 0.3 * S);
  let rgt = normalize(cam.right.xyz);
  let upv = normalize(cam.up.xyz);
  var p = c + (rgt * q[vi].x + upv * q[vi].y) * 1.8 * S;
  // pull the billboard towards the camera so the flame draws over sticks that sit inside it;
  // scaling about the eye keeps the on-screen footprint unchanged
  let rel = p - cam.pos.xyz;
  let dc = length(c - cam.pos.xyz);
  let k = max(dc - 0.75 * S, 0.1) / dc;
  var o: FO;
  o.pos = cam.viewProj * vec4f(rel * k, 1.0);
  return o;
}

// (density, temperature) of the flame volume at world position pw (fire base at the origin)
fn flameField(pw: vec3f, t: f32, S: f32) -> vec2f {
  let p = pw / S;
  let FH = 1.95 * scn.fire.z;
  let turb = scn.env.w;
  let fs0 = saturate(p.y / FH);
  var xz = p.xz - scn.fire.xy * fs0 * fs0 * 1.3;
  let adv = t * 0.9;
  let w1 = nz(vec3f(xz.x * 0.75, p.y * 0.5 - adv * 0.8, xz.y * 0.75)).rgb - 0.5;
  xz += w1.xz * turb * (0.1 + fs0);
  let w2 = nz(vec3f(xz.x * 1.9 + 0.4, p.y * 1.3 - adv * 1.9, xz.y * 1.9)).rgb - 0.5;
  xz += w2.xz * 0.35 * turb * fs0;
  let n2 = nz(vec3f(xz.x * 2.6 + 0.31, p.y * 0.95 - adv * 1.6, xz.y * 2.6 + 0.57)).rgb;
  let n3 = nz(vec3f(xz.x * 4.4 + 0.7, p.y * 1.5 - adv * 2.6, xz.y * 4.4 + 0.2));
  let lick = nz(vec3f(xz.x * 0.8 + 0.13, t * 0.25, xz.y * 0.8)).g;
  let Hloc = FH * (0.4 + 1.2 * lick);
  let f = p.y / Hloc;
  if (f > 1.0 || p.y < -0.05) { return vec2f(0.0); }
  let prof = mix(0.4, 1.0, smoothstep(0.0, 0.25, f)) * pow(1.0 - f, 0.75);
  let r = 0.82 * prof;
  let d = length(xz) / max(r, 0.02);
  var dens = (1.0 - d) * 1.6 - (n2.r - 0.48) * (0.7 + 1.9 * f) - (n3.g - 0.5) * (0.5 + 1.0 * f);
  dens = saturate(dens * 0.95);
  let T = saturate(dens * 1.05) * max(0.9 - 0.8 * pow(f, 1.15), 0.0) * max(1.0 - 0.45 * d * d, 0.0) * (0.55 + 0.75 * n3.b);
  return vec2f(dens, T);
}

fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

@fragment fn fs(in: FO) -> @location(0) vec4f {
  let S = scn.lightCol.w;
  let ro = cam.pos.xyz;
  let rd = camRay(cam, in.pos.xy);
  let Rb = 1.2 * S;
  let yTop = 2.6 * S;
  let yBot = -0.05;
  let a = dot(rd.xz, rd.xz);
  if (a < 1e-6) { discard; }
  let b = dot(ro.xz, rd.xz);
  let c = dot(ro.xz, ro.xz) - Rb * Rb;
  let disc = b * b - a * c;
  if (disc <= 0.0) { discard; }
  let sq = sqrt(disc);
  var t0 = (-b - sq) / a;
  var t1 = (-b + sq) / a;
  let ry = select(rd.y, 1e-5, abs(rd.y) < 1e-5);
  let ta = (yBot - ro.y) / ry;
  let tb = (yTop - ro.y) / ry;
  t0 = max(max(t0, min(ta, tb)), 0.0);
  t1 = min(t1, max(ta, tb));
  if (t1 <= t0) { discard; }

  let N = 72;
  let dt = (t1 - t0) / f32(N);
  var t = t0 + dt * ign(in.pos.xy + cam.misc.y * 7.13);
  let time = cam.pos.w;
  var L = vec3f(0.0);
  var trans = 1.0;
  for (var i = 0; i < N; i++) {
    let p = ro + rd * t;
    let fd = flameField(p, time, S);
    if (fd.x > 0.004) {
      let T = fd.y;
      var rad = flameRamp(T) * (0.22 + 7.5 * pow(T, 2.3));
      let low = 1.0 - smoothstep(0.0, 0.2, p.y / S);
      rad += vec3f(0.10, 0.26, 1.0) * low * 0.9;
      let w = pow(fd.x, 1.8) * dt;
      L += trans * rad * w * 3.2 * mix(0.3, 1.0, smoothstep(0.0, 0.8, p.y / S));
      trans *= exp(-w * (0.25 + 0.5 * (1.0 - T)));
    }
    t += dt;
  }
  return vec4f(L, trans);
}
`;

const PCG = /* wgsl */ `
const TAU: f32 = 6.28318530718;
fn pcg4(v0: vec4u) -> vec4u {
  var v = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  v ^= v >> vec4u(16u);
  v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
  return v;
}
fn rnd4(a: u32, b: u32, c: u32, d: u32) -> vec4f { return vec4f(pcg4(vec4u(a, b, c, d))) / 4294967295.0; }
`;

export const SIM = /* wgsl */ `
${PCG}
struct Pt { pos: vec4f, vel: vec4f, att: vec4f }
struct SimU { a: vec4f, b: vec4f, c: vec4f }   // a: time dt rate turb | b: wind.xy S flickH | c: frame smoke bend.xy
@group(0) @binding(0) var<uniform> U: SimU;
@group(0) @binding(1) var<storage, read_write> P: array<Pt>;
@group(0) @binding(2) var noiseT: texture_3d<f32>;
@group(0) @binding(3) var nsamp: sampler;
fn tn(p: vec3f) -> vec3f { return textureSampleLevel(noiseT, nsamp, p, 0.0).rgb - 0.5; }

// ---- sparks: tiny, long-lived, fast, cooling embers --------------------------------------
@compute @workgroup_size(64) fn sparks(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&P)) { return; }
  var p = P[i];
  let dt = U.a.y;
  let S = U.b.z;
  let age = p.pos.w + dt;
  if (age > p.vel.w + p.att.w) {
    let fr = u32(U.c.x);
    let r = rnd4(i, fr, 11u, 3u);
    let r2 = rnd4(i, fr, 17u, 5u);
    let ang = r.x * TAU;
    let rad = sqrt(r.y) * 0.4 * S;
    p.pos = vec4f(cos(ang) * rad + U.c.z * 0.2 * S, (0.3 + 0.9 * r.z) * S, sin(ang) * rad + U.c.w * 0.2 * S, 0.0);
    p.vel = vec4f((r2.x - 0.5) * 1.1, (1.0 + 2.4 * r2.y) * (0.7 + 0.4 * S), (r2.z - 0.5) * 1.1, 1.4 + 3.2 * r.w);
    p.att = vec4f(0.0, 0.008 + 0.013 * r2.w, r.x, r2.w * r.y * 70.0 / max(U.a.z, 0.03));
    P[i] = p;
    return;
  }
  if (age > p.vel.w) { p.pos.w = age; P[i] = p; return; }
  let k = age / p.vel.w;
  var v = p.vel.xyz;
  var pos = p.pos.xyz;
  let t = U.a.x;
  let n1 = tn(pos * 0.55 + vec3f(0.0, -t * 0.35, 0.0));
  let n2 = tn(pos * 1.9 + vec3f(0.3, -t * 0.8, 0.1));
  v += (n1 * 7.0 + n2 * 5.0) * U.a.w * dt;
  v.y += 1.5 * (1.0 - k) * dt;
  v.x += U.b.x * 1.8 * dt; v.z += U.b.y * 1.8 * dt;
  v *= exp(-0.55 * dt);
  pos += v * dt;
  var a2 = age;
  if (pos.y < 0.02) { a2 = p.vel.w; }
  P[i] = Pt(vec4f(pos, a2), vec4f(v, p.vel.w), p.att);
}

// ---- flame fragments: short-lived tongues torn off the flame tip --------------------------
@compute @workgroup_size(64) fn blobs(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&P)) { return; }
  var p = P[i];
  let dt = U.a.y;
  let S = U.b.z;
  let age = p.pos.w + dt;
  if (age > p.vel.w + p.att.w) {
    let fr = u32(U.c.x);
    let r = rnd4(i, fr, 23u, 3u);
    let r2 = rnd4(i, fr, 29u, 5u);
    let ang = r.x * TAU;
    let rad = sqrt(r.y) * 0.3 * S;
    let y = (0.85 + 0.55 * r.z) * S * U.b.w;
    let f = min(y / (1.55 * S * U.b.w), 1.0);
    p.pos = vec4f(cos(ang) * rad + U.c.z * f * f * S, y, sin(ang) * rad + U.c.w * f * f * S, 0.0);
    p.vel = vec4f((r2.x - 0.5) * 0.5, (0.8 + 1.0 * r2.y) * S, (r2.z - 0.5) * 0.5, 0.35 + 0.5 * r.w);
    p.att = vec4f(1.0, (0.03 + 0.05 * r2.w) * S, r.x, r2.w * r.y * 3.5 / max(U.a.z * 0.7 + 0.3, 0.1));
    P[i] = p;
    return;
  }
  if (age > p.vel.w) { p.pos.w = age; P[i] = p; return; }
  var v = p.vel.xyz;
  var pos = p.pos.xyz;
  let t = U.a.x;
  v += tn(pos * 1.2 + vec3f(0.0, -t * 0.9, 0.0)) * 9.0 * U.a.w * dt;
  v.y += 1.4 * dt;
  v.x += U.b.x * 1.2 * dt; v.z += U.b.y * 1.2 * dt;
  v *= exp(-1.1 * dt);
  pos += v * dt;
  P[i] = Pt(vec4f(pos, age), vec4f(v, p.vel.w), p.att);
}

// ---- smoke: large, slow, soft ------------------------------------------------------------
@compute @workgroup_size(64) fn smoke(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&P)) { return; }
  var p = P[i];
  let dt = U.a.y;
  let S = U.b.z;
  let age = p.pos.w + dt;
  if (age > p.vel.w + p.att.w) {
    let fr = u32(U.c.x);
    let r = rnd4(i, fr, 31u, 3u);
    let r2 = rnd4(i, fr, 37u, 5u);
    let ang = r.x * TAU;
    let rad = sqrt(r.y) * 0.3 * S;
    let y = (1.2 + 0.7 * r.z) * S * U.b.w;
    p.pos = vec4f(cos(ang) * rad + U.c.z * 0.9 * S, y, sin(ang) * rad + U.c.w * 0.9 * S, 0.0);
    p.vel = vec4f((r2.x - 0.5) * 0.3, 0.5 + 0.7 * r2.y, (r2.z - 0.5) * 0.3, 5.5 + 4.0 * r.w);
    p.att = vec4f(2.0, 0.2 + 0.22 * r2.w, r.x, r.y * 2.5);
    P[i] = p;
    return;
  }
  if (age > p.vel.w) { p.pos.w = age; P[i] = p; return; }
  var v = p.vel.xyz;
  var pos = p.pos.xyz;
  let t = U.a.x;
  let k = age / p.vel.w;
  v += tn(pos * 0.28 + vec3f(0.0, -t * 0.12, 0.0)) * 3.2 * U.a.w * dt;
  v.y += 0.38 * (1.0 - k) * dt;
  v.x += U.b.x * 1.0 * dt; v.z += U.b.y * 1.0 * dt;
  v *= exp(-0.28 * dt);
  pos += v * dt;
  P[i] = Pt(vec4f(pos, age), vec4f(v, p.vel.w), p.att);
}
`;

const PART_BIND = /* wgsl */ `
struct Pt { pos: vec4f, vel: vec4f, att: vec4f }
@group(1) @binding(0) var<storage, read> parts: array<Pt>;
var<private> QUAD: array<vec2f, 6> = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
`;

// sparks + flame fragments: additive streaks, stretched along their velocity in screen space
export const SPARK = /* wgsl */ `
${FIRE_COMMON}
${PART_BIND}
struct PO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) col: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PO {
  let pt = parts[ii];
  var o: PO;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  let age = pt.pos.w;
  let life = pt.vel.w;
  if (age >= life) { return o; }
  let k = age / life;
  let kind = pt.att.x;
  let rel = pt.pos.xyz - cam.pos.xyz;
  let clip = cam.viewProj * vec4f(rel, 1.0);
  if (clip.w <= 0.02) { return o; }
  let tanX = length(cam.right.xyz);
  let tanY = length(cam.up.xyz);
  let aspect = cam.res.x / cam.res.y;
  let corner = QUAD[vi];
  var size = pt.att.y;
  let speed = length(pt.vel.xyz);
  var stretch = 1.0 + speed * 1.1;
  var T = 1.0 - pow(k, 0.55);
  var bright = 1.0;
  if (kind > 0.5) {
    size *= (1.0 - 0.6 * k);
    stretch = 1.0 + speed * 0.9;
    T = 0.62 * (1.0 - k);
    bright = 0.22 * smoothstep(0.0, 0.12, k) * (1.0 - k);
  } else {
    size *= (1.0 - 0.5 * k);
    bright = (0.7 + 0.5 * sin(age * 38.0 + pt.att.z * 60.0)) * smoothstep(0.0, 0.04, k) * (1.0 - smoothstep(0.7, 1.0, k)) * scn.amb.w;
  }
  let ndcPerM = 1.0 / (tanY * clip.w);
  let clip2 = cam.viewProj * vec4f(rel + pt.vel.xyz * 0.05, 1.0);
  var sd = clip2.xy / clip2.w - clip.xy / clip.w;
  sd.x *= aspect;
  let sl = length(sd);
  let dir = select(vec2f(0.0, 1.0), sd / max(sl, 1e-6), sl > 1e-6);
  let perp = vec2f(-dir.y, dir.x);
  let off = dir * (corner.y * size * stretch) + perp * (corner.x * size);
  let ndcOff = vec2f(off.x / aspect, off.y) * ndcPerM;
  o.pos = vec4f((clip.xy / clip.w + ndcOff) * clip.w, clip.z, clip.w);
  o.uv = corner;
  o.col = flameRamp(T) * (0.3 + 8.0 * pow(T, 2.0)) * bright * select(1.0, 0.7, kind > 0.5);
  return o;
}
@fragment fn fs(in: PO) -> @location(0) vec4f {
  let g = exp(-dot(in.uv, in.uv) * 3.2);
  return vec4f(in.col * g, 1.0);
}
`;

export const SMOKE = /* wgsl */ `
${FIRE_COMMON}
${PART_BIND}
struct SO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) wp: vec3f, @location(2) k: f32, @location(3) seed: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SO {
  let pt = parts[ii];
  var o: SO;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  let age = pt.pos.w;
  let life = pt.vel.w;
  if (age >= life) { return o; }
  let k = age / life;
  let rgt = normalize(cam.right.xyz);
  let upv = normalize(cam.up.xyz);
  let size = pt.att.y * (1.0 + 4.5 * pow(k, 0.7));
  let corner = QUAD[vi];
  let wp = pt.pos.xyz;
  let rot = pt.att.z * 6.28 + age * 0.15;
  let cr = cos(rot); let sr = sin(rot);
  let c2 = vec2f(corner.x * cr - corner.y * sr, corner.x * sr + corner.y * cr);
  let w = wp + (rgt * c2.x + upv * c2.y) * size;
  o.pos = cam.viewProj * vec4f(w - cam.pos.xyz, 1.0);
  o.uv = corner;
  o.wp = wp;
  o.k = k;
  o.seed = pt.att.z;
  return o;
}
@fragment fn fs(in: SO) -> @location(0) vec4f {
  let r = length(in.uv);
  if (r > 1.0) { discard; }
  let n = nz(vec3f(in.uv * 0.55 + in.seed * 7.0, in.k * 0.6 + in.seed)).r;
  let n2 = nz(vec3f(in.uv * 1.7 + in.seed * 3.0, in.seed)).g;
  var a = (1.0 - smoothstep(0.15, 1.0, r)) * smoothstep(0.25, 0.8, n * 0.7 + n2 * 0.5 + 0.1 - r * 0.35);
  let fadeIn = smoothstep(0.0, 0.12, in.k);
  let fadeOut = pow(1.0 - in.k, 1.6);
  a *= fadeIn * fadeOut * 0.16 * scn.hor.w;
  let toL = scn.light.xyz - in.wp;
  let r2 = dot(toL, toL);
  let irr = scn.lightCol.rgb * scn.light.w / (r2 + 1.5) * 0.45;
  let lit = irr + scn.amb.rgb * 3.5 + scn.moonCol.rgb * 1.2 + scn.hor.rgb * 0.18;
  let col = vec3f(0.30, 0.29, 0.28) * lit;
  return vec4f(col * a, a);
}
`;

export const FIREFLY = /* wgsl */ `
${FIRE_COMMON}
struct FO { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) b: f32 };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FO {
  var q = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let f = f32(ii);
  let h = hash13(vec3f(f, 1.7, 3.1));
  let h2 = hash33(vec3f(f * 0.37, 5.0, 9.0));
  let t = cam.pos.w;
  let ang = h2.x * TAU;
  let rad = 4.5 + 26.0 * sqrt(h2.y);
  var p = vec3f(cos(ang) * rad + sin(t * 0.31 * (0.5 + h) + h * 40.0) * 1.6, 0.0, sin(ang) * rad + cos(t * 0.27 * (0.6 + h2.z) + h * 20.0) * 1.6);
  p.y = groundH(p.x, p.z) + 0.5 + 1.6 * h2.z + sin(t * 0.5 + h * 30.0) * 0.3;
  let ph = sin(t * (0.45 + 0.4 * h) + h * 70.0) * 0.5 + 0.5;
  let blink = pow(smoothstep(0.55, 1.0, ph), 2.0) * (0.35 + 0.65 * scn.env.x);
  var o: FO;
  let rgt = normalize(cam.right.xyz);
  let upv = normalize(cam.up.xyz);
  let w = p + (rgt * q[vi].x + upv * q[vi].y) * 0.05;
  o.pos = cam.viewProj * vec4f(w - cam.pos.xyz, 1.0);
  o.uv = q[vi];
  o.b = blink;
  return o;
}
@fragment fn fs(in: FO) -> @location(0) vec4f {
  let d2 = dot(in.uv, in.uv);
  let g = exp(-d2 * 7.0) + exp(-d2 * 1.6) * 0.1;
  return vec4f(vec3f(0.55, 1.0, 0.22) * 9.0 * g * in.b, 1.0);
}
`;
