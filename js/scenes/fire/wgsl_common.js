// Shared WGSL for every render shader of the fire scene: bindings, lighting, fog, noise volume.
import { CAM_WGSL, NOISE_WGSL } from '../../engine/wgsl.js';

export const FIRE_COMMON = /* wgsl */ `
${CAM_WGSL}
${NOISE_WGSL}
struct Scn {
  light: vec4f,      // xyz fire light position, w intensity
  lightCol: vec4f,   // rgb colour, w flame size S
  wind: vec4f,       // xy wind vector, z gust, w grass wind scale
  env: vec4f,        // x time of day, y mist, z heat haze, w turbulence
  moon: vec4f,       // xyz direction, w visibility
  zen: vec4f,        // rgb zenith, w star amount
  hor: vec4f,        // rgb horizon, w smoke amount
  amb: vec4f,        // rgb ambient, w ember amount
  fire: vec4f,       // xy flame bend, z flame height factor
  moonCol: vec4f,    // rgb moon light colour
  blk: array<vec4f, 24>, // shadow-casting spheres (stones + log segments): xyz, r
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<uniform> scn: Scn;
@group(0) @binding(2) var noiseT: texture_3d<f32>;
@group(0) @binding(3) var nsamp: sampler;

fn nz(p: vec3f) -> vec4f { return textureSampleLevel(noiseT, nsamp, p, 0.0); }

fn groundH(x: f32, z: f32) -> f32 {
  let r = length(vec2f(x, z));
  let h = 0.5 * (sin(x * 0.07 + 1.3) * cos(z * 0.055) + 0.5 * sin(x * 0.13 + z * 0.11 + 2.0));
  return h * smoothstep(3.0, 10.0, r);
}

// black-body-ish flame ramp, T in 0..1 -> linear radiance colour (unit brightness)
fn flameRamp(T: f32) -> vec3f {
  let t = clamp(T, 0.0, 1.0);
  var c = mix(vec3f(0.0), vec3f(0.55, 0.03, 0.0), smoothstep(0.0, 0.18, t));
  c = mix(c, vec3f(1.0, 0.16, 0.015), smoothstep(0.12, 0.38, t));
  c = mix(c, vec3f(1.0, 0.42, 0.05), smoothstep(0.32, 0.58, t));
  c = mix(c, vec3f(1.0, 0.72, 0.22), smoothstep(0.52, 0.80, t));
  c = mix(c, vec3f(1.0, 0.93, 0.68), smoothstep(0.76, 1.0, t));
  return c;
}

fn sphShadow(p: vec3f, L: vec3f, c: vec3f, r: f32) -> f32 {
  let d = L - p;
  let len2 = dot(d, d);
  let t = clamp(dot(c - p, d) / len2, 0.0, 1.0);
  let dist = length(p + d * t - c);
  let pen = clamp(0.28 * t / max(1.0 - t, 0.05), 0.03, 0.5);
  return smoothstep(r - pen, r + pen, dist);
}
fn fireShadow(p: vec3f) -> f32 {
  let L = scn.light.xyz;
  let d = L - p;
  if (dot(d, d) > 64.0) { return 1.0; }
  var s = 1.0;
  for (var i = 0; i < 24; i++) {
    let b = scn.blk[i];
    if (b.w > 0.0) { s *= sphShadow(p, L, b.xyz, b.w); }
  }
  return s;
}
fn fireIrr(p: vec3f, n: vec3f) -> vec3f {
  let d = scn.light.xyz - p;
  let r2 = dot(d, d);
  let l = d * inverseSqrt(r2);
  let wrap = saturate(dot(n, l) * 0.65 + 0.35);
  return scn.lightCol.rgb * scn.light.w * wrap / (r2 + 0.8);
}
fn skyBase(rd: vec3f) -> vec3f {
  let e = rd.y;
  var col = mix(scn.hor.rgb, scn.zen.rgb, pow(clamp(e, 0.0, 1.0), 0.45));
  let sunDir = normalize(vec3f(0.35, -0.10, -0.93));
  let sg = clamp(dot(rd, sunDir) * 0.5 + 0.5, 0.0, 1.0);
  let dusk = 1.0 - scn.env.x;
  col += vec3f(1.0, 0.40, 0.12) * pow(sg, 7.0) * exp(-max(e, 0.0) * 4.5) * dusk * dusk * 0.9;
  return col;
}
fn treeColor() -> vec3f { return scn.hor.rgb * 0.085 + scn.zen.rgb * 0.10 + vec3f(0.0015, 0.002, 0.0015); }

// in-scattered fire light along a ray through mist (closed-form integral of 1/(h^2+s^2))
fn fireScatter(ro: vec3f, rd: vec3f, tmax: f32) -> vec3f {
  let oc = scn.light.xyz - ro;
  let tc = dot(oc, rd);
  let h = max(length(oc - rd * tc), 0.35);
  let integral = (atan((tmax - tc) / h) - atan(-tc / h)) / h;
  return scn.lightCol.rgb * scn.light.w * integral * scn.env.y * 0.00042;
}

fn applyFog(col: vec3f, wp: vec3f) -> vec3f {
  let ro = cam.pos.xyz;
  let v = wp - ro;
  let d = length(v);
  let rd = v / max(d, 1e-4);
  let fogC = mix(scn.hor.rgb * 0.75 + scn.amb.rgb * 0.5, treeColor(), smoothstep(150.0, 700.0, d));
  let fogA = 1.0 - exp(-d * (0.0016 + 0.0028 * scn.env.y));
  return mix(col, fogC, fogA) + fireScatter(ro, rd, d);
}
`;
