// Shared WGSL snippets. Concatenate into a scene shader: `${CAM_WGSL}${NOISE_WGSL}...`.
//
// Camera uniform (192 bytes, written by the engine every frame, available as eng.camUBO).
// All matrices are CAMERA-RELATIVE: viewProj has no translation, so transform
// (worldPos - cam.pos) with it. This keeps float precision for large scenes.
// Depth is reverse-Z with an infinite far plane: clear depth to 0, compare 'greater'.
export const CAM_WGSL = /* wgsl */ `
struct Cam {
  viewProj: mat4x4f,
  pos: vec4f,        // xyz world position, w = time (s)
  right: vec4f,      // xyz = right * tan(fovX/2)
  up: vec4f,         // xyz = up * tan(fovY/2)
  fwd: vec4f,        // xyz forward
  res: vec4f,        // width, height, 1/width, 1/height (pixels)
  misc: vec4f,       // near, frameIndex, dt, 0
}
// world-space view ray for a fragment position (in.pos.xy)
fn camRay(c: Cam, frag: vec2f) -> vec3f {
  let ndc = vec2f(frag.x * c.res.z * 2.0 - 1.0, 1.0 - frag.y * c.res.w * 2.0);
  return normalize(c.fwd.xyz + ndc.x * c.right.xyz + ndc.y * c.up.xyz);
}
`;

export const FULLSCREEN_VS = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);   // z = 0 -> far plane in reverse-Z
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`;

export const NOISE_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn sat3(x: vec3f) -> vec3f { return clamp(x, vec3f(0.0), vec3f(1.0)); }
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn hash11(p: f32) -> f32 { var x = fract(p * 0.1031); x *= x + 33.33; x *= x + x; return fract(x); }
fn hash12(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
fn hash13(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
fn hash22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
fn hash33(p: vec3f) -> vec3f {
  var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}
fn vnoise2(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2f(1, 0)), u.x),
             mix(hash12(i + vec2f(0, 1)), hash12(i + vec2f(1, 1)), u.x), u.y);
}
fn vnoise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3f(1, 0, 0)), u.x),
                 mix(hash13(i + vec3f(0, 1, 0)), hash13(i + vec3f(1, 1, 0)), u.x), u.y),
             mix(mix(hash13(i + vec3f(0, 0, 1)), hash13(i + vec3f(1, 0, 1)), u.x),
                 mix(hash13(i + vec3f(0, 1, 1)), hash13(i + vec3f(1, 1, 1)), u.x), u.y), u.z);
}
fn fbm2(p0: vec2f, oct: i32) -> f32 {
  var p = p0; var a = 0.5; var s = 0.0;
  for (var i = 0; i < oct; i++) { s += a * vnoise2(p); p = p * 2.03 + vec2f(17.1, 9.2); a *= 0.5; }
  return s;
}
fn fbm3(p0: vec3f, oct: i32) -> f32 {
  var p = p0; var a = 0.5; var s = 0.0;
  for (var i = 0; i < oct; i++) { s += a * vnoise3(p); p = p * 2.03 + vec3f(17.1, 9.2, 4.7); a *= 0.5; }
  return s;
}
`;
