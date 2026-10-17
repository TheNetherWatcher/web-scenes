// Black hole shaders.
//   TRACE  pass A (render-scale res): per-pixel null geodesic through Schwarzschild curvature
//          (+ gravitomagnetic frame-drag term), accumulating the accretion disc on the way.
//          out0 = disc light (rgb) + transmittance (a), out1 = escape direction (xyz) + valid flag.
//   COMP   pass B (full res): lensed star field, Milky Way and companion star behind the disc,
//          with footprint-aware (anti-aliased) star splatting.
// All lengths are in Schwarzschild radii (rs = 1).
import { CAM_WGSL, FULLSCREEN_VS, NOISE_WGSL } from '../../engine/wgsl.js';

const COMMON = /* wgsl */ `
${CAM_WGSL}
${NOISE_WGSL}
${FULLSCREEN_VS}
struct S {
  a: vec4f,   // mass (unused), spin, disc brightness, disc temperature (K)
  b: vec4f,   // tilt (unused), thickness, max steps, time
  c: vec4f,   // pass-A resolution, r_in, r_out
  d: vec4f,   // disc normal xyz, base step
  e: vec4f,   // star density, A/out scale, out resolution
  f: vec4f,   // camera position (rs units), reference pixel angle
  g: vec4f,   // actual pixel angle
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<uniform> u: S;

// blackbody colour (Tanner Helland fit), linear, normalised to unit luminance
fn bbColor(T: f32) -> vec3f {
  let t = clamp(T, 1000.0, 40000.0) / 100.0;
  var r = 1.0; var g = 1.0; var b = 1.0;
  if (t <= 66.0) { g = clamp(0.39008157876 * log(t) - 0.63184144378, 0.0, 1.0); }
  else { r = clamp(1.29293618606 * pow(t - 60.0, -0.1332047592), 0.0, 1.0); g = clamp(1.12989086089 * pow(t - 60.0, -0.0755148492), 0.0, 1.0); }
  if (t < 66.0) { b = select(clamp(0.54320678911 * log(t - 10.0) - 1.19625408914, 0.0, 1.0), 0.0, t <= 19.0); }
  let c = pow(vec3f(r, g, b), vec3f(2.2));
  return c / max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-3);
}
`;

export const TRACE = COMMON + /* wgsl */ `
// periodic-in-angle value noise: radial coordinate rr (free), angle in turns, na angular cells
fn rnoise(rr: f32, turns: f32, na: f32) -> f32 {
  let a = turns * na;
  let i0 = floor(a);
  let f = fract(a);
  let i1 = select(i0 + 1.0, 0.0, i0 + 1.0 >= na);
  let ua = f * f * (3.0 - 2.0 * f);
  let ra = floor(rr);
  let fr = fract(rr);
  let ur = fr * fr * (3.0 - 2.0 * fr);
  let h00 = hash12(vec2f(ra, i0));
  let h10 = hash12(vec2f(ra + 1.0, i0));
  let h01 = hash12(vec2f(ra, i1));
  let h11 = hash12(vec2f(ra + 1.0, i1));
  return mix(mix(h00, h10, ur), mix(h01, h11, ur), ua);
}
fn ringFbm(r: f32, turns: f32) -> f32 {
  var s = 0.0;
  s += 0.36 * rnoise(r * 5.0, turns, 14.0);
  s += 0.28 * rnoise(r * 12.0 + 3.7, turns, 38.0);
  s += 0.20 * rnoise(r * 30.0 + 9.1, turns, 104.0);
  s += 0.16 * rnoise(r * 74.0 + 1.3, turns, 280.0);
  return s;
}

struct FO { @location(0) c: vec4f, @location(1) d: vec4f }

@fragment fn fs(in: VO) -> FO {
  let resA = u.c.xy;
  let ndc = vec2f(in.pos.x / resA.x * 2.0 - 1.0, 1.0 - in.pos.y / resA.y * 2.0);
  var v = normalize(cam.fwd.xyz + ndc.x * cam.right.xyz + ndc.y * cam.up.xyz);
  var p = u.f.xyz;
  let spin = u.a.y;
  let bright = u.a.z;
  let temp = u.a.w;
  let thick = u.b.y;
  let maxSteps = i32(u.b.z);
  let time = u.b.w;
  let rin = u.c.z;
  let rout = u.c.w;
  let n = u.d.xyz;
  let baseDt = u.d.w;
  let ex = vec3f(1.0, 0.0, 0.0);
  let ez = normalize(cross(ex, n));
  let J = n * (spin * 0.25);

  var Tr = 1.0;
  var col = vec3f(0.0);
  var status = 0;   // 0 out of steps, 1 captured, 2 escaped
  let Rb = max(rout * 2.2, length(p) * 1.02);

  for (var i = 0; i < maxSteps; i++) {
    let r2 = dot(p, p);
    let r = sqrt(r2);
    if (r < 1.0) { status = 1; break; }
    if (r > Rb && dot(p, v) > 0.0) { status = 2; break; }

    // Schwarzschild null geodesic in Cartesian form: a = -1.5 rs h^2 x / r^5
    let L = cross(p, v);
    let h2 = dot(L, L);
    var acc = -1.5 * h2 * p / (r2 * r2 * r);
    // gravitomagnetic (Lense-Thirring) frame dragging
    let rh = p / r;
    let Bg = (3.0 * dot(J, rh) * rh - J) / (r2 * r);
    acc += 2.0 * cross(v, Bg);

    // adaptive step: coarse far away, fine near the hole and inside the disc slab
    var dt = baseDt * (0.3 + 0.7 * r);
    dt = min(dt, 0.12 * (r - 0.85) + 0.01);
    let hh = dot(p, n);
    let rc = length(p - hh * n);
    if (rc < rout * 1.15 && rc > rin * 0.8) {
      let Hs = max(thick * 0.016 * rc, 0.004);
      let outside = max(abs(hh) - 3.0 * Hs, 0.0);
      let vh = abs(dot(v, n)) + 0.02;
      dt = min(dt, max(0.22 * Hs, 0.45 * outside / vh));
      dt = max(dt, 0.008);
    }

    // disc sample at the segment midpoint
    let pm = p + v * (0.5 * dt);
    let hm = dot(pm, n);
    let rvm = pm - hm * n;
    let rcm = length(rvm);
    let Hm = max(thick * 0.016 * rcm, 0.004);
    if (rcm > rin * 0.9 && rcm < rout && abs(hm) < 7.0 * Hm) {
      let zc = hm / Hm;
      var core = exp(-zc * zc);
      let cor = 0.07 * exp(-zc * zc / 6.25);
      let radial = smoothstep(rin * 0.9, rin * 1.08, rcm) * (1.0 - smoothstep(rout * 0.5, rout, rcm)) * pow(rin / rcm, 0.6);
      var nz = 0.5;
      if (core > 0.03) {
        let phi = atan2(dot(rvm, ez), dot(rvm, ex));
        let phis = phi + time * 2.2 * pow(rcm, -1.5);   // Keplerian shear: inner gas laps the outer
        nz = ringFbm(rcm, fract(phis / TAU));
        // large spiral arms: log-spiral twisted low-order mode
        let arm = rnoise(rcm * 1.6, fract(phis / TAU + 0.55 * log(rcm)), 6.0);
        nz = clamp(nz * (0.55 + 0.9 * arm), 0.0, 1.2);
        core *= 0.12 + 1.9 * pow(nz, 1.7);
      }
      let dens = (core + cor) * radial;
      let tau = 15.0 * dens * dt;
      if (tau > 1e-4) {
        // temperature profile of a thin disc
        let f = pow(rcm / rin, -0.75) * pow(max(1.0 - 0.75 * sqrt(rin / rcm), 0.0), 0.25) / 0.65;
        // emitter velocity (circular orbit) and combined Doppler + gravitational redshift
        let tang = cross(n, rvm) / rcm;
        let vmag = min(sqrt(0.5 / max(rcm - 1.0, 0.05)), 0.8);
        let gam = 1.0 / sqrt(1.0 - vmag * vmag);
        let gd = 1.0 / (gam * (1.0 + vmag * dot(tang, v)));
        let g = gd * sqrt(max(1.0 - 1.0 / rcm, 0.03));
        let Tcol = temp * f * pow(g, 0.9);
        let S = bright * f * f * pow(g, 3.6) * (0.25 + 1.6 * nz);
        let c = bbColor(Tcol) * S;
        let a = 1.0 - exp(-tau);
        col += Tr * a * c;
        Tr *= 1.0 - a;
        if (Tr < 0.004) { status = 1; break; }
      }
    }

    v = normalize(v + acc * dt);
    p += v * dt;
  }

  var o: FO;
  if (status == 2) {
    o.c = vec4f(col, Tr);
    o.d = vec4f(v, 1.0);
  } else {
    o.c = vec4f(col, 0.0);
    o.d = vec4f(0.0);
  }
  return o;
}
`;

export const COMP = COMMON + /* wgsl */ `
@group(0) @binding(2) var texA: texture_2d<f32>;
@group(0) @binding(3) var texD: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

fn ld(c: vec2i) -> vec4f {
  let d = vec2i(textureDimensions(texD));
  return textureLoad(texD, clamp(c, vec2i(0), d - vec2i(1)), 0);
}

const GN: vec3f = vec3f(0.2955, 0.9553, 0.0);          // galactic pole (normalised by hand)
const CD: vec3f = vec3f(-0.30, 0.10, -0.95);           // galactic-centre direction

struct Gal { col: vec3f, dens: f32, dust: f32 }

fn galaxy(dir: vec3f) -> Gal {
  let e1 = normalize(CD - GN * dot(CD, GN));
  let e2 = cross(GN, e1);
  let lon = atan2(dot(dir, e2), dot(dir, e1));
  let b = dot(dir, GN);
  let w = fbm3(dir * 3.0 + vec3f(1.7, 0.3, 2.9), 3) - 0.5;
  let bb = b + w * 0.16;
  let disc = exp(-pow(bb / 0.10, 2.0));
  let halo = exp(-pow(bb / 0.42, 2.0)) * 0.22;
  let bulge = exp(-lon * lon / 0.30 - pow(bb / 0.20, 2.0));
  let lonFall = 0.35 + 0.65 * exp(-lon * lon / 2.2);
  let st = fbm3(dir * 9.0 + vec3f(5.2, 1.1, 3.3), 5);
  let fine = fbm3(dir * 38.0 + vec3f(3.0, 8.0, 1.0), 3);
  let clouds = (0.12 + 2.0 * st * st) * (0.5 + 1.0 * fine);
  let dustN = fbm3(dir * 13.0 + vec3f(9.1, 2.2, 4.4), 5);
  let dust = smoothstep(0.46, 0.66, dustN) * exp(-pow(bb / 0.13, 2.0));
  let att = 1.0 - 0.9 * dust;
  let stellar = (disc * lonFall * clouds + halo * lonFall + 1.5 * bulge) * att;
  let warm = clamp(bulge * 1.2 + 0.4 * exp(-lon * lon / 1.2), 0.0, 1.0);
  var col = stellar * mix(vec3f(0.50, 0.66, 1.0), vec3f(1.0, 0.76, 0.48), warm);
  // emission nebulae near the plane
  let nb = fbm3(dir * 5.0 + vec3f(31.0, 7.0, 11.0), 4);
  let nmask = smoothstep(0.52, 0.78, nb) * exp(-pow(b / 0.38, 2.0));
  let nc = mix(vec3f(0.95, 0.22, 0.45), vec3f(0.12, 0.45, 0.95), smoothstep(0.35, 0.65, fbm3(dir * 3.0 + vec3f(2.0, 8.0, 4.0), 3)));
  col += nc * nmask * 0.22 * (0.4 + dust * 0.0 + 0.8 * st);
  var g: Gal;
  g.col = col * 0.05;
  g.dens = stellar;
  g.dust = att;
  return g;
}

fn facePoint(face: i32, uv: vec2f) -> vec3f {
  switch (face) {
    case 0: { return vec3f(1.0, uv.y, uv.x); }
    case 1: { return vec3f(-1.0, uv.y, uv.x); }
    case 2: { return vec3f(uv.x, 1.0, uv.y); }
    case 3: { return vec3f(uv.x, -1.0, uv.y); }
    case 4: { return vec3f(uv.x, uv.y, 1.0); }
    default: { return vec3f(uv.x, uv.y, -1.0); }
  }
}

// one layer of procedural stars on a cube-map grid; fp = sky footprint of this pixel (rad)
fn starLayer(dir: vec3f, N: f32, seed: f32, prob: f32, fluxBase: f32, fp: f32, sig0: f32, boost: f32) -> vec3f {
  let a = abs(dir);
  let m = max(a.x, max(a.y, a.z));
  var face = 5; var uv = vec2f(0.0);
  if (a.x == m) { face = select(1, 0, dir.x > 0.0); uv = vec2f(dir.z, dir.y) / m; }
  else if (a.y == m) { face = select(3, 2, dir.y > 0.0); uv = vec2f(dir.x, dir.z) / m; }
  else { face = select(5, 4, dir.z > 0.0); uv = vec2f(dir.x, dir.y) / m; }
  let cell = floor((uv * 0.5 + 0.5) * N);
  let key = vec3f(cell, f32(face) + seed * 7.0);
  let h = hash33(key);
  let ex = hash13(key + vec3f(3.1, 7.7, 1.3));
  let cellAng = 1.7 / N;
  let sigma2 = sig0 * sig0 + 0.25 * fp * fp;
  let p = prob * u.e.x * boost;
  // unresolved average radiance of this layer (used when the footprint spans many cells)
  let meanRad = p * fluxBase * 3.0 * sig0 * sig0 * N * N;
  let wide = smoothstep(0.12, 0.45, fp / cellAng);
  var res = vec3f(0.0);
  if (ex < p && wide < 0.999) {
    let suv = ((cell + 0.25 + 0.5 * h.xy) / N) * 2.0 - 1.0;
    let sdir = normalize(facePoint(face, suv));
    let d = dir - sdir;
    let d2 = dot(d, d);
    let F = fluxBase * (0.25 + 3.0 * pow(h.z, 3.0));
    let tk = pow(hash13(key + vec3f(11.0, 5.0, 2.0)), 1.6);
    let c = bbColor(mix(3200.0, 14000.0, tk));
    res = c * F * (sig0 * sig0 / sigma2) * exp(-0.5 * d2 / sigma2) * (1.0 - wide);
  }
  return res + vec3f(1.0, 0.94, 0.88) * meanRad * wide;
}

fn background(dir: vec3f, fp: f32) -> vec3f {
  let sig0 = 0.45 * max(u.f.w, u.g.x);
  let g = galaxy(dir);
  let boost = 1.0 + 5.0 * clamp(g.dens, 0.0, 1.5);
  var col = g.col;
  var st = vec3f(0.0);
  st += starLayer(dir, 22.0, 0.0, 0.40, 5.0, fp, sig0, 1.0);
  st += starLayer(dir, 58.0, 1.0, 0.30, 1.5, fp, sig0, boost * 0.5);
  st += starLayer(dir, 150.0, 2.0, 0.22, 0.55, fp, sig0, boost * 0.6);
  st += starLayer(dir, 380.0, 3.0, 0.12, 0.22, fp, sig0, boost * 0.7);
  st += starLayer(dir, 900.0, 4.0, 0.06, 0.12, fp, sig0, boost * 0.8);
  col += st * mix(1.0, g.dust, 0.8);
  // distant blue giant: lensed into an arc/ring when it lines up behind the hole
  let cd = normalize(vec3f(0.42, 0.16, -0.89));
  let dd = dir - cd;
  let d2 = dot(dd, dd);
  let s2 = sig0 * sig0 * 1.8 + 0.25 * fp * fp;
  col += vec3f(0.62, 0.78, 1.0) * 24.0 * (sig0 * sig0 * 1.8 / s2) * exp(-0.5 * d2 / s2);
  let hk = min(1.0, 4.0e-6 / (fp * fp + 1e-9));
  col += vec3f(0.55, 0.72, 1.0) * 0.0009 * hk / (d2 + 0.00006);
  return col;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let outRes = u.e.zw;
  let uv = in.pos.xy / outRes;
  let resA = u.c.xy;
  let ap = uv * resA - 0.5;
  let i0 = vec2i(floor(ap));
  let f = fract(ap);
  let d00 = ld(i0);
  let d10 = ld(i0 + vec2i(1, 0));
  let d01 = ld(i0 + vec2i(0, 1));
  let d11 = ld(i0 + vec2i(1, 1));
  let w00 = (1.0 - f.x) * (1.0 - f.y) * d00.w;
  let w10 = f.x * (1.0 - f.y) * d10.w;
  let w01 = (1.0 - f.x) * f.y * d01.w;
  let w11 = f.x * f.y * d11.w;
  let ws = w00 + w10 + w01 + w11;
  let ca = textureSampleLevel(texA, samp, uv, 0.0);
  var bg = vec3f(0.0);
  if (ws > 1e-4 && ca.a > 0.002) {
    let dir = normalize((d00.xyz * w00 + d10.xyz * w10 + d01.xyz * w01 + d11.xyz * w11) / ws);
    var fp = 0.0;
    if (d00.w > 0.5 && d10.w > 0.5) { fp = max(fp, length(d10.xyz - d00.xyz)); }
    if (d00.w > 0.5 && d01.w > 0.5) { fp = max(fp, length(d01.xyz - d00.xyz)); }
    if (d11.w > 0.5 && d10.w > 0.5) { fp = max(fp, length(d11.xyz - d10.xyz)); }
    if (d11.w > 0.5 && d01.w > 0.5) { fp = max(fp, length(d11.xyz - d01.xyz)); }
    fp = max(fp * u.e.y, 0.5 * u.g.x);
    bg = background(dir, fp);
  }
  return vec4f(ca.rgb + ca.a * bg, 1.0);
}
`;
