// Car, road and sky shaders. Everything lit by one analytic dusk environment (env()) so the
// body, glass, wheels and the wet road all reflect the same sky.
import { CAM_WGSL, NOISE_WGSL, FULLSCREEN_VS } from '../../engine/wgsl.js';

const ENV = /* wgsl */ `
${CAM_WGSL}
${NOISE_WGSL}
struct Env {
  sunDir: vec4f,   // xyz direction to the sun
  sunCol: vec4f,   // rgb sun tint, w = sun elevation (rad)
  zen: vec4f,      // zenith rgb
  hor: vec4f,      // horizon rgb, w = day amount
  paint: vec4f,    // rgb body paint (linear), w = flake amount
  opts: vec4f,     // x headlights, y floor reflectivity, z rim style, w mirror pass
  opts2: vec4f,    // x road scroll (m), y time, z, w
}
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<uniform> E: Env;
const SUN_E: f32 = 6.0;

fn hillH(az: f32) -> f32 { return 0.020 + 0.014 * sin(az * 2.0 + 0.7) + 0.010 * sin(az * 5.0 + 2.0) + 0.005 * sin(az * 13.0 + 1.0); }

fn skyOnly(d: vec3f, rough: f32) -> vec3f {
  let L = E.sunDir.xyz;
  let sd = dot(d, L);
  let t = clamp(d.y, 0.0, 1.0);
  var c = mix(E.hor.rgb, E.zen.rgb, 1.0 - exp(-t * 6.0));
  let sdp = max(sd, 0.0);
  c += E.sunCol.rgb * (pow(sdp, 4.0) * 0.10 + pow(sdp, 24.0) * 0.30) * (1.0 - 0.7 * t);
  c += E.hor.rgb * 0.35 * exp(-t * 18.0);
  // soft cloud streaks lit from below by the sun
  let az = atan2(d.z, d.x);
  let cl = fbm3(vec3f(d.x * 2.0, d.y * 16.0, d.z * 2.0) + vec3f(4.0, 1.0, 2.0), 5);
  let cm = smoothstep(0.52, 0.78, cl) * smoothstep(0.015, 0.12, d.y) * (1.0 - smoothstep(0.25, 0.6, d.y)) * (1.0 - rough);
  c = mix(c, c * 0.6 + E.hor.rgb * 0.45 * (0.5 + 0.5 * pow(sdp, 2.0)) + E.sunCol.rgb * 0.2 * pow(sdp, 3.0), cm * 0.8);
  // low hills on the horizon
  let bandM = exp(-pow((d.y - 0.075) / 0.018, 2.0)) * smoothstep(0.35, 0.65, fbm3(vec3f(d.x * 4.0, d.y * 30.0, d.z * 4.0), 3)) * (1.0 - rough);
  c += E.hor.rgb * 1.4 * bandM * (0.4 + 0.6 * pow(sdp, 2.0));
  let hh = hillH(az);
  let blur = 0.004 + 0.25 * rough;
  let hm = smoothstep(hh + blur, hh - blur, d.y);
  let hillCol = mix(E.hor.rgb * 0.22, vec3f(0.010, 0.014, 0.024), smoothstep(0.0, 0.05, hh - d.y));
  c = mix(c, hillCol, hm);
  let w = 0.00004 + rough * rough * 0.45;
  let amp = SUN_E / (TAU * w);
  let vis = (1.0 - hm) * smoothstep(-0.01, 0.01, d.y);
  c += E.sunCol.rgb * amp * exp(-(1.0 - sd) / w) * vis;
  return c;
}
// environment for reflections: sky above the horizon, dark road below
fn envDir(d: vec3f, rough: f32) -> vec3f {
  let s = smoothstep(-0.02 - 0.3 * rough, 0.02 + 0.3 * rough, d.y);
  let g = mix(E.hor.rgb * 0.12, vec3f(0.012, 0.011, 0.011), clamp(-d.y * 4.0, 0.0, 1.0));
  return mix(g, skyOnly(d, rough), s);
}
fn irr(n: vec3f) -> vec3f {
  let L = E.sunDir.xyz;
  let amb = mix(E.hor.rgb, E.zen.rgb, 0.55) * 0.8;
  var c = mix(vec3f(0.02, 0.018, 0.016) + E.hor.rgb * 0.05, amb, sat(n.y * 0.5 + 0.5));
  c += E.hor.rgb * 0.25 * sat(dot(n, normalize(vec3f(L.x, 0.0, L.z) + vec3f(1e-4, 0.0, 0.0))));
  return c;
}
fn fres(ndv: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0); }

fn paintShade(base: vec3f, flakeAmt0: f32, N: vec3f, V: vec3f, ao: f32, wp: vec3f, fp: f32) -> vec3f {
  let L = E.sunDir.xyz;
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  let fc = fres(ndv, 0.04);
  let so = mix(1.0, ao, 0.75);
  let coat = envDir(R, 0.03) * fc * so;
  let cell = floor(wp * 1400.0);
  let hv = hash33(cell);
  let nf = normalize(N + (hv - 0.5) * 0.9);
  let flakeAmt = flakeAmt0 * (1.0 - smoothstep(0.0015, 0.006, fp));
  let flake = envDir(reflect(-V, nf), 0.12) * pow(hv.x, 3.0) * flakeAmt * 0.45;
  let metal = base * envDir(R, 0.30) * 0.5;
  let ndl = sat(dot(N, L));
  let diff = base * (irr(N) * ao + E.sunCol.rgb * SUN_E * ndl / PI) * 0.55;
  return coat + (1.0 - fc) * (diff + (metal + base * flake) * so);
}

// projection frame for procedural surface patterns, by dominant normal axis
struct Proj { uv: vec2f, a: vec3f, b: vec3f }
fn projOf(lp: vec3f, N: vec3f) -> Proj {
  let an = abs(N);
  if (an.y > an.x && an.y > an.z) { return Proj(lp.xz, vec3f(1, 0, 0), vec3f(0, 0, 1)); }
  if (an.x > an.z) { return Proj(lp.zy, vec3f(0, 0, 1), vec3f(0, 1, 0)); }
  return Proj(lp.xy, vec3f(1, 0, 0), vec3f(0, 1, 0));
}

fn carbonShade(N: vec3f, V: vec3f, ao: f32, lp: vec3f, fp: f32) -> vec3f {
  let pr = projOf(lp, N);
  let tow = 0.0042;
  let g = pr.uv / tow;
  let ci = floor(g); let f = fract(g);
  let k = (i32(ci.x) + i32(ci.y)) & 3;
  let vert = k < 2;
  let s = select(f.y, f.x, vert);
  let lod = 1.0 - smoothstep(0.0008, 0.0035, fp);
  let tilt = cos(PI * s) * 0.55 * lod;
  let Nw = normalize(N + select(pr.b, pr.a, vert) * tilt);
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  let Rw = reflect(-V, Nw);
  let fc = fres(ndv, 0.04);
  let so = mix(1.0, ao, 0.75);
  let fibre = mix(1.0, 0.55 + 0.45 * sin(PI * s), lod);
  let L = E.sunDir.xyz;
  let diff = vec3f(0.012) * fibre * (irr(N) * ao + E.sunCol.rgb * SUN_E * sat(dot(N, L)) / PI);
  let sheen = envDir(Rw, 0.42) * 0.12 * fibre;
  return envDir(R, 0.03) * fc * so + (1.0 - fc) * (diff + sheen * so);
}

fn glassShade(N: vec3f, V: vec3f, ao: f32) -> vec3f {
  let ndv = max(dot(N, V), 1e-3);
  let R = reflect(-V, N);
  let fg = fres(ndv, 0.08);
  let inside = vec3f(0.004, 0.0055, 0.008) + E.hor.rgb * 0.012 * (1.0 - ndv);
  return envDir(R, 0.015) * fg + (1.0 - fg) * inside * mix(0.4, 1.0, ao);
}

fn cavShade(N: vec3f, V: vec3f, ao: f32, lp: vec3f) -> vec3f {
  let pr = projOf(lp, N);
  let q = pr.uv * (TAU / 0.011);
  let f = cos(q.x) + cos(dot(q, vec2f(-0.5, 0.866))) + cos(dot(q, vec2f(-0.5, -0.866)));
  let hole = smoothstep(1.2, 1.9, f);
  let mesh = 1.0 - hole;
  let R = reflect(-V, N);
  return vec3f(0.004) * (0.3 + hole) + mesh * (envDir(R, 0.5) * 0.02 + vec3f(0.012) * irr(N)) * ao;
}

fn gapLines(lp: vec3f, N: vec3f, fp: f32) -> f32 {
  let ap = abs(vec3f(lp.x, lp.y, abs(lp.z)));
  var d = 1e3;
  let sideF = smoothstep(0.4, 0.7, abs(N.z));
  let topF = smoothstep(0.4, 0.7, N.y);
  // door outline
  let q2 = vec2f(lp.x - 0.19, lp.y - 0.55);
  let dd = abs(q2) - vec2f(0.58, 0.245);
  let sdDoor = length(max(dd, vec2f(0.0))) + min(max(dd.x, dd.y), 0.0);
  d = min(d, mix(1e3, abs(sdDoor), sideF * smoothstep(0.7, 0.8, abs(lp.z))));
  // hood and engine cover shut lines
  let hq = abs(vec2f(lp.x - 1.58, abs(lp.z))) - vec2f(0.48, 0.70);
  d = min(d, mix(1e3, abs(length(max(hq, vec2f(0.0))) + min(max(hq.x, hq.y), 0.0)), topF));
  let eq = abs(vec2f(lp.x + 1.40, abs(lp.z))) - vec2f(0.46, 0.60);
  d = min(d, mix(1e3, abs(length(max(eq, vec2f(0.0))) + min(max(eq.x, eq.y), 0.0)), topF));
  return 1.0 - smoothstep(0.0011, 0.0011 + fp * 1.2, d);
}

struct VIn {
  @location(0) pos: vec3f, @location(1) nrm: vec3f, @location(2) m1: vec4f, @location(3) m2: vec2f,
  @location(4) i0: vec4f, @location(5) i1: vec4f, @location(6) i2: vec4f, @location(7) i3: vec4f,
}
struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) wp: vec3f, @location(1) n: vec3f, @location(2) m1: vec4f, @location(3) m2: vec2f,
  @location(4) lp: vec3f, @location(5) ln: vec3f,
  @location(6) tx: vec3f, @location(7) ty: vec3f, @location(8) tz: vec3f,
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
  o.tx = normalize(v.i0.xyz); o.ty = normalize(v.i1.xyz); o.tz = normalize(v.i2.xyz);
  return o;
}
fn viewVec(wp: vec3f) -> vec3f {
  var cp = cam.pos.xyz;
  if (E.opts.w > 0.5) { cp.y = -cp.y; }
  return normalize(cp - wp);
}

@fragment fn fs_body(in: VOut) -> @location(0) vec4f {
  let V = viewVec(in.wp);
  var N = normalize(in.n);
  if (dot(N, V) < 0.0) { N = -N; }
  let fp = length(fwidth(in.wp));
  let glass = smoothstep(0.4, 0.6, in.m1.x);
  let carbon = smoothstep(0.4, 0.6, in.m1.y);
  let cav = smoothstep(0.4, 0.6, in.m1.z);
  let lamp = smoothstep(0.4, 0.6, in.m1.w);
  let trim = smoothstep(0.4, 0.6, in.m2.y);
  var ao = in.m2.x * mix(0.35, 1.0, smoothstep(0.02, 0.32, in.wp.y));
  let gap = gapLines(in.lp, N, fp);
  var col = paintShade(E.paint.rgb, E.paint.w, N, V, ao, in.lp, fp);
  col = mix(col, carbonShade(N, V, ao, in.lp, fp), carbon);
  col *= mix(1.0, 0.12, gap * (1.0 - glass));
  if (trim > 0.01) { col = mix(col, paintShade(vec3f(0.006), 0.0, N, V, ao, in.lp, fp), trim); }
  if (glass > 0.01) { col = mix(col, glassShade(N, V, ao), glass); }
  if (cav > 0.01) { col = mix(col, cavShade(N, V, ao, in.lp), cav); }
  if (lamp > 0.01) {
    let ndv = max(dot(N, V), 1e-3);
    let lens = envDir(reflect(-V, N), 0.02) * fres(ndv, 0.05);
    var em: vec3f;
    if (in.lp.x > 0.0) { em = vec3f(0.85, 0.93, 1.0) * (0.25 + 12.0 * E.opts.x); }
    else { em = vec3f(1.0, 0.025, 0.012) * (1.2 + 9.0 * (0.35 + 0.65 * E.opts.x)); }
    col = mix(col, em + lens, lamp);
  }
  return vec4f(col, 1.0);
}

@fragment fn fs_wheel(in: VOut) -> @location(0) vec4f {
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
  // cross-drilled disc: real see-through holes
  if (disc > 0.5 && r > 0.12 && r < 0.186) {
    let stp = TAU / 28.0;
    let row = select(0.136, 0.168, (i32(floor(ang / stp)) & 1) == 0);
    let a2 = (fract(ang / stp) - 0.5) * stp * r;
    if (length(vec2f(a2, r - row)) < 0.0056) { discard; }
  }
  // dark under the fender and inside the arch
  var ao = in.m2.x * mix(0.18, 1.0, smoothstep(0.62, 0.20, in.wp.y)) * mix(0.55, 1.0, smoothstep(0.0, 0.2, in.wp.y));
  var col: vec3f;
  let L = E.sunDir.xyz;
  if (tire > 0.5 && rim < 0.5) {
    // tread blocks and sidewall rings
    let onTread = smoothstep(0.65, 0.35, abs(in.ln.z));
    let lod = 1.0 - smoothstep(0.002, 0.008, fp);
    let gz = abs(abs(lp.z) - 0.045);
    let g1 = 1.0 - smoothstep(0.004, 0.006, gz);
    let g2 = 1.0 - smoothstep(0.004, 0.006, abs(abs(lp.z) - 0.105));
    let sipe = smoothstep(0.42, 0.49, abs(fract(ang * 56.0 / TAU + lp.z * 6.0) - 0.5));
    let groove = max(max(g1, g2) * onTread, sipe * onTread * 0.55) * lod;
    let rings = 0.5 + 0.5 * sin(r * 520.0);
    let side = (1.0 - onTread) * lod;
    let alb = vec3f(0.018) * (1.0 - 0.7 * groove) * (1.0 - 0.18 * rings * side);
    let ndv = max(dot(N, V), 1e-3);
    let R = reflect(-V, N);
    col = alb * (irr(N) * ao + E.sunCol.rgb * SUN_E * sat(dot(N, L)) / PI) + envDir(R, 0.62) * fres(ndv, 0.035) * 0.45 * mix(1.0, ao, 0.8) * (1.0 - 0.8 * groove);
  } else if (rim > 0.5 || (tire <= 0.5 && disc <= 0.5 && cal <= 0.5)) {
    let ndv = max(dot(N, V), 1e-3);
    let R = reflect(-V, N);
    let base = vec3f(0.55, 0.52, 0.47);
    let lip = smoothstep(0.236, 0.250, r) * smoothstep(0.08, 0.15, lp.z);
    let bc = mix(base, vec3f(0.75, 0.72, 0.68), lip * 0.8);
    col = bc * envDir(R, 0.26) * mix(1.0, ao, 0.8) * 0.85 + bc * irr(N) * ao * 0.15 + envDir(R, 0.04) * fres(ndv, 0.04) * 0.5 * ao;
  } else if (disc > 0.5) {
    let ndv = max(dot(N, V), 1e-3);
    let R = reflect(-V, N);
    let brush = 0.82 + 0.18 * hash11(floor(r * 1400.0));
    var bc = vec3f(0.42, 0.41, 0.40) * brush;
    if (r < 0.118) { bc = vec3f(0.04); }
    col = bc * envDir(R, 0.34) * mix(1.0, ao, 0.8) * 0.9 + bc * irr(N) * ao * 0.1;
  } else {
    col = paintShade(vec3f(0.55, 0.012, 0.01), 0.1, N, V, ao, in.wp, fp);
  }
  return vec4f(col, 1.0);
}

@fragment fn fs_sky(in: VO) -> @location(0) vec4f {
  var d = camRay(cam, in.pos.xy);
  d.y = abs(d.y);          // below the horizon: mirrored sky (the road reflects it)
  return vec4f(skyOnly(d, 0.0), 1.0);
}
${FULLSCREEN_VS}

struct GV { @builtin(position) clip: vec4f, @location(0) wp: vec3f }
@vertex fn gvs(@builtin(vertex_index) i: u32) -> GV {
  var p = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  let w = vec3f(cam.pos.x + p[i].x * 4000.0, 0.0, cam.pos.z + p[i].y * 4000.0);
  var o: GV;
  o.clip = cam.viewProj * vec4f(w - cam.pos.xyz, 1.0);
  o.wp = w;
  return o;
}

fn sdBox2(p: vec2f, b: vec2f) -> f32 { let d = abs(p) - b; return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0); }
// rough proxy of the car for the road's contact shadow / sun shadow
fn carProxy(p: vec3f) -> f32 {
  let a = sdBox(p - vec3f(0.0, 0.45, 0.0), vec3f(1.9, 0.30, 0.82)) - 0.16;
  let b = sdBox(p - vec3f(-0.15, 0.88, 0.0), vec3f(0.75, 0.14, 0.50)) - 0.14;
  return min(a, b);
}
fn sdBox(p: vec3f, b: vec3f) -> f32 { let q = abs(p) - b; return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0); }

@fragment fn gfs(in: GV) -> @location(0) vec4f {
  let p = in.wp;
  let V = normalize(cam.pos.xyz - p);
  let L = E.sunDir.xyz;
  let fpx = length(fwidth(p.xz));
  let sc = vec2f(p.x + E.opts2.x, p.z);
  // asphalt: aggregate + large scale stains
  let agg = vnoise2(sc * 220.0);
  let stains = fbm2(sc * 0.7, 4);
  var alb = vec3f(0.050, 0.048, 0.047) * (0.55 + 0.9 * stains) * mix(0.8, 1.2, agg);
  let speck = step(0.93, hash12(floor(sc * 160.0))) * (1.0 - smoothstep(0.004, 0.02, fpx));
  alb += vec3f(0.06) * speck;
  // lane paint: dashed centre lines at +-3.6 m, solid edge lines at +-7.2 m
  let dash = step(fract(sc.x / 9.0), 0.33);
  let ln1 = (1.0 - smoothstep(0.06, 0.06 + fpx * 1.5, abs(abs(p.z) - 3.6))) * dash;
  let ln2 = 1.0 - smoothstep(0.08, 0.08 + fpx * 1.5, abs(abs(p.z) - 7.2));
  let paintM = max(ln1, ln2) * (0.55 + 0.45 * stains);
  alb = mix(alb, vec3f(0.55, 0.53, 0.48), paintM);
  // wetness: puddles and a damp film
  let wetN = fbm2(sc * 0.32 + vec2f(3.1, 7.7), 5);
  let wet = clamp(smoothstep(0.38, 0.55, wetN) * 0.85 + 0.35, 0.0, 1.0);
  let wetAlb = alb * mix(1.0, 0.45, wet);
  let ndv = max(V.y, 1e-3);
  let F = fres(ndv, 0.02);
  var Rw = F * mix(0.20, 1.0, wet) * E.opts.y;
  Rw = clamp(max(Rw, 0.05 * E.opts.y), 0.0, 1.0);
  Rw *= 1.0 - 0.75 * paintM;
  // contact ambient occlusion and soft sun shadow from the car
  let dc = carProxy(vec3f(p.x, 0.0, p.z));
  let aoG = mix(0.06, 1.0, 1.0 - exp(-max(dc, 0.0) * 3.2));
  var sh = 1.0;
  var t = 0.03;
  for (var i = 0; i < 20; i++) {
    let hp = p + L * t + vec3f(0.0, 0.002, 0.0);
    let h = carProxy(hp);
    sh = min(sh, 9.0 * h / t);
    t += clamp(h, 0.04, 0.5);
    if (t > 7.0 || sh < 0.01) { break; }
  }
  sh = clamp(sh, 0.0, 1.0);
  let ndl = L.y;
  let sunD = E.sunCol.rgb * SUN_E * max(ndl, 0.0) / PI * mix(sh, 1.0, 0.0);
  let diff = wetAlb * (irr(vec3f(0.0, 1.0, 0.0)) * aoG + sunD);
  let rough = envDir(reflect(-V, vec3f(0.0, 1.0, 0.0)) * vec3f(1.0, 1.0, 1.0), 0.5) * F * 0.5 * (1.0 - wet) * aoG;
  let src = diff * (1.0 - Rw) + rough;
  // fade into the sky colour at the horizon
  let far = smoothstep(300.0, 2500.0, length(p.xz - cam.pos.xz));
  let a = clamp(mix(Rw * mix(0.25, 1.0, aoG), 1.0, far), 0.0, 1.0);
  return vec4f(src * (1.0 - far), a);
}
`;

export const CAR_SHADER = ENV;
