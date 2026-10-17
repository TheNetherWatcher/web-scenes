// Opaque world: sky + stars + moon + treeline, ground, grass blades, logs and stones.
import { FIRE_COMMON } from './wgsl_common.js';
import { FULLSCREEN_VS } from '../../engine/wgsl.js';

export const SKY = /* wgsl */ `
${FIRE_COMMON}
${FULLSCREEN_VS}

fn starField(rd: vec3f, t: f32) -> vec3f {
  let lat = asin(clamp(rd.y, -1.0, 1.0));
  let lon = atan2(rd.x, rd.z);
  let q = vec2f(lon * max(cos(lat), 0.05), lat) / 0.0105;
  let id = floor(q);
  let h = hash12(id);
  if (h > 0.12) { return vec3f(0.0); }
  let c = hash22(id + 3.3) * 0.5 + 0.25;
  let d = length(fract(q) - c);
  let mag = pow(hash12(id + 9.1), 5.0);
  let b = (0.12 + 3.2 * mag) * exp(-d * d / (2.0 * 0.11 * 0.11));
  let tw = 0.85 + 0.15 * sin(t * (2.0 + h * 30.0) + h * 80.0);
  let tint = mix(vec3f(1.0, 0.78, 0.6), vec3f(0.7, 0.8, 1.0), hash12(id + 1.7));
  return tint * b * tw * 0.55;
}

fn milkyWay(rd: vec3f) -> vec3f {
  let G = normalize(vec3f(0.35, 0.78, 0.52));
  let lat = dot(rd, G);
  let band = exp(-lat * lat * 14.0);
  let n1 = nz(rd * 1.6 + vec3f(0.3, 0.1, 0.7));
  let n2 = nz(rd * 4.1 + vec3f(0.9, 0.4, 0.2));
  let cloud = smoothstep(0.30, 0.85, n1.r * 0.65 + n2.g * 0.45);
  let dust = smoothstep(0.55, 0.80, n2.b) * smoothstep(0.2, 0.9, band);
  let core = exp(-pow(dot(rd, normalize(vec3f(-0.6, 0.45, 0.65))) - 1.0, 2.0) * 90.0);
  var col = vec3f(0.55, 0.62, 0.85) * band * cloud * 0.045 + vec3f(0.9, 0.75, 0.55) * core * band * 0.04;
  col *= 1.0 - dust * 0.65;
  return col;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let ro = cam.pos.xyz;
  let t = cam.pos.w;
  var rd = camRay(cam, in.pos.xy);

  // heat haze: wobble the view ray where it passes through the hot air above the fire
  let oc = scn.light.xyz - ro;
  let tc = dot(oc, rd);
  let hl = length(oc - rd * tc);
  let yc = ro.y + rd.y * tc;
  let hz = exp(-hl * hl * 2.5) * step(0.0, tc) / (1.0 + tc * 0.4) * smoothstep(0.1, 0.9, yc) * (1.0 - smoothstep(2.0, 4.0, yc));
  if (hz > 0.002) {
    let w = nz(vec3f(rd.x * 55.0, rd.y * 40.0 - t * 0.9, rd.z * 55.0)).rgb - 0.5;
    rd = normalize(rd + w * 0.03 * scn.env.z * hz);
  }

  var col = skyBase(rd);
  let tod = scn.env.x;
  let up = rd.y;

  if (up > -0.02) {
    // thin streaky clouds catching the last afterglow
    let pc = rd.xz / (up + 0.18);
    let cn = nz(vec3f(pc.x * 0.11 + 0.3, pc.y * 0.26, 0.2)).r * 0.65 + nz(vec3f(pc.x * 0.34, pc.y * 0.8, 0.6)).g * 0.35;
    let cm = smoothstep(0.5, 0.78, cn) * smoothstep(0.0, 0.22, up) * (1.0 - smoothstep(0.55, 1.0, up));
    let glowK = (1.0 - tod) * exp(-up * 2.2);
    let ccol = mix(scn.zen.rgb * 1.4 + vec3f(0.01), vec3f(0.85, 0.34, 0.16) * 0.5, glowK);
    col = mix(col, ccol, cm * 0.6);

    let sv = scn.zen.w * smoothstep(0.0, 0.1, up);
    col += (starField(rd, t) + milkyWay(rd)) * sv * (1.0 - cm * 0.8);

    // moon
    let md = dot(rd, scn.moon.xyz);
    let ang = acos(clamp(md, -1.0, 1.0));
    let mr = 0.017;
    let tdir = normalize(cross(vec3f(0.0, 1.0, 0.0), scn.moon.xyz));
    let bdir = cross(scn.moon.xyz, tdir);
    let muv = vec2f(dot(rd, tdir), dot(rd, bdir)) / mr;
    let surf = nz(vec3f(muv * 0.7 + 0.5, 0.15));
    let crat = nz(vec3f(muv * 3.2 + 0.2, 0.55));
    let alb = mix(0.95, 0.38, smoothstep(0.42, 0.58, surf.r) * 0.8) * (0.7 + 0.55 * crat.g) * (0.85 + 0.3 * crat.a);
    let shade = 0.78 + 0.22 * clamp(muv.x, -1.0, 1.0);
    let disc = 1.0 - smoothstep(mr * 0.96, mr, ang);
    let mv = scn.moon.w;
    col += vec3f(1.0, 0.95, 0.85) * alb * shade * 4.2 * disc * mv;
    col += vec3f(0.55, 0.65, 0.9) * (exp(-ang * ang / (2.0 * 0.05 * 0.05)) * 0.05 + exp(-ang * ang / (2.0 * 0.3 * 0.3)) * 0.012) * mv;
  }

  // far treeline silhouette
  let az = atan2(rd.x, -rd.z);
  let lump = nz(vec3f(az * 0.9, 0.3, 0.1)).r;
  let det = nz(vec3f(az * 7.0, 0.7, 0.4)).g;
  let fine = nz(vec3f(az * 34.0, 0.2, 0.6)).b;
  let treeH = 0.012 + 0.07 * lump * lump + 0.014 * det + 0.008 * fine;
  let edge = max(fwidth(rd.y), 1e-4) * 1.3;
  let tm = 1.0 - smoothstep(treeH - edge, treeH + edge, rd.y);
  let tcol = mix(treeColor(), skyBase(rd) * 0.35, smoothstep(0.55, 1.0, rd.y / treeH) * 0.6);
  col = mix(col, tcol, tm);

  // fire light scattered by mist along the whole ray
  let tmax = min(500.0, 28.0 / max(rd.y, 0.02));
  col += fireScatter(ro, rd, tmax) * (1.0 - tm * 0.6) * 0.45;
  return vec4f(col, 1.0);
}
`;

export const GROUND = /* wgsl */ `
${FIRE_COMMON}
struct GO { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> GO {
  let N = 256u;
  let row = vi / (N + 1u);
  let ci = vi % (N + 1u);
  let g = vec2f(f32(ci), f32(row)) / f32(N) * 2.0 - 1.0;
  let w = sign(g) * pow(abs(g), vec2f(3.0)) * 3000.0;
  let wp = vec3f(w.x, groundH(w.x, w.y), w.y);
  var o: GO;
  o.pos = cam.viewProj * vec4f(wp - cam.pos.xyz, 1.0);
  o.wp = wp;
  return o;
}
@fragment fn fs(in: GO) -> @location(0) vec4f {
  let wp = in.wp;
  let t = cam.pos.w;
  let e = 0.15;
  let n = normalize(vec3f(groundH(wp.x - e, wp.z) - groundH(wp.x + e, wp.z), 2.0 * e, groundH(wp.x, wp.z - e) - groundH(wp.x, wp.z + e)));
  let n0 = nz(vec3f(wp.xz * 0.05, 0.3));
  let n1 = nz(vec3f(wp.xz * 0.45, 0.6));
  let n2 = nz(vec3f(wp.xz * 3.1, 0.1));
  let n3 = nz(vec3f(wp.xz * 14.0, 0.9));
  var alb = mix(vec3f(0.032, 0.040, 0.016), vec3f(0.085, 0.078, 0.034), n0.r * 0.6 + n1.g * 0.5);
  alb = mix(alb, vec3f(0.05, 0.034, 0.02), smoothstep(0.55, 0.8, n1.b) * 0.5);
  alb *= 0.65 + 0.7 * n2.r * (0.6 + 0.8 * n3.g);

  // ash bed + glowing coals
  let r = length(wp.xz);
  let ashM = 1.0 - smoothstep(0.62, 1.2, r + (n1.r - 0.5) * 0.55 + (n2.g - 0.5) * 0.2);
  let ashC = mix(vec3f(0.030, 0.028, 0.026), vec3f(0.11, 0.105, 0.10), n2.b * n3.r);
  alb = mix(alb, ashC, ashM);
  let cell = nz(vec3f(wp.xz * 4.4, 0.35)).a;
  let cb = smoothstep(0.52, 0.86, cell) * (1.0 - smoothstep(0.35, 0.95, r)) * smoothstep(0.05, 0.9, ashM);
  let pulse = 0.55 + 0.45 * nz(vec3f(wp.xz * 2.2, t * 0.25)).b + 0.2 * sin(t * 7.0 + cell * 40.0);
  let emis = flameRamp(0.30 + 0.28 * pulse) * cb * pulse * 1.6;

  let sh = fireShadow(wp + vec3f(0.0, 0.03, 0.0));
  var lit = fireIrr(wp, n) * mix(0.2, 1.0, sh);
  lit += scn.amb.rgb * (0.55 + 0.45 * n.y);
  lit += scn.moonCol.rgb * saturate(dot(n, scn.moon.xyz));
  var col = alb * lit + emis;
  col = applyFog(col, wp);
  return vec4f(col, 1.0);
}
`;

export const GRASS = /* wgsl */ `
${FIRE_COMMON}
const SEG: u32 = 5u;
struct GOut { @builtin(position) pos: vec4f, @location(0) col: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> GOut {
  let t = cam.pos.w;
  let row = min(vi / 2u, SEG);
  let tip = vi == 2u * SEG;
  let side = f32(vi & 1u) * 2.0 - 1.0;
  let u = f32(row) / f32(SEG);
  let root2 = inst.xy;
  let yaw = inst.z;
  let sd = inst.w;
  let gh = groundH(root2.x, root2.y);
  let dcam = length(root2 - cam.pos.xz);
  let clump = nz(vec3f(root2 * 0.09, 0.4));
  let fade = 1.0 - smoothstep(30.0, 52.0, dcam);
  let h = (0.2 + 0.5 * pow(hash11(sd * 91.7), 1.6)) * (0.55 + 0.9 * clump.r) * fade;
  let wdt = 0.019 * (0.7 + 0.6 * hash11(sd * 13.1));
  let dir = vec2f(cos(yaw), sin(yaw));
  let perp = vec2f(-dir.y, dir.x);
  let lean = (0.1 + 0.4 * hash11(sd * 37.3)) * h;

  let wv = scn.wind.xy;
  let ws = length(wv);
  let wdir = wv / max(ws, 1e-3);
  let phase = dot(root2, wdir) * 0.35 - t * (1.4 + ws * 2.5) + clump.g * 6.0;
  let gust = 0.5 + 0.5 * sin(phase) * (0.6 + 0.4 * sin(phase * 0.37 + t * 0.3));
  let sway = wv * (0.18 + 0.5 * gust) * scn.wind.w;
  let flutter = vec2f(sin(t * 4.1 + sd * 50.0), cos(t * 3.7 + sd * 31.0)) * 0.012 * (0.3 + ws);
  let bendXZ = (dir * lean + sway * h) * u * u + flutter * u;
  let dBend = (dir * lean + sway * h) * 2.0 * u + flutter;
  let wscale = select((1.0 - u * u) * 0.9 + 0.1, 0.0, tip);
  let xz = root2 + bendXZ + perp * side * wdt * wscale;
  let y = gh + h * u * (1.0 - 0.18 * u);
  let wp = vec3f(xz.x, y, xz.y);

  let tang = normalize(vec3f(dBend.x, h * (1.0 - 0.36 * u) + 1e-3, dBend.y));
  var n = normalize(cross(vec3f(perp.x, 0.0, perp.y), tang));
  let toCam = cam.pos.xyz - wp;
  if (dot(n, toCam) < 0.0) { n = -n; }
  n = normalize(mix(n, vec3f(0.0, 1.0, 0.0), 0.4));

  let var1 = hash11(sd * 7.7);
  let rootC = vec3f(0.018, 0.026, 0.010);
  var tipC = mix(vec3f(0.075, 0.115, 0.030), vec3f(0.16, 0.125, 0.05), smoothstep(0.55, 0.95, var1));
  tipC = mix(tipC, vec3f(0.05, 0.08, 0.025), clump.b * 0.5);
  let alb = mix(rootC, tipC, pow(u, 0.75));
  let ao = mix(0.22, 1.0, pow(u, 0.7));

  let L = scn.light.xyz;
  let dl = L - wp;
  let r2 = dot(dl, dl);
  let ldir = dl * inverseSqrt(r2);
  var sh = 1.0;
  if (r2 < 64.0) { sh = fireShadow(wp + vec3f(0.0, 0.05, 0.0)); }
  var irr = fireIrr(wp, n) * mix(0.2, 1.0, sh);
  // translucency: light shining through the blade towards the viewer
  let vdir = normalize(-toCam);
  let back = pow(saturate(dot(ldir, vdir)), 2.5) * 0.55;
  irr += scn.lightCol.rgb * scn.light.w * back * sh / (r2 + 0.3) * 0.8;
  var lit = irr * (0.35 + 0.65 * ao);
  lit += scn.amb.rgb * ao * (0.6 + 0.4 * n.y);
  lit += scn.moonCol.rgb * (0.35 + 0.65 * saturate(dot(n, scn.moon.xyz))) * ao;
  var col = alb * lit;
  col = applyFog(col, wp);

  var o: GOut;
  o.pos = cam.viewProj * vec4f(wp - cam.pos.xyz, 1.0);
  if (fade <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
  o.col = col;
  return o;
}
@fragment fn fs(in: GOut) -> @location(0) vec4f { return vec4f(in.col, 1.0); }
`;

export const SOLID = /* wgsl */ `
${FIRE_COMMON}
struct SO {
  @builtin(position) pos: vec4f,
  @location(0) wp: vec3f,
  @location(1) n: vec3f,
  @location(2) uv: vec2f,
  @location(3) ms: vec2f,
};
@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f, @location(3) ms: vec2f) -> SO {
  var o: SO;
  o.pos = cam.viewProj * vec4f(p - cam.pos.xyz, 1.0);
  o.wp = p; o.n = n; o.uv = uv; o.ms = ms;
  return o;
}

fn barkH(uv: vec2f, seed: f32) -> f32 {
  let q = vec3f(cos(uv.x) * 0.55 + seed * 0.13, sin(uv.x) * 0.55 + seed * 0.07, uv.y * 1.3);
  let a = nz(q).r;
  let b = nz(q * vec3f(3.1, 3.1, 1.4) + 0.37).g;
  let c = nz(q * vec3f(8.0, 8.0, 3.0) + 0.71).b;
  return a * 0.55 + b * 0.35 + c * 0.1;
}

@fragment fn fs(in: SO) -> @location(0) vec4f {
  let t = cam.pos.w;
  let wp = in.wp;
  var n = normalize(in.n);
  let mat = in.ms.x;
  let seed = in.ms.y;
  let isBark = mat < 0.5;
  let isStone = mat > 0.5 && mat < 1.5;

  // surface height used for bump + cracks (derivatives kept in uniform control flow)
  var hh = barkH(in.uv, seed);
  let sp = wp * 3.2 + seed;
  let sh0 = nz(sp * 0.7).r * 0.6 + nz(sp * 2.6).g * 0.4;
  hh = select(hh, sh0, isStone);
  let rr = length(in.uv);
  let ringsN = nz(vec3f(in.uv * 0.3 + seed, 0.2)).r;
  hh = select(hh, 0.5 + 0.12 * sin(rr * 55.0 + ringsN * 5.0), mat > 1.5);

  let dpx = dpdx(wp); let dpy = dpdy(wp);
  let dhx = dpdx(hh); let dhy = dpdy(hh);
  let r1 = cross(dpy, n); let r2 = cross(n, dpx);
  let det = dot(dpx, r1);
  let grad = sign(det) * (dhx * r1 + dhy * r2);
  let bump = select(0.006, 0.012, isStone);
  n = normalize(abs(det) * n - grad * bump);

  // heat from the fire core
  let core = length(wp.xz);
  let heat = exp(-core * 2.0) * (0.3 + 0.7 * exp(-wp.y * 1.3));
  let flick = nz(vec3f(wp.xz * 2.4 + seed * 0.1, t * 0.45)).b;
  let crack = smoothstep(0.50, 0.30, hh);
  let glow = saturate(crack * heat * (0.45 + 0.9 * flick) * 1.5);

  let charM = saturate(heat * 1.6 + 0.15 * flick);
  var alb = mix(vec3f(0.060, 0.038, 0.024), vec3f(0.11, 0.075, 0.045), hh);
  alb = mix(alb, vec3f(0.012, 0.011, 0.010), charM * 0.88);
  alb += vec3f(0.06, 0.058, 0.055) * smoothstep(0.55, 0.8, hh) * charM * 0.5;
  // stone
  var stoneC = mix(vec3f(0.055, 0.052, 0.048), vec3f(0.12, 0.115, 0.105), sh0);
  stoneC = mix(stoneC, vec3f(0.012), saturate(heat * 1.8) * 0.8);
  // end grain
  let endC = mix(vec3f(0.17, 0.10, 0.05), vec3f(0.30, 0.19, 0.09), hh) * (1.0 - charM * 0.82);
  alb = select(alb, stoneC, isStone);
  alb = select(alb, endC, mat > 1.5);

  var emis = flameRamp(0.18 + 0.5 * glow) * glow * select(1.8, 0.0, isStone);
  emis += select(vec3f(0.0), flameRamp(0.2 + 0.4 * heat) * heat * saturate(rr) * 1.2 * (0.6 + flick), mat > 1.5);

  let ao = mix(0.28, 1.0, smoothstep(0.0, 0.45, wp.y)) * mix(0.55, 1.0, saturate(core * 1.5));
  var lit = fireIrr(wp, n) * (0.55 + 0.45 * ao);
  lit += scn.amb.rgb * ao * (0.6 + 0.4 * n.y);
  lit += scn.moonCol.rgb * saturate(dot(n, scn.moon.xyz)) * ao;
  var col = alb * lit + emis;
  col = applyFog(col, wp);
  return vec4f(col, 1.0);
}
`;
