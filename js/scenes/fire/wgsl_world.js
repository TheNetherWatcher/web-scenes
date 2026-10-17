// Opaque world: sky + stars + moon + treeline, ground, grass blades, logs and stones.
import { FIRE_COMMON } from './wgsl_common.js';
import { FULLSCREEN_VS } from '../../engine/wgsl.js';

export const SKY = /* wgsl */ `
${FIRE_COMMON}
${FULLSCREEN_VS}

fn starField(rd: vec3f, t: f32) -> vec3f {
  // isotropic: 3-D cells on a sphere shell, one possible star per cell
  let p = rd * 118.0;
  let id = floor(p);
  let h = hash13(id);
  if (h > 0.34) { return vec3f(0.0); }
  let c = hash33(id + 3.3) * 0.5 + 0.25;
  let q = fract(p) - c;
  let d2 = dot(q, q);
  let mag = pow(hash13(id + 9.1), 4.0);
  let b = (0.10 + 3.4 * mag) * exp(-d2 / (2.0 * 0.095 * 0.095));
  let tw = 0.85 + 0.15 * sin(t * (2.0 + h * 30.0) + h * 80.0);
  let tint = mix(vec3f(1.0, 0.78, 0.6), vec3f(0.7, 0.8, 1.0), hash13(id + 1.7));
  return tint * b * tw * 0.6;
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

  // bare, trodden soil around the pit: grit, pebbles and dry litter instead of meadow
  let r = length(wp.xz);
  let soilM = 1.0 - smoothstep(1.4, 4.8, r + (n1.g - 0.5) * 1.6);
  let grit = ns(vec3f(wp.xz * 9.0, 0.2));
  var soil = mix(vec3f(0.026, 0.021, 0.016), vec3f(0.070, 0.058, 0.045), n2.r * (0.5 + n3.g));
  soil = mix(soil, vec3f(0.095, 0.082, 0.064), smoothstep(0.62, 0.8, grit.b) * 0.8);
  soil = mix(soil, vec3f(0.060, 0.045, 0.022), smoothstep(0.55, 0.75, n1.b) * 0.6);
  alb = mix(alb, soil, soilM);
  // scorched ash bed + glowing coals: charcoal fragments, grey ash, hot cracks
  let ashM = 1.0 - smoothstep(0.55, 1.25, r + (n1.r - 0.5) * 0.6 + (n2.g - 0.5) * 0.25);
  let scorch = 1.0 - smoothstep(1.0, 2.3, r + (n1.g - 0.5) * 1.0);
  alb = mix(alb, alb * 0.2 + vec3f(0.003), scorch * 0.9);
  let chunks = ns(vec3f(wp.xz * 3.0, 0.45)).b;
  let ashC = mix(vec3f(0.010, 0.0095, 0.009), vec3f(0.060, 0.058, 0.055), smoothstep(0.2, 0.9, n2.b * n3.r + 0.25 * (1.0 - chunks)));
  alb = mix(alb, ashC, ashM);
  let wq = wp.xz + (nz(vec3f(wp.xz * 1.3, 0.2)).rg - 0.5) * 0.45;
  let net = 1.0 - smoothstep(0.0, 0.17, ns(vec3f(wq * 2.2, 0.3)).r);
  let net2 = 1.0 - smoothstep(0.0, 0.22, ns(vec3f(wq * 5.0 + 0.2, 0.7)).g);
  let cell = ns(vec3f(wq * 1.1, 0.8)).b;
  let cover = smoothstep(0.30, 0.65, nz(vec3f(wq * 1.9, 0.55)).a + 0.25 * (n2.r - 0.5));   // ash lying over the embers
  let hotZone = (1.0 - smoothstep(0.30, 1.05, r)) * smoothstep(0.1, 0.8, ashM);
  let pulse = 0.5 + 0.5 * nz(vec3f(wq * 2.2, t * 0.25)).b + 0.25 * sin(t * 6.0 + cell * 40.0);
  let cb = saturate(max(net, net2 * 0.6) * (0.2 + 1.2 * cell) * hotZone * (1.0 - 0.8 * cover));
  let emis = flameRamp(0.16 + 0.4 * pulse * cb) * cb * pulse * 1.5;

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
  let rpit = length(root2);
  let trample = mix(0.4, 1.0, smoothstep(1.5, 4.2, rpit));   // worn, sparser grass around the ring
  let keep = step(hash11(sd * 3.7 + 0.31), mix(0.10, 1.0, smoothstep(1.5, 4.5, rpit)));
  let h = (0.2 + 0.55 * pow(hash11(sd * 91.7), 1.6)) * (0.55 + 0.9 * clump.r) * fade * trample;
  let wdt = 0.017 * (0.7 + 0.6 * hash11(sd * 13.1)) * (0.5 + 0.5 * trample);
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
  let rootC = vec3f(0.020, 0.022, 0.009);
  var tipC = mix(vec3f(0.112, 0.090, 0.040), vec3f(0.185, 0.140, 0.062), smoothstep(0.35, 0.95, var1));
  tipC = mix(tipC, vec3f(0.050, 0.068, 0.024), smoothstep(0.45, 0.8, clump.b) * 0.75);
  tipC *= 0.65 + 0.7 * hash11(sd * 53.9);
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

  // heat shimmer: blades seen through the hot plume wobble sideways / up
  let vv = wp - cam.pos.xyz;
  let dv = length(vv);
  let rdv = vv / max(dv, 1e-3);
  let oc = scn.light.xyz - cam.pos.xyz;
  let tc = dot(oc, rdv);
  var wpD = wp;
  if (tc > 0.0 && tc < dv) {
    let hl = length(oc - rdv * tc);
    let yc = cam.pos.y + rdv.y * tc;
    let hz = exp(-hl * hl * 2.2) / (1.0 + tc * 0.3) * smoothstep(0.1, 0.9, yc) * (1.0 - smoothstep(2.0, 4.0, yc));
    let wob = nz(vec3f(wp.x * 2.5, wp.y * 2.0 - t * 1.4, wp.z * 2.5)).rgb - 0.5;
    wpD += (normalize(cam.right.xyz) * wob.x + normalize(cam.up.xyz) * wob.y) * 0.12 * hz * scn.env.z;
  }
  var o: GOut;
  o.pos = cam.viewProj * vec4f(wpD - cam.pos.xyz, 1.0);
  if (fade <= 0.0 || keep < 0.5) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); }
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

@fragment fn fs(in: SO) -> @location(0) vec4f {
  let t = cam.pos.w;
  let wp = in.wp;
  var n = normalize(in.n);
  let mat = in.ms.x;
  let seed = in.ms.y;
  let isBark = mat < 0.5;
  let isStone = mat > 0.5 && mat < 1.5;
  let isEnd = mat > 1.5;

  // ---- procedural surface: charred bark blocks with glowing cracks / stone facets / end grain ----
  let ang = in.uv.x;
  let along = in.uv.y;
  let cs = vec3f(cos(ang) * 0.8 + seed * 0.31, sin(ang) * 0.8 + seed * 0.17, along * 1.5 + seed * 0.7);
  let fibre = nz(vec3f(cos(ang) * 0.5 + seed * 0.13, sin(ang) * 0.5 + seed * 0.07, along * 0.45)).r * 0.6
            + nz(vec3f(cos(ang) * 1.9 + seed, sin(ang) * 1.9, along * 0.9)).g * 0.4;
  let e1 = ns(cs * vec3f(0.30, 0.30, 0.38)).r;       // coarse char blocks (elongated along the log)
  let e2 = ns(cs * vec3f(0.30, 0.30, 0.55) + 0.37).g; // finer crazing
  let crack = max(1.0 - smoothstep(0.0, 0.17, e1), (1.0 - smoothstep(0.0, 0.22, e2)) * 0.6);
  var hh = mix(0.12, 0.78, smoothstep(0.0, 0.4, e1)) * 0.75 + fibre * 0.25;
  // stone: faceted lumps
  let sp = wp * 2.4 + seed;
  let facets = ns(sp * 0.45).b;
  let sh0 = ns(sp * 0.45).a * 0.6 + nz(sp * 1.1).g * 0.3 + ns(sp * 2.6).a * 0.1;
  hh = select(hh, sh0 + facets * 0.35, isStone);
  // end grain: growth rings + radial checks
  let rr = length(in.uv);
  let ringsN = nz(vec3f(in.uv * 0.3 + seed, 0.2)).r;
  let radial = ns(vec3f(in.uv * 0.7 + seed * 0.1, 0.3)).r;
  hh = select(hh, 0.5 + 0.06 * sin(rr * 24.0 + ringsN * 6.0) + 0.14 * smoothstep(0.0, 0.3, radial), isEnd);

  let dpx = dpdx(wp); let dpy = dpdy(wp);
  let dhx = dpdx(hh); let dhy = dpdy(hh);
  let r1 = cross(dpy, n); let r2 = cross(n, dpx);
  let det = dot(dpx, r1);
  let grad = sign(det) * (dhx * r1 + dhy * r2);
  let bump = select(0.016, 0.014, isStone);
  n = normalize(abs(det) * n - grad * bump);

  // ---- heat: closer to the flame and facing it = more charred and more glowing ----------------
  let core = length(wp.xz);
  let toL = scn.light.xyz - wp;
  let Ld = normalize(toL);
  let facing = saturate(dot(normalize(in.n), Ld) * 0.5 + 0.5);
  let heat = exp(-core * 1.7) * (0.30 + 0.70 * exp(-wp.y * 1.0)) * (0.55 + 0.45 * facing);
  let flick = nz(vec3f(wp.xz * 2.2 + seed * 0.1, t * 0.4 + wp.y)).b;
  let cell = ns(cs * vec3f(0.3, 0.3, 0.38) * 0.7 + 0.2).b;
  let hotCell = smoothstep(0.30, 0.85, cell);
  let glow = saturate(crack * smoothstep(0.08, 0.55, heat) * (0.20 + 1.0 * flick) * (0.04 + 1.2 * hotCell * hotCell) * 1.7);

  let soot = ns(vec3f(cos(ang) * 0.4 + seed * 0.2, sin(ang) * 0.4, along * 0.35)).a;
  let charM = saturate(heat * 3.0 + 0.25 * flick + 0.22 + 0.6 * smoothstep(0.30, 0.65, soot));
  // raw bark: dry grey-brown with lighter fibre
  var alb = mix(vec3f(0.016, 0.012, 0.009), vec3f(0.070, 0.052, 0.036), pow(fibre, 3.0) * 3.0) * mix(0.55, 1.1, hh);
  // charcoal: near black blocks with dusty grey ash on the ridges
  let ash = smoothstep(0.55, 0.9, hh) * (0.4 + 0.6 * flick);
  var charC = mix(vec3f(0.008, 0.007, 0.006), vec3f(0.035, 0.034, 0.033), ash * charM);
  charC = mix(charC, vec3f(0.003), crack * 0.9);
  alb = mix(alb, charC, saturate(charM * 1.15));
  // stone: grey granite with mica speckle and soot on the fire-facing side
  let speck = smoothstep(0.80, 0.90, ns(wp * 35.0 + seed).a) * 0.08;
  var stoneC = mix(vec3f(0.016, 0.016, 0.015), vec3f(0.058, 0.054, 0.049), sh0 * sh0 * 1.6);
  stoneC *= 0.55 + 0.9 * ns(wp * 1.3 + seed).a;
  stoneC = mix(stoneC, vec3f(0.060, 0.050, 0.040), smoothstep(0.5, 0.8, facets) * 0.5) + speck;
  stoneC = mix(stoneC, vec3f(0.006, 0.0055, 0.005), saturate(heat * 2.6 + (1.0 - saturate(n.y * 1.2 + 0.3)) * 0.35) * 0.9);
  // end grain
  let endC = mix(vec3f(0.17, 0.10, 0.05), vec3f(0.30, 0.19, 0.09), hh) * (1.0 - charM * 0.85);
  alb = select(alb, stoneC, isStone);
  alb = select(alb, endC, isEnd);

  // ember emission: bright orange-yellow in the cracks nearest the fire, dull red further out
  var emis = flameRamp(0.14 + 0.62 * glow) * glow * select(2.0, 0.0, isStone);
  let endCrack = saturate(smoothstep(0.35, 0.0, radial) * 0.8 + 0.25 * saturate(rr));
  emis += select(vec3f(0.0), flameRamp(0.16 + 0.5 * heat) * heat * endCrack * 2.0 * (0.55 + flick), isEnd);

  let ao = mix(0.30, 1.0, smoothstep(0.0, 0.45, wp.y)) * mix(0.5, 1.0, saturate(core * 1.4));
  var lit = fireIrr(wp, n) * (0.5 + 0.5 * ao);
  lit += scn.amb.rgb * ao * (0.6 + 0.4 * n.y);
  lit += scn.moonCol.rgb * saturate(dot(n, scn.moon.xyz)) * ao;
  var col = alb * lit + emis;
  // sheen on charcoal scales / wet stone catching the fire
  let V = normalize(cam.pos.xyz - wp);
  let Hh = normalize(Ld + V);
  let rough = select(30.0, 14.0, isStone);
  let spec = pow(saturate(dot(n, Hh)), rough) * select(charM * (1.0 - crack), 0.5, isStone);
  col += scn.lightCol.rgb * scn.light.w / (dot(toL, toL) + 0.8) * spec * 0.09 * ao;
  col = applyFog(col, wp);
  return vec4f(col, 1.0);
}
`;
