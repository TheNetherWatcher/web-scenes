// New York skyline from the Hudson: procedural Manhattan, hand-built landmarks and bridges,
// physically-flavoured sky, mirrored-geometry river reflections, height-map sun shadows.
import { boot } from '../../engine/engine.js';
import { D2R, clamp, smoothstep } from '../../engine/math.js';
import { buildCity, gridToWorld, GRID_ROT, HMAP, LAND, MAT, STY } from './gen.js';
import { BUILDINGS } from './shaders.js';
import { BEAMS, SPRITES, GROUND, SKYPASS } from './shaders2.js';

const GU = GPUBufferUsage, TU = GPUTextureUsage;

// ---- sky colour keyframes by sun elevation (deg): zenith / horizon away / horizon toward sun --
const KF = [
  [-18, [0.0012, 0.0025, 0.008], [0.008, 0.010, 0.018], [0.012, 0.013, 0.022]],
  [-12, [0.003, 0.007, 0.022], [0.016, 0.022, 0.050], [0.035, 0.032, 0.062]],
  [-6, [0.015, 0.03, 0.10], [0.07, 0.07, 0.13], [0.35, 0.16, 0.12]],
  [-2, [0.04, 0.08, 0.24], [0.26, 0.18, 0.24], [1.2, 0.48, 0.24]],
  [0, [0.06, 0.12, 0.32], [0.42, 0.30, 0.32], [2.2, 0.85, 0.32]],
  [3, [0.07, 0.16, 0.46], [0.55, 0.40, 0.42], [2.5, 1.25, 0.50]],
  [8, [0.10, 0.23, 0.62], [0.50, 0.50, 0.58], [1.9, 1.4, 0.95]],
  [15, [0.14, 0.30, 0.74], [0.52, 0.66, 0.86], [1.2, 1.1, 1.0]],
  [30, [0.19, 0.38, 0.88], [0.60, 0.78, 1.0], [0.85, 0.92, 1.05]],
  [60, [0.20, 0.42, 0.95], [0.58, 0.78, 1.0], [0.70, 0.88, 1.1]],
];
function kf(e, col, out) {
  let i = 1;
  while (i < KF.length - 1 && e > KF[i][0]) i++;
  const a = KF[i - 1], b = KF[i];
  const t = clamp((e - a[0]) / (b[0] - a[0]), 0, 1);
  for (let k = 0; k < 3; k++) out[k] = a[col][k] + (b[col][k] - a[col][k]) * t;
}
// sunlight colour after the air mass it crosses (Rayleigh + a little aerosol)
function sunColor(eDeg, out, intensity) {
  const e = Math.max(eDeg, -1);
  const m = 1 / (Math.sin(e * D2R) + 0.15 * Math.pow(e + 3.885, -1.253));
  const a = Math.min(m, 40);
  out[0] = Math.exp(-a * (0.033 + 0.02)); out[1] = Math.exp(-a * (0.075 + 0.02)); out[2] = Math.exp(-a * (0.17 + 0.02));
  const k = intensity * smoothstep(-1.5, 0.8, eDeg);
  out[0] *= k; out[1] *= k; out[2] *= k;
}
const dirFrom = (bearingDeg, elevDeg) => {
  const b = bearingDeg * D2R, e = elevDeg * D2R;
  return [Math.sin(b) * Math.cos(e), Math.sin(e), -Math.cos(b) * Math.cos(e)];
};

// ---- camera presets (grid km) ------------------------------------------------------------------
const lookAt = (eng, s, a, y, ts, ta, ty, fov) => {
  const [x, z] = gridToWorld(s * 1000, a * 1000), [tx, tz] = gridToWorld(ts * 1000, ta * 1000);
  const c = eng.cam;
  c.x = x; c.y = y; c.z = z;
  c.yaw = Math.atan2(tx - x, -(tz - z));
  c.pitch = Math.atan2(ty - y, Math.hypot(tx - x, tz - z));
  if (fov) c.fovY = fov * D2R;
};
const PRESETS = {
  1: { name: 'Lower Manhattan to Midtown, dusk', cam: [1.4, -2.8, 55, -0.25, 2.97, 150, 20], sun: [-1.3, 285], post: { exposure: 0.8, contrast: 1.24, sat: 1.15 } },
  2: { name: 'Midtown, golden hour', cam: [-3.15, 5.0, 75, -0.32, 7.06, 175, 30], sun: [5.0, 280] },
  3: { name: 'Downtown from Jersey City', cam: [-2.1, -0.15, 70, -0.5, 1.5, 230, 36], sun: [4.0, 282] },
  4: { name: 'Brooklyn Bridge', cam: [1.12, 1.28, 17, 1.12, 1.9, 58, 62], sun: [6.0, 262] },
  5: { name: 'Midtown at night', cam: [-3.15, 5.0, 75, -0.32, 7.06, 175, 30], sun: [-14, 280] },
  6: { name: 'Sunset over the Hudson', cam: [-2.7, 5.55, 55, -5.1, 3.75, 66, 46], sun: [3.0, 262] },
  7: { name: 'High above, dusk', cam: [-3.4, 3.6, 850, 0.0, 6.0, 120, 48], sun: [-2.5, 280] },
  8: { name: 'Empire State close-up', cam: [-1.15, 5.75, 140, -0.06, 6.45, 290, 50], sun: [4.0, 280] },
};

// ---- moving boats (paths in grid km, closed loops) ----------------------------------------------
const BOATS = [
  { path: [[0.15, -1.75], [1.3, -2.0], [1.3, -2.12], [0.15, -1.87]], v: 7.5, len: 72, wid: 20, mat: MAT.ORANGE, cab: MAT.WHITE },
  { path: [[-3.35, 7.0], [-2.05, 7.2], [-2.05, 7.35], [-3.35, 7.15]], v: 7, len: 38, wid: 11, mat: MAT.BOAT, cab: MAT.WHITE },
  { path: [[-3.0, 5.5], [-2.15, 5.65], [-2.15, 5.78], [-3.0, 5.63]], v: 6, len: 42, wid: 12, mat: MAT.BOAT, cab: MAT.WHITE },
  { path: [[-2.7, 1.2], [-2.7, 9.0], [-2.95, 9.0], [-2.95, 1.2]], v: 5.5, len: 30, wid: 9, mat: MAT.DARK, cab: MAT.ORANGE },
  { path: [[0.9, 0.35], [1.14, 1.5], [1.5, 2.5], [1.95, 3.6], [2.05, 3.6], [1.6, 2.5], [1.25, 1.5], [1.0, 0.35]], v: 6, len: 34, wid: 9, mat: MAT.BOAT, cab: MAT.WHITE },
  { path: [[-0.2, -0.7], [-0.8, -4.8], [-0.95, -4.8], [-0.35, -0.7]], v: 8.5, len: 70, wid: 18, mat: MAT.ORANGE, cab: MAT.WHITE },
  { path: [[-2.4, 3.0], [-2.4, 5.2], [-2.55, 5.2], [-2.55, 3.0]], v: 4, len: 14, wid: 4.5, mat: MAT.WHITE, cab: MAT.WHITE },
  { path: [[-0.3, -0.6], [0.4, -1.8], [0.05, -2.6], [-1.0, -1.4]], v: 5, len: 32, wid: 8, mat: MAT.BOAT, cab: MAT.BOAT },
  { path: [[-2.3, 2.0], [-2.3, 0.4], [-2.45, 0.4], [-2.45, 2.0]], v: 3, len: 46, wid: 12, mat: MAT.DARK, cab: MAT.MECH },
];
for (const b of BOATS) {
  b.pts = b.path.map(([s, a]) => gridToWorld(s * 1000, a * 1000));
  b.seg = []; b.total = 0;
  for (let i = 0; i < b.pts.length; i++) {
    const p = b.pts[i], q = b.pts[(i + 1) % b.pts.length];
    const l = Math.hypot(q[0] - p[0], q[1] - p[1]);
    b.seg.push(l); b.total += l;
  }
  b.off = Math.random() * b.total;
}
function boatPos(b, t, out) {
  let d = (t * b.v + b.off) % b.total;
  let i = 0;
  while (d > b.seg[i]) { d -= b.seg[i]; i++; }
  const p = b.pts[i], q = b.pts[(i + 1) % b.pts.length], u = d / b.seg[i];
  out[0] = p[0] + (q[0] - p[0]) * u; out[1] = p[1] + (q[1] - p[1]) * u;
  out[2] = (q[0] - p[0]) / b.seg[i]; out[3] = (q[1] - p[1]) / b.seg[i];
}

const scene = {
  name: 'city',
  msaa: 4,
  params: {
    sunElev: 5.0, sunAz: 280, haze: 0.8, glow: 1.0, clouds: 0.38, rough: 0.16, timeLapse: 0,
    exposure: 1.05, bloom: 0.16, vignette: 0.14, sat: 1.28, contrast: 1.16, warmth: 0.0, grain: 0.0,
  },
  sliders: [
    ['sunElev', 'Time of day · sun elevation (deg)', -18, 60, 0.1],
    ['sunAz', 'Sun azimuth (deg)', 0, 360, 1],
    ['timeLapse', 'Auto time-lapse (deg/s)', 0, 3, 0.05],
    ['haze', 'Haze', 0.1, 4, 0.01],
    ['glow', 'Window glow', 0, 2, 0.01],
    ['clouds', 'Cloud cover', 0, 1, 0.01],
    ['rough', 'Water roughness', 0.02, 1, 0.01],
    ['exposure', 'Exposure', 0.2, 3, 0.01],
    ['bloom', 'Bloom', 0, 0.4, 0.002],
  ],
  camera: { mode: 'fly', x: 0, y: 70, z: 0, yaw: 0, pitch: 0, fov: 30, speed: 80, near: 1.0, minY: 3 },
  help: 'click: mouse look · WASD move · Space/E up · Q/C down · Shift fast · wheel speed · 1-8 camera presets · T time-lapse · H panel',

  async init(eng) {
    const { device, queue, P } = eng;
    const t0 = performance.now();
    const city = buildCity((m) => console.log(m));
    this.city = city;
    this.t0 = performance.now() - t0;

    // ---- textures ----
    this.landTex = device.createTexture({ size: [LAND.w, LAND.h], format: 'r8unorm', usage: TU.TEXTURE_BINDING | TU.COPY_DST, label: 'land sdf' });
    queue.writeTexture({ texture: this.landTex }, city.landTex, { bytesPerRow: LAND.w }, [LAND.w, LAND.h]);
    this.hmapTex = device.createTexture({ size: [HMAP.w, HMAP.h], format: 'r16float', mipLevelCount: HMAP.levels, usage: TU.TEXTURE_BINDING | TU.COPY_DST, label: 'height map' });
    city.hmapLevels.forEach((lv, i) => queue.writeTexture({ texture: this.hmapTex, mipLevel: i }, lv.data, { bytesPerRow: lv.w * 2 }, [lv.w, lv.h]));
    this.landView = this.landTex.createView();
    this.hmapView = this.hmapTex.createView();

    // ---- buffers ----
    const vb = (data, label) => {
      const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GU.VERTEX | GU.COPY_DST, label });
      queue.writeBuffer(b, 0, data);
      return b;
    };
    this.instBuf = vb(city.inst, 'buildings');
    this.beamBuf = vb(city.beams, 'beams');
    this.sprBuf = vb(city.sprites, 'lights');
    this.maxBoatInst = 40;
    this.boatInst = new Float32Array(this.maxBoatInst * 12);
    this.boatInstBuf = device.createBuffer({ size: this.boatInst.byteLength, usage: GU.VERTEX | GU.COPY_DST, label: 'boats' });
    this.dynSpr = new Float32Array(48 * 8);
    this.dynSprBuf = device.createBuffer({ size: this.dynSpr.byteLength, usage: GU.VERTEX | GU.COPY_DST, label: 'dynamic lights' });
    this.nBoatInst = 0; this.nDynSpr = 0;
    this.scnF = new Float32Array(104);
    this.scnBuf = [0, 1].map((i) => eng.mkBuf(416, GU.UNIFORM | GU.COPY_DST, 'scene ubo ' + i));
    this.boatTmp = new Float32Array(4);

    // ---- pipelines (main pass, 4x MSAA) and mirrored reflection pass (1x) ----
    const mods = await Promise.all([
      eng.shader(BUILDINGS, 'buildings'), eng.shader(BEAMS, 'beams'), eng.shader(SPRITES, 'sprites'),
      eng.shader(GROUND, 'ground'), eng.shader(SKYPASS, 'sky'),
    ]);
    const [mB, mBm, mS, mG, mSky] = mods;
    const additive = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    const inst = (stride, n) => ({ arrayStride: stride, stepMode: 'instance', attributes: Array.from({ length: n }, (_, i) => ({ shaderLocation: i, offset: i * 16, format: 'float32x4' })) });
    const make = (ms) => Promise.all([
      eng.renderPipe({ label: 'buildings', vertex: { module: mB, entryPoint: 'vs', buffers: [inst(48, 3)] }, fragment: { module: mB, entryPoint: 'fs' }, multisample: { count: ms } }),
      eng.renderPipe({ label: 'beams', vertex: { module: mBm, entryPoint: 'vs', buffers: [inst(32, 2)] }, fragment: { module: mBm, entryPoint: 'fs' }, multisample: { count: ms } }),
      eng.renderPipe({
        label: 'sprites', vertex: { module: mS, entryPoint: 'vs', buffers: [inst(32, 2)] },
        fragment: { module: mS, entryPoint: 'fs', targets: [{ format: eng.HDR, blend: additive }] },
        depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'greater' }, multisample: { count: ms },
      }),
    ]);
    const [main, refl] = await Promise.all([make(eng.msaa), make(1)]);
    this.pipes = { main, refl };
    this.gndPipe = await eng.renderPipe({ label: 'ground', vertex: { module: mG, entryPoint: 'vs' }, fragment: { module: mG, entryPoint: 'fs' } });
    this.skyPipe = await eng.renderPipe({
      label: 'sky', vertex: { module: mSky, entryPoint: 'vs' }, fragment: { module: mSky, entryPoint: 'fs' },
      depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'equal' },
    });
    const mk = (pipes, k) => pipes.map((p, i) => eng.bg(p, [{ buffer: eng.camUBO }, { buffer: this.scnBuf[k] }, ...(i === 0 ? [this.hmapView] : [])], 'world' + k + i));
    this.bgMain = mk(main, 0);
    this.bgRefl = mk(refl, 1);
    this.skyBG = eng.bg(this.skyPipe, [{ buffer: eng.camUBO }, { buffer: this.scnBuf[0] }]);
    this.sampL = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.sampR = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.t = 0;
    this.tmp3 = new Float32Array(3);
    this.scnF2 = new Float32Array(104);
    const pn = eng.qs.get('preset') || '1';
    const pr = PRESETS[pn] || PRESETS[1];
    if (!eng.qs.has('cam')) lookAt(eng, ...pr.cam);
    for (const [k, v] of [['sunElev', pr.sun[0]], ['sunAz', pr.sun[1]], ...Object.entries(pr.post || {})]) if (!eng.qs.has(k)) P[k] = v;
    if (eng.qs.has('fov')) eng.cam.fovY = Number(eng.qs.get('fov')) * D2R;
  },

  onResize(eng) {
    const { device } = eng;
    const w = Math.max(2, eng.W >> 1), h = Math.max(2, eng.H >> 1);
    for (const t of [this.reflTex, this.reflDepth]) t && t.destroy();
    this.reflTex = device.createTexture({ size: [w, h], format: eng.HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'reflection' });
    this.reflDepth = device.createTexture({ size: [w, h], format: eng.DEPTH, usage: TU.RENDER_ATTACHMENT, label: 'reflection depth' });
    this.reflView = this.reflTex.createView();
    this.reflDepthView = this.reflDepth.createView();
    this.gndBG = eng.bg(this.gndPipe, [{ buffer: eng.camUBO }, { buffer: this.scnBuf[0] }, this.hmapView, this.landView, this.sampL, this.reflView, this.sampR], 'ground');
  },

  onKey(eng, e) {
    if (e.code === 'KeyT') { eng.P.timeLapse = eng.P.timeLapse > 0 ? 0 : 0.6; return; }
    const m = /^Digit([1-8])$/.exec(e.code);
    if (!m) return;
    const p = PRESETS[m[1]];
    lookAt(eng, ...p.cam);
    eng.P.sunElev = p.sun[0]; eng.P.sunAz = p.sun[1];
    Object.assign(eng.P, { exposure: 1.05, contrast: 1.16, sat: 1.28 }, p.post || {});
  },

  update(eng, dt, t) {
    const P = eng.P;
    this.t = t;
    if (eng.qs.has('bench')) {
      const n = performance.now();
      if (this.bn === undefined) { this.bn = n; this.bf = 0; }
      if (++this.bf === 60) { fetch('/__log', { method: 'POST', body: 'bench ms/frame ' + ((n - this.bn) / 60).toFixed(2) }); this.bn = n; this.bf = 0; }
    }
    if (P.timeLapse > 0) {
      P.sunElev -= P.timeLapse * dt;
      if (P.sunElev < -17) P.sunElev = 38;
    }
    const e = P.sunElev;
    const f = this.scnF;
    const sun = dirFrom(P.sunAz, e);
    const nt = smoothstep(4, -8, e);
    const tmp = this.tmp3;
    // sun / moon
    f.set([sun[0], sun[1], sun[2], nt], 0);
    sunColor(e, tmp, 6.0); f.set([tmp[0], tmp[1], tmp[2], 1], 4);
    const zen = [0, 0, 0], hor = [0, 0, 0], hs = [0, 0, 0];
    kf(e, 1, zen); kf(e, 2, hor); kf(e, 3, hs);
    const glowAmt = P.glow * (0.01 + 0.99 * smoothstep(6, -4, e));
    const gl = nt * clamp(P.glow, 0, 1.5);
    // light-pollution glow on the horizon at night
    hor[0] += 0.03 * gl; hor[1] += 0.016 * gl; hor[2] += 0.007 * gl;
    hs[0] += 0.02 * gl; hs[1] += 0.011 * gl; hs[2] += 0.005 * gl;
    f.set([zen[0], zen[1], zen[2], 5.2e-5 * P.haze], 8);
    f.set([hor[0], hor[1], hor[2], glowAmt], 12);
    f.set([hs[0], hs[1], hs[2], P.clouds], 16);
    const moon = dirFrom(108, 27);
    f.set([moon[0], moon[1], moon[2], nt], 20);
    // ambient hemisphere
    const sy = Math.max(sun[1], 0);
    const up = [zen[0] * 0.75 + hor[0] * 0.2, zen[1] * 0.75 + hor[1] * 0.2, zen[2] * 0.75 + hor[2] * 0.2];
    const side = [0, 0, 0], down = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      side[k] = (hor[k] * 0.65 + hs[k] * 0.35) * 0.38 * [0.82, 0.95, 1.3][k];
      down[k] = side[k] * 0.28 + tmp[k] * sy * 0.07;
    }
    // faint moon / sky glow so the night isn't pitch black
    up[0] += 0.006 * nt; up[1] += 0.011 * nt; up[2] += 0.026 * nt;
    side[0] += 0.004 * nt + 0.01 * gl; side[1] += 0.006 * nt + 0.006 * gl; side[2] += 0.012 * nt + 0.003 * gl;
    down[0] += 0.02 * gl; down[1] += 0.011 * gl; down[2] += 0.005 * gl;
    f.set([up[0], up[1], up[2], P.rough], 24);
    f.set([side[0], side[1], side[2], t], 28);
    f.set([down[0], down[1], down[2], 0], 32);
    sunColor(e + 2.6, tmp, 1.5); f.set([tmp[0], tmp[1], tmp[2], 0], 36);

    // boats
    let bi = 0, si = 0;
    const bd = this.boatInst, sd = this.dynSpr, b4 = this.boatTmp;
    f.fill(0, 40, 104);
    BOATS.forEach((b, j) => {
      boatPos(b, t, b4);
      if (j < 8) { f[40 + j * 4] = b4[0]; f[41 + j * 4] = b4[1]; f[42 + j * 4] = b4[2]; f[43 + j * 4] = b4[3]; f[72 + j * 4] = b.len; f[73 + j * 4] = b.v; }
      const rot = Math.atan2(b4[3], b4[2]);
      const bob = Math.sin(t * 0.9 + j) * 0.12;
      const put = (ox, y0, h, hw, hd, mat, taper) => {
        bd.set([b4[0] + b4[2] * ox, b4[1] + b4[3] * ox, y0 + bob, h, hw, hd, j * 3.3 + ox, mat, taper, rot, 0, STY.NONE], bi * 12); bi++;
      };
      put(0, 0.2, 3.4, b.len * 0.5, b.wid * 0.5, b.mat, 0.8);
      put(-b.len * 0.06, 3.6, 3.6, b.len * 0.3, b.wid * 0.38, b.cab, 0.9);
      put(-b.len * 0.12, 7.2, 3.4, b.len * 0.07, b.wid * 0.16, MAT.DARK, 0.8);
      // nav lights: red port, green starboard, white masthead
      const bow = b.len * 0.42, sp = b.wid * 0.5;
      const px = -b4[3], pz = b4[2];
      sd.set([b4[0] + b4[2] * bow + px * sp, 4.6 + bob, b4[1] + b4[3] * bow + pz * sp, 1.6, 3.2, 0.1, 0.1, 0], si * 8); si++;
      sd.set([b4[0] + b4[2] * bow - px * sp, 4.6 + bob, b4[1] + b4[3] * bow - pz * sp, 1.6, 0.1, 3.0, 0.2, 0], si * 8); si++;
      sd.set([b4[0] - b4[2] * b.len * 0.05, 11.4 + bob, b4[1] - b4[3] * b.len * 0.05, 2.0, 3.4, 3.4, 3.0, 0], si * 8); si++;
      if (b.len > 30) { sd.set([b4[0] - b4[2] * b.len * 0.2, 5.6 + bob, b4[1] - b4[3] * b.len * 0.2, 2.2, 3.4, 2.7, 1.8, 0], si * 8); si++; }
    });
    // helicopter and a jet on approach
    const [hx, hz] = gridToWorld(-100 + Math.cos(t * 0.03) * 1700, 7300 + Math.sin(t * 0.03) * 1700);
    sd.set([hx, 420, hz, 2.4, 3, 3, 3, -1.1], si * 8); si++;
    sd.set([hx + 6, 421, hz, 2.0, 3.2, 0.2, 0.1, 0.9], si * 8); si++;
    const jt = (t * 70) % 14000;
    const [jx, jz] = gridToWorld(-5200 + jt, 9500 - jt * 0.55);
    sd.set([jx, 900 - jt * 0.02, jz, 3.5, 6, 6, 6, 0], si * 8); si++;
    sd.set([jx, 902 - jt * 0.02, jz, 3.5, 6, 0.2, 0.2, -0.8], si * 8); si++;
    this.nBoatInst = bi; this.nDynSpr = si;
    eng.queue.writeBuffer(this.boatInstBuf, 0, bd, 0, bi * 12);
    eng.queue.writeBuffer(this.dynSprBuf, 0, sd, 0, si * 8);
    const fr = this.scnF2; fr.set(f);
    eng.queue.writeBuffer(this.scnBuf[0], 0, f);
    fr[32 + 3] = 1;
    eng.queue.writeBuffer(this.scnBuf[1], 0, fr);
  },

  drawWorld(rp, pipes, bgs) {
    const c = this.city;
    rp.setPipeline(pipes[0]); rp.setBindGroup(0, bgs[0]);
    rp.setVertexBuffer(0, this.instBuf); rp.draw(30, c.instCount);
    if (this.nBoatInst) { rp.setVertexBuffer(0, this.boatInstBuf); rp.draw(30, this.nBoatInst); }
    rp.setPipeline(pipes[1]); rp.setBindGroup(0, bgs[1]);
    rp.setVertexBuffer(0, this.beamBuf); rp.draw(6, c.beamCount);
  },
  drawLights(rp, pipes, bgs) {
    const c = this.city;
    rp.setPipeline(pipes[2]); rp.setBindGroup(0, bgs[2]);
    rp.setVertexBuffer(0, this.sprBuf); rp.draw(6, c.spriteCount);
    rp.setVertexBuffer(0, this.dynSprBuf); rp.draw(6, this.nDynSpr);
  },

  compute(eng, enc) {
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.reflView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: this.reflDepthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    this.drawWorld(rp, this.pipes.refl, this.bgRefl);
    this.drawLights(rp, this.pipes.refl, this.bgRefl);
    rp.end();
  },
  draw(eng, rp) {
    const bgs = this.bgMain;
    this.drawWorld(rp, this.pipes.main, bgs);
    rp.setPipeline(this.gndPipe); rp.setBindGroup(0, this.gndBG); rp.draw(6);
    rp.setPipeline(this.skyPipe); rp.setBindGroup(0, this.skyBG); rp.draw(3);
    this.drawLights(rp, this.pipes.main, bgs);
  },
};

boot(scene);
