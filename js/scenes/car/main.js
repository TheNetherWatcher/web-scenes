// "Vesper" - an original mid-engine hypercar on wet asphalt at golden hour. Raw WebGPU.
// The body, wheels and brake calipers are signed-distance models meshed once at start-up
// (sdf.js -> mesher.js); the frame is plain raster with a mirrored second pass for the road
// reflection, an analytic dusk environment (shaders.js) and the engine's HDR/bloom/AgX post.
import { boot } from '../../engine/engine.js';
import { CAR_SHADER } from './shaders.js';
import { BODY_PART, WHEEL_PART, CALIPER_PART } from './sdf.js';
import { meshFromPart } from './mesher.js';
import { clamp } from '../../engine/math.js';

const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const qs = new URLSearchParams(location.search);
const RES = (Number(qs.get('res')) || 1.4) / 100;       // body grid cell (m)

const hsl = (h, s, l) => {
  const f = (n) => { const k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)].map((c) => Math.pow(c, 2.2));
};
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

const WHEELS = [   // x, z, scale  (front axle, rear axle)
  { x: 1.40, z: 0.77, s: 1.04 }, { x: 1.40, z: -0.77, s: 1.04 },
  { x: -1.32, z: 0.80, s: 1.075 }, { x: -1.32, z: -0.80, s: 1.075 },
];
const WHEEL_R = 0.345;

const scene = {
  name: 'car',
  msaa: 4,
  params: {
    hue: 0.60, light: 0.27, flake: 1.0, sunElev: 8, sunAzim: -65, wheelSpeed: 0.5, rimStyle: 0,
    headlights: 1, floorRefl: 1.0, exposure: 1.1, bloom: 0.12, vignette: 0.16, sat: 1.08, contrast: 1.12, warmth: -0.02, grain: 0.3,
  },
  sliders: [
    ['hue', 'Paint hue', 0, 1, 0.005], ['light', 'Paint lightness', 0.08, 0.7, 0.005], ['flake', 'Metal flake', 0, 1.5, 0.01],
    ['sunElev', 'Sun elevation (deg)', -4, 45, 0.1], ['sunAzim', 'Sun azimuth (deg)', -180, 180, 1],
    ['wheelSpeed', 'Wheel spin (rev/s)', 0, 8, 0.05], ['rimStyle', 'Rim style', 0, 2, 1],
    ['headlights', 'Headlights', 0, 1, 1], ['floorRefl', 'Floor reflectivity', 0, 1.5, 0.01],
    ['exposure', 'Exposure', 0.2, 3, 0.01], ['bloom', 'Bloom', 0, 0.4, 0.005],
  ],
  camera: { mode: 'orbit', target: [0.1, 0.5, 0], dist: 6.8, yaw: -128, pitch: -7, fov: 36, near: 0.05,
    minDist: 2.4, maxDist: 26, minPitch: -1.3, maxPitch: 0.2, autoRotate: 5 },
  help: 'drag: orbit · wheel: zoom · WASD: slide view · H: panel',

  async init(eng) {
    const { device, queue } = eng;
    const t0 = performance.now();
    const hud = document.getElementById('fps');
    if (hud) hud.textContent = 'modelling the car…';
    const [body, wheel, cal] = await Promise.all([
      meshFromPart(eng, BODY_PART, { min: [-2.45, 0.02, -1.15], max: [2.45, 1.40, 1.15], h: RES, band: 0.06, label: 'body' }),
      meshFromPart(eng, WHEEL_PART, { min: [-0.38, -0.38, -0.18], max: [0.38, 0.38, 0.18], h: 0.006, band: 0.025, style: eng.P.rimStyle, label: 'wheel' }),
      meshFromPart(eng, CALIPER_PART, { min: [-0.22, 0.04, -0.03], max: [0.10, 0.27, 0.09], h: 0.004, band: 0.012, label: 'caliper' }),
    ]);
    console.log(`car meshed in ${(performance.now() - t0).toFixed(0)} ms: body ${body.vertexCount} verts / ${body.indices.length / 3} tris`);
    const gpuMesh = (m, label) => {
      const vb = eng.bufFrom(m.verts, GU.VERTEX, label + ' vb');
      const ib = eng.bufFrom(m.indices, GU.INDEX, label + ' ib');
      return { vb, ib, count: m.indices.length };
    };
    this.mBody = gpuMesh(body, 'body');
    this.mWheel = gpuMesh(wheel, 'wheel');
    this.mCal = gpuMesh(cal, 'caliper');

    // instances (model matrices)
    this.instBody = eng.mkBuf(64, GU.VERTEX | GU.COPY_DST, 'inst body');
    this.instWheel = eng.mkBuf(64 * 4, GU.VERTEX | GU.COPY_DST, 'inst wheels');
    this.instCal = eng.mkBuf(64 * 4, GU.VERTEX | GU.COPY_DST, 'inst calipers');
    this.wheelM = new Float32Array(16 * 4);
    this.calM = new Float32Array(16 * 4);
    this.bodyM = new Float32Array(16);
    this.spin = 0;
    this.writeMatrices(0);
    const ident = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    queue.writeBuffer(this.instBody, 0, ident);
    queue.writeBuffer(this.instCal, 0, this.calM);

    // pipelines
    const mod = await eng.shader(CAR_SHADER, 'car');
    const vbuf = [
      { arrayStride: 48, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x4' }, { shaderLocation: 3, offset: 40, format: 'float32x2' }] },
      { arrayStride: 64, stepMode: 'instance', attributes: [4, 5, 6, 7].map((l, i) => ({ shaderLocation: l, offset: i * 16, format: 'float32x4' })) },
    ];
    const carPipe = (fs, label) => eng.renderPipe({ label, vertex: { module: mod, entryPoint: 'cvs', buffers: vbuf }, fragment: { module: mod, entryPoint: fs } });
    const blend = { color: { srcFactor: 'one', dstFactor: 'src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' } };
    [this.pBody, this.pWheel, this.pSky, this.pGround] = await Promise.all([
      carPipe('fs_body', 'car body'),
      carPipe('fs_wheel', 'car wheel'),
      eng.renderPipe({ label: 'sky', vertex: { module: mod, entryPoint: 'vs' }, fragment: { module: mod, entryPoint: 'fs_sky' },
        depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'equal' } }),
      eng.renderPipe({ label: 'road', vertex: { module: mod, entryPoint: 'gvs' }, fragment: { module: mod, entryPoint: 'gfs', targets: [{ format: eng.HDR, blend }] } }),
    ]);
    this.uboN = eng.mkBuf(112, GU.UNIFORM | GU.COPY_DST, 'env');
    this.uboM = eng.mkBuf(112, GU.UNIFORM | GU.COPY_DST, 'env mirror');
    const mkBG = (pipe, ubo) => eng.bg(pipe, [{ buffer: eng.camUBO }, { buffer: ubo }]);
    this.bg = {};
    for (const [k, p] of [['body', this.pBody], ['wheel', this.pWheel], ['sky', this.pSky], ['ground', this.pGround]]) {
      this.bg[k] = [mkBG(p, this.uboN), mkBG(p, this.uboM)];
    }
    this.env = new Float32Array(28);
    this.scroll = 0;
    this.lastStyle = eng.P.rimStyle;
    console.log(`car ready in ${(performance.now() - t0).toFixed(0)} ms`);
  },

  // wheel/caliper model matrices: translate * scale(mirror in z) * rotZ(spin)
  writeMatrices(spin) {
    for (let i = 0; i < 4; i++) {
      const w = WHEELS[i];
      const sz = w.z < 0 ? -1 : 1;
      const a = -spin, c = Math.cos(a) * w.s, s = Math.sin(a) * w.s;
      this.wheelM.set([c, s, 0, 0, -s, c, 0, 0, 0, 0, w.s * sz, 0, w.x, WHEEL_R * w.s, w.z, 1], i * 16);
      this.calM.set([w.s, 0, 0, 0, 0, w.s, 0, 0, 0, 0, w.s * sz, 0, w.x, WHEEL_R * w.s, w.z, 1], i * 16);
    }
  },

  async onParam(key, eng) {
    if (key === 'rimStyle' && eng.P.rimStyle !== this.lastStyle) {
      this.lastStyle = eng.P.rimStyle;
      const m = await meshFromPart(eng, WHEEL_PART, { min: [-0.38, -0.38, -0.18], max: [0.38, 0.38, 0.18], h: 0.006, band: 0.025, style: eng.P.rimStyle, label: 'wheel' });
      this.mWheel.vb.destroy(); this.mWheel.ib.destroy();
      this.mWheel = { vb: eng.bufFrom(m.verts, GU.VERTEX, 'wheel vb'), ib: eng.bufFrom(m.indices, GU.INDEX, 'wheel ib'), count: m.indices.length };
    }
  },

  update(eng, dt) {
    const P = eng.P, C = eng.cam;
    // keep the orbit camera above the road
    C.maxPitch = Math.asin(clamp((C.target[1] - 0.16) / C.dist, -1, 1));
    if (C.pitch > C.maxPitch) C.pitch = C.maxPitch;

    this.spin += dt * P.wheelSpeed * Math.PI * 2;
    this.scroll += dt * P.wheelSpeed * Math.PI * 2 * WHEEL_R * 1.02;
    this.writeMatrices(this.spin);
    eng.queue.writeBuffer(this.instWheel, 0, this.wheelM);

    // sun / sky model driven by elevation
    const e = P.sunElev, az = P.sunAzim * D2R, er = e * D2R;
    const t = clamp(e / 35, 0, 1);
    const dim = 0.25 + 0.75 * sstep(-3, 10, e);
    const sunCol = mix([1.0, 0.40, 0.12], [1.0, 0.93, 0.82], Math.pow(t, 0.6)).map((v) => v * dim);
    const day = clamp((e + 6) / 40, 0, 1);
    const zen = mix([0.012, 0.025, 0.075], [0.085, 0.19, 0.48], Math.pow(day, 0.8));
    let hor = mix([0.07, 0.05, 0.09], [1.25, 0.52, 0.20], sstep(-5, 3, e));
    hor = mix(hor, [0.60, 0.70, 0.85], sstep(8, 40, e));
    const paint = hsl(P.hue, 0.8, P.light);
    const f = this.env;
    f.set([Math.cos(er) * Math.cos(az), Math.sin(er), Math.cos(er) * Math.sin(az), 0], 0);
    f.set([...sunCol, er], 4);
    f.set([...zen, 0], 8);
    f.set([...hor, day], 12);
    f.set([...paint, P.flake], 16);
    f.set([P.headlights, P.floorRefl, P.rimStyle, 0], 20);
    f.set([this.scroll, eng.time, 0, 0], 24);
    eng.queue.writeBuffer(this.uboN, 0, f);
    f[23] = 1;
    eng.queue.writeBuffer(this.uboM, 0, f);
  },

  drawCar(rp, mirror) {
    const i = mirror ? 1 : 0;
    rp.setPipeline(this.pBody);
    rp.setBindGroup(0, this.bg.body[i]);
    rp.setVertexBuffer(0, this.mBody.vb); rp.setVertexBuffer(1, this.instBody);
    rp.setIndexBuffer(this.mBody.ib, 'uint32');
    rp.drawIndexed(this.mBody.count, 1);
    rp.setPipeline(this.pWheel);
    rp.setBindGroup(0, this.bg.wheel[i]);
    rp.setVertexBuffer(0, this.mWheel.vb); rp.setVertexBuffer(1, this.instWheel);
    rp.setIndexBuffer(this.mWheel.ib, 'uint32');
    rp.drawIndexed(this.mWheel.count, 4);
    rp.setVertexBuffer(0, this.mCal.vb); rp.setVertexBuffer(1, this.instCal);
    rp.setIndexBuffer(this.mCal.ib, 'uint32');
    rp.drawIndexed(this.mCal.count, 4);
  },

  draw(eng, rp) {
    this.drawCar(rp, true);         // reflection of the car under the road
    this.drawCar(rp, false);
    rp.setPipeline(this.pSky);      // sky (mirrored below the horizon) fills what is left
    rp.setBindGroup(0, this.bg.sky[0]);
    rp.draw(3);
    rp.setPipeline(this.pGround);   // road blends over the mirrored scene
    rp.setBindGroup(0, this.bg.ground[0]);
    rp.draw(6);
  },
};

boot(scene);
