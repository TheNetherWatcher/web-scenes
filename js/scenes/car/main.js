// "Vesper" - an original mid-engine hypercar on wet asphalt at golden hour. Raw WebGPU.
// The body, wheels and brake calipers are signed-distance models meshed once at start-up
// (sdf.js -> mesher.js). Per frame:
//   1. (when the sky changes) bake the dusk environment into a prefiltered cube map
//   2. render the car mirrored under the road into an offscreen target and mip-chain it
//   3. main pass: car, analytic sky + street lamps, wet road (blurred mirror image, sky/lamp/sun
//      reflection, sun shadow from a coarse car distance texture)
//   then the engine's HDR bloom + AgX post.
import { boot } from '../../engine/engine.js';
import { CAR_SHADER, ENV_BAKE, ENV_MIP, MIP2D } from './shaders.js';
import { BODY_PART, WHEEL_PART, CALIPER_PART, SDF_FUNCS, shadowFieldMain } from './sdf.js';
import { meshFromPart } from './mesher.js';
import { clamp } from '../../engine/math.js';

const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const TU = GPUTextureUsage;
const qs = new URLSearchParams(location.search);
const RES = (Number(qs.get('res')) || 0.8) / 100;       // body grid cell (m)
const ENV_SIZE = 512;
const ENV_MIPS = 10;
const SDF_ORG = [-2.6, -0.05, -1.25], SDF_CELL = 0.025, SDF_DIMS = [208, 60, 100];

const hsl = (h, s, l) => {
  const f = (n) => { const k = (n + h * 12) % 12; return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0), f(8), f(4)].map((c) => Math.pow(c, 2.2));
};
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

const WHEELS = [   // x, z, scale  (front axle, rear axle)
  { x: 1.40, z: 0.77, s: 0.98 }, { x: 1.40, z: -0.77, s: 0.98 },
  { x: -1.32, z: 0.80, s: 1.07 }, { x: -1.32, z: -0.80, s: 1.07 },
];
const WHEEL_R = 0.345;

const scene = {
  name: 'car',
  msaa: 4,
  params: {
    hue: 0.0, light: 0.30, flake: 1.0, sunElev: 8, sunAzim: 60, wheelSpeed: 0.5, rimStyle: 0,
    headlights: 1, floorRefl: 1.0, lamps: 1, peel: 1, drops: 1, exposure: 1.1, bloom: 0.12, vignette: 0.16, sat: 1.08, contrast: 1.12, warmth: -0.02, grain: 0.3,
  },
  sliders: [
    ['hue', 'Paint hue', 0, 1, 0.005], ['light', 'Paint lightness', 0.08, 0.7, 0.005], ['flake', 'Metal flake', 0, 1.5, 0.01],
    ['peel', 'Orange peel', 0, 2, 0.01], ['drops', 'Rain beads', 0, 1.5, 0.01],
    ['sunElev', 'Sun elevation (deg)', -4, 45, 0.1], ['sunAzim', 'Sun azimuth (deg)', -180, 180, 1],
    ['wheelSpeed', 'Wheel spin (rev/s)', 0, 8, 0.05], ['rimStyle', 'Rim style', 0, 2, 1],
    ['headlights', 'Headlights', 0, 1, 1], ['lamps', 'Street lamps', 0, 2, 0.01], ['floorRefl', 'Floor reflectivity', 0, 1.5, 0.01],
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
      meshFromPart(eng, BODY_PART, { min: [-2.45, 0.02, -0.045], max: [2.45, 1.40, 1.15], h: RES, band: 0.06, label: 'body', zKeep: -0.012, chunks: 5 }),
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
    this.instBody = eng.mkBuf(128, GU.VERTEX | GU.COPY_DST, 'inst body');
    this.instWheel = eng.mkBuf(64 * 4, GU.VERTEX | GU.COPY_DST, 'inst wheels');
    this.instCal = eng.mkBuf(64 * 4, GU.VERTEX | GU.COPY_DST, 'inst calipers');
    this.wheelM = new Float32Array(16 * 4);
    this.calM = new Float32Array(16 * 4);
    this.spin = 0;
    this.writeMatrices(0);
    const ident = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    queue.writeBuffer(this.instBody, 0, new Float32Array([...ident, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]));
    queue.writeBuffer(this.instCal, 0, this.calM);

    // ---- car distance texture (sun shadows) ----
    this.sdfTex = device.createTexture({ size: SDF_DIMS, dimension: '3d', format: 'rgba16float', usage: TU.STORAGE_BINDING | TU.TEXTURE_BINDING, label: 'car sdf' });
    {
      const code = SDF_FUNCS + BODY_PART + shadowFieldMain(WHEELS, WHEEL_R);
      const pipe = await eng.computePipe(code, 'car sdf field');
      const ub = new ArrayBuffer(48);
      const uf = new Float32Array(ub), uu = new Uint32Array(ub);
      uf.set(SDF_ORG, 0); uf[4] = SDF_CELL; uu.set(SDF_DIMS, 8);
      const gb = eng.bufFrom(new Uint8Array(ub), GU.UNIFORM, 'sdf grid');
      const bgp = eng.bg(pipe, [{ buffer: gb }, this.sdfTex.createView()]);
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe); cp.setBindGroup(0, bgp);
      cp.dispatchWorkgroups(Math.ceil(SDF_DIMS[0] / 4), Math.ceil(SDF_DIMS[1] / 4), Math.ceil(SDF_DIMS[2] / 4));
      cp.end();
      queue.submit([enc.finish()]);
    }

    // ---- shared bind group layout: camera, env, sdf, sampler, mirror colour, mirror height, env cube ----
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, F = GPUShaderStage.FRAGMENT;
    this.bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 2, visibility: F, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: F, sampler: { type: 'filtering' } },
      { binding: 4, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 5, visibility: F, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: 'cube' } },
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });

    // ---- environment cube map ----
    this.envTex = device.createTexture({ size: [ENV_SIZE, ENV_SIZE, 6], format: eng.HDR, mipLevelCount: ENV_MIPS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label: 'env cube' });
    this.envCube = this.envTex.createView({ dimension: 'cube' });
    this.envFace = [];
    for (let l = 0; l < ENV_MIPS; l++) {
      this.envFace.push([0, 1, 2, 3, 4, 5].map((f) => this.envTex.createView({ dimension: '2d', baseMipLevel: l, mipLevelCount: 1, baseArrayLayer: f, arrayLayerCount: 1 })));
    }
    const bakeMod = await eng.shader(ENV_BAKE, 'env bake');
    const mipMod = await eng.shader(ENV_MIP, 'env mip');
    const noDepth = { primitive: { topology: 'triangle-list' } };
    this.pBake = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: bakeMod, entryPoint: 'bvs' }, fragment: { module: bakeMod, entryPoint: 'bfs', targets: [{ format: eng.HDR }] }, ...noDepth });
    this.pEnvMip = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: mipMod, entryPoint: 'bvs' }, fragment: { module: mipMod, entryPoint: 'bfs', targets: [{ format: eng.HDR }] }, ...noDepth });
    this.envMipBG = [];
    for (let l = 1; l < ENV_MIPS; l++) {
      const ub = eng.mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'env mip ' + l);
      queue.writeBuffer(ub, 0, new Float32Array([1 / (ENV_SIZE >> l), 0, 0, 0]));
      const src = this.envTex.createView({ dimension: 'cube', baseMipLevel: l - 1, mipLevelCount: 1 });
      this.envMipBG[l] = device.createBindGroup({ layout: this.pEnvMip.getBindGroupLayout(0), entries: [
        { binding: 0, resource: src }, { binding: 1, resource: eng.sampClamp }, { binding: 2, resource: { buffer: ub } }] });
    }
    this.envKey = '';

    // ---- car + road pipelines ----
    const mod = await eng.shader(CAR_SHADER, 'car');
    const vbuf = [
      { arrayStride: 56, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x4' }, { shaderLocation: 3, offset: 40, format: 'float32x4' }] },
      { arrayStride: 64, stepMode: 'instance', attributes: [4, 5, 6, 7].map((l, i) => ({ shaderLocation: l, offset: i * 16, format: 'float32x4' })) },
    ];
    const carPipe = (fs, label) => eng.renderPipe({ label, layout, vertex: { module: mod, entryPoint: 'cvs', buffers: vbuf }, fragment: { module: mod, entryPoint: fs } });
    const mirrorPipe = (fs, label) => device.createRenderPipelineAsync({
      label, layout, primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: eng.DEPTH, depthWriteEnabled: true, depthCompare: 'greater' },
      vertex: { module: mod, entryPoint: 'cvs', buffers: vbuf },
      fragment: { module: mod, entryPoint: fs, targets: [{ format: eng.HDR }, { format: eng.HDR }] },
    });
    [this.pBody, this.pWheel, this.pSky, this.pGround, this.pBodyM, this.pWheelM] = await Promise.all([
      carPipe('fs_body', 'car body'),
      carPipe('fs_wheel', 'car wheel'),
      eng.renderPipe({ label: 'sky', layout, vertex: { module: mod, entryPoint: 'vs' }, fragment: { module: mod, entryPoint: 'fs_sky' },
        depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'equal' } }),
      eng.renderPipe({ label: 'road', layout, vertex: { module: mod, entryPoint: 'gvs' }, fragment: { module: mod, entryPoint: 'gfs' } }),
      mirrorPipe('fs_body_m', 'mirror body'),
      mirrorPipe('fs_wheel_m', 'mirror wheel'),
    ]);
    const mipPipeMod = await eng.shader(MIP2D, 'mip2d');
    this.pMip2d = await device.createRenderPipelineAsync({ layout: 'auto', vertex: { module: mipPipeMod, entryPoint: 'vs' }, fragment: { module: mipPipeMod, entryPoint: 'fs', targets: [{ format: eng.HDR }] } });

    this.uboN = eng.mkBuf(176, GU.UNIFORM | GU.COPY_DST, 'env');
    this.uboM = eng.mkBuf(176, GU.UNIFORM | GU.COPY_DST, 'env mirror');
    this.env = new Float32Array(44);
    this.scroll = 0;
    this.lastStyle = eng.P.rimStyle;
    // 1x1 stand-ins for the mirror textures while the mirror pass itself renders into them
    this.dummy = device.createTexture({ size: [1, 1], format: eng.HDR, usage: TU.TEXTURE_BINDING, label: 'dummy' }).createView();
    this.makeGroups(eng, null);
    console.log(`car ready in ${(performance.now() - t0).toFixed(0)} ms`);
  },

  makeGroups(eng, mirror) {
    const dev = eng.device;
    const mk = (ubo, a, b) => dev.createBindGroup({ layout: this.bgl, entries: [
      { binding: 0, resource: { buffer: eng.camUBO } }, { binding: 1, resource: { buffer: ubo } },
      { binding: 2, resource: this.sdfTex.createView() }, { binding: 3, resource: eng.sampClamp },
      { binding: 4, resource: a }, { binding: 5, resource: b }, { binding: 6, resource: this.envCube }] });
    this.bgMirror = mk(this.uboM, this.dummy, this.dummy);
    this.bgMain = mk(this.uboN, mirror ? mirror.viewA : this.dummy, mirror ? mirror.viewB : this.dummy);
  },

  // offscreen mirrored-scene targets (colour, height) with a full mip chain, rebuilt on resize
  onResize(eng) {
    const dev = eng.device;
    const m = this.mirror;
    if (m) { m.a.destroy(); m.b.destroy(); m.depth.destroy(); }
    const W = eng.W, H = eng.H;
    const levels = Math.min(8, Math.floor(Math.log2(Math.max(W, H))) + 1);
    const mk = (label) => dev.createTexture({ size: [W, H], format: eng.HDR, mipLevelCount: levels, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING, label });
    const a = mk('mirror color'), b = mk('mirror height');
    const depth = dev.createTexture({ size: [W, H], format: eng.DEPTH, usage: TU.RENDER_ATTACHMENT, label: 'mirror depth' });
    const lv = (t, l) => t.createView({ baseMipLevel: l, mipLevelCount: 1 });
    const chain = (t) => {
      const out = [];
      for (let l = 1; l < levels; l++) {
        out.push({ view: lv(t, l), bg: dev.createBindGroup({ layout: this.pMip2d.getBindGroupLayout(0), entries: [{ binding: 0, resource: lv(t, l - 1) }, { binding: 1, resource: eng.sampClamp }] }) });
      }
      return out;
    };
    this.mirror = { a, b, depth, levels, viewA: a.createView(), viewB: b.createView(), v0A: lv(a, 0), v0B: lv(b, 0), depthView: depth.createView(), chainA: chain(a), chainB: chain(b) };
    this.makeGroups(eng, this.mirror);
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
    f.set([this.scroll, eng.time, P.lamps * (1 - 0.6 * sstep(8, 30, e)), 0], 24);
    f.set([SDF_ORG[0], SDF_ORG[1], SDF_ORG[2], 0], 28);
    f.set([1 / (SDF_DIMS[0] * SDF_CELL), 1 / (SDF_DIMS[1] * SDF_CELL), 1 / (SDF_DIMS[2] * SDF_CELL), 0], 32);
    f.set([P.peel * 0.8, 60, 1, P.drops], 36);
    f[23] = 0;
    eng.queue.writeBuffer(this.uboN, 0, f);
    f[23] = 1;
    eng.queue.writeBuffer(this.uboM, 0, f);
    this.envKeyNow = [P.sunElev, P.sunAzim, P.lamps].join(',');
    if (qs.get('perf')) {      // dev: log the wall-clock frame time every 120 frames
      const n = performance.now();
      if (this.perfT) { this.perfAcc = (this.perfAcc || 0) + (n - this.perfT); this.perfN = (this.perfN || 0) + 1; }
      this.perfT = n;
      if (this.perfN === 120) { fetch('/__log', { method: 'POST', body: `perf ${(this.perfAcc / 120).toFixed(2)} ms/frame` }); this.perfN = 0; this.perfAcc = 0; }
    }
  },

  // bake + prefilter the environment when the sky changed, then render the mirrored car
  compute(eng, enc) {
    const dev = eng.device;
    if (this.envKeyNow !== this.envKey) {
      this.envKey = this.envKeyNow;
      for (let f = 0; f < 6; f++) {
        const p = enc.beginRenderPass({ colorAttachments: [{ view: this.envFace[0][f], loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }] });
        p.setPipeline(this.pBake);
        p.setBindGroup(0, dev.createBindGroup({ layout: this.pBake.getBindGroupLayout(0), entries: [{ binding: 1, resource: { buffer: this.uboN } }] }));
        p.draw(3, 1, 0, f);
        p.end();
      }
      for (let l = 1; l < ENV_MIPS; l++) {
        for (let f = 0; f < 6; f++) {
          const p = enc.beginRenderPass({ colorAttachments: [{ view: this.envFace[l][f], loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }] });
          p.setPipeline(this.pEnvMip);
          p.setBindGroup(0, this.envMipBG[l]);
          p.draw(3, 1, 0, f);
          p.end();
        }
      }
    }
    const m = this.mirror;
    if (!m || (qs.get('off') || '').includes('mirror')) return;
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: m.v0A, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' },
        { view: m.v0B, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
      depthStencilAttachment: { view: m.depthView, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    this.drawCar(rp, true);
    rp.end();
    for (const chain of [m.chainA, m.chainB]) {
      for (const lvl of chain) {
        const p = enc.beginRenderPass({ colorAttachments: [{ view: lvl.view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
        p.setPipeline(this.pMip2d); p.setBindGroup(0, lvl.bg); p.draw(3); p.end();
      }
    }
  },

  drawCar(rp, mirror) {
    rp.setBindGroup(0, mirror ? this.bgMirror : this.bgMain);
    rp.setPipeline(mirror ? this.pBodyM : this.pBody);
    rp.setVertexBuffer(0, this.mBody.vb); rp.setVertexBuffer(1, this.instBody);
    rp.setIndexBuffer(this.mBody.ib, 'uint32');
    rp.drawIndexed(this.mBody.count, 2);
    rp.setPipeline(mirror ? this.pWheelM : this.pWheel);
    rp.setVertexBuffer(0, this.mWheel.vb); rp.setVertexBuffer(1, this.instWheel);
    rp.setIndexBuffer(this.mWheel.ib, 'uint32');
    rp.drawIndexed(this.mWheel.count, 4);
    rp.setVertexBuffer(0, this.mCal.vb); rp.setVertexBuffer(1, this.instCal);
    rp.setIndexBuffer(this.mCal.ib, 'uint32');
    rp.drawIndexed(this.mCal.count, 4);
  },

  draw(eng, rp) {
    const off = qs.get('off') || '';       // dev: ?off=car,sky,road,mirror switches passes off for profiling
    if (!off.includes('car')) this.drawCar(rp, false);
    rp.setBindGroup(0, this.bgMain);
    if (!off.includes('sky')) { rp.setPipeline(this.pSky); rp.draw(3); }      // sky fills what the car and road leave
    if (!off.includes('road')) { rp.setPipeline(this.pGround); rp.draw(6); }  // road
  },
};

boot(scene);
