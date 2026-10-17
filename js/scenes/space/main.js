// Black hole: relativistic ray tracing (see wgsl.js). Orbit camera around the hole.
import { boot } from '../../engine/engine.js';
import { log } from '../../engine/gpu.js';
import { TU, GU } from '../../engine/gpu.js';
import { D2R } from '../../engine/math.js';
import { TRACE, COMP } from './wgsl.js';

// prograde ISCO of a Kerr hole (Bardeen et al. 1972), in units of M, returned in rs = 2M
function isco(a) {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return (3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2))) * 0.5;
}

const scene = {
  name: 'space',
  msaa: 1,
  params: {
    exposure: 1.0, bloom: 0.14, vignette: 0.22, sat: 1.08, contrast: 1.06, warmth: 0.04, grain: 0.35,
    spin: 0.7, mass: 1.0, bright: 1.0, temp: 6500, tilt: 0, thick: 1.0, steps: 320, starDensity: 1.0, scale: 1.0, auto: 1,
  },
  sliders: [
    ['spin', 'Black hole spin', 0, 0.99, 0.01],
    ['mass', 'Mass (rs scale)', 0.4, 2.5, 0.01],
    ['bright', 'Disc brightness', 0.1, 4, 0.01],
    ['temp', 'Disc temperature (K)', 2500, 12000, 50],
    ['tilt', 'Disc tilt (deg)', -60, 60, 0.5],
    ['thick', 'Disc thickness', 0.3, 4, 0.01],
    ['steps', 'Lensing steps', 80, 640, 10],
    ['starDensity', 'Star density', 0, 2.5, 0.01],
    ['exposure', 'Exposure', 0.1, 4, 0.01],
    ['bloom', 'Bloom', 0, 0.4, 0.005],
    ['scale', 'Render scale', 0.4, 1.5, 0.05],
    ['auto', 'Auto scale (0/1)', 0, 1, 1],
  ],
  camera: { mode: 'orbit', target: [0, 0, 0], dist: 26, minDist: 1.6, maxDist: 250, yaw: 24, pitch: -7, fov: 52, autoRotate: 1.2, minPitch: -1.45, maxPitch: 1.45 },
  help: 'drag: orbit · wheel: zoom (lensing grows as you fall in) · WASD/QE slide target · H panel',

  async init(eng) {
    const { device } = eng;
    this.ubo = eng.mkBuf(128, GU.UNIFORM | GU.COPY_DST, 'bh ubo');
    this.u = new Float32Array(32);
    const trMod = await eng.shader(TRACE, 'bh trace');
    this.tracePipe = await device.createRenderPipelineAsync({
      label: 'bh trace', layout: 'auto',
      vertex: { module: trMod, entryPoint: 'vs' },
      fragment: { module: trMod, entryPoint: 'fs', targets: [{ format: 'rgba16float' }, { format: 'rgba32float' }] },
    });
    const cpMod = await eng.shader(COMP, 'bh composite');
    this.compPipe = await eng.renderPipe({
      label: 'bh composite',
      vertex: { module: cpMod, entryPoint: 'vs' },
      fragment: { module: cpMod, entryPoint: 'fs' },
      depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'always' },
    });
    // very large canvases (hi-dpi) start at a lower trace resolution; auto-scale trims further if needed
    if (eng.W * eng.H > 3.0e6 && !eng.qs.has('scale') && !eng.qs.has('shot')) eng.P.scale = 0.6;
    this.avg = 0; this.acc = 0;
    this.traceBG = eng.bg(this.tracePipe, [{ buffer: eng.camUBO }, { buffer: this.ubo }], 'trace bg');
    this.alloc(eng);
  },

  // pass-A targets follow the output size x render scale
  alloc(eng) {
    const { device } = eng;
    const w = Math.max(16, Math.round(eng.W * eng.P.scale)), h = Math.max(16, Math.round(eng.H * eng.P.scale));
    if (this.texA && this.aw === w && this.ah === h) return;
    this.texA && this.texA.destroy(); this.texD && this.texD.destroy();
    this.aw = w; this.ah = h;
    const usage = TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING;
    this.texA = device.createTexture({ size: [w, h], format: 'rgba16float', usage });
    this.texD = device.createTexture({ size: [w, h], format: 'rgba32float', usage });
    this.passA = {
      colorAttachments: [
        { view: this.texA.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
        { view: this.texD.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
      ],
    };
    // the composite shader does not read the camera block, so auto layout starts at binding 1
    this.compBG = device.createBindGroup({
      label: 'comp bg', layout: this.compPipe.getBindGroupLayout(0),
      entries: [{ binding: 1, resource: { buffer: this.ubo } }, { binding: 2, resource: this.texA.createView() },
        { binding: 3, resource: this.texD.createView() }, { binding: 4, resource: eng.sampClamp }],
    });
  },
  // ?bench=1 : GPU time of the trace pass and the composite pass (ms/frame), logged once
  async bench(eng) {
    const { device } = eng;
    const col = device.createTexture({ size: [eng.W, eng.H], format: 'rgba16float', usage: TU.RENDER_ATTACHMENT });
    const dep = device.createTexture({ size: [eng.W, eng.H], format: eng.DEPTH, usage: TU.RENDER_ATTACHMENT });
    const pb = { colorAttachments: [{ view: col.createView(), loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: dep.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' } };
    const time = async (fn, n = 20) => {
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      for (let i = 0; i < n; i++) { const e = device.createCommandEncoder(); fn(e); device.queue.submit([e.finish()]); }
      await device.queue.onSubmittedWorkDone();
      return (performance.now() - t0) / n;
    };
    const a = await time((e) => this.compute(eng, e));
    const b = await time((e) => { const r = e.beginRenderPass(pb); this.draw(eng, r); r.end(); });
    log(`bench ${eng.W}x${eng.H} scale ${eng.P.scale}: trace ${a.toFixed(2)} ms, composite ${b.toFixed(2)} ms`);
  },
  onResize(eng) { this.alloc(eng); },
  onParam(key, eng) { if (key === 'scale') this.alloc(eng); },

  update(eng) {
    const P = eng.P, C = eng.cam, u = this.u;
    C.dist = Math.max(C.dist, 1.5 * P.mass);
    const t = P.tilt * D2R;
    const rs = P.mass;
    u[0] = 1; u[1] = P.spin; u[2] = P.bright; u[3] = P.temp;
    u[4] = 0; u[5] = P.thick; u[6] = P.steps; u[7] = eng.time;
    u[8] = this.aw; u[9] = this.ah; u[10] = isco(P.spin) / 1.0; u[11] = 11;
    u[12] = 0; u[13] = Math.cos(t); u[14] = Math.sin(t); u[15] = 0.05 * Math.min(2.5, Math.max(0.6, 320 / P.steps));
    u[16] = P.starDensity; u[17] = this.aw / eng.W; u[18] = eng.W; u[19] = eng.H;
    u[20] = C.x / rs; u[21] = C.y / rs; u[22] = C.z / rs;
    u[23] = 2 * eng.basis.tanY / 1080;
    u[24] = 2 * eng.basis.tanY / eng.H;
    eng.queue.writeBuffer(this.ubo, 0, u);
    // auto scale: if frames run slow for a second, drop the trace resolution (never raises it again)
    if (P.auto && !eng.qs.has('shot') && eng.dt > 0) {
      this.avg = this.avg ? this.avg * 0.95 + eng.dt * 0.05 : eng.dt;
      this.acc += eng.dt;
      if (this.acc > 1.0) {
        this.acc = 0;
        if (this.avg > 0.0205 && P.scale > 0.5) { P.scale = Math.max(0.5, +(P.scale - 0.1).toFixed(2)); this.alloc(eng); this.avg = 0; }
      }
    }
    if (eng.qs.has('bench') && eng.frame === 5) this.bench(eng);
  },

  compute(eng, enc) {
    const rp = enc.beginRenderPass(this.passA);
    rp.setPipeline(this.tracePipe);
    rp.setBindGroup(0, this.traceBG);
    rp.draw(3);
    rp.end();
  },

  draw(eng, rp) {
    rp.setPipeline(this.compPipe);
    rp.setBindGroup(0, this.compBG);
    rp.draw(3);
  },
};

boot(scene);
