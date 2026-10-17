// Campfire at dusk on an open meadow: raymarched turbulent flame volume, charred logs with glowing
// cracks, GPU particle sparks / flame fragments / smoke, flickering point light with analytic stone +
// log shadows, instanced grass blades lit by the fire, procedural dusk-to-night sky with stars, Milky
// Way, moon and a distant treeline.
import { boot } from '../../engine/engine.js';
import { GU, TU } from '../../engine/gpu.js';
import { buildSolids, buildGrass, groundHeight, NBLK } from './mesh.js';
import { SKY, GROUND, GRASS, SOLID } from './wgsl_world.js';
import { NOISE_GEN, NOISE_GEN2, FLAME, SIM, SPARK, SMOKE, FIREFLY } from './wgsl_fx.js';

const NOISE_RES = 128;
const MAX_GRASS = 200000;
const N_SPARK = 2200, N_BLOB = 900, N_SMOKE = 2400, N_FLY = 220;
const STAGE = GPUShaderStage;

// ---- tiny smooth 1-D value noise for the flicker signals -----------------------------------
const h1 = (n) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };
function vn(x) {
  const i = Math.floor(x), f = x - i;
  const u = f * f * (3 - 2 * f);
  return h1(i) * (1 - u) + h1(i + 1) * u;
}
const mix = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const mix3 = (a, b, t) => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];

// everything that varies with time / sliders: flame size, light flicker, wind, sky colours
function fireState(P, t) {
  const S = P.size;
  const lightFlick = 1 + 0.2 * (2 * vn(t * 8.3) - 1) + 0.1 * (2 * vn(t * 2.9 + 7) - 1) + 0.05 * Math.sin(t * 21);
  const heightF = Math.min(1.3, Math.max(0.7, 0.92 + 0.28 * vn(t * 1.4 + 3) + 0.14 * (2 * vn(t * 5.7 + 9) - 1)));
  const wMag = P.wind * (0.55 + 0.7 * vn(t * 0.17));
  const wAng = 0.7 + 0.5 * Math.sin(t * 0.05);
  const wind = [Math.cos(wAng) * wMag, Math.sin(wAng) * wMag];
  const gust = vn(t * 0.35);
  const bend = [
    Math.sin(t * 1.3) * 0.08 + Math.sin(t * 2.9 + 1.7) * 0.04 + (vn(t * 0.8) - 0.5) * 0.22 + wind[0] * 0.6,
    Math.cos(t * 1.1) * 0.07 + Math.cos(t * 3.3) * 0.03 + (vn(t * 0.7 + 5) - 0.5) * 0.2 + wind[1] * 0.6,
  ];
  const light = [
    bend[0] * 0.25 * S + (vn(t * 5.1 + 3) - 0.5) * 0.16,
    0.25 + 0.55 * S * heightF + (vn(t * 6.3 + 1) - 0.5) * 0.1,
    bend[1] * 0.25 * S + (vn(t * 4.7 + 8) - 0.5) * 0.16,
  ];
  const intensity = 18 * P.intensity * S * S * lightFlick * (0.7 + 0.3 * heightF);

  const k = P.tod;
  const night = sstep(0.0, 0.75, k);
  const zen = mix3([0.05, 0.095, 0.21], [0.0025, 0.005, 0.015], Math.pow(sstep(0, 1, k), 0.8));
  const hor = mix3([0.62, 0.26, 0.13], [0.010, 0.016, 0.03], Math.pow(night, 0.7));
  const amb = mix3([0.024, 0.030, 0.050], [0.0016, 0.0022, 0.0055], Math.pow(night, 0.6));
  const moonVis = sstep(0.2, 0.7, k);
  const moonCol = [0.50 * moonVis * 0.33, 0.62 * moonVis * 0.33, 0.9 * moonVis * 0.33];
  for (let i = 0; i < 3; i++) { amb[i] *= 0.6; }
  return { S, lightFlick, heightF, wind, gust, bend, light, intensity, zen, hor, amb, moonVis, moonCol, stars: sstep(0.05, 0.7, k) };
}

const scene = {
  name: 'fire',
  msaa: 4,
  params: {
    size: 1.0, intensity: 1.0, wind: 0.3, turb: 1.0, embers: 1.0, smoke: 1.0, grass: 1.0, grassWind: 1.0,
    tod: 0.62, mist: 0.85, haze: 1.0, exposure: 1.3, bloom: 0.2, vignette: 0.3, sat: 1.05, contrast: 1.1, warmth: 0.0, grain: 0.35,
  },
  sliders: [
    ['size', 'Fire size', 0.5, 1.6, 0.01],
    ['intensity', 'Fire light', 0.2, 2.5, 0.01],
    ['turb', 'Flame turbulence', 0.2, 2.0, 0.01],
    ['wind', 'Wind', 0, 1.2, 0.01],
    ['embers', 'Embers / sparks', 0, 3, 0.01],
    ['smoke', 'Smoke', 0, 2.5, 0.01],
    ['grass', 'Grass density', 0.05, 1, 0.01],
    ['grassWind', 'Grass wind', 0, 3, 0.01],
    ['mist', 'Mist', 0, 3, 0.01],
    ['haze', 'Heat haze', 0, 3, 0.01],
    ['tod', 'Dusk → night', 0, 1, 0.005],
    ['exposure', 'Exposure', 0.2, 6, 0.01],
    ['bloom', 'Bloom', 0, 0.4, 0.002],
  ],
  camera: { mode: 'walk', x: 2.3, y: 1.2, z: 2.75, yaw: -40, pitch: 5, fov: 56, speed: 1.6, eye: 1.15, near: 0.03 },
  groundHeight,
  help: 'click: mouse look · WASD walk · Shift run · wheel speed · H panel',

  constrainCamera(eng) {
    const c = eng.cam;
    const r = Math.hypot(c.x, c.z);
    const R = 1.5 * Math.max(1, this.S * 0.85);
    if (r < R) { const k = R / Math.max(r, 1e-3); c.x = r < 1e-3 ? R : c.x * k; c.z *= k; }
  },

  async init(eng) {
    const { device, queue, P } = eng;
    if (eng.qs.has('eye')) eng.cam.eye = Number(eng.qs.get('eye'));
    this.S = P.size;
    this.eng_queue = queue;

    // ---- explicit layouts so one bind group serves every render pipeline ------------------
    const VF = STAGE.VERTEX | STAGE.FRAGMENT;
    const bglR = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 2, visibility: VF, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: VF, sampler: { type: 'filtering' } },
      { binding: 4, visibility: VF, texture: { sampleType: 'float', viewDimension: '3d' } },
    ] });
    const bglP = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: STAGE.VERTEX, buffer: { type: 'read-only-storage' } }] });
    const bglC = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: STAGE.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: STAGE.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: STAGE.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: STAGE.COMPUTE, sampler: { type: 'filtering' } },
    ] });
    const bglN = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: STAGE.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' } },
    ] });
    const plR = device.createPipelineLayout({ bindGroupLayouts: [bglR] });
    const plRP = device.createPipelineLayout({ bindGroupLayouts: [bglR, bglP] });
    const plC = device.createPipelineLayout({ bindGroupLayouts: [bglC] });
    const plN = device.createPipelineLayout({ bindGroupLayouts: [bglN] });

    // ---- noise volumes ---------------------------------------------------------------------
    const mkVolume = async (code, label) => {
      const tex = device.createTexture({ size: [NOISE_RES, NOISE_RES, NOISE_RES], dimension: '3d', format: 'rgba8unorm', usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING });
      const mod = await eng.shader(code, label);
      const pipe = await device.createComputePipelineAsync({ layout: plN, compute: { module: mod, entryPoint: 'main' } });
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe);
      cp.setBindGroup(0, device.createBindGroup({ layout: bglN, entries: [{ binding: 0, resource: tex.createView({ dimension: '3d' }) }] }));
      cp.dispatchWorkgroups(NOISE_RES / 4, NOISE_RES / 4, NOISE_RES / 4);
      cp.end();
      queue.submit([enc.finish()]);
      return tex.createView({ dimension: '3d' });
    };
    const noiseView = await mkVolume(NOISE_GEN, 'noise gen');
    const noiseView2 = await mkVolume(NOISE_GEN2, 'noise gen 2');
    const nsamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });

    // ---- uniforms --------------------------------------------------------------------------
    this.scnF = new Float32Array((10 + NBLK) * 4);
    this.scnBuf = eng.mkBuf(this.scnF.byteLength, GU.UNIFORM | GU.COPY_DST, 'scene ubo');
    this.simF = new Float32Array(12);
    this.simBuf = eng.mkBuf(48, GU.UNIFORM | GU.COPY_DST, 'sim ubo');
    const solids = buildSolids();
    this.scnF.set(solids.blockers, 40);
    this.bgR = device.createBindGroup({ layout: bglR, entries: [
      { binding: 0, resource: { buffer: eng.camUBO } }, { binding: 1, resource: { buffer: this.scnBuf } },
      { binding: 2, resource: noiseView }, { binding: 3, resource: nsamp }, { binding: 4, resource: noiseView2 },
    ] });

    // ---- geometry --------------------------------------------------------------------------
    this.solidVB = eng.bufFrom(solids.vertices, GU.VERTEX, 'solid vb');
    this.solidIB = eng.bufFrom(solids.indices, GU.INDEX, 'solid ib');
    this.solidCount = solids.indices.length;
    const GN = 256;
    const gi = new Uint32Array(GN * GN * 6);
    for (let j = 0, k = 0; j < GN; j++) {
      for (let i = 0; i < GN; i++) {
        const a = i + j * (GN + 1), b = a + 1, c = a + GN + 1, d = c + 1;
        gi[k++] = a; gi[k++] = c; gi[k++] = b; gi[k++] = b; gi[k++] = c; gi[k++] = d;
      }
    }
    this.groundIB = eng.bufFrom(gi, GU.INDEX, 'ground ib');
    this.groundCount = gi.length;
    this.grassVB = eng.bufFrom(buildGrass(MAX_GRASS), GU.VERTEX, 'grass');

    // ---- particles -------------------------------------------------------------------------
    const mkPool = (n, delaySpread) => {
      const data = new Float32Array(n * 12);
      for (let i = 0; i < n; i++) { data[i * 12 + 3] = 1000; data[i * 12 + 7] = 1; data[i * 12 + 11] = Math.random() * delaySpread; }
      const buf = eng.bufFrom(data, GU.STORAGE, 'pool');
      return {
        n, buf,
        bgC: device.createBindGroup({ layout: bglC, entries: [
          { binding: 0, resource: { buffer: this.simBuf } }, { binding: 1, resource: { buffer: buf } },
          { binding: 2, resource: noiseView }, { binding: 3, resource: nsamp },
        ] }),
        bgP: device.createBindGroup({ layout: bglP, entries: [{ binding: 0, resource: { buffer: buf } }] }),
      };
    };
    this.pools = { sparks: mkPool(N_SPARK, 6), blobs: mkPool(N_BLOB, 2), smoke: mkPool(N_SMOKE, 6) };
    const simMod = await eng.shader(SIM, 'sim');
    this.simPipes = {};
    for (const name of ['sparks', 'blobs', 'smoke']) {
      this.simPipes[name] = await device.createComputePipelineAsync({ layout: plC, compute: { module: simMod, entryPoint: name } });
    }

    // ---- render pipelines ------------------------------------------------------------------
    const noWrite = { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'greater' };
    const addBlend = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    const modOf = async (code, label) => eng.shader(code, label);
    const [mSky, mGround, mGrass, mSolid, mFlame, mSpark, mSmoke, mFly] = await Promise.all([
      modOf(SKY, 'sky'), modOf(GROUND, 'ground'), modOf(GRASS, 'grass'), modOf(SOLID, 'solid'),
      modOf(FLAME, 'flame'), modOf(SPARK, 'spark'), modOf(SMOKE, 'smoke'), modOf(FIREFLY, 'firefly'),
    ]);
    const mk = (label, mod, extra = {}, layout = plR) => eng.renderPipe({
      label, layout, vertex: { module: mod, entryPoint: 'vs', ...(extra.buffers ? { buffers: extra.buffers } : {}) },
      fragment: { module: mod, entryPoint: 'fs', ...(extra.targets ? { targets: extra.targets } : {}) },
      ...(extra.depthStencil ? { depthStencil: extra.depthStencil } : {}),
      ...(extra.primitive ? { primitive: extra.primitive } : {}),
    });
    this.pSky = await mk('sky', mSky, { depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'equal' } });
    this.pGround = await mk('ground', mGround);
    this.pGrass = await mk('grass', mGrass, {
      buffers: [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }],
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
    });
    this.pSolid = await mk('solid', mSolid, {
      buffers: [{ arrayStride: 40, attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32x2' }, { shaderLocation: 3, offset: 32, format: 'float32x2' },
      ] }],
    });
    this.pFlame = await mk('flame', mFlame, {
      depthStencil: noWrite,
      targets: [{ format: eng.HDR, blend: { color: { srcFactor: 'one', dstFactor: 'src-alpha', operation: 'add' }, alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' } } }],
    });
    this.pSpark = await mk('spark', mSpark, { depthStencil: noWrite, targets: [{ format: eng.HDR, blend: addBlend }] }, plRP);
    this.pSmoke = await mk('smoke', mSmoke, {
      depthStencil: noWrite,
      targets: [{ format: eng.HDR, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } } }],
    }, plRP);
    this.pFly = await mk('firefly', mFly, { depthStencil: noWrite, targets: [{ format: eng.HDR, blend: addBlend }] });

    // ---- warm-up: simulate ~14 s so smoke and sparks are already alive on the first frame -----
    const t0 = eng.time - 14;
    for (let i = 0; i < 420; i++) {
      this.writeSim(P, t0 + i * 0.033, 0.033, i);
      const enc = device.createCommandEncoder();
      this.encodeSim(enc);
      queue.submit([enc.finish()]);
    }
    this.simFrame = 420;
  },

  writeSim(P, t, dt, frame) {
    const st = fireState(P, t);
    const f = this.simF;
    f.set([t, dt, Math.max(P.embers, 0.03), P.turb, st.wind[0], st.wind[1], st.S, st.heightF, frame, P.smoke, st.bend[0], st.bend[1]]);
    this.eng_queue.writeBuffer(this.simBuf, 0, f);
  },

  encodeSim(enc) {
    const cp = enc.beginComputePass();
    for (const name of ['sparks', 'blobs', 'smoke']) {
      const pool = this.pools[name];
      cp.setPipeline(this.simPipes[name]);
      cp.setBindGroup(0, pool.bgC);
      cp.dispatchWorkgroups(Math.ceil(pool.n / 64));
    }
    cp.end();
  },

  update(eng, dt, t) {
    this.eng_queue = eng.queue;
    const P = eng.P;
    const st = fireState(P, t);
    this.S = st.S;
    const f = this.scnF;
    f.set([st.light[0], st.light[1], st.light[2], st.intensity], 0);
    f.set([1.0, 0.46, 0.17, st.S], 4);
    f.set([st.wind[0], st.wind[1], st.gust, P.grassWind], 8);
    f.set([P.tod, P.mist, P.haze, P.turb], 12);
    const e = 0.36, a = -0.84;   // moon: elevation, azimuth
    f.set([Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a), st.moonVis], 16);
    f.set([st.zen[0], st.zen[1], st.zen[2], st.stars], 20);
    f.set([st.hor[0], st.hor[1], st.hor[2], P.smoke], 24);
    f.set([st.amb[0], st.amb[1], st.amb[2], P.embers], 28);
    f.set([st.bend[0], st.bend[1], st.heightF, 0], 32);
    f.set([st.moonCol[0], st.moonCol[1], st.moonCol[2], 0], 36);
    eng.queue.writeBuffer(this.scnBuf, 0, f);
    this.writeSim(P, t, Math.min(dt, 0.05), this.simFrame++);
  },

  compute(eng, enc) { this.encodeSim(enc); },

  draw(eng, rp) {
    const P = eng.P;
    rp.setBindGroup(0, this.bgR);
    rp.setPipeline(this.pSolid);
    rp.setVertexBuffer(0, this.solidVB);
    rp.setIndexBuffer(this.solidIB, 'uint32');
    rp.drawIndexed(this.solidCount);

    rp.setPipeline(this.pGround);
    rp.setIndexBuffer(this.groundIB, 'uint32');
    rp.drawIndexed(this.groundCount);

    rp.setPipeline(this.pGrass);
    rp.setVertexBuffer(0, this.grassVB);
    rp.draw(11, Math.max(1, Math.floor(P.grass * MAX_GRASS)));

    rp.setPipeline(this.pSky);
    rp.draw(3);

    rp.setPipeline(this.pSmoke);
    rp.setBindGroup(1, this.pools.smoke.bgP);
    rp.draw(6, N_SMOKE);

    rp.setPipeline(this.pFlame);
    rp.setBindGroup(0, this.bgR);
    rp.draw(6);

    rp.setPipeline(this.pSpark);
    rp.setBindGroup(1, this.pools.blobs.bgP);
    if (eng.qs.get('blobs') !== '0') rp.draw(6, N_BLOB);
    rp.setBindGroup(1, this.pools.sparks.bgP);
    rp.draw(6, N_SPARK);

    rp.setPipeline(this.pFly);
    rp.draw(6, N_FLY);
  },
};

boot(scene);
