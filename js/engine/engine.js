// Scene host: device, HDR + MSAA targets, camera controllers, bloom + AgX composite, HUD / panel,
// URL parameters and the screenshot hook. A scene is a plain object (see js/scenes/_template).
import { initGPU, makeHelpers, fail, log, GU, TU, HDR, DEPTH } from './gpu.js';
import { BLOOM_DOWN, BLOOM_UP, COMPOSITE } from './post.js';
import { mat4Mul, D2R, clamp } from './math.js';

const BLOOM_LEVELS = 8;
const NAV = [['car', 'Supercar'], ['city', 'New York'], ['space', 'Black hole'], ['fire', 'Campfire']];

export async function run(scene) {
  const qs = new URLSearchParams(location.search);
  const canvas = document.getElementById('c');
  const { device, ctx, swapFormat } = await initGPU(canvas);
  const helpers = makeHelpers(device);
  const queue = device.queue;

  // ------------------------------------------------------------------------------------
  // Engine object handed to the scene
  // ------------------------------------------------------------------------------------
  const P = { exposure: 1.0, bloom: 0.1, vignette: 0.1, sat: 1.05, contrast: 1.06, warmth: 0, grain: 0, ...scene.params };
  for (const k of Object.keys(P)) if (qs.has(k)) P[k] = Number(qs.get(k));
  const msaa = scene.msaa || 4;
  const camUBO = helpers.mkBuf(192, GU.UNIFORM | GU.COPY_DST, 'camera');
  const eng = {
    device, queue, canvas, ctx, swapFormat, P, msaa, qs, camUBO,
    ...helpers, HDR, DEPTH,
    W: 0, H: 0, time: Number(qs.get('t')) || 0, frame: 0, dt: 0,
    keys: Object.create(null),
    cam: null,
    sampLinear: device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 8 }),
    sampClamp: device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' }),
    sampNearest: device.createSampler({ magFilter: 'nearest', minFilter: 'nearest', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' }),
    // render pipeline defaults matching the main pass (HDR colour, reverse-Z depth, MSAA)
    mainTargets: { fragmentTargets: [{ format: HDR }], depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: 'greater' }, multisample: { count: msaa } },
    // draw a render pipeline description pre-filled with the main-pass state
    renderPipe(desc) {
      return device.createRenderPipelineAsync({
        layout: 'auto',
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: 'greater' },
        multisample: { count: msaa },
        ...desc,
        fragment: desc.fragment && { targets: [{ format: HDR }], ...desc.fragment },
      });
    },
  };

  // ------------------------------------------------------------------------------------
  // Camera
  // ------------------------------------------------------------------------------------
  const C = { mode: 'fly', x: 0, y: 2, z: 0, yaw: 0, pitch: 0, fov: 60, speed: 5, near: 0.05, minY: -1e9,
    target: [0, 0, 0], dist: 5, minDist: 1, maxDist: 100, minPitch: -1.45, maxPitch: 1.45, eye: 1.7, autoRotate: 0, ...scene.camera };
  C.yaw *= D2R; C.pitch *= D2R; C.fovY = C.fov * D2R;
  if (qs.get('cam')) {
    const v = qs.get('cam').split(',').map(Number);
    if (C.mode === 'orbit') { C.target = [v[0], v[1], v[2]]; C.yaw = (v[3] || 0) * D2R; C.pitch = (v[4] || 0) * D2R; if (v[5]) C.dist = v[5]; }
    else { C.x = v[0]; C.y = v[1]; C.z = v[2]; C.yaw = (v[3] || 0) * D2R; C.pitch = (v[4] || 0) * D2R; }
  }
  if (qs.get('fov')) C.fovY = Number(qs.get('fov')) * D2R;
  eng.cam = C;
  const basis = { fx: 0, fy: 0, fz: -1, rx: 1, ry: 0, rz: 0, ux: 0, uy: 1, uz: 0 };
  eng.basis = basis;
  let idleT = 0;

  const keys = eng.keys;
  addEventListener('keydown', (e) => { keys[e.code] = true; if (e.code === 'KeyH') panel.classList.toggle('hidden'); if (scene.onKey) scene.onKey(eng, e); });
  addEventListener('keyup', (e) => { keys[e.code] = false; });
  addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
  const lookMode = () => C.mode !== 'orbit';
  canvas.addEventListener('click', () => { if (lookMode() && document.pointerLockElement !== canvas) canvas.requestPointerLock(); });
  let dragging = false;
  canvas.addEventListener('mousedown', (e) => { if (e.button === 2 || (e.button === 0 && !lookMode())) dragging = true; idleT = 0; });
  addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas || dragging) {
      C.yaw += e.movementX * 0.0022;
      C.pitch = clamp(C.pitch - e.movementY * 0.0022, C.mode === 'orbit' ? C.minPitch : -1.5, C.mode === 'orbit' ? C.maxPitch : 1.5);
      idleT = 0;
    }
  });
  addEventListener('wheel', (e) => {
    const k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    if (C.mode === 'orbit') C.dist = clamp(C.dist / k, C.minDist, C.maxDist);
    else C.speed = clamp(C.speed * k, 0.05, 5000);
    idleT = 0;
  }, { passive: true });
  // touch: one finger look/orbit, two fingers pinch zoom
  let tch = null;
  canvas.addEventListener('touchstart', (e) => { tch = [...e.touches].map((t) => [t.clientX, t.clientY]); idleT = 0; }, { passive: true });
  canvas.addEventListener('touchmove', (e) => {
    const cur = [...e.touches].map((t) => [t.clientX, t.clientY]);
    if (tch && cur.length === 1 && tch.length === 1) {
      C.yaw += (cur[0][0] - tch[0][0]) * 0.005;
      C.pitch = clamp(C.pitch - (cur[0][1] - tch[0][1]) * 0.005, C.minPitch, C.maxPitch);
    } else if (tch && cur.length === 2 && tch.length === 2) {
      const d0 = Math.hypot(tch[0][0] - tch[1][0], tch[0][1] - tch[1][1]);
      const d1 = Math.hypot(cur[0][0] - cur[1][0], cur[0][1] - cur[1][1]);
      if (C.mode === 'orbit') C.dist = clamp(C.dist * d0 / Math.max(d1, 1), C.minDist, C.maxDist);
    }
    tch = cur; idleT = 0;
  }, { passive: true });

  function updateCamera(dt) {
    const cy = Math.cos(C.yaw), sy = Math.sin(C.yaw), cp = Math.cos(C.pitch), sp = Math.sin(C.pitch);
    idleT += dt;
    const fwd = (keys.KeyW || keys.ArrowUp ? 1 : 0) - (keys.KeyS || keys.ArrowDown ? 1 : 0);
    const str = (keys.KeyD || keys.ArrowRight ? 1 : 0) - (keys.KeyA || keys.ArrowLeft ? 1 : 0);
    const up = (keys.KeyE || keys.Space ? 1 : 0) - (keys.KeyQ || keys.KeyC ? 1 : 0);
    const boost = keys.ShiftLeft || keys.ShiftRight ? 5 : 1;
    if (C.mode === 'orbit') {
      if (C.autoRotate && idleT > 3 && !dragging) C.yaw += C.autoRotate * D2R * dt;
      if (fwd || str || up) {            // WASD/QE slide the orbit target in the view plane
        const v = C.dist * 0.6 * boost * dt;
        C.target[0] += (str * cy + fwd * sy) * v;
        C.target[2] += (str * sy - fwd * cy) * v;
        C.target[1] += up * v;
        idleT = 0;
      }
      const fx = cp * sy, fy = sp, fz = -cp * cy;
      C.x = C.target[0] - fx * C.dist; C.y = C.target[1] - fy * C.dist; C.z = C.target[2] - fz * C.dist;
      return;
    }
    const v = C.speed * boost * dt;
    if (C.mode === 'walk') {
      C.x += (fwd * sy + str * cy) * v;
      C.z += (-fwd * cy + str * sy) * v;
      const g = scene.groundHeight ? scene.groundHeight(C.x, C.z) : 0;
      const target = g + C.eye;
      C.y += (target - C.y) * Math.min(1, dt * 14);
    } else {
      C.x += (fwd * cp * sy + str * cy) * v;
      C.y += (fwd * sp + up) * v;
      C.z += (-fwd * cp * cy + str * sy) * v;
      if (C.y < C.minY) C.y = C.minY;
    }
    if (scene.constrainCamera) scene.constrainCamera(eng, dt);
  }

  const proj = new Float32Array(16), view = new Float32Array(16), viewProj = new Float32Array(16);
  const camF = new Float32Array(48);
  eng.viewProj = viewProj;
  function writeCamera() {
    const aspect = eng.W / eng.H;
    const tanY = Math.tan(C.fovY * 0.5), tanX = tanY * aspect;
    proj.fill(0);
    proj[0] = 1 / tanX; proj[5] = 1 / tanY; proj[11] = -1; proj[14] = C.near;
    const cy = Math.cos(C.yaw), sy = Math.sin(C.yaw), cp = Math.cos(C.pitch), sp = Math.sin(C.pitch);
    const fx = cp * sy, fy = sp, fz = -cp * cy;
    const rx = cy, ry = 0, rz = sy;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    view.fill(0);
    view[0] = rx; view[4] = ry; view[8] = rz;
    view[1] = ux; view[5] = uy; view[9] = uz;
    view[2] = -fx; view[6] = -fy; view[10] = -fz;
    view[15] = 1;
    mat4Mul(viewProj, proj, view);
    Object.assign(basis, { fx, fy, fz, rx, ry, rz, ux, uy, uz, tanX, tanY });
    const f = camF;
    f.set(viewProj, 0);
    f[16] = C.x; f[17] = C.y; f[18] = C.z; f[19] = eng.time;
    f[20] = rx * tanX; f[21] = ry * tanX; f[22] = rz * tanX;
    f[24] = ux * tanY; f[25] = uy * tanY; f[26] = uz * tanY;
    f[28] = fx; f[29] = fy; f[30] = fz;
    f[32] = eng.W; f[33] = eng.H; f[34] = 1 / eng.W; f[35] = 1 / eng.H;
    f[36] = C.near; f[37] = eng.frame; f[38] = eng.dt; f[39] = 0;
    queue.writeBuffer(camUBO, 0, f);
  }

  // ------------------------------------------------------------------------------------
  // Render targets
  // ------------------------------------------------------------------------------------
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const compUBO = helpers.mkBuf(32, GU.UNIFORM | GU.COPY_DST, 'composite ubo');
  const compData = new Float32Array(8);
  const bloomUBOs = [];
  for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(helpers.mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'bloom ubo'));
  const [downMod, upMod, compMod] = await Promise.all([
    helpers.shader(BLOOM_DOWN, 'bloom down'), helpers.shader(BLOOM_UP, 'bloom up'), helpers.shader(COMPOSITE, 'composite'),
  ]);
  const blendAdd = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
  const [downPipe, upPipe, compPipe] = await Promise.all([
    device.createRenderPipelineAsync({ label: 'bloom down', layout: 'auto', vertex: { module: downMod, entryPoint: 'vs' }, fragment: { module: downMod, entryPoint: 'fs', targets: [{ format: HDR }] } }),
    device.createRenderPipelineAsync({ label: 'bloom up', layout: 'auto', vertex: { module: upMod, entryPoint: 'vs' }, fragment: { module: upMod, entryPoint: 'fs', targets: [{ format: HDR, blend: blendAdd }] } }),
    device.createRenderPipelineAsync({ label: 'composite', layout: 'auto', vertex: { module: compMod, entryPoint: 'vs' }, fragment: { module: compMod, entryPoint: 'fs', targets: [{ format: swapFormat }] } }),
  ]);

  let colorMS, depthMS, hdrTex, bloomTex;
  const mainPass = {
    colorAttachments: [{ view: null, resolveTarget: undefined, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: msaa > 1 ? 'discard' : 'store' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  const compPass = { colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] };
  let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null;

  function resize() {
    const fw = Number(qs.get('w')) || 0, fh = Number(qs.get('h')) || 0;
    const dpr = Math.min(window.devicePixelRatio || 1, Number(qs.get('dpr')) || 2);
    const w = fw || Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = fh || Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (w === eng.W && h === eng.H) return false;
    const W = eng.W = w, H = eng.H = h;
    canvas.width = W; canvas.height = H;
    for (const t of [colorMS, depthMS, hdrTex, bloomTex]) t && t.destroy();
    colorMS = msaa > 1 ? device.createTexture({ size: [W, H], format: HDR, sampleCount: msaa, usage: TU.RENDER_ATTACHMENT }) : null;
    depthMS = device.createTexture({ size: [W, H], format: DEPTH, sampleCount: msaa, usage: TU.RENDER_ATTACHMENT });
    hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const hdrView = hdrTex.createView();
    mainPass.colorAttachments[0].view = msaa > 1 ? colorMS.createView() : hdrView;
    mainPass.colorAttachments[0].resolveTarget = msaa > 1 ? hdrView : undefined;
    mainPass.depthStencilAttachment.view = depthMS.createView();
    const bv = [];
    for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    downPasses = []; upPasses = []; downBGs = []; upBGs = [];
    const tmp = new Float32Array(4);
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const srcW = m === 0 ? W : Math.max(1, bw >> (m - 1));
      const srcH = m === 0 ? H : Math.max(1, bh >> (m - 1));
      tmp[0] = 1 / srcW; tmp[1] = 1 / srcH; tmp[2] = m === 0 ? 1 : 0; tmp[3] = 0;
      queue.writeBuffer(bloomUBOs[m], 0, tmp);
      downBGs.push(helpers.bg(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
      downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    }
    for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
      const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
      tmp[0] = 1 / sw; tmp[1] = 1 / sh; tmp[2] = 1.0; tmp[3] = 1.0;
      const ubo = bloomUBOs[BLOOM_LEVELS + m];
      queue.writeBuffer(ubo, 0, tmp);
      upBGs.push(helpers.bg(upPipe, [bv[m], linSamp, { buffer: ubo }]));
      upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: 'load', storeOp: 'store' }] });
    }
    bgComp = helpers.bg(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }]);
    eng.hdrTex = hdrTex;
    return true;
  }

  // ------------------------------------------------------------------------------------
  // HUD + panel
  // ------------------------------------------------------------------------------------
  const panel = document.getElementById('panel');
  const nav = document.getElementById('nav');
  const suf = location.pathname.endsWith('.min.html') ? '.min.html' : '.html';
  if (nav) {
    nav.innerHTML = NAV.map(([id, label]) => `<a href="${id}${suf}"${id === scene.name ? ' class="on"' : ''}>${label}</a>`).join('');
  }
  const refresh = (key) => { if (scene.onParam) scene.onParam(key, eng); };
  for (const [key, label, min, max, step] of scene.sliders || []) {
    const row = document.createElement('label');
    const name = document.createElement('span'); name.textContent = label;
    const val = document.createElement('output'); val.textContent = String(P[key]);
    const inp = document.createElement('input');
    Object.assign(inp, { type: 'range', min, max, step, value: P[key] });
    inp.addEventListener('input', () => { P[key] = Number(inp.value); val.textContent = inp.value; refresh(key); });
    row.append(name, inp, val);
    panel.appendChild(row);
  }
  if (!panel.children.length) panel.style.display = 'none';
  if (qs.get('ui') === '0') { for (const id of ['hud', 'nav']) { const el = document.getElementById(id); if (el) el.style.display = 'none'; } panel.classList.add('hidden'); }
  const helpEl = document.getElementById('help');
  if (helpEl) helpEl.textContent = scene.help || '';
  const hudFps = document.getElementById('fps');

  // ------------------------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------------------------
  resize();
  await scene.init(eng);
  addEventListener('resize', () => { if (resize() && scene.onResize) scene.onResize(eng); });
  if (scene.onResize) scene.onResize(eng);
  if (hudFps) hudFps.textContent = 'ready';

  const shotName = qs.get('shot');
  const shotFrames = Number(qs.get('frames')) || 240;
  const fixedDt = shotName ? 1 / 60 : 0;
  const paused = qs.get('pause') === '1';
  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fpsT = 0, shooting = false;

  function frame(now) {
    if (shooting) return;
    requestAnimationFrame(frame);
    if (resize() && scene.onResize) scene.onResize(eng);
    let dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    fpsAcc += dt; fpsN++; fpsT += dt;
    if (fpsT > 0.5) {
      const fps = fpsN / fpsAcc;
      if (hudFps) hudFps.textContent = `${fps.toFixed(0)} fps  ${(1000 / fps).toFixed(1)} ms  |  ${eng.W}x${eng.H}`;
      fpsAcc = 0; fpsN = 0; fpsT = 0;
    }
    if (fixedDt) dt = fixedDt;
    if (paused) dt = 0;
    eng.dt = dt;
    updateCamera(dt);
    eng.time += dt;
    eng.frame++;
    writeCamera();
    if (scene.update) scene.update(eng, dt, eng.time);

    const enc = device.createCommandEncoder();
    if (scene.compute) scene.compute(eng, enc);
    const rp = enc.beginRenderPass(mainPass);
    scene.draw(eng, rp);
    rp.end();

    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const p = enc.beginRenderPass(downPasses[m]);
      p.setPipeline(downPipe); p.setBindGroup(0, downBGs[m]); p.draw(3); p.end();
    }
    for (let i = 0; i < upPasses.length; i++) {
      const p = enc.beginRenderPass(upPasses[i]);
      p.setPipeline(upPipe); p.setBindGroup(0, upBGs[i]); p.draw(3); p.end();
    }
    compData[0] = P.exposure; compData[1] = P.bloom; compData[2] = eng.frame % 64; compData[3] = P.vignette;
    compData[4] = P.sat; compData[5] = P.contrast; compData[6] = P.warmth; compData[7] = P.grain;
    queue.writeBuffer(compUBO, 0, compData);
    const swapTex = ctx.getCurrentTexture();
    compPass.colorAttachments[0].view = swapTex.createView();
    const pc = enc.beginRenderPass(compPass);
    pc.setPipeline(compPipe); pc.setBindGroup(0, bgComp); pc.draw(3); pc.end();

    let readBuf = null, bpr = 0;
    if (shotName && eng.frame >= shotFrames) {
      shooting = true;
      bpr = Math.ceil(eng.W * 4 / 256) * 256;
      readBuf = device.createBuffer({ size: bpr * eng.H, usage: GU.MAP_READ | GU.COPY_DST });
      enc.copyTextureToBuffer({ texture: swapTex }, { buffer: readBuf, bytesPerRow: bpr }, [eng.W, eng.H]);
    }
    queue.submit([enc.finish()]);
    if (readBuf) saveShot(readBuf, bpr);
  }

  async function saveShot(buf, bpr) {
    const { W, H } = eng;
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const img = new ImageData(W, H);
    const bgra = swapFormat === 'bgra8unorm';
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = y * bpr + x * 4, d = (y * W + x) * 4;
        img.data[d] = src[s + (bgra ? 2 : 0)];
        img.data[d + 1] = src[s + 1];
        img.data[d + 2] = src[s + (bgra ? 0 : 2)];
        img.data[d + 3] = 255;
      }
    }
    buf.unmap();
    const oc = new OffscreenCanvas(W, H);
    oc.getContext('2d').putImageData(img, 0, 0);
    const blob = await oc.convertToBlob({ type: 'image/png' });
    await fetch('/__shot?name=' + encodeURIComponent(shotName), { method: 'POST', body: blob });
    log('shot saved ' + shotName);
    document.title = 'SHOT DONE';
    shooting = false;
    if (qs.get('keep') === '1') requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

export function boot(scene) {
  run(scene).catch((e) => { log('init failed: ' + (e && e.stack || e)); const el = document.getElementById('err'); if (el && !el.style.display) { el.style.display = 'block'; el.textContent = String(e && e.message || e); } });
}
