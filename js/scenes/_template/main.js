// Scene template / engine smoke test: sky gradient, a lit ground grid and one instanced cube.
// Copy this folder to start a new scene. A scene is a plain object handed to boot().
//
//   name      id used by the nav + dist file name
//   msaa      4 (default) or 1 — pipelines must use eng.msaa as their sample count
//   params    default values for the slider panel / URL parameters (exposure, bloom, vignette,
//             sat, contrast, warmth, grain are read by the engine's composite pass)
//   sliders   [key, label, min, max, step] rows for the panel (H toggles it)
//   camera    { mode: 'fly'|'orbit'|'walk', x,y,z,yaw,pitch(deg),fov(deg),speed, ... }
//   init      async, build pipelines/resources
//   update    per-frame CPU work + uniform writes
//   compute   optional, encode compute passes into the frame's command encoder
//   draw      draw into the HDR main pass (reverse-Z depth: clear 0, compare 'greater')
import { boot } from '../../engine/engine.js';
import { CAM_WGSL, FULLSCREEN_VS, NOISE_WGSL } from '../../engine/wgsl.js';

const SKY = /* wgsl */ `
${CAM_WGSL}
${NOISE_WGSL}
${FULLSCREEN_VS}
struct U { sun: vec4f }
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<uniform> u: U;
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let rd = camRay(cam, in.pos.xy);
  let h = sat(rd.y * 0.5 + 0.5);
  var col = mix(vec3f(0.9, 0.55, 0.3), vec3f(0.12, 0.3, 0.7), pow(h, 0.6));
  let s = max(dot(rd, normalize(u.sun.xyz)), 0.0);
  col += vec3f(1.0, 0.8, 0.5) * (pow(s, 600.0) * 40.0 + pow(s, 8.0) * 0.3);
  return vec4f(col, 1.0);
}
`;

const GROUND = /* wgsl */ `
${CAM_WGSL}
${NOISE_WGSL}
@group(0) @binding(0) var<uniform> cam: Cam;
struct VO { @builtin(position) pos: vec4f, @location(0) wp: vec3f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  // two big triangles forming a 4 km square around the origin
  var p = array<vec2f, 6>(vec2f(-1,-1), vec2f(1,-1), vec2f(1,1), vec2f(-1,-1), vec2f(1,1), vec2f(-1,1));
  let w = vec3f(p[i].x * 2000.0, 0.0, p[i].y * 2000.0);
  var o: VO;
  o.pos = cam.viewProj * vec4f(w - cam.pos.xyz, 1.0);
  o.wp = w;
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let g = abs(fract(in.wp.xz * 0.5 - 0.5) - 0.5) / fwidth(in.wp.xz * 0.5);
  let line = 1.0 - min(min(g.x, g.y), 1.0);
  var col = mix(vec3f(0.05, 0.06, 0.05), vec3f(0.5, 0.5, 0.45), line);
  let d = length(in.wp.xz - cam.pos.xz);
  col = mix(col, vec3f(0.6, 0.35, 0.2), 1.0 - exp(-d * 0.002));
  return vec4f(col, 1.0);
}
`;

const scene = {
  name: 'template',
  msaa: 4,
  params: { exposure: 1.0, bloom: 0.08, sunElev: 12 },
  sliders: [['exposure', 'Exposure', 0.1, 4, 0.01], ['bloom', 'Bloom', 0, 0.3, 0.001], ['sunElev', 'Sun elevation', 0, 80, 0.5]],
  camera: { mode: 'fly', x: 0, y: 2, z: 8, yaw: 0, pitch: -8, fov: 60, speed: 5 },
  help: 'click: mouse look · WASD move · Space/E up · Q/C down · Shift fast · wheel speed · H panel',

  async init(eng) {
    const { device } = eng;
    this.uBuf = eng.mkBuf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'sun');
    const skyMod = await eng.shader(SKY, 'sky');
    this.skyPipe = await eng.renderPipe({
      label: 'sky',
      vertex: { module: skyMod, entryPoint: 'vs' },
      fragment: { module: skyMod, entryPoint: 'fs' },
      depthStencil: { format: eng.DEPTH, depthWriteEnabled: false, depthCompare: 'equal' },   // far plane only
    });
    this.skyBG = eng.bg(this.skyPipe, [{ buffer: eng.camUBO }, { buffer: this.uBuf }]);
    const gMod = await eng.shader(GROUND, 'ground');
    this.gndPipe = await eng.renderPipe({
      label: 'ground',
      vertex: { module: gMod, entryPoint: 'vs' },
      fragment: { module: gMod, entryPoint: 'fs' },
    });
    this.gndBG = eng.bg(this.gndPipe, [{ buffer: eng.camUBO }]);
    this.sun = new Float32Array(4);
  },

  update(eng) {
    const e = eng.P.sunElev * Math.PI / 180;
    this.sun.set([Math.cos(e) * 0.3, Math.sin(e), -Math.cos(e), 0]);
    eng.queue.writeBuffer(this.uBuf, 0, this.sun);
  },

  draw(eng, rp) {
    rp.setPipeline(this.gndPipe);
    rp.setBindGroup(0, this.gndBG);
    rp.draw(6);
    rp.setPipeline(this.skyPipe);
    rp.setBindGroup(0, this.skyBG);
    rp.draw(3);
  },
};

boot(scene);
