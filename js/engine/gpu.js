// WebGPU device bootstrap and small resource helpers shared by every scene.
export const GU = GPUBufferUsage;
export const TU = GPUTextureUsage;
export const HDR = 'rgba16float';
export const DEPTH = 'depth32float';

export function log(msg) {
  console.log(msg);
  try { fetch('/__log', { method: 'POST', body: String(msg) }).catch(() => {}); } catch (e) { /* offline */ }
}

export function fail(msg) {
  const el = document.getElementById('err');
  if (el) { el.style.display = 'block'; el.textContent = msg; }
  log('FATAL ' + msg);
  throw new Error(msg);
}

export async function initGPU(canvas) {
  if (!navigator.gpu) fail('WebGPU is not available in this browser. Use a recent Chrome.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter found.');
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTS ? ['timestamp-query'] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
      maxStorageTexturesPerShaderStage: Math.min(adapter.limits.maxStorageTexturesPerShaderStage, 8),
    },
  });
  device.lost.then((i) => log('device lost: ' + i.message));
  device.onuncapturederror = (e) => log('GPU error: ' + e.error.message);
  const ctx = canvas.getContext('webgpu');
  const swapFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: swapFormat, alphaMode: 'opaque', usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC });
  return { adapter, device, ctx, swapFormat, hasTS };
}

// Helpers bound to one device. Scenes get these on `eng` (eng.shader, eng.computePipe, ...).
export function makeHelpers(device) {
  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') {
        const line = code.split('\n')[msg.lineNum - 1];
        log(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${line}`);
      }
    }
    if (bad) fail('Shader compile error in ' + label + ' (see console)');
    return m;
  }
  async function computePipe(code, label, entryPoint = 'main') {
    const module = await shader(code, label);
    return device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint } });
  }
  const mkBuf = (size, usage, label) => device.createBuffer({ size, usage, label });
  // bind group from a pipeline's auto layout; entries are listed in binding order
  const bg = (pipe, entries, label, group = 0) => device.createBindGroup({
    label, layout: pipe.getBindGroupLayout(group),
    entries: entries.map((r, i) => ({ binding: i, resource: r })),
  });
  // upload a typed array into a new buffer
  const bufFrom = (data, usage, label) => {
    const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage: usage | GU.COPY_DST, label });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  return { shader, computePipe, mkBuf, bg, bufFrom };
}
