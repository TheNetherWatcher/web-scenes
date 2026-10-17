// Turns an SDF part (see sdf.js) into a triangle mesh: the field, material channels and ambient
// occlusion are evaluated on a grid by a compute shader, then CPU surface nets extract the surface.
// Vertex layout (12 floats): position, normal, material1 (4), material2 (2).
import { SDF_COMMON, FIELD_MAIN } from './sdf.js';

const GUsage = GPUBufferUsage;
const pipeCache = new Map();

export async function meshFromPart(eng, partCode, o) {
  const { min, max, h, band, style = 0, label = 'part' } = o;
  const nx = Math.ceil((max[0] - min[0]) / h) + 1;
  const ny = Math.ceil((max[1] - min[1]) / h) + 1;
  const nz = Math.ceil((max[2] - min[2]) / h) + 1;
  const N = nx * ny * nz;
  const dev = eng.device;
  let pipe = pipeCache.get(label);
  if (!pipe) { pipe = await eng.computePipe(SDF_COMMON + partCode + FIELD_MAIN, 'sdf ' + label); pipeCache.set(label, pipe); }

  const ub = new ArrayBuffer(48);
  const uf = new Float32Array(ub), uu = new Uint32Array(ub);
  uf[0] = min[0]; uf[1] = min[1]; uf[2] = min[2];
  uu[4] = nx; uu[5] = ny; uu[6] = nz;
  uf[8] = h; uf[9] = band; uf[10] = style;
  const gbuf = eng.mkBuf(48, GUsage.UNIFORM | GUsage.COPY_DST, label + ' grid');
  dev.queue.writeBuffer(gbuf, 0, ub);
  const mk = (usage) => eng.mkBuf(N * 4, usage, label + ' field');
  const bF = mk(GUsage.STORAGE | GUsage.COPY_SRC), b1 = mk(GUsage.STORAGE | GUsage.COPY_SRC), b2 = mk(GUsage.STORAGE | GUsage.COPY_SRC);
  const rF = mk(GUsage.MAP_READ | GUsage.COPY_DST), r1 = mk(GUsage.MAP_READ | GUsage.COPY_DST), r2 = mk(GUsage.MAP_READ | GUsage.COPY_DST);
  const bgroup = eng.bg(pipe, [{ buffer: gbuf }, { buffer: bF }, { buffer: b1 }, { buffer: b2 }], label);
  const enc = dev.createCommandEncoder();
  const cp = enc.beginComputePass();
  cp.setPipeline(pipe);
  cp.setBindGroup(0, bgroup);
  cp.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 4), Math.ceil(nz / 4));
  cp.end();
  enc.copyBufferToBuffer(bF, 0, rF, 0, N * 4);
  enc.copyBufferToBuffer(b1, 0, r1, 0, N * 4);
  enc.copyBufferToBuffer(b2, 0, r2, 0, N * 4);
  dev.queue.submit([enc.finish()]);
  await Promise.all([rF.mapAsync(GPUMapMode.READ), r1.mapAsync(GPUMapMode.READ), r2.mapAsync(GPUMapMode.READ)]);
  const F = new Float32Array(rF.getMappedRange().slice(0));
  const M1 = new Uint8Array(r1.getMappedRange().slice(0));
  const M2 = new Uint8Array(r2.getMappedRange().slice(0));
  for (const b of [rF, r1, r2, bF, b1, b2, gbuf]) { try { b.unmap?.(); } catch (e) { /* not mapped */ } b.destroy(); }
  return surfaceNets(F, M1, M2, nx, ny, nz, min, h);
}

function surfaceNets(F, M1, M2, nx, ny, nz, org, h) {
  const cx = nx - 1, cy = ny - 1, cz = nz - 1;
  const cellIdx = new Int32Array(cx * cy * cz).fill(-1);
  const sy = nx, sz = nx * ny;
  const cof = [0, 1, sy, 1 + sy, sz, 1 + sz, sy + sz, 1 + sy + sz];
  const corner = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  let cap = 1 << 16, nv = 0;
  let gp = new Float32Array(cap * 3);
  let cc = new Int32Array(cap * 3);
  const cv = new Float32Array(8);
  for (let k = 0; k < cz; k++) {
    for (let j = 0; j < cy; j++) {
      let base = nx * (j + ny * k);
      for (let i = 0; i < cx; i++, base++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) { const v = F[base + cof[c]]; cv[c] = v; if (v < 0) mask |= 1 << c; }
        if (mask === 0 || mask === 255) continue;
        let ax = 0, ay = 0, az = 0, cnt = 0;
        for (let e = 0; e < 12; e++) {
          const a = edges[e][0], b = edges[e][1];
          if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
          const t = cv[a] / (cv[a] - cv[b]);
          ax += corner[a][0] + t * (corner[b][0] - corner[a][0]);
          ay += corner[a][1] + t * (corner[b][1] - corner[a][1]);
          az += corner[a][2] + t * (corner[b][2] - corner[a][2]);
          cnt++;
        }
        if (nv >= cap) {
          cap *= 2;
          const g2 = new Float32Array(cap * 3); g2.set(gp); gp = g2;
          const c2 = new Int32Array(cap * 3); c2.set(cc); cc = c2;
        }
        gp[nv * 3] = i + ax / cnt; gp[nv * 3 + 1] = j + ay / cnt; gp[nv * 3 + 2] = k + az / cnt;
        cc[nv * 3] = i; cc[nv * 3 + 1] = j; cc[nv * 3 + 2] = k;
        cellIdx[i + cx * (j + cy * k)] = nv++;
      }
    }
  }
  // quads for every grid edge that crosses the surface
  const idx = [];
  const ci = (i, j, k) => cellIdx[i + cx * (j + cy * k)];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) idx.push(a, b, c, a, c, d); else idx.push(a, c, b, a, d, c);
  };
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      let base = nx * (j + ny * k);
      for (let i = 0; i < nx; i++, base++) {
        const s0 = F[base] < 0;
        if (i < nx - 1 && j > 0 && k > 0 && j < ny - 1 && k < nz - 1 && s0 !== (F[base + 1] < 0)) {
          quad(ci(i, j - 1, k - 1), ci(i, j, k - 1), ci(i, j, k), ci(i, j - 1, k), s0);
        }
        if (j < ny - 1 && i > 0 && k > 0 && i < nx - 1 && k < nz - 1 && s0 !== (F[base + sy] < 0)) {
          quad(ci(i - 1, j, k - 1), ci(i, j, k - 1), ci(i, j, k), ci(i - 1, j, k), !s0);
        }
        if (k < nz - 1 && i > 0 && j > 0 && i < nx - 1 && j < ny - 1 && s0 !== (F[base + sz] < 0)) {
          quad(ci(i - 1, j - 1, k), ci(i, j - 1, k), ci(i, j, k), ci(i - 1, j, k), s0);
        }
      }
    }
  }
  // normals (trilinear central-difference gradient) and materials
  const gradAt = (i, j, k, out) => {
    const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, nx - 1);
    const j0 = Math.max(j - 1, 0), j1 = Math.min(j + 1, ny - 1);
    const k0 = Math.max(k - 1, 0), k1 = Math.min(k + 1, nz - 1);
    out[0] = F[i1 + nx * (j + ny * k)] - F[i0 + nx * (j + ny * k)];
    out[1] = F[i + nx * (j1 + ny * k)] - F[i + nx * (j0 + ny * k)];
    out[2] = F[i + nx * (j + ny * k1)] - F[i + nx * (j + ny * k0)];
  };
  const verts = new Float32Array(nv * 12);
  const g = [0, 0, 0];
  for (let v = 0; v < nv; v++) {
    const i = cc[v * 3], j = cc[v * 3 + 1], k = cc[v * 3 + 2];
    const u = Math.min(Math.max(gp[v * 3] - i, 0), 1), w = Math.min(Math.max(gp[v * 3 + 1] - j, 0), 1), t = Math.min(Math.max(gp[v * 3 + 2] - k, 0), 1);
    let nxv = 0, nyv = 0, nzv = 0, m10 = 0, m11 = 0, m12 = 0, m13 = 0, m20 = 0, m21 = 0;
    for (let c = 0; c < 8; c++) {
      const ox = corner[c][0], oy = corner[c][1], oz = corner[c][2];
      const wt = (ox ? u : 1 - u) * (oy ? w : 1 - w) * (oz ? t : 1 - t);
      if (wt === 0) continue;
      gradAt(i + ox, j + oy, k + oz, g);
      const gl = Math.hypot(g[0], g[1], g[2]) || 1;
      nxv += wt * g[0] / gl; nyv += wt * g[1] / gl; nzv += wt * g[2] / gl;
      const o = (i + ox + nx * (j + oy + ny * (k + oz))) * 4;
      m10 += wt * M1[o]; m11 += wt * M1[o + 1]; m12 += wt * M1[o + 2]; m13 += wt * M1[o + 3];
      m20 += wt * M2[o]; m21 += wt * M2[o + 1];
    }
    const nl = Math.hypot(nxv, nyv, nzv) || 1;
    const o = v * 12;
    verts[o] = org[0] + gp[v * 3] * h; verts[o + 1] = org[1] + gp[v * 3 + 1] * h; verts[o + 2] = org[2] + gp[v * 3 + 2] * h;
    verts[o + 3] = nxv / nl; verts[o + 4] = nyv / nl; verts[o + 5] = nzv / nl;
    verts[o + 6] = m10 / 255; verts[o + 7] = m11 / 255; verts[o + 8] = m12 / 255; verts[o + 9] = m13 / 255;
    verts[o + 10] = m20 / 255; verts[o + 11] = m21 / 255;
  }
  return { verts, indices: new Uint32Array(idx), vertexCount: nv };
}
