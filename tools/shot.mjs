// Headless screenshot helper: serves the repo, loads a page in headless Chrome with ?shot=...,
// waits for the page to POST the PNG into ./shots, then exits.
//   node tools/shot.mjs <page> <outName.png> [query] [port]
//   node tools/shot.mjs template.html t.png "w=1280&h=720&frames=60&cam=0,2,8,0,-8"
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const [page, out, query = '', portArg] = process.argv.slice(2);
if (!page || !out) { console.error('usage: node tools/shot.mjs <page> <out.png> [query] [port]'); process.exit(2); }
const port = Number(portArg) || 8080 + Math.floor(Math.random() * 900);
const chrome = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const shotPath = path.join(root, 'shots', path.basename(out));
fs.mkdirSync(path.join(root, 'shots'), { recursive: true });
fs.rmSync(shotPath, { force: true });

const server = spawn('node', [path.join(root, 'server.mjs'), String(port)], { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
server.stdout.on('data', (d) => { serverLog += d; });
server.stderr.on('data', (d) => { serverLog += d; });
await new Promise((r) => setTimeout(r, 400));

const q = `shot=${encodeURIComponent(out)}&${query}`;
const url = `http://localhost:${port}/${page}?${q}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shot-'));
const br = spawn(chrome, [
  '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist',
  '--use-angle=metal', '--window-size=1280,720', `--user-data-dir=${profile}`, '--no-first-run', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  url,
], { stdio: 'ignore' });

const timeout = Number(process.env.SHOT_TIMEOUT || 90) * 1000;
const t0 = Date.now();
let ok = false;
while (Date.now() - t0 < timeout) {
  await new Promise((r) => setTimeout(r, 500));
  if (fs.existsSync(shotPath) && fs.statSync(shotPath).size > 1000) { ok = true; break; }
  if (/FATAL|init failed|error/i.test(serverLog) && Date.now() - t0 > 5000) break;
}
br.kill('SIGKILL');
server.kill();
fs.rmSync(profile, { recursive: true, force: true });
const lines = serverLog.split('\n').filter((l) => l.startsWith('[client]'));
if (lines.length) console.log(lines.join('\n'));
console.log(ok ? `ok -> ${shotPath}` : 'FAILED (no screenshot)');
process.exit(ok ? 0 : 1);
