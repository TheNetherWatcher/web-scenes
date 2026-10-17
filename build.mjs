// Bundles each scene page + its js modules into self-contained pages that open directly from
// disk (file://) without a server:
//   dist/<scene>.html       readable bundle
//   dist/<scene>.min.html   minified bundle (WGSL comments/whitespace stripped, JS minified)
//   dist/index.html         landing page
//   node build.mjs [scene ...]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import esbuild from 'esbuild';

const root = path.dirname(fileURLToPath(import.meta.url));
const SCENES = ['car', 'city', 'space', 'fire'];
const only = process.argv.slice(2);
const dist = path.join(root, 'dist');
fs.mkdirSync(dist, { recursive: true });

// WGSL has no string literals, so comments and whitespace can be removed safely. Spaces are
// kept around + - / so that e.g. "a - -b" never fuses into other tokens.
function minifyWGSL(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/ ?([{}()[\],;:=<>*&|^!%@.]) ?/g, '$1')
    .trim();
}
// Minify every template literal in a module in place; ${...} interpolations are masked with
// identifier-like placeholders first so tokens never fuse across them.
function minifyTemplates(src) {
  return src.replace(/`([^`]*)`/g, (_, body) => {
    const parts = [];
    const masked = body.replace(/\$\{[^}]*\}/g, (m) => `__P${parts.push(m) - 1}__`);
    const out = minifyWGSL(masked).replace(/__P(\d+)__/g, (_, i) => parts[Number(i)]);
    return '`' + out + '`';
  });
}
const wgslPlugin = (min) => ({
  name: 'wgsl-minify',
  setup(build) {
    if (!min) return;
    // only modules that hold shaders (`/* wgsl */` tagged templates) are rewritten
    build.onLoad({ filter: /\.js$/ }, (args) => {
      const src = fs.readFileSync(args.path, 'utf8');
      if (!src.includes('/* wgsl */')) return null;
      return { contents: minifyTemplates(src), loader: 'js' };
    });
  },
});

async function bundle(scene, min) {
  const res = await esbuild.build({
    entryPoints: [path.join(root, 'js/scenes', scene, 'main.js')],
    bundle: true, write: false, format: 'esm', target: 'es2022', legalComments: 'none',
    minify: min, plugins: [wgslPlugin(min)],
  });
  return res.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
}

const css = fs.readFileSync(path.join(root, 'css/ui.css'), 'utf8');
const inlineCss = (html, min) => html.replace(/<link rel="stylesheet" href="css\/ui.css">/,
  () => `<style>${min ? css.replace(/\s+/g, ' ').replace(/ ?([{}:;,>]) ?/g, '$1').trim() : '\n' + css}</style>`);

for (const scene of SCENES) {
  if (only.length && !only.includes(scene)) continue;
  const page = fs.readFileSync(path.join(root, `${scene}.html`), 'utf8');
  const tag = `<script type="module" src="js/scenes/${scene}/main.js"></script>`;
  if (!page.includes(tag)) throw new Error(`${scene}.html is missing its script tag`);
  for (const min of [false, true]) {
    const code = await bundle(scene, min);
    let html = inlineCss(page, min).replace(tag, () => `<script type="module">${min ? '' : '\n'}${code}${min ? '' : '\n'}</script>`);
    if (min) html = html.replace(/>\s+</g, '><');
    const name = `${scene}${min ? '.min' : ''}.html`;
    fs.writeFileSync(path.join(dist, name), html);
    console.log(`wrote dist/${name}`, (html.length / 1024).toFixed(0) + ' KB');
  }
}
if (fs.existsSync(path.join(root, 'index.html'))) fs.copyFileSync(path.join(root, 'index.html'), path.join(dist, 'index.html'));
