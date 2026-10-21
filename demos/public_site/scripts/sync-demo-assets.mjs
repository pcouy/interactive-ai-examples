import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const demosRoot = path.resolve(scriptDir, '../..');
const demoDist = path.resolve(scriptDir, '../../dist');
const tokenFile = path.resolve(scriptDir, '../../src/styles/token-values.ts');

const PUBLIC_DEMO = path.join(projectRoot, 'public/demo');
const PUBLIC_KATEX = path.join(projectRoot, 'public/katex');
const PUBLIC_KATEX_FONTS = path.join(PUBLIC_KATEX, 'fonts');
const PALETTE_FILE = path.join(projectRoot, 'src/styles/palette.generated.css');

const MIN_NODE = [22, 22, 0];
const COLOUR_KEYS = [
  'bg',
  'surface',
  'fg',
  'muted',
  'accent',
  'accent-2',
  'arrow',
  'border',
  'fg-2',
  'muted-2',
  'warn',
  'danger',
  'danger-2',
];
const SITE_KEYS = {
  'site-bg': 'bg',
  'site-surface': 'surface',
  'site-fg': 'fg',
  'site-muted': 'muted',
  'site-accent': 'accent',
  'site-accent-2': 'accent-2',
  'site-border': 'border',
};
const MIN_CONTRAST = 4.5;
const DEMO_SOURCE_FILES = [
  path.join(demosRoot, 'vite.config.ts'),
  path.join(demosRoot, 'package.json'),
  path.join(demosRoot, 'tsconfig.json'),
];

function fail(message) {
  console.error(`sync-demo-assets: ${message}`);
  process.exit(1);
}

function checkNodeVersion() {
  const current = process.versions.node.split('.').map(Number);
  for (let index = 0; index < MIN_NODE.length; index += 1) {
    if (current[index] > MIN_NODE[index]) return;
    if (current[index] < MIN_NODE[index]) {
      fail(
        `Node ${MIN_NODE.join('.')} or newer is required. ` +
          `This process runs Node ${process.versions.node}.`,
      );
    }
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function hexToRgb(hex) {
  return {
    r: Number.parseInt(hex.slice(1, 3), 16),
    g: Number.parseInt(hex.slice(3, 5), 16),
    b: Number.parseInt(hex.slice(5, 7), 16),
  };
}

function relativeLuminance(hex) {
  const { r, g, b } = hexToRgb(hex);
  const channels = [r, g, b].map((value) => {
    const channel = value / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrastRatio(first, second) {
  const lighter = Math.max(relativeLuminance(first), relativeLuminance(second));
  const darker = Math.min(relativeLuminance(first), relativeLuminance(second));
  return (lighter + 0.05) / (darker + 0.05);
}

function validatePalette(name, palette) {
  if (palette === null || typeof palette !== 'object') {
    fail(`drlPalette.${name} is missing from ${tokenFile}.`);
  }

  for (const key of COLOUR_KEYS) {
    const value = palette[key];
    if (typeof value !== 'string' || !/^#[0-9a-f]{6}$/i.test(value)) {
      fail(
        `drlPalette.${name}.${key} must be a #rrggbb string. ` +
          `Received ${JSON.stringify(value)}.`,
      );
    }
  }
}

function displayDemoPath(file) {
  return path.join('demos', path.relative(demosRoot, file));
}

async function findNewestFile(directory, current = null) {
  const entries = await readdir(directory, { withFileTypes: true });
  let newest = current;

  for (const entry of entries) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      newest = await findNewestFile(absolute, newest);
      continue;
    }

    const details = await stat(absolute);
    if (details.isFile() && (newest === null || details.mtimeMs > newest.mtimeMs)) {
      newest = { file: absolute, mtimeMs: details.mtimeMs };
    }
  }

  return newest;
}

async function findNewestDemoSource() {
  let newest = await findNewestFile(path.join(demosRoot, 'src'));

  for (const file of DEMO_SOURCE_FILES) {
    const details = await stat(file);
    if (newest === null || details.mtimeMs > newest.mtimeMs) {
      newest = { file, mtimeMs: details.mtimeMs };
    }
  }

  return newest;
}

function renderBlock({ selector, indent, palette, onAccent, colorScheme }) {
  const selectorIndent = selector.match(/^\s*/)[0];
  const lines = [`${indent}color-scheme: ${colorScheme};`];

  for (const [siteKey, paletteKey] of Object.entries(SITE_KEYS)) {
    lines.push(`${indent}--${siteKey}: ${palette[paletteKey]};`);
  }
  lines.push(`${indent}--site-on-accent: ${onAccent};`);

  for (const key of COLOUR_KEYS) {
    lines.push(`${indent}--drl-theme-${key}: ${palette[key]};`);
  }

  return `${selector} {\n${lines.join('\n')}\n${selectorIndent}}`;
}

function renderPaletteCss({ light, dark, onAccent }) {
  const root = renderBlock({
    selector: ':root',
    indent: '  ',
    palette: light,
    onAccent: onAccent.light,
    colorScheme: 'light',
  });
  const explicitDark = renderBlock({
    selector: ":root[data-theme='dark']",
    indent: '  ',
    palette: dark,
    onAccent: onAccent.dark,
    colorScheme: 'dark',
  });

  const mediaLines = renderBlock({
    selector: '  :root:not([data-theme])',
    indent: '    ',
    palette: dark,
    onAccent: onAccent.dark,
    colorScheme: 'dark',
  });
  const mediaDark = `@media (prefers-color-scheme: dark) {\n${mediaLines}\n}`;

  return [
    '/* Generated by scripts/sync-demo-assets.mjs. Do not edit. */',
    '',
    root,
    '',
    explicitDark,
    '',
    mediaDark,
    '',
  ].join('\n');
}

async function syncDemoBundle() {
  const bundlePath = path.join(demoDist, 'drl-demos.js');
  let bundleStat;
  try {
    bundleStat = await stat(bundlePath);
    if (!bundleStat.isFile()) throw new Error('not a file');
  } catch {
    fail(
      `Missing ${bundlePath}. ` +
        'Run `npm --prefix .. run build` to build the demo bundle first.',
    );
  }

  let newestSource;
  try {
    newestSource = await findNewestDemoSource();
  } catch (error) {
    fail(`Cannot read the demo source tree: ${error.message}`);
  }

  if (process.env.SYNC_DEBUG === '1') {
    console.log(
      `[sync] bundle mtime: ${new Date(bundleStat.mtimeMs).toISOString()} ` +
        `(${bundleStat.mtimeMs})`,
    );
    console.log(
      `[sync] newest source: ${displayDemoPath(newestSource.file)} ` +
        `${new Date(newestSource.mtimeMs).toISOString()} (${newestSource.mtimeMs})`,
    );
  }

  if (bundleStat.mtimeMs < newestSource.mtimeMs) {
    fail(
      `demos/dist/drl-demos.js is older than ${displayDemoPath(newestSource.file)}. ` +
        'Rebuild the demos: npm --prefix .. run build',
    );
  }

  const entries = await readdir(demoDist, { withFileTypes: true });
  const bundleFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.js'))
    .map((entry) => entry.name)
    .sort();

  await rm(PUBLIC_DEMO, { recursive: true, force: true });
  await mkdir(PUBLIC_DEMO, { recursive: true });

  const sizes = new Map();
  for (const name of bundleFiles) {
    const source = path.join(demoDist, name);
    const destination = path.join(PUBLIC_DEMO, name);
    await copyFile(source, destination);
    sizes.set(name, (await stat(destination)).size);
  }

  const bundle = await readFile(path.join(PUBLIC_DEMO, 'drl-demos.js'), 'utf8');
  const dynamicTargets = new Set(
    [...bundle.matchAll(/\.\/([A-Za-z0-9_.-]+\.js)/g)].map((match) => match[1]),
  );
  for (const target of dynamicTargets) {
    if (!bundleFiles.includes(target)) {
      fail(
        `drl-demos.js references ./${target}, but that file is missing from ${demoDist}. ` +
          'Run `npm --prefix .. run build` again.',
      );
    }
  }

  return { bundleFiles, sizes, dynamicTargets };
}

async function syncKatex() {
  let katexCss;
  try {
    katexCss = require.resolve('katex/dist/katex.min.css');
  } catch {
    fail('Cannot resolve katex/dist/katex.min.css. Run `npm install` in demos/public_site.');
  }

  const katexFonts = path.join(path.dirname(katexCss), 'fonts');
  const fontNames = (await readdir(katexFonts))
    .filter((name) => /\.(woff2|woff|ttf)$/i.test(name))
    .sort();

  await rm(PUBLIC_KATEX_FONTS, { recursive: true, force: true });
  await mkdir(PUBLIC_KATEX_FONTS, { recursive: true });

  const cssDestination = path.join(PUBLIC_KATEX, 'katex.min.css');
  await copyFile(katexCss, cssDestination);

  let fontBytes = 0;
  for (const name of fontNames) {
    const destination = path.join(PUBLIC_KATEX_FONTS, name);
    await copyFile(path.join(katexFonts, name), destination);
    fontBytes += (await stat(destination)).size;
  }

  return {
    cssBytes: (await stat(cssDestination)).size,
    fontNames,
    fontBytes,
  };
}

checkNodeVersion();

const tokenModule = await import(pathToFileURL(tokenFile).href);
const { drlPalette, drlTokenValues } = tokenModule;

if (!drlTokenValues || typeof drlTokenValues !== 'object') {
  fail(`drlTokenValues is missing from ${tokenFile}.`);
}
if (!drlPalette || typeof drlPalette !== 'object') {
  fail(`drlPalette is missing from ${tokenFile}.`);
}

validatePalette('light', drlPalette.light);
validatePalette('dark', drlPalette.dark);

const ratios = {};
for (const theme of ['light', 'dark']) {
  const palette = drlPalette[theme];
  ratios[theme] = {
    'fg/bg': contrastRatio(palette.fg, palette.bg),
    'muted/bg': contrastRatio(palette.muted, palette.bg),
    'accent/bg': contrastRatio(palette.accent, palette.bg),
  };

  for (const [label, ratio] of Object.entries(ratios[theme])) {
    if (ratio < MIN_CONTRAST) {
      fail(
        `${theme} ${label} contrast is ${ratio.toFixed(2)}, ` +
          `below the required ${MIN_CONTRAST}. Adjust ${tokenFile}.`,
      );
    }
  }
}

const onAccentCandidates = [drlPalette.light.surface, drlPalette.dark.bg];
const onAccentFor = (palette) =>
  onAccentCandidates.reduce((best, candidate) =>
    contrastRatio(candidate, palette.accent) > contrastRatio(best, palette.accent) ? candidate : best,
  );
const onAccent = {
  light: onAccentFor(drlPalette.light),
  dark: onAccentFor(drlPalette.dark),
};

for (const theme of ['light', 'dark']) {
  const ratio = contrastRatio(onAccent[theme], drlPalette[theme].accent);
  if (ratio < MIN_CONTRAST) {
    fail(
      `[sync] on-accent contrast for ${theme} is ${ratio.toFixed(2)}, ` +
        `below the required ${MIN_CONTRAST}. Adjust ${tokenFile}.`,
    );
  }
}

const demoSummary = await syncDemoBundle();
const katexSummary = await syncKatex();

await writeFile(
  PALETTE_FILE,
  renderPaletteCss({ light: drlPalette.light, dark: drlPalette.dark, onAccent }),
  'utf8',
);

const paletteBytes = (await stat(PALETTE_FILE)).size;
const ratioText = (theme) =>
  Object.entries(ratios[theme])
    .map(([label, ratio]) => `${label} ${ratio.toFixed(2)}`)
    .join(', ');

console.log('sync-demo-assets: OK');
console.log(
  `  demo: ${demoSummary.bundleFiles.length} files ` +
    `(${demoSummary.bundleFiles
      .map((name) => `${name} ${formatBytes(demoSummary.sizes.get(name))}`)
      .join(', ')})`,
);
console.log(
  `  dynamic imports: ${[...demoSummary.dynamicTargets].sort().join(', ') || 'none'}`,
);
console.log(
  `  katex: katex.min.css ${formatBytes(katexSummary.cssBytes)}, ` +
    `${katexSummary.fontNames.length} fonts (${formatBytes(katexSummary.fontBytes)})`,
);
console.log(`  contrast light: ${ratioText('light')}`);
console.log(`  contrast dark: ${ratioText('dark')}`);
console.log(
  `  --site-on-accent: light ${onAccent.light} ` +
    `(${contrastRatio(onAccent.light, drlPalette.light.accent).toFixed(2)}), ` +
    `dark ${onAccent.dark} ` +
    `(${contrastRatio(onAccent.dark, drlPalette.dark.accent).toFixed(2)})`,
);
console.log(`  wrote src/styles/palette.generated.css (${formatBytes(paletteBytes)})`);
