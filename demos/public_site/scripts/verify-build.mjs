import { createHash } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const distDir = path.join(projectRoot, 'dist');
const globalCssPath = path.join(projectRoot, 'src/styles/global.css');
const demoDistBundlePath = path.resolve(projectRoot, '../dist/drl-demos.js');

const BUNDLE_MARKERS = [
  '--drl-theme-bg',
  'Boucle agent-environnement',
  'Dernière mise à jour',
  'Rejeu',
];

const REQUIRED_FILES = [
  'index.html',
  'deep-blue.html',
  'frozen-lake.html',
  'favicon.svg',
  'demo/drl-demos.js',
  'demo/drl-three.js',
  'demo/drl-index.js',
  'katex/katex.min.css',
  'katex/fonts/KaTeX_Main-Regular.woff2',
];

const DEMO_PAGES = {
  'deep-blue.html': 'drl-deep-blue-page',
  'frozen-lake.html': 'drl-frozen-lake',
};

const ROOT_ATTRIBUTE = /\b(?:href|src)\s*=\s*["']\/(?!\/)/i;
const ROOT_CSS_URL = /url\(\s*["']?\/(?!\/)/i;
const REFERENCE =
  /(?:\b(?:href|src)\s*=\s*["'](\.\/[^"']*)["'])|(?:url\(\s*["']?(\.\/[^"')]*)["']?\s*\))/gi;
const COLOUR_LITERAL = /#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(/i;

const failures = [];
let checks = 0;

function check(condition, message) {
  checks += 1;
  if (!condition) failures.push(message);
  return condition;
}

async function walkHtml(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];

  for (const entry of entries.sort((first, second) => first.name.localeCompare(second.name))) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walkHtml(absolute)));
    } else if (entry.isFile() && entry.name.endsWith('.html')) {
      files.push(absolute);
    }
  }

  return files;
}

async function fileIsPresent(relative) {
  try {
    const details = await stat(path.join(distDir, relative));
    return details.isFile() && details.size > 0;
  } catch {
    return false;
  }
}

async function referenceExists(pageFile, reference) {
  const target = reference.split(/[?#]/, 1)[0];
  try {
    return (await stat(path.resolve(path.dirname(pageFile), target))).isFile();
  } catch {
    return false;
  }
}

function checkThemeScriptOrder(html, label) {
  const headEnd = html.indexOf('</head>');
  check(headEnd !== -1, `${label}: missing </head>.`);

  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)];
  const themeScript = scripts.find(
    (script) => script[1].includes('drl-theme') && script[1].includes('dataset.theme'),
  );
  if (!check(Boolean(themeScript), `${label}: missing the inline theme init script.`)) return;

  const scriptIndex = themeScript.index ?? -1;
  check(
    headEnd === -1 || scriptIndex < headEnd,
    `${label}: the theme init script is not inside <head>.`,
  );

  const styleIndex = html.search(/<style\b/i);
  const stylesheet = /<link\b[^>]*\brel=["']?stylesheet/i.exec(html);
  const firstStyleIndex = Math.min(
    styleIndex === -1 ? Number.POSITIVE_INFINITY : styleIndex,
    stylesheet ? stylesheet.index : Number.POSITIVE_INFINITY,
  );

  check(
    scriptIndex < firstStyleIndex,
    `${label}: the theme init script must run before the first style or stylesheet link.`,
  );
}

check(await fileIsPresent('demo/drl-demos.js'), 'dist/demo/drl-demos.js is missing.');

const bundlePath = path.join(distDir, 'demo/drl-demos.js');
const bundleBuffer = await readFile(bundlePath).catch(() => null);
const bundle = bundleBuffer?.toString('utf8') ?? '';

for (const marker of BUNDLE_MARKERS) {
  check(
    bundle.includes(marker),
    `dist/demo/drl-demos.js does not contain ${JSON.stringify(marker)}. ` +
      'The copied bundle is stale. Run npm --prefix .. run build, then npm run build.',
  );
}

const demoDistBundleBuffer = await readFile(demoDistBundlePath).catch(() => null);
if (bundleBuffer === null || demoDistBundleBuffer === null) {
  const missingPath = bundleBuffer === null ? bundlePath : demoDistBundlePath;
  check(
    false,
    `Cannot compare demo bundles: ${missingPath} is missing or unreadable. ` +
      'The copied bundle is stale.',
  );
} else {
  const copiedHash = createHash('sha256').update(bundleBuffer).digest('hex');
  const buildHash = createHash('sha256').update(demoDistBundleBuffer).digest('hex');
  check(
    bundleBuffer.length === demoDistBundleBuffer.length && copiedHash === buildHash,
    `Copied bundle is stale: ${bundlePath} ` +
      `(${bundleBuffer.length} bytes, sha256 ${copiedHash}) differs from ` +
      `${demoDistBundlePath} (${demoDistBundleBuffer.length} bytes, sha256 ${buildHash}). ` +
      'Run npm --prefix .. run build, then npm run build.',
  );
}

for (const relative of REQUIRED_FILES) {
  check(await fileIsPresent(relative), `Missing required build file: dist/${relative}.`);
}

let htmlFiles = [];
try {
  htmlFiles = await walkHtml(distDir);
} catch {
  failures.push(`Cannot read the build directory ${distDir}. Run \`npm run build\` first.`);
  checks += 1;
}

const globalCss = await readFile(globalCssPath, 'utf8').catch(() => null);
check(globalCss !== null, 'Missing src/styles/global.css.');
check(
  globalCss === null || !COLOUR_LITERAL.test(globalCss),
  'src/styles/global.css contains a colour literal; use --site-* tokens instead.',
);

for (const file of htmlFiles) {
  const label = path.relative(distDir, file);
  const html = await readFile(file, 'utf8');

  check(/<html\b[^>]*\blang=["']fr["']/i.test(html), `${label}: <html> must use lang="fr".`);

  const h1Count = (html.match(/<h1\b/gi) ?? []).length;
  check(h1Count === 1, `${label}: expected exactly one <h1>, found ${h1Count}.`);

  check(
    /<main\b[^>]*\bid=["']main["']/i.test(html),
    `${label}: missing <main id="main">.`,
  );
  check(
    /<a\b[^>]*\bhref=["']#main["']/i.test(html),
    `${label}: missing the skip link to #main.`,
  );

  const toggle = /<button\b[^>]*\bdata-theme-toggle\b[^>]*>/i.exec(html)?.[0] ?? '';
  check(toggle.length > 0, `${label}: missing the theme toggle button.`);
  check(/\baria-pressed=/i.test(toggle), `${label}: theme toggle is missing aria-pressed.`);

  check(
    /<link\b[^>]*\bhref=["']\.\/katex\/katex\.min\.css["']/i.test(html),
    `${label}: missing the local KaTeX stylesheet link.`,
  );

  const currentCount = (html.match(/\baria-current=["']page["']/gi) ?? []).length;
  check(
    currentCount === 1,
    `${label}: expected exactly one aria-current="page", found ${currentCount}.`,
  );

  check(
    !ROOT_ATTRIBUTE.test(html) && !ROOT_CSS_URL.test(html),
    `${label}: a root-absolute href, src, or url() remains.`,
  );
  check(!html.includes('/src/'), `${label}: contains a /src/ reference.`);

  for (const match of html.matchAll(REFERENCE)) {
    const reference = match[1] ?? match[2];
    if (!reference) continue;
    check(
      await referenceExists(file, reference),
      `${label}: reference ${JSON.stringify(reference)} does not resolve to a file.`,
    );
  }

  checkThemeScriptOrder(html, label);

  check(html.includes('--site-bg:'), `${label}: missing --site-bg in the inlined styles.`);
  check(
    /\[data-theme=(?:"dark"|'dark'|dark)\]/i.test(html),
    `${label}: missing the dark theme block.`,
  );

  check(
    /<link\b[^>]*rel=["']icon["'][^>]*href=["']\.\/favicon\.svg["']/i.test(html),
    `${label}: missing the favicon link ./favicon.svg.`,
  );

  const blogLink =
    /href="https:\/\/pierre-couy\.dev"[^>]*target="_blank"[^>]*rel="noopener noreferrer"[^>]*>Mon site/;
  const githubLink =
    /href="https:\/\/github\.com\/pcouy\/interactive-ai-examples"[^>]*target="_blank"[^>]*rel="noopener noreferrer"[^>]*>Dépôt GitHub/;
  check(
    blogLink.test(html) && githubLink.test(html),
    `${label}: external links are missing or lose their new-tab attributes`,
  );

  const baseName = path.basename(file);
  const demoTag = DEMO_PAGES[baseName];
  if (demoTag) {
    check(
      /<script\b[^>]*\bsrc=["']\.\/demo\/drl-demos\.js["']/i.test(html),
      `${label}: missing the demo bundle script ./demo/drl-demos.js.`,
    );
    check(
      /<section\b[^>]*class=(["'])[^"']*\bdemo-surface\b[^"']*\1/i.test(html),
      `${label}: missing the <section class="demo-surface"> element.`,
    );
    check(
      new RegExp(`<${demoTag}(?=[\\s>])`, 'i').test(html),
      `${label}: missing the <${demoTag}> custom element.`,
    );
    check(
      /class=["']katex["']/i.test(html),
      `${label}: expected at least one rendered KaTeX expression.`,
    );
    check(/<math\b/i.test(html), `${label}: expected at least one <math> element.`);
  }
}

if (failures.length > 0) {
  console.error(`verify-build: ${failures.length} check(s) failed.`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}

console.log(`verify-build: OK (${htmlFiles.length} pages, ${checks} checks)`);
