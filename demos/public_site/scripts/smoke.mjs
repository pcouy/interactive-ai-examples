#!/usr/bin/env node
/**
 * HTTP smoke test for the built static site.
 *
 * The test serves `dist/` under the nested prefix `/nested/site/`, then fetches
 * the pages and key assets over real HTTP. It uses Node built-ins only.
 *
 * Run from `demos/public_site`:
 *   node scripts/smoke.mjs
 *   npm run smoke
 */

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const distDir = path.join(projectRoot, 'dist');

const PREFIX = '/nested/site/';
const REDIRECT_PATH = '/nested/site';

const BUNDLE_MARKERS = [
  '--drl-theme-bg',
  'Boucle agent-environnement',
  'Dernière mise à jour',
  'Rejeu',
];

const MIME_TYPES = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.woff2', 'font/woff2'],
  ['.woff', 'font/woff'],
  ['.ttf', 'font/ttf'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.ico', 'image/x-icon'],
]);

const PAGES = [
  { label: 'index.html', path: `${PREFIX}` },
  { label: 'deep-blue.html', path: `${PREFIX}deep-blue.html` },
  { label: 'frozen-lake.html', path: `${PREFIX}frozen-lake.html` },
];

const ASSETS = [
  { path: `${PREFIX}demo/drl-demos.js`, type: 'text/javascript' },
  { path: `${PREFIX}demo/drl-three.js`, type: 'text/javascript' },
  { path: `${PREFIX}demo/drl-index.js`, type: 'text/javascript' },
  { path: `${PREFIX}katex/katex.min.css`, type: 'text/css' },
  { path: `${PREFIX}katex/fonts/KaTeX_Main-Regular.woff2`, type: 'font/woff2' },
];

const requestedPaths = [];
const failures = [];
let checks = 0;

function check(condition, message) {
  checks += 1;
  if (condition) {
    console.log(`PASS ${message}`);
  } else {
    failures.push(message);
    console.log(`FAIL ${message}`);
  }
  return Boolean(condition);
}

function contentTypeFor(filePath) {
  return MIME_TYPES.get(path.extname(filePath).toLowerCase()) ?? 'application/octet-stream';
}

function resolveRequestPath(requestPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(requestPath);
  } catch {
    return null;
  }

  let relative = decoded.startsWith(PREFIX) ? decoded.slice(PREFIX.length) : decoded;
  if (relative === '' || relative.endsWith('/')) {
    relative = `${relative}index.html`;
  }

  const absolute = path.resolve(distDir, relative);
  const inside = absolute === distDir || absolute.startsWith(`${distDir}${path.sep}`);
  return inside ? absolute : null;
}

function createStaticServer() {
  return createServer(async (request, response) => {
    const url = new URL(request.url ?? '/', 'http://localhost');
    const pathname = url.pathname;
    requestedPaths.push(pathname);

    if (pathname === REDIRECT_PATH) {
      response.writeHead(308, { location: PREFIX });
      response.end();
      return;
    }

    if (!pathname.startsWith(PREFIX)) {
      response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      response.end('Not found');
      return;
    }

    const filePath = resolveRequestPath(pathname);
    if (filePath === null) {
      response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      response.end('Not found');
      return;
    }

    try {
      const body = await readFile(filePath);
      response.writeHead(200, {
        'content-type': contentTypeFor(filePath),
        'content-length': body.byteLength,
        'cache-control': 'no-store',
      });
      response.end(body);
    } catch {
      response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      response.end('Not found');
    }
  });
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      server.off('error', reject);
      resolve();
    });
  });
  const address = server.address();
  if (address === null || typeof address === 'string') {
    throw new Error('The HTTP server did not expose a TCP address.');
  }
  return address.port;
}

async function closeServer(server) {
  if (server === null) {
    return;
  }
  server.closeAllConnections?.();
  await new Promise((resolve) => {
    server.close(() => resolve());
  });
}

async function fetchPath(baseUrl, requestPath) {
  const response = await fetch(`${baseUrl}${requestPath}`, { redirect: 'manual' });
  const body = await response.text();
  return {
    path: requestPath,
    status: response.status,
    contentType: response.headers.get('content-type') ?? '',
    body,
  };
}

function countMatches(value, pattern) {
  return (value.match(pattern) ?? []).length;
}

function checkHtmlDocument(label, html) {
  check(
    /<html\b[^>]*\blang=["']fr["']/i.test(html),
    `${label}: html element uses lang="fr"`,
  );

  const h1Count = countMatches(html, /<h1\b/gi);
  check(h1Count === 1, `${label}: has exactly one h1 (found ${h1Count})`);

  check(
    /<button\b[^>]*\bdata-theme-toggle\b/i.test(html),
    `${label}: theme toggle button exists`,
  );

  check(/<main\b[^>]*\bid=["']main["']/i.test(html), `${label}: has <main id="main">`);

  const headEnd = html.indexOf('</head>');
  check(headEnd !== -1, `${label}: has a closing </head>`);

  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)];
  const themeScript = scripts.find(
    (script) => script[1].includes('drl-theme') && script[1].includes('localStorage'),
  );
  check(Boolean(themeScript), `${label}: inline theme init script exists`);

  if (themeScript) {
    const scriptIndex = themeScript.index ?? -1;
    check(
      headEnd !== -1 && scriptIndex >= 0 && scriptIndex < headEnd,
      `${label}: theme init script appears before </head>`,
    );
    check(
      scriptIndex >= 0 &&
        (html.indexOf('<style') === -1 || scriptIndex < html.indexOf('<style')),
      `${label}: theme init script appears before the first style block`,
    );
  }

  const rootAttributes = countMatches(html, /\b(?:href|src)\s*=\s*["']\/(?!\/)/gi);
  check(
    rootAttributes === 0,
    `${label}: has no root-absolute href or src (found ${rootAttributes})`,
  );

  const rootCssUrls = countMatches(html, /url\(\s*["']?\/(?!\/)/gi);
  check(
    rootCssUrls === 0,
    `${label}: has no root-absolute CSS url() (found ${rootCssUrls})`,
  );

  check(!html.includes('/src/'), `${label}: has no /src/ reference`);
}

function checkDeepBluePage(html) {
  const label = 'deep-blue.html';

  check(html.includes('drl-deep-blue-page'), `${label}: contains drl-deep-blue-page`);
  check(html.includes('data-demo-log'), `${label}: contains data-demo-log`);
  check(html.includes('demo/drl-demos.js'), `${label}: loads demo/drl-demos.js`);
  check(html.includes('class="katex"'), `${label}: contains rendered KaTeX markup`);
}

function checkFrozenLakePage(html) {
  const label = 'frozen-lake.html';

  check(html.includes('drl-frozen-lake'), `${label}: contains drl-frozen-lake`);
  check(
    html.includes('drl-frozen-lake-replay'),
    `${label}: contains drl-frozen-lake-replay`,
  );
  check(html.includes('demo/drl-demos.js'), `${label}: loads demo/drl-demos.js`);
  check(html.includes('class="katex"'), `${label}: contains rendered KaTeX markup`);

  const demoOpen = html.indexOf('<drl-frozen-lake');
  const demoClose = html.indexOf('</drl-frozen-lake>');
  const controls = html.search(/class="[^"]*\bdemo-controls\b/);
  check(
    demoOpen !== -1 && demoClose > demoOpen && controls > demoOpen && controls < demoClose,
    `${label}: controls sit inside the demo element`,
  );

  const sliderCount = countMatches(html, /\bdata-param=/g);
  check(sliderCount === 7, `${label}: has seven data-param sliders (found ${sliderCount})`);
}

function checkHomePage(html) {
  check(
    html.includes("Jouer pour comprendre l'IA"),
    "index.html: contains the title text \"Jouer pour comprendre l'IA\"",
  );
}

async function main() {
  console.log(`[smoke] Project: ${projectRoot}`);
  console.log(`[smoke] Serving dist at ${PREFIX} over HTTP.`);

  const server = createStaticServer();
  try {
    const port = await listen(server);
    const baseUrl = `http://127.0.0.1:${port}`;
    console.log(`[smoke] Server listening on ${baseUrl}${PREFIX}`);

    const redirect = await fetch(`${baseUrl}${REDIRECT_PATH}`, { redirect: 'manual' });
    check(
      redirect.status === 308,
      `${REDIRECT_PATH} redirects with 308 (got ${redirect.status})`,
    );
    check(
      redirect.headers.get('location') === PREFIX,
      `${REDIRECT_PATH} redirects to ${PREFIX}`,
    );

    const pageResponses = new Map();
    for (const page of PAGES) {
      const result = await fetchPath(baseUrl, page.path);
      pageResponses.set(page.label, result);
      check(
        result.status === 200,
        `${page.path} returns 200 (got ${result.status})`,
      );
      check(
        result.contentType.startsWith('text/html'),
        `${page.path} is served as text/html (got ${result.contentType || 'none'})`,
      );
    }

    for (const asset of ASSETS) {
      const result = await fetchPath(baseUrl, asset.path);
      check(
        result.status === 200,
        `${asset.path} returns 200 (got ${result.status})`,
      );
      check(
        result.contentType.startsWith(asset.type),
        `${asset.path} is served as ${asset.type} (got ${result.contentType || 'none'})`,
      );
      if (asset.path.endsWith('.js')) {
        check(result.body.length > 0, `${asset.path} has a non-empty body`);
      }
    }

    const servedBundle = await fetchPath(baseUrl, `${PREFIX}demo/drl-demos.js`);
    for (const marker of BUNDLE_MARKERS) {
      checks += 1;
      if (servedBundle.status === 200 && servedBundle.body.includes(marker)) {
        console.log(`PASS served demo/drl-demos.js contains "${marker}"`);
      } else {
        const message = `served demo/drl-demos.js is stale: missing "${marker}"`;
        failures.push(message);
        console.log(`FAIL ${message}`);
      }
    }

    for (const page of PAGES) {
      const result = pageResponses.get(page.label);
      if (result) {
        checkHtmlDocument(page.label, result.body);
      }
    }

    const deepBlue = pageResponses.get('deep-blue.html');
    if (deepBlue) {
      checkDeepBluePage(deepBlue.body);
    }

    const frozenLake = pageResponses.get('frozen-lake.html');
    if (frozenLake) {
      checkFrozenLakePage(frozenLake.body);
    }

    const home = pageResponses.get('index.html');
    if (home) {
      checkHomePage(home.body);
    }

    // The redirect probe uses the bare prefix without a trailing slash. Every
    // other request must stay under /nested/site/.
    const strayPaths = requestedPaths.filter(
      (requestPath) => requestPath !== REDIRECT_PATH && !requestPath.startsWith(PREFIX),
    );
    check(
      strayPaths.length === 0,
      `all request paths stay under ${PREFIX} (stray: ${strayPaths.join(', ') || 'none'})`,
    );
    check(
      requestedPaths.filter((requestPath) => requestPath.startsWith(PREFIX)).length >=
        PAGES.length + ASSETS.length,
      `the server received every expected request (${requestedPaths.length} total)`,
    );
  } finally {
    await closeServer(server);
    console.log('[smoke] Server stopped.');
  }

  console.log(`[smoke] ${checks - failures.length}/${checks} checks passed.`);
  if (failures.length > 0) {
    console.log('[smoke] Failures:');
    for (const failure of failures) {
      console.log(`[smoke] - ${failure}`);
    }
    process.exitCode = 1;
  } else {
    console.log('[smoke] All checks passed.');
  }
}

main().catch((error) => {
  console.error(`[smoke] Unexpected error: ${error instanceof Error ? error.message : error}`);
  process.exitCode = 1;
});
