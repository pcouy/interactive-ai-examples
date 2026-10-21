import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { rewriteHtml } from './lib/url-rewrite.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(scriptDir, '../dist');

const ATTRIBUTE_ROOT = /\b(?:href|src)\s*=\s*["']\/(?!\/)/gi;
const CSS_ROOT = /url\(\s*["']?\/(?!\/)/gi;
const REFERENCE =
  /(?:\b(?:href|src)\s*=\s*["'](\.\/[^"']*)["'])|(?:url\(\s*["']?(\.\/[^"')]*)["']?\s*\))/gi;

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

function countRootAbsolute(html) {
  return (html.match(ATTRIBUTE_ROOT) ?? []).length + (html.match(CSS_ROOT) ?? []).length;
}

function hasRootAbsolute(html) {
  ATTRIBUTE_ROOT.lastIndex = 0;
  CSS_ROOT.lastIndex = 0;
  return ATTRIBUTE_ROOT.test(html) || CSS_ROOT.test(html);
}

function relativePath(directory, absolute) {
  return path.relative(directory, absolute) || path.basename(absolute);
}

async function fileExists(absolute) {
  try {
    return (await stat(absolute)).isFile();
  } catch {
    return false;
  }
}

let htmlFiles;
try {
  htmlFiles = await walkHtml(distDir);
} catch {
  console.error(`finalize-build: cannot read ${distDir}. Run \`npm run build\` first.`);
  process.exit(1);
}

const errors = [];
let rewrittenUrls = 0;
let rewrittenFiles = 0;
let checkedReferences = 0;

for (const file of htmlFiles) {
  const before = await readFile(file, 'utf8');
  const after = rewriteHtml(before);
  const rewritten = countRootAbsolute(before);

  if (after !== before) {
    await writeFile(file, after, 'utf8');
    rewrittenFiles += 1;
  }
  rewrittenUrls += rewritten;

  const label = relativePath(distDir, file);
  if (hasRootAbsolute(after)) {
    errors.push(`${label}: a root-absolute href, src, or url() remains after rewriting.`);
  }

  for (const match of after.matchAll(REFERENCE)) {
    const reference = match[1] ?? match[2];
    if (!reference) continue;

    const target = reference.split(/[?#]/, 1)[0];
    const absolute = path.resolve(path.dirname(file), target);
    checkedReferences += 1;

    if (!(await fileExists(absolute))) {
      errors.push(
        `${label}: reference ${JSON.stringify(reference)} does not resolve ` +
          `to a file in dist.`,
      );
    }
  }
}

const summary =
  `Rewrote ${rewrittenUrls} URLs in ${rewrittenFiles} files. ` +
  `Checked ${checkedReferences} references.`;
console.log(summary);

if (errors.length > 0) {
  for (const error of errors) console.error(`finalize-build: ${error}`);
  process.exit(1);
}
