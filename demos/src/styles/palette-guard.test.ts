import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const DEMO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

const SCAN_ROOTS = [
  'src',
  'playground',
  'plop-templates',
  'scripts',
  'public_site/src',
  'public_site/scripts',
];

const SKIPPED_DIRECTORIES = new Set(['node_modules', 'dist', '.astro', '.cache', '.git', 'public']);

const SCAN_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.mjs',
  '.cjs',
  '.html',
  '.hbs',
  '.css',
  '.astro',
]);

const ALLOWED_FILES = new Set([
  'src/styles/token-values.ts',
  'src/components/drl-mnist-cnn/drl-mnist-cnn.test.ts',
]);

const SKIPPED_FILES = new Set(['public_site/src/styles/palette.generated.css']);

const HEX_PATTERN = /#[0-9a-fA-F]{3,8}\b/g;
const URL_REFERENCE = /url\([^)\n]*#[^)\n]*\)/g;
const HREF_REFERENCE = /\bhref\s*=\s*(?:"[^"\n]*#[^"\n]*"|'[^'\n]*#[^'\n]*')/g;

function blank(match: string): string {
  return match.replace(/[^\n]/g, ' ');
}

/** Strips colour references such as `url(#id)` and `href="#anchor"`. */
function stripReferences(content: string): string {
  return content.replace(URL_REFERENCE, blank).replace(HREF_REFERENCE, blank);
}

/** Returns every hex colour literal with its 1-based line number. */
function hexLiterals(content: string): Array<{ line: number; hex: string }> {
  const found: Array<{ line: number; hex: string }> = [];
  const lines = stripReferences(content).split('\n');
  for (let index = 0; index < lines.length; index += 1) {
    for (const match of lines[index].matchAll(HEX_PATTERN)) {
      found.push({ line: index + 1, hex: match[0] });
    }
  }
  return found;
}

function walk(root: string, files: string[]): void {
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const fullPath = join(root, entry.name);
    if (entry.isDirectory()) {
      if (!SKIPPED_DIRECTORIES.has(entry.name)) {
        walk(fullPath, files);
      }
    } else if (entry.isFile() && SCAN_EXTENSIONS.has(extname(entry.name).toLowerCase())) {
      files.push(fullPath);
    }
  }
}

function toKey(file: string): string {
  return relative(DEMO_ROOT, file).split(sep).join('/');
}

interface ScanResult {
  readonly visited: readonly string[];
  readonly offenders: readonly string[];
}

function scanRepository(): ScanResult {
  const files: string[] = [];
  for (const root of SCAN_ROOTS) {
    const absolute = resolve(DEMO_ROOT, root);
    if (existsSync(absolute)) {
      walk(absolute, files);
    }
  }
  const visited: string[] = [];
  const offenders: string[] = [];
  for (const file of files) {
    const key = toKey(file);
    if (SKIPPED_FILES.has(key) || ALLOWED_FILES.has(key)) {
      continue;
    }
    visited.push(key);
    for (const { line, hex } of hexLiterals(readFileSync(file, 'utf8'))) {
      offenders.push(`${key}:${line} ${hex}`);
    }
  }
  return { visited, offenders };
}

describe('palette guard', () => {
  it('finds hex colours but ignores SVG references, rgba() and named colours', () => {
    const sample = [
      'a { fill: url(\u0023abcdef); }',
      'b { color: \u0023123abc; }',
      '<a href="\u0023top">x</a>',
      '<use href="\u0023abcdef" />',
      'c { background: rgba(0, 0, 0, 0.5); color: rebeccapurple; }',
    ].join('\n');
    expect(hexLiterals(sample)).toEqual([{ line: 2, hex: '\u0023123abc' }]);
  });

  it('keeps demos/ free of colour literals outside the allowlist', () => {
    const { visited, offenders } = scanRepository();
    expect(visited).toContain('src/styles/tokens.ts');
    expect(offenders).toEqual([]);
  });
});
