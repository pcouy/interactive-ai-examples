import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  CACHE_DIR_TAG,
  CACHE_DIR_TAG_NAME,
  cacheDirTagPath,
  writeCacheDirTag,
} from './cachedir-tag.mjs';

const SIGNATURE = 'Signature: 8a477f597d28d172789f06886806bc55';

/** Répertoires temporaires créés par les tests, nettoyés après chaque test. */
const tempDirs = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir() {
  const dir = await mkdtemp(join(tmpdir(), 'cachedir-tag-'));
  tempDirs.push(dir);
  return dir;
}

describe('CACHE_DIR_TAG', () => {
  it('starts with the exact 43-byte cachedir signature', () => {
    expect(CACHE_DIR_TAG.split('\n')[0]).toBe(SIGNATURE);
    expect(Buffer.byteLength(SIGNATURE, 'utf8')).toBe(43);
  });
});

describe('cacheDirTagPath', () => {
  it('places the tag under the dataset directory', () => {
    expect(cacheDirTagPath('/data/mnist')).toBe(join('/data/mnist', CACHE_DIR_TAG_NAME));
  });
});

describe('writeCacheDirTag', () => {
  it('creates the directory and writes the tag', async () => {
    const directory = join(await makeTempDir(), 'nested', 'dataset');

    expect(await writeCacheDirTag(directory)).toBe(true);
    expect(await readFile(cacheDirTagPath(directory), 'utf8')).toBe(CACHE_DIR_TAG);
  });

  it('is idempotent and leaves an up-to-date tag untouched', async () => {
    const directory = await makeTempDir();
    expect(await writeCacheDirTag(directory)).toBe(true);
    const before = await stat(cacheDirTagPath(directory));

    expect(await writeCacheDirTag(directory)).toBe(false);
    const after = await stat(cacheDirTagPath(directory));
    expect(after.mtimeMs).toBe(before.mtimeMs);
  });

  it('replaces a tag with different content', async () => {
    const directory = await makeTempDir();
    await writeFile(cacheDirTagPath(directory), 'not a cachedir tag\n');

    expect(await writeCacheDirTag(directory)).toBe(true);
    expect(await readFile(cacheDirTagPath(directory), 'utf8')).toBe(CACHE_DIR_TAG);
  });
});
