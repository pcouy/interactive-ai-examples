import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  buildIndex,
  gzipSync,
  main,
  parseNpyHeader,
  QUICKDRAW_CLASS_NAMES,
  readNpyImages,
  shardRanges,
  splitIndices,
  writeIdxImagesBytes,
  writeIdxLabelsBytes,
} from './convert-quickdraw.mjs';

const IMAGE_SIZE = 784;

/** En-tête `.npy` de 80 octets, version 1.0, avec les champs donnés. */
function buildNpyHeader({ descr = '<u1', fortranOrder = false, shape }) {
  const body = `{'descr': '${descr}', 'fortran_order': ${fortranOrder ? 'True' : 'False'}, 'shape': (${shape.join(', ')}), }`;
  const headerLength = 80 - 10;
  const padded = `${body.padEnd(headerLength - 1, ' ')}\n`;
  const header = new Uint8Array(80);
  const magic = '\x93NUMPY';
  for (let index = 0; index < magic.length; index += 1) {
    header[index] = magic.charCodeAt(index);
  }
  header[6] = 1;
  header[7] = 0;
  header[8] = headerLength & 0xff;
  header[9] = headerLength >> 8;
  for (let index = 0; index < padded.length; index += 1) {
    header[10 + index] = padded.charCodeAt(index);
  }
  return header;
}

/** Buffer `.npy` complet : en-tête de 80 octets puis `N` images remplies de `pixelValue`. */
function buildNpyBuffer({ descr, fortranOrder, shape, pixelValue = 11 }) {
  const header = buildNpyHeader({ descr, fortranOrder, shape });
  const payload = new Uint8Array(shape[0] * IMAGE_SIZE).fill(pixelValue);
  return new Uint8Array([...header, ...payload]);
}

describe('parseNpyHeader', () => {
  it('reads a hand-built 80-byte header', () => {
    const header = parseNpyHeader(buildNpyBuffer({ shape: [2, 784] }));
    expect(header.descr).toBe('<u1');
    expect(header.fortranOrder).toBe(false);
    expect(header.shape).toEqual([2, 784]);
    expect(header.dataOffset).toBe(80);
  });

  it('rejects a bad magic', () => {
    const buffer = buildNpyBuffer({ shape: [2, 784] });
    buffer[0] = 0x00;
    expect(() => parseNpyHeader(buffer)).toThrow(/magic/);
  });

  it('rejects a wrong dtype', () => {
    const buffer = buildNpyBuffer({ descr: '<u2', shape: [2, 784] });
    expect(() => parseNpyHeader(buffer)).toThrow(/<u1/);
  });

  it('accepts the byte-order-agnostic uint8 descriptor', () => {
    const header = parseNpyHeader(buildNpyBuffer({ descr: '|u1', shape: [2, 784] }));
    expect(header.descr).toBe('|u1');
    expect(header.shape).toEqual([2, 784]);
  });

  it('rejects a wrong shape', () => {
    const buffer = buildNpyBuffer({ shape: [2, 32] });
    expect(() => parseNpyHeader(buffer)).toThrow(/shape/);
  });

  it('rejects a fortran-ordered header', () => {
    const buffer = buildNpyBuffer({ fortranOrder: true, shape: [2, 784] });
    expect(() => parseNpyHeader(buffer)).toThrow(/fortran/);
  });
});

describe('readNpyImages', () => {
  it('returns the payload as a Uint8Array of N*784 bytes', () => {
    const images = readNpyImages(buildNpyBuffer({ shape: [3, 784], pixelValue: 200 }));
    expect(images).toBeInstanceOf(Uint8Array);
    expect(images.length).toBe(3 * IMAGE_SIZE);
    expect(images.every((byte) => byte === 200)).toBe(true);
  });

  it('rejects a truncated payload', () => {
    const buffer = buildNpyBuffer({ shape: [2, 784] });
    expect(() => readNpyImages(buffer.subarray(0, 100))).toThrow(/truncated/);
  });
});

describe('splitIndices', () => {
  it('is deterministic for a given seed', () => {
    expect(splitIndices(1000, 42)).toEqual(splitIndices(1000, 42));
  });

  it('splits 1000 indices into 800 train and 200 test', () => {
    const { train, test } = splitIndices(1000, 42);
    expect(train.length).toBe(800);
    expect(test.length).toBe(200);
  });

  it('produces disjoint groups that cover every index once', () => {
    const { train, test } = splitIndices(1000, 42);
    const seen = new Set();
    for (const index of train) seen.add(index);
    for (const index of test) {
      expect(seen.has(index)).toBe(false);
      seen.add(index);
    }
    expect(seen.size).toBe(1000);
  });

  it('uses the floor of 80 percent as the train count', () => {
    expect(splitIndices(5, 42).train.length).toBe(4);
    expect(splitIndices(5, 42).test.length).toBe(1);
  });
});

describe('shardRanges', () => {
  it('splits 2500 images into 1000, 1000 and 500', () => {
    expect(shardRanges(2500, 1000)).toEqual([
      { start: 0, count: 1000 },
      { start: 1000, count: 1000 },
      { start: 2000, count: 500 },
    ]);
  });

  it('returns an empty list for zero images', () => {
    expect(shardRanges(0, 1000)).toEqual([]);
  });
});

describe('writeIdxImagesBytes', () => {
  it('writes the IDX magic 2051, three dimensions and the right count', () => {
    const idx = writeIdxImagesBytes(new Uint8Array(2 * IMAGE_SIZE).fill(5));
    const view = new DataView(idx.buffer);
    expect(view.getUint32(0)).toBe(2051);
    expect(view.getUint32(4)).toBe(2);
    expect(view.getUint32(8)).toBe(28);
    expect(view.getUint32(12)).toBe(28);
    expect(idx.length).toBe(16 + 2 * IMAGE_SIZE);
    expect(idx[16]).toBe(5);
  });

  it('rejects a payload that is not a multiple of 784', () => {
    expect(() => writeIdxImagesBytes(new Uint8Array(10))).toThrow(/multiple/);
  });
});

describe('writeIdxLabelsBytes', () => {
  it('writes the IDX label magic 2049 and the label count', () => {
    const idx = writeIdxLabelsBytes(new Uint8Array([0, 1, 1]));
    const view = new DataView(idx.buffer);
    expect(view.getUint32(0)).toBe(2049);
    expect(view.getUint32(4)).toBe(3);
    expect(idx.length).toBe(11);
    expect(idx[8]).toBe(0);
  });
});

describe('gzipSync', () => {
  it('starts with the gzip magic bytes', () => {
    const compressed = gzipSync(new Uint8Array([1, 2, 3, 4]));
    expect(compressed[0]).toBe(0x1f);
    expect(compressed[1]).toBe(0x8b);
  });

  it('is deterministic for the same input', () => {
    expect(gzipSync(new Uint8Array([1, 2, 3, 4]))).toEqual(gzipSync(new Uint8Array([1, 2, 3, 4])));
  });
});

describe('buildIndex', () => {
  it('computes split counts and shard entries matching the ranges', () => {
    const index = buildIndex({
      classNames: ['a', 'b'],
      trainCounts: [2500, 1000],
      testCounts: [500, 500],
      shardSize: 1000,
    });
    expect(index.version).toBe(1);
    expect(index.dataset).toBe('quickdraw');
    expect(index.classNames).toEqual(['a', 'b']);
    expect(index.shardSize).toBe(1000);
    expect(index.splits.train.count).toBe(3500);
    expect(index.splits.test.count).toBe(1000);
    expect(index.splits.train.shards).toEqual([
      { file: 'train-c0-s00.idx.gz', classIndex: 0, start: 0, count: 1000 },
      { file: 'train-c0-s01.idx.gz', classIndex: 0, start: 1000, count: 1000 },
      { file: 'train-c0-s02.idx.gz', classIndex: 0, start: 2000, count: 500 },
      { file: 'train-c1-s00.idx.gz', classIndex: 1, start: 2500, count: 1000 },
    ]);
    expect(index.splits.test.shards).toEqual([
      { file: 'test-c0-s00.idx.gz', classIndex: 0, start: 0, count: 500 },
      { file: 'test-c1-s00.idx.gz', classIndex: 1, start: 500, count: 500 },
    ]);
    expect(index.gallery).toEqual({
      imagesFile: 'gallery.idx.gz',
      labelsFile: 'gallery-labels-idx1-ubyte.gz',
      perClass: 10,
    });
  });
});

/** Répertoires temporaires créés par les tests de `main`, nettoyés après chaque test. */
const tempDirs = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function makeTempDir(prefix) {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

describe('main with --only', () => {
  it('writes the selected class shards under its global class index', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');

    const images = new Uint8Array(2500 * IMAGE_SIZE);
    for (let index = 0; index < images.length; index += 1) images[index] = index % 256;
    const npy = buildNpyBuffer({ shape: [2500, 784] });
    npy.subarray(80).set(images);
    await writeFile(join(cacheDir, 'moon.npy'), npy);

    const md5 = createHash('md5').update(npy).digest('hex');
    const files = [
      { name: 'book.npy', size: 1, md5: '0'.repeat(32) },
      { name: 'moon.npy', size: npy.length, md5 },
      { name: 'cat.npy', size: 1, md5: '0'.repeat(32) },
      { name: 'umbrella.npy', size: 1, md5: '0'.repeat(32) },
      { name: 'cake.npy', size: 1, md5: '0'.repeat(32) },
    ];
    const fetchImpl = async () => {
      throw new Error('no network in tests');
    };

    expect(await main(['--only', 'moon'], { files, outputDir, cacheDir, fetchImpl })).toBe(0);

    const names = (await readdir(outputDir)).filter((name) => name.endsWith('.idx.gz')).sort();
    expect(names).toContain('train-c1-s00.idx.gz');
    expect(names).toContain('train-c1-s01.idx.gz');
    expect(names).toContain('test-c1-s00.idx.gz');
    expect(names.some((name) => name.startsWith('train-c0-'))).toBe(false);
    expect(names.some((name) => name.startsWith('test-c0-'))).toBe(false);

    const index = JSON.parse(await readFile(join(outputDir, 'index.json'), 'utf8'));
    expect(index.classNames).toEqual(QUICKDRAW_CLASS_NAMES);
    expect(index.splits.train.count).toBe(2000);
    expect(index.splits.test.count).toBe(500);
    expect(index.splits.train.shards.map((shard) => shard.file)).toEqual([
      'train-c1-s00.idx.gz',
      'train-c1-s01.idx.gz',
    ]);
  });
});
