import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
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
import { QUICKDRAW_FILES } from './dataset-sources.mjs';

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
    expect(index.version).toBe(2);
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
    expect(index.galleries).toEqual({
      train: {
        imagesFile: 'train-gallery.idx.gz',
        labelsFile: 'train-gallery-labels-idx1-ubyte.gz',
        perClass: 10,
      },
      test: {
        imagesFile: 'gallery.idx.gz',
        labelsFile: 'gallery-labels-idx1-ubyte.gz',
        perClass: 10,
      },
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
    const files = QUICKDRAW_FILES.map((entry) => ({
      name: entry.name,
      size: 1,
      md5: '0'.repeat(32),
    }));
    files[1] = { name: 'moon.npy', size: npy.length, md5 };
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

/** `.npy` synthétique de `count` images : chaque image est remplie d'une valeur unique. */
function buildSyntheticNpy(classIndex, count = 50) {
  const npy = buildNpyBuffer({ shape: [count, IMAGE_SIZE] });
  const payload = npy.subarray(80);
  for (let image = 0; image < count; image += 1) {
    payload.fill((classIndex * count + image) % 256, image * IMAGE_SIZE, (image + 1) * IMAGE_SIZE);
  }
  return npy;
}

/** Écrit les quinze `.npy` synthétiques et renvoie les entrées de `files`. */
async function writeSyntheticQuickdraw(cacheDir) {
  const files = [];
  for (let classIndex = 0; classIndex < QUICKDRAW_FILES.length; classIndex += 1) {
    const npy = buildSyntheticNpy(classIndex);
    const { name } = QUICKDRAW_FILES[classIndex];
    await writeFile(join(cacheDir, name), npy);
    files.push({ name, size: npy.length, md5: createHash('md5').update(npy).digest('hex') });
  }
  return files;
}

/** Compte les libellés valant `label` dans un fichier IDX de libellés décompressé. */
function countLabel(idxBytes, label) {
  return [...idxBytes.subarray(8)].filter((value) => value === label).length;
}

describe('main with fifteen classes', () => {
  it('writes index v2 with distinct train and test galleries, then deletes the raw files', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');
    const files = await writeSyntheticQuickdraw(cacheDir);
    let fetchCount = 0;
    const fetchImpl = async () => {
      fetchCount += 1;
      throw new Error('no network in tests');
    };

    expect(await main([], { files, outputDir, cacheDir, fetchImpl })).toBe(0);
    expect(fetchCount).toBe(0);

    const index = JSON.parse(await readFile(join(outputDir, 'index.json'), 'utf8'));
    expect(index.version).toBe(2);
    expect(index.classNames).toEqual(QUICKDRAW_CLASS_NAMES);
    expect(index.galleries).toEqual({
      train: {
        imagesFile: 'train-gallery.idx.gz',
        labelsFile: 'train-gallery-labels-idx1-ubyte.gz',
        perClass: 10,
      },
      test: {
        imagesFile: 'gallery.idx.gz',
        labelsFile: 'gallery-labels-idx1-ubyte.gz',
        perClass: 10,
      },
    });

    const trainGallery = gunzipSync(await readFile(join(outputDir, 'train-gallery.idx.gz')));
    const testGallery = gunzipSync(await readFile(join(outputDir, 'gallery.idx.gz')));
    expect(trainGallery.length).toBe(16 + 150 * IMAGE_SIZE);
    expect(testGallery.length).toBe(16 + 150 * IMAGE_SIZE);
    expect(trainGallery.subarray(16, 16 + IMAGE_SIZE)).not.toEqual(
      testGallery.subarray(16, 16 + IMAGE_SIZE),
    );
    const trainLabels = gunzipSync(
      await readFile(join(outputDir, 'train-gallery-labels-idx1-ubyte.gz')),
    );
    for (let classIndex = 0; classIndex < QUICKDRAW_CLASS_NAMES.length; classIndex += 1) {
      expect(countLabel(trainLabels, classIndex)).toBe(10);
    }

    const remainingRaw = (await readdir(cacheDir)).filter((name) => name.endsWith('.npy'));
    expect(remainingRaw).toEqual([]);

    // Les sorties sont complètes et les `.npy` ont disparu : le second passage ne
    // télécharge rien et ne change pas l'index.
    expect(await main([], { files, outputDir, cacheDir, fetchImpl })).toBe(0);
    expect(fetchCount).toBe(0);
  });

  it('re-downloads and rewrites a shard that is missing', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');
    const files = await writeSyntheticQuickdraw(cacheDir);
    const offlineFetch = async () => {
      throw new Error('no network in tests');
    };

    expect(await main([], { files, outputDir, cacheDir, fetchImpl: offlineFetch })).toBe(0);

    const missingShard = 'train-c0-s00.idx.gz';
    await rm(join(outputDir, missingShard));

    const rawByName = new Map(
      QUICKDRAW_FILES.map((entry, classIndex) => [entry.name, buildSyntheticNpy(classIndex)]),
    );
    let fetchCount = 0;
    const fetchImpl = async (url) => {
      fetchCount += 1;
      const name = url.slice(url.lastIndexOf('/') + 1);
      return new Response(rawByName.get(name), { status: 200, statusText: 'OK' });
    };

    expect(await main([], { files, outputDir, cacheDir, fetchImpl })).toBe(0);
    expect(fetchCount).toBe(QUICKDRAW_FILES.length);

    const names = await readdir(outputDir);
    expect(names).toContain(missingShard);

    const index = JSON.parse(await readFile(join(outputDir, 'index.json'), 'utf8'));
    expect(index.version).toBe(2);
    const referenced = [
      ...['train', 'test'].flatMap((split) => [
        index.splits[split].labelsFile,
        ...index.splits[split].shards.map((shard) => shard.file),
      ]),
      index.galleries.train.imagesFile,
      index.galleries.train.labelsFile,
      index.galleries.test.imagesFile,
      index.galleries.test.labelsFile,
    ];
    expect(referenced.every((name) => names.includes(name))).toBe(true);
  });

  it('writes index.json before deleting any raw file', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');
    const files = await writeSyntheticQuickdraw(cacheDir);
    const indexPath = join(outputDir, 'index.json');
    const indexExistedAtDeletion = [];
    const logSpy = vi.spyOn(console, 'log').mockImplementation((...args) => {
      if (String(args[0]).includes(': deleted from cache')) {
        indexExistedAtDeletion.push(existsSync(indexPath));
      }
    });
    const fetchImpl = async () => {
      throw new Error('no network in tests');
    };
    let exitCode;
    try {
      exitCode = await main([], { files, outputDir, cacheDir, fetchImpl });
    } finally {
      logSpy.mockRestore();
    }

    expect(exitCode).toBe(0);
    expect(indexExistedAtDeletion).toHaveLength(QUICKDRAW_FILES.length);
    expect(indexExistedAtDeletion.every(Boolean)).toBe(true);
    expect((await readdir(cacheDir)).filter((name) => name.endsWith('.npy'))).toEqual([]);
  });

  it('keeps raw files and writes no index when a later class fails', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');
    const bookRaw = buildSyntheticNpy(0);
    const moonRaw = new Uint8Array([1, 2, 3, 4, 5, 6]);
    await writeFile(join(cacheDir, 'book.npy'), bookRaw);
    await writeFile(join(cacheDir, 'moon.npy'), moonRaw);

    const files = QUICKDRAW_FILES.map((entry) => ({
      name: entry.name,
      size: 1,
      md5: '0'.repeat(32),
    }));
    files[0] = {
      name: 'book.npy',
      size: bookRaw.length,
      md5: createHash('md5').update(bookRaw).digest('hex'),
    };
    files[1] = {
      name: 'moon.npy',
      size: moonRaw.length,
      md5: createHash('md5').update(moonRaw).digest('hex'),
    };
    const fetchImpl = async () => {
      throw new Error('no network in tests');
    };
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    let exitCode;
    let errorMessages;
    try {
      exitCode = await main([], { files, outputDir, cacheDir, fetchImpl });
    } finally {
      errorMessages = errorSpy.mock.calls.map(([message]) => String(message));
      errorSpy.mockRestore();
    }

    expect(exitCode).toBe(1);
    expect(errorMessages.some((message) => message.includes('convert-quickdraw: FAILED'))).toBe(
      true,
    );
    expect((await readdir(cacheDir)).filter((name) => name.endsWith('.npy')).sort()).toEqual([
      'book.npy',
      'moon.npy',
    ]);
    const outputNames = await readdir(outputDir);
    expect(outputNames).not.toContain('index.json');
    expect(outputNames).toContain('train-c0-s00.idx.gz');
  });

  it('keeps the raw files with --keep-raw', async () => {
    const cacheDir = await makeTempDir('qd-cache-');
    const outputDir = await makeTempDir('qd-out-');
    const files = await writeSyntheticQuickdraw(cacheDir);
    const fetchImpl = async () => {
      throw new Error('no network in tests');
    };

    expect(await main(['--keep-raw'], { files, outputDir, cacheDir, fetchImpl })).toBe(0);

    const remainingRaw = (await readdir(cacheDir)).filter((name) => name.endsWith('.npy'));
    expect(remainingRaw.sort()).toEqual(QUICKDRAW_FILES.map((file) => file.name).sort());
  });
});
