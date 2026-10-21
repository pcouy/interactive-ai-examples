import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { deflateRawSync, gunzipSync } from 'node:zlib';
import { afterEach, describe, expect, it } from 'vitest';
import {
  buildShardedIndex,
  groupByClass,
  MEDMNIST_CLASS_NAMES,
  main,
  mergeTrainVal,
  parseNpy,
  parseNpyImages,
  parseZipEntries,
  shardRanges,
} from './convert-medmnist.mjs';

const IMAGE_SIZE = 784;
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** Somme de contrôle CRC-32 d'un buffer, attendue par le format zip. */
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** Écrit un entier 16 bits petit-boutiste. */
function writeU16(view, offset, value) {
  view.setUint16(offset, value, true);
}

/** Écrit un entier 32 bits petit-boutiste. */
function writeU32(view, offset, value) {
  view.setUint32(offset, value, true);
}

/**
 * Construit un buffer `.npy` : en-tête de version 1.0 puis charge utile.
 *
 * `descr` vaut `<u1` ou `<i8` ; `shape` est la forme complète ; `values`
 * fournit les octets (`u1`) ou les entiers (`i8`) dans l'ordre ligne-major.
 */
function buildNpy({ descr = '<u1', fortranOrder = false, shape, values }) {
  const headerLength = 128 - 10;
  const body = `{'descr': '${descr}', 'fortran_order': ${fortranOrder ? 'True' : 'False'}, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const padded = `${body.padEnd(headerLength - 1, ' ')}\n`;
  const header = new Uint8Array(128);
  const magic = '\x93NUMPY';
  for (let index = 0; index < magic.length; index += 1) header[index] = magic.charCodeAt(index);
  header[6] = 1;
  header[7] = 0;
  writeU16(new DataView(header.buffer), 8, headerLength);
  for (let index = 0; index < padded.length; index += 1)
    header[10 + index] = padded.charCodeAt(index);

  const items = shape.reduce((product, dimension) => product * dimension, 1);
  const payload = new Uint8Array(items * (descr === '<u1' ? 1 : 8));
  const view = new DataView(payload.buffer);
  for (let index = 0; index < items; index += 1) {
    if (descr === '<u1') payload[index] = values[index];
    else view.setBigInt64(index * 8, BigInt(values[index]), true);
  }
  return new Uint8Array([...header, ...payload]);
}

/**
 * Construit une archive zip en mémoire : en-têtes locaux, annuaire central et
 * enregistrement de fin. Chaque membre est compressé en flate brut (méthode 8).
 */
function buildZip(members) {
  const nameBytes = members.map((member) => new TextEncoder().encode(member.name));
  const compressed = members.map((member) => deflateRawSync(member.data));
  const localSize = members.reduce(
    (sum, _, index) => sum + 30 + nameBytes[index].length + compressed[index].length,
    0,
  );
  const centralOffset = localSize;
  const centralSize = members.reduce((sum, _, index) => sum + 46 + nameBytes[index].length, 0);
  const total = centralOffset + centralSize + 22;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  let offset = 0;
  const offsets = [];
  members.forEach((member, index) => {
    offsets.push(offset);
    const data = compressed[index];
    const crc = crc32(member.data);
    writeU32(view, offset, 0x04034b50);
    writeU16(view, offset + 4, 20);
    writeU16(view, offset + 6, 0);
    writeU16(view, offset + 8, 8);
    writeU16(view, offset + 10, 0);
    writeU16(view, offset + 12, 0);
    writeU32(view, offset + 14, crc);
    writeU32(view, offset + 18, data.length);
    writeU32(view, offset + 22, member.data.length);
    writeU16(view, offset + 26, nameBytes[index].length);
    writeU16(view, offset + 28, 0);
    out.set(nameBytes[index], offset + 30);
    out.set(data, offset + 30 + nameBytes[index].length);
    offset += 30 + nameBytes[index].length + data.length;
  });
  members.forEach((member, index) => {
    const crc = crc32(member.data);
    writeU32(view, offset, 0x02014b50);
    writeU16(view, offset + 4, 20);
    writeU16(view, offset + 6, 20);
    writeU16(view, offset + 8, 0);
    writeU16(view, offset + 10, 8);
    writeU16(view, offset + 12, 0);
    writeU16(view, offset + 14, 0);
    writeU32(view, offset + 16, crc);
    writeU32(view, offset + 20, compressed[index].length);
    writeU32(view, offset + 24, member.data.length);
    writeU16(view, offset + 28, nameBytes[index].length);
    writeU16(view, offset + 30, 0);
    writeU16(view, offset + 32, 0);
    writeU16(view, offset + 34, 0);
    writeU16(view, offset + 36, 0);
    writeU32(view, offset + 38, 0);
    writeU32(view, offset + 42, offsets[index]);
    out.set(nameBytes[index], offset + 46);
    offset += 46 + nameBytes[index].length;
  });
  writeU32(view, offset, 0x06054b50);
  writeU16(view, offset + 4, 0);
  writeU16(view, offset + 6, 0);
  writeU16(view, offset + 8, members.length);
  writeU16(view, offset + 10, members.length);
  writeU32(view, offset + 12, centralSize);
  writeU32(view, offset + 16, centralOffset);
  writeU16(view, offset + 20, 0);
  return out;
}

/** Valeur de pixel déterministe pour l'image `index` de la classe `classIndex`. */
function pixel(index, classIndex) {
  return (index * 7 + classIndex * 13 + 1) % 256;
}

/** Images `(N, 28, 28, 1)` uint8 pour `counts` images réparties sur deux classes. */
function buildImages(counts) {
  const total = counts.reduce((sum, count) => sum + count, 0);
  const values = [];
  let imageIndex = 0;
  counts.forEach((count, classIndex) => {
    for (let index = 0; index < count; index += 1) {
      for (let pixelIndex = 0; pixelIndex < IMAGE_SIZE; pixelIndex += 1) {
        values.push(pixel(imageIndex * IMAGE_SIZE + pixelIndex, classIndex));
      }
      imageIndex += 1;
    }
  });
  return buildNpy({ descr: '<u1', shape: [total, 28, 28, 1], values });
}

/** Libellés `(N, 1)` uint8 pour `counts` images réparties sur deux classes. */
function buildLabels(counts) {
  const values = [];
  counts.forEach((count, classIndex) => {
    for (let index = 0; index < count; index += 1) values.push(classIndex);
  });
  return buildNpy({ descr: '<u1', shape: [values.length, 1], values });
}

/** Archive `.npz` MedMNIST synthétique : six membres, deux classes. */
function buildArchive({ trainCounts = [8, 4], valCounts = [3, 3], testCounts = [10, 10] } = {}) {
  return buildZip([
    { name: 'train_images.npy', data: buildImages(trainCounts) },
    { name: 'val_images.npy', data: buildImages(valCounts) },
    { name: 'test_images.npy', data: buildImages(testCounts) },
    { name: 'train_labels.npy', data: buildLabels(trainCounts) },
    { name: 'val_labels.npy', data: buildLabels(valCounts) },
    { name: 'test_labels.npy', data: buildLabels(testCounts) },
  ]);
}

describe('parseZipEntries', () => {
  it('returns the entry names and the inflated bytes', () => {
    const archive = buildArchive();
    const entries = parseZipEntries(archive);
    expect([...entries.keys()]).toEqual([
      'train_images.npy',
      'val_images.npy',
      'test_images.npy',
      'train_labels.npy',
      'val_labels.npy',
      'test_labels.npy',
    ]);
    const images = entries.get('train_images.npy');
    expect(images).toBeInstanceOf(Uint8Array);
    expect(images.length).toBe((8 + 4) * IMAGE_SIZE + 128);
    expect(images[0]).toBe(0x93);
  });

  it('rejects a buffer that is not a zip archive', () => {
    expect(() => parseZipEntries(new Uint8Array(64))).toThrow(/central directory/);
  });
});

describe('parseNpy', () => {
  it('reads a (2,28,28,1) uint8 image array', () => {
    const values = Array.from({ length: 2 * IMAGE_SIZE }, (_, index) => index % 256);
    const parsed = parseNpy(buildNpy({ descr: '<u1', shape: [2, 28, 28, 1], values }));
    expect(parsed.descr).toBe('<u1');
    expect(parsed.shape).toEqual([2, 28, 28, 1]);
    expect(parsed.payload.length).toBe(2 * IMAGE_SIZE);
    expect([...parsed.payload.slice(0, 4)]).toEqual([0, 1, 2, 3]);
  });

  it('reads a (2,1) int64 label array', () => {
    const parsed = parseNpy(buildNpy({ descr: '<i8', shape: [2, 1], values: [1, 0] }));
    expect(parsed.descr).toBe('<i8');
    expect(parsed.shape).toEqual([2, 1]);
    expect([...parsed.payload]).toEqual([1, 0]);
  });

  it('rejects a bad magic', () => {
    const buffer = buildNpy({ descr: '<u1', shape: [1, 1], values: [0] });
    buffer[0] = 0x00;
    expect(() => parseNpy(buffer)).toThrow(/bad magic/);
  });

  it('rejects a fortran-order header', () => {
    const buffer = buildNpy({ descr: '<u1', fortranOrder: true, shape: [1, 1], values: [0] });
    expect(() => parseNpy(buffer)).toThrow(/fortran/);
  });

  it('rejects an unsupported dtype', () => {
    const buffer = buildNpy({ descr: '<f8', shape: [1, 1], values: [0] });
    expect(() => parseNpyImages(buffer, 'x.npz', 'train_images')).toThrow(/dtype/);
  });
});

describe('mergeTrainVal', () => {
  it('concatenates the images and the labels and reports the counts', () => {
    const train = {
      images: new Uint8Array([1, 2, 3, 4]),
      labels: new Uint8Array([0, 1]),
    };
    const val = {
      images: new Uint8Array([5, 6]),
      labels: new Uint8Array([0]),
    };
    const merged = mergeTrainVal(train, val);
    expect([...merged.images]).toEqual([1, 2, 3, 4, 5, 6]);
    expect([...merged.labels]).toEqual([0, 1, 0]);
    expect(merged.trainCount).toBe(2);
    expect(merged.valCount).toBe(1);
    expect(merged.count).toBe(3);
  });
});

describe('groupByClass', () => {
  it('reorders by class and keeps a stable order within a class', () => {
    // Deux classes de deux images ; l'image d'indice `i` commence par l'octet `10 + i`.
    const images = new Uint8Array(4 * IMAGE_SIZE);
    for (let index = 0; index < 4; index += 1) images[index * IMAGE_SIZE] = 10 + index;
    const labels = new Uint8Array([1, 0, 1, 0]);
    const grouped = groupByClass(images, labels);
    expect([...grouped.labels]).toEqual([0, 0, 1, 1]);
    expect([
      grouped.images[0],
      grouped.images[IMAGE_SIZE],
      grouped.images[2 * IMAGE_SIZE],
      grouped.images[3 * IMAGE_SIZE],
    ]).toEqual([11, 13, 10, 12]);
    expect(grouped.counts).toEqual([2, 2]);
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

describe('buildShardedIndex', () => {
  it('produces contiguous start values and shard counts matching shardRanges', () => {
    const index = buildShardedIndex({
      dataset: 'pneumoniamnist',
      classNames: ['normal', 'pneumonia'],
      trainCounts: [2500, 1000],
      testCounts: [500, 500],
      shardSize: 1000,
    });
    expect(index.version).toBe(2);
    expect(index.dataset).toBe('pneumoniamnist');
    expect(index.classNames).toEqual(['normal', 'pneumonia']);
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

/** Instantané du contenu de chaque fichier d'un répertoire. */
async function snapshot(dir) {
  const files = (await readdir(dir)).sort();
  const entries = {};
  for (const file of files) {
    const info = await stat(join(dir, file));
    entries[file] = { bytes: await readFile(join(dir, file)), mtimeMs: info.mtimeMs };
  }
  return entries;
}

describe('main', () => {
  it('writes index v2 and both galleries, deletes the raw archive and skips on a second run', async () => {
    const archive = buildArchive({ trainCounts: [10, 8], valCounts: [3, 3] });
    const cacheDir = await makeTempDir('mm-cache-');
    const outputDir = await makeTempDir('mm-out-');
    const md5 = createHash('md5').update(archive).digest('hex');
    const sources = [
      { dataset: 'pneumoniamnist', name: 'pneumoniamnist.npz', size: archive.length, md5 },
    ];
    const fetchImpl = async () => new Response(archive, { status: 200, statusText: 'OK' });

    expect(await main([], { sources, targetRoot: outputDir, cacheRoot: cacheDir, fetchImpl })).toBe(
      0,
    );

    const datasetDir = join(outputDir, 'pneumoniamnist');
    const names = (await readdir(datasetDir)).sort();
    expect(names).toContain('index.json');
    expect(names).toContain('train-labels-idx1-ubyte.gz');
    expect(names).toContain('test-labels-idx1-ubyte.gz');
    expect(names).toContain('train-gallery.idx.gz');
    expect(names).toContain('train-gallery-labels-idx1-ubyte.gz');
    expect(names).toContain('gallery.idx.gz');
    expect(names).toContain('gallery-labels-idx1-ubyte.gz');
    expect(names).toContain('train-c0-s00.idx.gz');
    expect(names).toContain('train-c1-s00.idx.gz');

    const index = JSON.parse(await readFile(join(datasetDir, 'index.json'), 'utf8'));
    expect(index.version).toBe(2);
    expect(index.dataset).toBe('pneumoniamnist');
    expect(index.classNames).toEqual(MEDMNIST_CLASS_NAMES.pneumoniamnist);
    expect(index.splits.train.count).toBe(24);
    expect(index.splits.test.count).toBe(20);
    expect(index.splits.train.shards.map((shard) => shard.count)).toEqual([13, 11]);
    expect(index.splits.test.shards.map((shard) => shard.count)).toEqual([10, 10]);
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

    const trainGallery = gunzipSync(await readFile(join(datasetDir, 'train-gallery.idx.gz')));
    const testGallery = gunzipSync(await readFile(join(datasetDir, 'gallery.idx.gz')));
    expect(trainGallery.length).toBe(16 + 20 * IMAGE_SIZE);
    expect(testGallery.length).toBe(16 + 20 * IMAGE_SIZE);
    const trainGalleryLabels = gunzipSync(
      await readFile(join(datasetDir, 'train-gallery-labels-idx1-ubyte.gz')),
    );
    expect([...trainGalleryLabels.subarray(8)].filter((label) => label === 0)).toHaveLength(10);
    expect([...trainGalleryLabels.subarray(8)].filter((label) => label === 1)).toHaveLength(10);

    expect((await readdir(cacheDir)).filter((name) => name.endsWith('.npz'))).toEqual([]);

    const first = await snapshot(datasetDir);
    const shardBefore = first['train-c0-s00.idx.gz'].mtimeMs;

    const offlineFetch = async () => {
      throw new Error('no network in tests');
    };
    expect(
      await main([], {
        sources,
        targetRoot: outputDir,
        cacheRoot: cacheDir,
        fetchImpl: offlineFetch,
      }),
    ).toBe(0);

    const second = await snapshot(datasetDir);
    expect(Object.keys(second).sort()).toEqual(Object.keys(first).sort());
    for (const file of Object.keys(first)) {
      expect(second[file].bytes.equals(first[file].bytes)).toBe(true);
    }
    // Le morceau existant n'est pas réécrit : son horodatage ne change pas.
    expect(second['train-c0-s00.idx.gz'].mtimeMs).toBe(shardBefore);
  });
});
