import { gzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ShardedDatasetDefinition } from '../../core/dataset-registry.js';
import type { RoundRef } from '../../core/mnist-cnn-segmented.js';
import type { ShardEntry, ShardedIndex, ShardedSplitIndex } from './sharded-assets.js';

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;

function u32Bytes(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function imageIdx(count: number, fillBase = 0): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(16 + count * 28 * 28));
  bytes.set([...u32Bytes(IMAGE_MAGIC), ...u32Bytes(count), ...u32Bytes(28), ...u32Bytes(28)]);
  for (let index = 0; index < count * 28 * 28; index += 1) {
    bytes[16 + index] = (fillBase + index * 7) % 256;
  }
  return bytes;
}

function labelIdx(labels: number[]): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(8 + labels.length));
  bytes.set([...u32Bytes(LABEL_MAGIC), ...u32Bytes(labels.length)]);
  for (let index = 0; index < labels.length; index += 1) {
    bytes[index + 8] = labels[index];
  }
  return bytes;
}

function gz(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  return new Uint8Array(gzipSync(bytes));
}

/** Jeu segmenté de test : deux classes, des fragments de deux images. */
const SHARDED_DATASET: ShardedDatasetDefinition = {
  id: 'quickdraw',
  name: 'QuickDraw',
  description: 'jeu segmenté de test',
  classNames: ['livre', 'lune'],
  trainCount: 12,
  testCount: 12,
  defaultTrainPercent: 100,
  defaultValidationPercent: 100,
  license: 'test',
  attribution: 'test',
  kind: 'sharded',
  indexFile: 'index.json',
  shardSize: 2,
  downloadCommand: 'npm run quickdraw:data',
};

const TRAIN_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
const TEST_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

/** Index valide : deux classes, trois fragments pleins par classe et par fraction. */
function baseIndex(): Record<string, unknown> {
  const shards = (prefix: string) => [
    { file: `${prefix}-c0-s00.idx.gz`, classIndex: 0, start: 0, count: 2 },
    { file: `${prefix}-c0-s01.idx.gz`, classIndex: 0, start: 2, count: 2 },
    { file: `${prefix}-c0-s02.idx.gz`, classIndex: 0, start: 4, count: 2 },
    { file: `${prefix}-c1-s00.idx.gz`, classIndex: 1, start: 6, count: 2 },
    { file: `${prefix}-c1-s01.idx.gz`, classIndex: 1, start: 8, count: 2 },
    { file: `${prefix}-c1-s02.idx.gz`, classIndex: 1, start: 10, count: 2 },
  ];
  return {
    version: 2,
    dataset: 'quickdraw',
    classNames: ['livre', 'lune'],
    shardSize: 2,
    splits: {
      train: { count: 12, labelsFile: 'train-labels.idx.gz', shards: shards('train') },
      test: { count: 12, labelsFile: 'test-labels.idx.gz', shards: shards('test') },
    },
    galleries: {
      train: {
        imagesFile: 'train-gallery.idx.gz',
        labelsFile: 'train-gallery-labels.idx.gz',
        perClass: 3,
      },
      test: { imagesFile: 'gallery.idx.gz', labelsFile: 'gallery-labels.idx.gz', perClass: 3 },
    },
  };
}

/** Libellés des galeries de test : l'entraînement et le test diffèrent. */
const GALLERY_LABELS = new Map([
  ['train-gallery-labels.idx.gz', [1, 1, 1, 0, 0, 0]],
  ['gallery-labels.idx.gz', [0, 0, 0, 1, 1, 1]],
]);

/** Ressources complètes pour l'index donné : index, libellés, fragments et galeries. */
function assetsForIndex(index: Record<string, unknown>): Map<string, Uint8Array<ArrayBuffer>> {
  const assets = new Map<string, Uint8Array<ArrayBuffer>>();
  assets.set('index.json', new TextEncoder().encode(JSON.stringify(index)));
  assets.set('train-labels.idx.gz', gz(labelIdx(TRAIN_LABELS)));
  assets.set('test-labels.idx.gz', gz(labelIdx(TEST_LABELS)));
  const splits = index.splits as {
    train: { shards: { file: string; count: number }[] };
    test: { shards: { file: string; count: number }[] };
  };
  for (const split of [splits.train, splits.test]) {
    for (const shard of split.shards) {
      assets.set(shard.file, gz(imageIdx(shard.count)));
    }
  }
  const galleries = index.galleries as {
    train: { imagesFile: string; labelsFile: string };
    test: { imagesFile: string; labelsFile: string };
  };
  for (const gallery of [galleries.train, galleries.test]) {
    const labels = GALLERY_LABELS.get(gallery.labelsFile) ?? [0, 0, 0, 1, 1, 1];
    const fillBase = gallery.imagesFile.startsWith('train') ? 100 : 0;
    assets.set(gallery.imagesFile, gz(imageIdx(labels.length, fillBase)));
    assets.set(gallery.labelsFile, gz(labelIdx(labels)));
  }
  return assets;
}

function stubFetch(assets: Map<string, Uint8Array<ArrayBuffer>>): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const name = String(input).split('/').pop() ?? '';
    const body = assets.get(name);
    if (body === undefined) {
      return new Response('missing', { status: 404, statusText: 'Not Found' });
    }
    return new Response(body, { status: 200 });
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

async function loadLoader(): Promise<typeof import('./sharded-assets.js')> {
  vi.resetModules();
  return import('./sharded-assets.js');
}

/** Noms de fichiers des fragments effectivement téléchargés. */
function fetchedShardFiles(mock: ReturnType<typeof vi.fn>): string[] {
  return mock.mock.calls
    .map((call) => String(call[0]).split('/').pop() ?? '')
    .filter((name) => /-s\d+\.idx\.gz$/.test(name));
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('parseIndexJson', () => {
  it('accepts a valid version 2 index with both galleries', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = parseIndexJson(JSON.stringify(baseIndex()));
    expect(index.version).toBe(2);
    expect(index.classNames).toEqual(['livre', 'lune']);
    expect(index.shardSize).toBe(2);
    expect(index.splits.train.count).toBe(12);
    expect(index.splits.train.shards).toHaveLength(6);
    expect(index.splits.test.shards).toHaveLength(6);
    expect(index.galleries.train.imagesFile).toBe('train-gallery.idx.gz');
    expect(index.galleries.test.labelsFile).toBe('gallery-labels.idx.gz');
    expect(index.galleries.test.perClass).toBe(3);
  });

  it('accepts the expected dataset and rejects a different one', async () => {
    const { parseIndexJson } = await loadLoader();
    const text = JSON.stringify(baseIndex());
    expect(parseIndexJson(text, 'quickdraw').dataset).toBe('quickdraw');
    expect(() => parseIndexJson(text, 'other')).toThrow(/dataset/);
    expect(() => parseIndexJson(text, 'other')).toThrow(RangeError);
  });

  it('rejects a shard whose classIndex is not below the class count', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    const splits = index.splits as {
      train: { shards: { classIndex: number }[] };
    };
    splits.train.shards[0].classIndex = 2;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(RangeError);
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/train-c0-s00\.idx\.gz/);
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/classIndex 2/);
  });

  it('rejects a missing splits field', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    delete index.splits;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/splits/);
  });

  it('rejects a missing galleries field', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    delete index.galleries;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/galleries/);
  });

  it('rejects a missing train gallery', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    const galleries = index.galleries as { train?: unknown };
    delete galleries.train;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/galleries\.train/);
  });

  it('rejects a version 1 index and names both rerun commands', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    index.version = 1;
    const parse = () => parseIndexJson(JSON.stringify(index));
    expect(parse).toThrow(RangeError);
    expect(parse).toThrow(/version 1/);
    expect(parse).toThrow(/npm run quickdraw:data/);
    expect(parse).toThrow(/npm run medmnist:data/);
  });

  it('rejects an unsupported version', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    index.version = 3;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/version/);
  });

  it('rejects a class count of 1 or 21', async () => {
    const { parseIndexJson } = await loadLoader();
    const one = baseIndex();
    one.classNames = ['seule'];
    expect(() => parseIndexJson(JSON.stringify(one))).toThrow(/classes/);
    const twentyOne = baseIndex();
    twentyOne.classNames = Array.from({ length: 21 }, (_unused, index) => `c${index}`);
    expect(() => parseIndexJson(JSON.stringify(twentyOne))).toThrow(/classes/);
  });

  it('rejects a shard that ends beyond the split count', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    const splits = index.splits as {
      train: { shards: { start: number; count: number }[] };
    };
    splits.train.shards[0].start = 11;
    splits.train.shards[0].count = 2;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/beyond the split count/);
  });

  it('rejects a non-positive shardSize', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    index.shardSize = 0;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/shardSize/);
  });
});

describe('shardCountForPercent', () => {
  it('rounds to the expected shard count', async () => {
    const { shardCountForPercent } = await loadLoader();
    expect(shardCountForPercent(1, 100)).toBe(1);
    expect(shardCountForPercent(2, 100)).toBe(2);
    expect(shardCountForPercent(50, 100)).toBe(50);
    expect(shardCountForPercent(100, 100)).toBe(100);
    expect(shardCountForPercent(100, 1)).toBe(1);
  });

  it('never returns zero and caps at the available shards', async () => {
    const { shardCountForPercent } = await loadLoader();
    for (let percent = 1; percent <= 100; percent += 1) {
      expect(shardCountForPercent(percent, 100)).toBeGreaterThanOrEqual(1);
      expect(shardCountForPercent(percent, 100)).toBeLessThanOrEqual(100);
    }
    expect(shardCountForPercent(99, 3)).toBe(3);
  });

  it('rejects an invalid percent or shard count', async () => {
    const { shardCountForPercent } = await loadLoader();
    for (const percent of [0, 101, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => shardCountForPercent(percent, 100)).toThrow(RangeError);
    }
    expect(() => shardCountForPercent(50, 0)).toThrow(RangeError);
  });
});

describe('selectShardEntries', () => {
  const split: ShardedSplitIndex = {
    count: 12,
    labelsFile: 'train-labels.idx.gz',
    shards: [
      { file: 'z-first.idx.gz', classIndex: 0, start: 0, count: 2 },
      { file: 'a-second.idx.gz', classIndex: 0, start: 2, count: 2 },
      { file: 'm-third.idx.gz', classIndex: 0, start: 4, count: 2 },
      { file: 'b-first.idx.gz', classIndex: 1, start: 6, count: 2 },
      { file: 'y-second.idx.gz', classIndex: 1, start: 8, count: 2 },
      { file: 'c-third.idx.gz', classIndex: 1, start: 10, count: 2 },
    ],
  };

  it('keeps the first shard of each class at 1 %', async () => {
    const { selectShardEntries } = await loadLoader();
    const selected = selectShardEntries(split, 1);
    expect(selected.map((shard) => shard.file)).toEqual(['z-first.idx.gz', 'b-first.idx.gz']);
  });

  it('keeps two shards of each class at 67 %', async () => {
    const { selectShardEntries } = await loadLoader();
    const selected = selectShardEntries(split, 67);
    expect(selected.map((shard) => shard.file)).toEqual([
      'z-first.idx.gz',
      'a-second.idx.gz',
      'b-first.idx.gz',
      'y-second.idx.gz',
    ]);
  });

  it('returns the shards in class order, not in file name order', async () => {
    const { selectShardEntries } = await loadLoader();
    const selected = selectShardEntries(split, 100);
    expect(selected.map((shard) => shard.file)).toEqual([
      'z-first.idx.gz',
      'a-second.idx.gz',
      'm-third.idx.gz',
      'b-first.idx.gz',
      'y-second.idx.gz',
      'c-third.idx.gz',
    ]);
  });

  const sixClasses: ShardedSplitIndex = {
    count: 12,
    labelsFile: 'train-labels.idx.gz',
    shards: [0, 1, 2, 3, 4, 5].map((classIndex) => ({
      file: `c${classIndex}.idx.gz`,
      classIndex,
      start: classIndex * 2,
      count: 2,
    })),
  };

  it('keeps only the selected classes, in sorted order', async () => {
    const { selectShardEntries } = await loadLoader();
    const selected = selectShardEntries(sixClasses, 100, [2, 5]);
    expect(selected.map((shard) => shard.file)).toEqual(['c2.idx.gz', 'c5.idx.gz']);
  });

  it('sorts and deduplicates the selected classes', async () => {
    const { selectShardEntries } = await loadLoader();
    const selected = selectShardEntries(sixClasses, 100, [5, 2, 5]);
    expect(selected.map((shard) => shard.classIndex)).toEqual([2, 5]);
  });

  it('treats an empty selection as every class', async () => {
    const { selectShardEntries } = await loadLoader();
    expect(selectShardEntries(sixClasses, 100, [])).toEqual(selectShardEntries(sixClasses, 100));
  });

  it('rejects a selected class without a shard', async () => {
    const { selectShardEntries } = await loadLoader();
    expect(() => selectShardEntries(sixClasses, 100, [2, 9])).toThrow(RangeError);
    expect(() => selectShardEntries(sixClasses, 100, [2, 9])).toThrow(/class 9/);
  });
});

describe('mergeShards', () => {
  const entryA = { file: 'a.idx.gz', classIndex: 0, start: 0, count: 2 };
  const entryB = { file: 'b.idx.gz', classIndex: 1, start: 3, count: 1 };
  const labels = Uint8Array.from([10, 11, 12, 13, 14, 15]);

  it('concatenates the images and slices the labels from the right start', async () => {
    const { mergeShards } = await loadLoader();
    const imageA = new Uint8Array(2 * 784).fill(1);
    const imageB = new Uint8Array(784).fill(2);
    const merged = mergeShards([entryA, entryB], [imageA, imageB], labels);
    expect(merged.count).toBe(3);
    expect(merged.images).toBeInstanceOf(Uint8Array);
    expect(merged.images).toHaveLength(3 * 784);
    expect(merged.images).not.toBe(imageA);
    expect(merged.images[0]).toBe(1);
    expect(merged.images[2 * 784]).toBe(2);
    expect(Array.from(merged.labels)).toEqual([10, 11, 13]);
  });

  it('rejects an image with the wrong byte length', async () => {
    const { mergeShards } = await loadLoader();
    expect(() => mergeShards([entryA], [new Uint8Array(784)], labels)).toThrow(/image bytes/);
  });

  it('rejects a labels slice beyond the split labels', async () => {
    const { mergeShards } = await loadLoader();
    const outOfRange = { file: 'b.idx.gz', classIndex: 1, start: 10, count: 1 };
    expect(() => mergeShards([outOfRange], [new Uint8Array(784)], labels)).toThrow(/needs labels/);
  });

  it('remaps the sliced labels through the provided function', async () => {
    const { mergeShards } = await loadLoader();
    const remapped = mergeShards(
      [entryA, entryB],
      [new Uint8Array(2 * 784), new Uint8Array(784)],
      Uint8Array.from([0, 1, 0, 1]),
      (label) => 1 - label,
    );
    expect(Array.from(remapped.labels)).toEqual([1, 0, 0]);
  });

  it('rejects a remapped label outside the class count', async () => {
    const { mergeShards } = await loadLoader();
    const images = [new Uint8Array(2 * 784), new Uint8Array(784)];
    const sliced = Uint8Array.from([0, 1, 0, 1]);
    expect(() => mergeShards([entryA, entryB], images, sliced, () => 2)).toThrow(RangeError);
    expect(() => mergeShards([entryA, entryB], images, sliced, () => 2)).toThrow(
      /outside \[0, 2\)/,
    );
    expect(() => mergeShards([entryA, entryB], images, sliced, () => 0.5)).toThrow(RangeError);
  });
});

describe('loadShardedSplits', () => {
  it('fetches the index, two labels files and the selected shards', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedSplits } = await loadLoader();

    const splits = await loadShardedSplits(SHARDED_DATASET, 100, 100);
    expect(mock).toHaveBeenCalledTimes(15);
    expect(splits.train.count).toBe(12);
    expect(Array.from(splits.train.labels)).toEqual(TRAIN_LABELS);
    expect(splits.test.count).toBe(12);
    expect(Array.from(splits.test.labels)).toEqual(TEST_LABELS);
    // Tous les fragments sont sélectionnés à 100 % : aucun fichier hors sélection.
    expect(fetchedShardFiles(mock)).toHaveLength(12);
    for (const call of mock.mock.calls) {
      expect(String(call[0])).toContain('/datasets/quickdraw/');
    }
  });

  it('fetches one shard per class at 2 % and reports progress', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedSplits } = await loadLoader();
    const progress: [number, number][] = [];

    const splits = await loadShardedSplits(SHARDED_DATASET, 2, 2, (loaded, total) => {
      progress.push([loaded, total]);
    });
    // Index + deux libellés + un fragment par classe et par fraction.
    expect(mock).toHaveBeenCalledTimes(7);
    expect(splits.train.count).toBe(4);
    expect(progress.at(-1)).toEqual([7, 7]);
  });

  it('downloads only the selected classes and remaps their labels to 0..N-1', async () => {
    const fixture = segmentFixture(15, 4);
    const assets = new Map(fixture.assets);
    assets.set('index.json', new TextEncoder().encode(JSON.stringify(fixture.json)));
    const mock = stubFetch(assets);
    const { loadShardedSplits } = await loadLoader();
    const progress: [number, number][] = [];

    const splits = await loadShardedSplits(
      SHARDED_DATASET,
      100,
      100,
      (loaded, total) => {
        progress.push([loaded, total]);
      },
      [2, 5],
    );

    // Deux classes, quatre fragments de deux images par fraction.
    expect(splits.train.count).toBe(16);
    expect(splits.test.count).toBe(16);
    const expectedLabels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    expect(Array.from(splits.train.labels)).toEqual(expectedLabels);
    expect(Array.from(splits.test.labels)).toEqual(expectedLabels);

    const shards = fetchedShardFiles(mock);
    expect(shards).toHaveLength(16);
    for (const name of shards) {
      expect(name).toMatch(/-c(2|5)-/);
    }
    // Index + deux libellés + seize fragments (deux classes, quatre fragments,
    // deux fractions).
    expect(progress.at(-1)).toEqual([19, 19]);
  });

  it('keeps every class when the selection is null or empty', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedSplits } = await loadLoader();

    const withNull = await loadShardedSplits(SHARDED_DATASET, 100, 100, undefined, null);
    expect(Array.from(withNull.train.labels)).toEqual(TRAIN_LABELS);
    expect(Array.from(withNull.test.labels)).toEqual(TEST_LABELS);

    const withEmpty = await loadShardedSplits(SHARDED_DATASET, 100, 100, undefined, []);
    expect(Array.from(withEmpty.train.labels)).toEqual(TRAIN_LABELS);
    expect(fetchedShardFiles(mock)).toHaveLength(12);
  });

  it('rejects a labels file that carries an unselected class', async () => {
    const fixture = segmentFixture(15, 4);
    const labels = [...fixture.labels];
    // Premier fragment de la classe 2 : le libellé 7 n'est pas sélectionné.
    labels[16] = 7;
    const assets = new Map(fixture.assets);
    assets.set('train-labels.idx.gz', gz(labelIdx(labels)));
    assets.set('index.json', new TextEncoder().encode(JSON.stringify(fixture.json)));
    stubFetch(assets);
    const { loadShardedSplits } = await loadLoader();

    const error = await loadShardedSplits(SHARDED_DATASET, 2, 2, undefined, [2, 5]).then(
      () => {
        throw new Error('expected the load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toMatch(/label 7/);
  });

  it('caches a second call with the same percentages', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedSplits } = await loadLoader();

    await loadShardedSplits(SHARDED_DATASET, 100, 100);
    expect(mock).toHaveBeenCalledTimes(15);
    await loadShardedSplits(SHARDED_DATASET, 100, 100);
    expect(mock).toHaveBeenCalledTimes(15);
  });

  it('fetches no shard already cached for a smaller second call', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedSplits } = await loadLoader();

    await loadShardedSplits(SHARDED_DATASET, 50, 50);
    expect(mock).toHaveBeenCalledTimes(11);
    await loadShardedSplits(SHARDED_DATASET, 100, 100);
    // Les fragments déjà téléchargés ne sont pas rechargés.
    expect(mock).toHaveBeenCalledTimes(15);
    await loadShardedSplits(SHARDED_DATASET, 2, 2);
    expect(mock).toHaveBeenCalledTimes(15);
  });

  it('never runs more than four downloads at once', async () => {
    const assets = assetsForIndex(baseIndex());
    let active = 0;
    let maxActive = 0;
    const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active -= 1;
      const name = String(input).split('/').pop() ?? '';
      const body = assets.get(name);
      if (body === undefined) {
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      return new Response(body, { status: 200 });
    });
    vi.stubGlobal('fetch', mock);
    const { loadShardedSplits } = await loadLoader();

    await loadShardedSplits(SHARDED_DATASET, 100, 100);
    expect(maxActive).toBeGreaterThan(1);
    expect(maxActive).toBeLessThanOrEqual(4);
  });

  it('names the shard file and the dataset download command on a 404, and allows a retry', async () => {
    const assets = assetsForIndex(baseIndex());
    let failures = 1;
    const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const name = String(input).split('/').pop() ?? '';
      if (failures > 0 && name === 'train-c0-s00.idx.gz') {
        failures -= 1;
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      const body = assets.get(name);
      if (body === undefined) {
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      return new Response(body, { status: 200 });
    });
    vi.stubGlobal('fetch', mock);
    const { loadShardedSplits } = await loadLoader();

    const error = await loadShardedSplits(SHARDED_DATASET, 100, 100).then(
      () => {
        throw new Error('expected the load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain('train-c0-s00.idx.gz');
    expect(error.message).toContain('HTTP 404');
    expect(error.message).toContain(SHARDED_DATASET.downloadCommand);

    const splits = await loadShardedSplits(SHARDED_DATASET, 100, 100);
    expect(splits.train.count).toBe(12);
  });

  it('rejects a labels file whose count differs from the split count', async () => {
    const assets = assetsForIndex(baseIndex());
    assets.set('train-labels.idx.gz', gz(labelIdx(TRAIN_LABELS.slice(0, 11))));
    stubFetch(assets);
    const { loadShardedSplits } = await loadLoader();

    const error = await loadShardedSplits(SHARDED_DATASET, 100, 100).then(
      () => {
        throw new Error('expected the load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain('train-labels.idx.gz');
    expect(error.message).toMatch(/11 labels, expected 12/);
  });

  it('does not share the cache across dataset ids', async () => {
    const assets = assetsForIndex(baseIndex());
    const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = String(input);
      const name = url.split('/').pop() ?? '';
      if (name === 'index.json') {
        const datasetId = url.includes('/datasets/quickdraw-a/') ? 'quickdraw-a' : 'quickdraw-b';
        return new Response(
          new TextEncoder().encode(JSON.stringify({ ...baseIndex(), dataset: datasetId })),
          { status: 200 },
        );
      }
      const body = assets.get(name);
      if (body === undefined) {
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      return new Response(body, { status: 200 });
    });
    vi.stubGlobal('fetch', mock);
    const { loadShardedSplits } = await loadLoader();
    const first = { ...SHARDED_DATASET, id: 'quickdraw-a' };
    const second = { ...SHARDED_DATASET, id: 'quickdraw-b' };

    await loadShardedSplits(first, 100, 100);
    await loadShardedSplits(second, 100, 100);

    const shardCalls = mock.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.endsWith('train-c0-s00.idx.gz'));
    expect(shardCalls).toHaveLength(2);
    expect(shardCalls.some((url) => url.includes('/datasets/quickdraw-a/'))).toBe(true);
    expect(shardCalls.some((url) => url.includes('/datasets/quickdraw-b/'))).toBe(true);
  });
});

describe('loadShardedGallery', () => {
  it('returns the test gallery with perClass times classCount normalised samples', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedGallery } = await loadLoader();

    const gallery = await loadShardedGallery(SHARDED_DATASET);
    expect(gallery.count).toBe(6);
    expect(gallery.images).toBeInstanceOf(Float32Array);
    expect(gallery.images).toHaveLength(6 * 28 * 28);
    expect(Array.from(gallery.labels)).toEqual([0, 0, 0, 1, 1, 1]);
    // Le jeu de test n'a pas le décalage de 100 réservé à l'entraînement.
    expect(gallery.images[0]).toBe(0);
    for (const value of gallery.images) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    // Aucun fragment n'est nécessaire pour la galerie.
    expect(fetchedShardFiles(mock)).toEqual([]);
  });

  it('returns the train gallery when the train split is asked', async () => {
    stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedGallery } = await loadLoader();

    const gallery = await loadShardedGallery(SHARDED_DATASET, 'train');
    expect(gallery.count).toBe(6);
    // Libellés et pixels différents de ceux du jeu de test : la fraction
    // demandée est bien celle qui est lue.
    expect(Array.from(gallery.labels)).toEqual([1, 1, 1, 0, 0, 0]);
    expect(gallery.images[0]).toBeCloseTo(100 / 255);
  });

  it('fetches the test gallery file names from the index', async () => {
    const index = baseIndex();
    const galleries = index.galleries as {
      test: { imagesFile: string; labelsFile: string };
    };
    galleries.test.imagesFile = 'index-gallery-images.idx.gz';
    galleries.test.labelsFile = 'index-gallery-labels-idx1-ubyte.gz';
    const mock = stubFetch(assetsForIndex(index));
    const { loadShardedGallery } = await loadLoader();

    const gallery = await loadShardedGallery(SHARDED_DATASET);
    expect(gallery.count).toBe(6);
    const urls = mock.mock.calls.map((call) => String(call[0]));
    expect(urls.some((url) => url.endsWith('index-gallery-images.idx.gz'))).toBe(true);
    expect(urls.some((url) => url.endsWith('index-gallery-labels-idx1-ubyte.gz'))).toBe(true);
  });

  it('rejects a gallery whose image count differs from perClass times classCount', async () => {
    const index = baseIndex();
    const galleries = index.galleries as { test: { perClass: number } };
    galleries.test.perClass = 4;
    stubFetch(assetsForIndex(index));
    const { loadShardedGallery } = await loadLoader();

    const error = await loadShardedGallery(SHARDED_DATASET).then(
      () => {
        throw new Error('expected the gallery load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toMatch(/expected 8/);
  });

  it('reports a missing train gallery and still loads the test gallery', async () => {
    const assets = assetsForIndex(baseIndex());
    assets.delete('train-gallery.idx.gz');
    stubFetch(assets);
    const { loadShardedGallery } = await loadLoader();

    const error = await loadShardedGallery(SHARDED_DATASET, 'train').then(
      () => {
        throw new Error('expected the train gallery load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain('train-gallery.idx.gz');
    expect(error.message).toContain(SHARDED_DATASET.downloadCommand);

    const gallery = await loadShardedGallery(SHARDED_DATASET, 'test');
    expect(gallery.count).toBe(6);
  });

  it('caches a second call with the same dataset, split and perClass', async () => {
    const mock = stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedGallery } = await loadLoader();

    await loadShardedGallery(SHARDED_DATASET, 'test');
    // Index + deux fichiers de galerie.
    expect(mock).toHaveBeenCalledTimes(3);
    await loadShardedGallery(SHARDED_DATASET, 'test');
    expect(mock).toHaveBeenCalledTimes(3);
  });
});

/** Index synthétique : `classCount` classes, `shardsPerClass` fragments pleins. */
interface SegmentFixture {
  readonly json: Record<string, unknown>;
  readonly labels: number[];
  readonly assets: Map<string, Uint8Array<ArrayBuffer>>;
}

function segmentFixture(classCount: number, shardsPerClass: number, perShard = 2): SegmentFixture {
  const classNames = Array.from({ length: classCount }, (_unused, index) => `classe-${index}`);
  const labels: number[] = [];
  const trainShards: ShardEntry[] = [];
  const testShards: ShardEntry[] = [];
  let start = 0;
  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    for (let shardIndex = 0; shardIndex < shardsPerClass; shardIndex += 1) {
      const suffix = `c${classIndex}-s${String(shardIndex).padStart(2, '0')}.idx.gz`;
      trainShards.push({ file: `train-${suffix}`, classIndex, start, count: perShard });
      testShards.push({ file: `test-${suffix}`, classIndex, start, count: perShard });
      for (let index = 0; index < perShard; index += 1) {
        labels.push(classIndex);
      }
      start += perShard;
    }
  }
  const json = {
    version: 2,
    dataset: 'quickdraw',
    classNames,
    shardSize: perShard,
    splits: {
      train: { count: start, labelsFile: 'train-labels.idx.gz', shards: trainShards },
      test: { count: start, labelsFile: 'test-labels.idx.gz', shards: testShards },
    },
    galleries: {
      train: {
        imagesFile: 'train-gallery.idx.gz',
        labelsFile: 'train-gallery-labels.idx.gz',
        perClass: 1,
      },
      test: { imagesFile: 'gallery.idx.gz', labelsFile: 'gallery-labels.idx.gz', perClass: 1 },
    },
  };
  const assets = new Map<string, Uint8Array<ArrayBuffer>>();
  assets.set('train-labels.idx.gz', gz(labelIdx(labels)));
  assets.set('test-labels.idx.gz', gz(labelIdx(labels)));
  let fill = 1;
  for (const split of [trainShards, testShards]) {
    for (const shard of split) {
      assets.set(shard.file, gz(imageIdx(shard.count, fill % 200)));
      fill += 7;
    }
  }
  return { json, labels, assets };
}

/**
 * Variante dont la fraction de test a `testShardsPerClass` fragments et un
 * fichier de libellés propre, pour prouver que le magasin lit la fraction
 * demandée et pas toujours l'entraînement.
 */
function differingSplitFixture(testShardsPerClass: number): SegmentFixture {
  const base = segmentFixture(15, 4);
  const testLabels: number[] = [];
  const testShards: ShardEntry[] = [];
  let start = 0;
  for (let classIndex = 0; classIndex < 15; classIndex += 1) {
    for (let shardIndex = 0; shardIndex < testShardsPerClass; shardIndex += 1) {
      const suffix = `c${classIndex}-s${String(shardIndex).padStart(2, '0')}.idx.gz`;
      testShards.push({ file: `test-${suffix}`, classIndex, start, count: 2 });
      testLabels.push(classIndex, classIndex);
      start += 2;
    }
  }
  const splits = base.json.splits as {
    train: ShardedSplitIndex;
    test: ShardedSplitIndex;
  };
  const json = {
    ...base.json,
    splits: {
      train: splits.train,
      test: { count: start, labelsFile: 'test-labels.idx.gz', shards: testShards },
    },
  };
  const assets = new Map(base.assets);
  assets.set('test-labels.idx.gz', gz(labelIdx(testLabels)));
  return { json, labels: base.labels, assets };
}

/** URL publique d'une ressource du jeu de test `quickdraw`. */
function segmentUrl(file: string): string {
  return `/datasets/quickdraw/${file}`;
}

/** Cache Cache API en mémoire, suffisant pour les tests du magasin. */
class TestCache {
  readonly entries = new Map<string, Response>();

  async match(url: string): Promise<Response | undefined> {
    const response = this.entries.get(url);
    return response === undefined ? undefined : response.clone();
  }

  async put(url: string, response: Response): Promise<void> {
    this.entries.set(url, response.clone());
  }
}

class TestCacheStorage {
  readonly caches = new Map<string, TestCache>();

  async open(name: string): Promise<TestCache> {
    let cache = this.caches.get(name);
    if (cache === undefined) {
      cache = new TestCache();
      this.caches.set(name, cache);
    }
    return cache;
  }
}

function stubCaches(): TestCacheStorage {
  const storage = new TestCacheStorage();
  vi.stubGlobal('caches', storage);
  return storage;
}

/** Un round : un fragment par classe sélectionnée, à l'ordinal donné. */
function segmentRound(
  index: ShardedIndex,
  classIndices: readonly number[],
  shardOrdinal: number,
  roundIndex = 0,
  split: 'train' | 'test' = 'train',
): RoundRef {
  const entries = classIndices.map((classIndex) => {
    const shards = index.splits[split].shards.filter((shard) => shard.classIndex === classIndex);
    const shard = shards[shardOrdinal % shards.length];
    if (shard === undefined) {
      throw new Error(`fixture: no shard for class ${classIndex}`);
    }
    return {
      file: shard.file,
      classIndex: shard.classIndex,
      start: shard.start,
      count: shard.count,
    };
  });
  return { index: roundIndex, entries };
}

const FIFTEEN_CLASSES = Array.from({ length: 15 }, (_unused, index) => index);

/** Importe le module et construit un magasin sur l'index synthétique. */
async function createSegmentStore(
  fixture: SegmentFixture,
  classIndices: readonly number[],
  maxShards?: number,
): Promise<{
  module: typeof import('./sharded-assets.js');
  index: ShardedIndex;
  store: import('./sharded-assets.js').SegmentStore;
}> {
  const module = await loadLoader();
  const index = module.parseIndexJson(JSON.stringify(fixture.json));
  const store = new module.SegmentStore({
    dataset: SHARDED_DATASET,
    index,
    classIndices,
    cacheName: 'test-cache',
    ...(maxShards === undefined ? {} : { maxShards }),
  });
  return { module, index, store };
}

/** Liste les fichiers de fragments demandés au faux `fetch`. */
function fetchedSegmentShards(mock: ReturnType<typeof vi.fn>): string[] {
  return mock.mock.calls
    .map((call) => String(call[0]).split('/').pop() ?? '')
    .filter((name) => /^train-c\d+-s\d+\.idx\.gz$/.test(name));
}

describe('SegmentStore', () => {
  it('serves a round from a pre-populated cache without fetching', async () => {
    const fixture = segmentFixture(15, 4);
    const storage = stubCaches();
    const cache = await storage.open('test-cache');
    const { index, store } = await createSegmentStore(fixture, [0, 1]);
    const ref = segmentRound(index, [0, 1], 0);
    for (const entry of ref.entries) {
      cache.entries.set(segmentUrl(entry.file), new Response(gz(imageIdx(entry.count))));
    }
    cache.entries.set(
      segmentUrl('train-labels.idx.gz'),
      new Response(gz(labelIdx(fixture.labels))),
    );
    const mock = stubFetch(new Map());

    const round = await store.loadRound(ref);

    expect(mock).not.toHaveBeenCalled();
    expect(round.count).toBe(4);
    expect(Array.from(round.labels)).toEqual([0, 0, 1, 1]);
  });

  it('fetches once, fills the cache, and a new store reads the cache', async () => {
    const fixture = segmentFixture(15, 4);
    const storage = stubCaches();
    const cache = await storage.open('test-cache');
    const mock = stubFetch(fixture.assets);
    const first = await createSegmentStore(fixture, [0, 1]);
    const ref = segmentRound(first.index, [0, 1], 0);

    await first.store.loadRound(ref);
    // Deux fragments et le fichier de libellés.
    expect(mock).toHaveBeenCalledTimes(3);
    for (const entry of ref.entries) {
      expect(cache.entries.has(segmentUrl(entry.file))).toBe(true);
    }
    expect(cache.entries.has(segmentUrl('train-labels.idx.gz'))).toBe(true);

    const second = await createSegmentStore(fixture, [0, 1]);
    const round = await second.store.loadRound(ref);
    expect(mock).toHaveBeenCalledTimes(3);
    expect(round.count).toBe(4);
  });

  it('loads when cache.put rejects with QuotaExceededError', async () => {
    const fixture = segmentFixture(15, 4);
    const storage = stubCaches();
    const cache = await storage.open('test-cache');
    cache.put = async () => {
      const error = new Error('quota exceeded');
      error.name = 'QuotaExceededError';
      throw error;
    };
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, [0, 1]);

    const round = await store.loadRound(segmentRound(index, [0, 1], 0));

    expect(round.count).toBe(4);
    expect(Array.from(round.labels)).toEqual([0, 0, 1, 1]);
  });

  it('loads when the Cache API is unavailable', async () => {
    const fixture = segmentFixture(15, 4);
    vi.stubGlobal('caches', undefined);
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, [0, 1]);

    const round = await store.loadRound(segmentRound(index, [0, 1], 0));

    expect(round.count).toBe(4);
  });

  it('never keeps more than 60 resident shards with 15 classes and 4 shards each', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, FIFTEEN_CLASSES);

    for (let ordinal = 0; ordinal < 4; ordinal += 1) {
      await store.loadRound(segmentRound(index, FIFTEEN_CLASSES, ordinal, ordinal));
      expect(store.residentShards).toBeLessThanOrEqual(60);
    }
    expect(store.residentShards).toBe(60);
  });

  it('evicts the least recently used shards above maxShards', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, FIFTEEN_CLASSES, 20);

    await store.loadRound(segmentRound(index, FIFTEEN_CLASSES, 0, 0));
    expect(store.residentShards).toBe(15);
    await store.loadRound(segmentRound(index, FIFTEEN_CLASSES, 1, 1));

    expect(store.residentShards).toBe(20);
  });

  it('evicts a consumed shard after releaseRound', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, FIFTEEN_CLASSES);
    const first = segmentRound(index, FIFTEEN_CLASSES, 0, 0);
    const second = segmentRound(index, FIFTEEN_CLASSES, 1, 1);
    const third = segmentRound(index, FIFTEEN_CLASSES, 2, 2);

    await store.loadRound(first);
    await store.loadRound(second);
    expect(store.residentShards).toBe(30);

    store.releaseRound(first);
    expect(store.residentShards).toBe(15);

    store.prefetchRound(third);
    await store.loadRound(third);
    // Sans éviction, le premier round resterait : 45 fragments.
    expect(store.residentShards).toBe(30);
    expect(store.residentShards).toBeLessThanOrEqual(60);
  });

  it('remaps the selected classes to 0..N-1 and slices labels per entry', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, [2, 5]);
    const ref = segmentRound(index, [2, 5], 0);

    const round = await store.loadRound(ref);

    expect(store.totalCount).toBe(16);
    expect(Array.from(round.labels)).toEqual([0, 0, 1, 1]);
    const batch = round.readBatch(1, 3);
    expect(batch.count).toBe(2);
    expect(Array.from(batch.labels)).toEqual([0, 1]);
    expect(batch.images).toBeInstanceOf(Float32Array);
    expect(batch.images).toHaveLength(2 * 28 * 28);
    for (const value of batch.images) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(round.readBatch(0, 4).count).toBe(4);
    expect(() => round.readBatch(3, 5)).toThrow(RangeError);
  });

  it('fetches no shard outside the selected classes', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    const mock = stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, [2, 5]);
    const ref = segmentRound(index, [2, 5], 1);

    await store.loadRound(ref);

    expect(fetchedSegmentShards(mock).sort()).toEqual(
      ref.entries.map((entry) => entry.file).sort(),
    );
  });

  it('accepts a labels file that is already remapped', async () => {
    const fixture = segmentFixture(15, 4);
    const module = await loadLoader();
    const index = module.parseIndexJson(JSON.stringify(fixture.json));
    const ref = segmentRound(index, [2, 5], 0);
    const remapped = [...fixture.labels];
    for (const entry of ref.entries) {
      const position = [2, 5].indexOf(entry.classIndex);
      for (let offset = 0; offset < entry.count; offset += 1) {
        remapped[entry.start + offset] = position;
      }
    }
    fixture.assets.set('train-labels.idx.gz', gz(labelIdx(remapped)));
    stubCaches();
    stubFetch(fixture.assets);
    const store = new module.SegmentStore({
      dataset: SHARDED_DATASET,
      index,
      classIndices: [2, 5],
      cacheName: 'test-cache',
    });

    const round = await store.loadRound(ref);

    expect(Array.from(round.labels)).toEqual([0, 0, 1, 1]);
  });

  it('rejects a shard whose payload is not count * 784 bytes', async () => {
    const fixture = segmentFixture(15, 4);
    const module = await loadLoader();
    const index = module.parseIndexJson(JSON.stringify(fixture.json));
    const ref = segmentRound(index, [0], 0);
    const entry = ref.entries[0];
    const truncated = new Uint8Array(16 + 28 * 28);
    truncated.set(imageIdx(entry.count).subarray(0, 16 + 28 * 28));
    fixture.assets.set(entry.file, gz(truncated));
    stubCaches();
    stubFetch(fixture.assets);
    const store = new module.SegmentStore({
      dataset: SHARDED_DATASET,
      index,
      classIndices: [0],
      cacheName: 'test-cache',
    });

    const error = await store.loadRound(ref).then(
      () => {
        throw new Error('expected the round to fail');
      },
      (reason: Error) => reason,
    );

    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain(entry.file);
    expect(error.message).toMatch(/784 image bytes, expected 1568/);
  });

  it('shares the in-flight prefetch promise with loadRound', async () => {
    const fixture = segmentFixture(15, 4);
    stubCaches();
    const mock = stubFetch(fixture.assets);
    const { index, store } = await createSegmentStore(fixture, [0, 1]);
    const ref = segmentRound(index, [0, 1], 0);

    store.prefetchRound(ref);
    const round = await store.loadRound(ref);

    expect(round.count).toBe(4);
    for (const entry of ref.entries) {
      const calls = mock.mock.calls.filter((call) => String(call[0]).endsWith(entry.file));
      expect(calls).toHaveLength(1);
    }
  });

  it('names the shard file on a 404 and allows a retry', async () => {
    const fixture = segmentFixture(15, 4);
    const module = await loadLoader();
    const index = module.parseIndexJson(JSON.stringify(fixture.json));
    const ref = segmentRound(index, [0, 1], 0);
    const failing = ref.entries[0].file;
    let failures = 1;
    const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const name = String(input).split('/').pop() ?? '';
      if (failures > 0 && name === failing) {
        failures -= 1;
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      const body = fixture.assets.get(name);
      if (body === undefined) {
        return new Response('missing', { status: 404, statusText: 'Not Found' });
      }
      return new Response(body, { status: 200 });
    });
    stubCaches();
    vi.stubGlobal('fetch', mock);
    const store = new module.SegmentStore({
      dataset: SHARDED_DATASET,
      index,
      classIndices: [0, 1],
      cacheName: 'test-cache',
    });

    const error = await store.loadRound(ref).then(
      () => {
        throw new Error('expected the round to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain(failing);
    expect(error.message).toContain('HTTP 404');
    expect(error.message).toContain(SHARDED_DATASET.downloadCommand);

    const round = await store.loadRound(ref);
    expect(round.count).toBe(4);
  });

  it('serves the test split and keeps train as the default', async () => {
    const fixture = differingSplitFixture(2);
    stubCaches();
    const mock = stubFetch(fixture.assets);
    const module = await loadLoader();
    const index = module.parseIndexJson(JSON.stringify(fixture.json));
    const options = {
      dataset: SHARDED_DATASET,
      index,
      classIndices: [2, 5],
      cacheName: 'test-cache',
    };

    const trainStore = new module.SegmentStore(options);
    expect(trainStore.totalCount).toBe(16);
    const trainRef = segmentRound(index, [2, 5], 0);
    const trainRound = await trainStore.loadRound(trainRef);
    expect(Array.from(trainRound.labels)).toEqual([0, 0, 1, 1]);
    const trainNames = mock.mock.calls.map((call) => String(call[0]).split('/').pop() ?? '');
    expect(trainNames.length).toBeGreaterThan(0);
    expect(trainNames.every((name) => name.startsWith('train-'))).toBe(true);

    const before = mock.mock.calls.length;
    const testStore = new module.SegmentStore({ ...options, split: 'test' });
    expect(testStore.totalCount).toBe(8);
    const testRef = segmentRound(index, [2, 5], 0, 0, 'test');
    const testRound = await testStore.loadRound(testRef);
    expect(Array.from(testRound.labels)).toEqual([0, 0, 1, 1]);

    const testNames = mock.mock.calls.slice(before).map((call) => String(call[0]).split('/').pop());
    expect(testNames).toContain(testRef.entries[0].file);
    expect(testNames).toContain('test-labels.idx.gz');
    expect(testNames.some((name) => name?.startsWith('train-'))).toBe(false);
  });
});
