import { gzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ShardedDatasetDefinition } from '../../core/dataset-registry.js';
import type { ShardedSplitIndex } from './sharded-assets.js';

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;

function u32Bytes(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function imageIdx(count: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(16 + count * 28 * 28));
  bytes.set([...u32Bytes(IMAGE_MAGIC), ...u32Bytes(count), ...u32Bytes(28), ...u32Bytes(28)]);
  for (let index = 0; index < count * 28 * 28; index += 1) {
    bytes[16 + index] = (index * 7) % 256;
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
    version: 1,
    dataset: 'quickdraw',
    classNames: ['livre', 'lune'],
    shardSize: 2,
    splits: {
      train: { count: 12, labelsFile: 'train-labels.idx.gz', shards: shards('train') },
      test: { count: 12, labelsFile: 'test-labels.idx.gz', shards: shards('test') },
    },
    gallery: { imagesFile: 'gallery.idx.gz', labelsFile: 'gallery-labels.idx.gz', perClass: 3 },
  };
}

/** Ressources complètes pour l'index donné : index, libellés, fragments et galerie. */
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
  const gallery = index.gallery as {
    imagesFile: string;
    labelsFile: string;
    perClass: number;
  };
  assets.set(gallery.imagesFile, gz(imageIdx(6)));
  assets.set(gallery.labelsFile, gz(labelIdx([0, 0, 0, 1, 1, 1])));
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
  it('accepts a valid index', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = parseIndexJson(JSON.stringify(baseIndex()));
    expect(index.version).toBe(1);
    expect(index.classNames).toEqual(['livre', 'lune']);
    expect(index.shardSize).toBe(2);
    expect(index.splits.train.count).toBe(12);
    expect(index.splits.train.shards).toHaveLength(6);
    expect(index.splits.test.shards).toHaveLength(6);
  });

  it('rejects a missing splits field', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    delete index.splits;
    expect(() => parseIndexJson(JSON.stringify(index))).toThrow(/splits/);
  });

  it('rejects an unsupported version', async () => {
    const { parseIndexJson } = await loadLoader();
    const index = baseIndex();
    index.version = 2;
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
    const mock = stubFetch(assetsForIndex(baseIndex()));
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
  it('returns perClass times classCount normalised samples with the right labels', async () => {
    stubFetch(assetsForIndex(baseIndex()));
    const { loadShardedGallery } = await loadLoader();

    const gallery = await loadShardedGallery(SHARDED_DATASET);
    expect(gallery.count).toBe(6);
    expect(gallery.images).toBeInstanceOf(Float32Array);
    expect(gallery.images).toHaveLength(6 * 28 * 28);
    expect(Array.from(gallery.labels)).toEqual([0, 0, 0, 1, 1, 1]);
    for (const value of gallery.images) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('fetches the gallery file names from the index', async () => {
    const index = baseIndex();
    const galleryIndex = index.gallery as { imagesFile: string; labelsFile: string };
    galleryIndex.imagesFile = 'index-gallery-images.idx.gz';
    galleryIndex.labelsFile = 'index-gallery-labels-idx1-ubyte.gz';
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
    const galleryIndex = index.gallery as { perClass: number };
    galleryIndex.perClass = 4;
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
});
