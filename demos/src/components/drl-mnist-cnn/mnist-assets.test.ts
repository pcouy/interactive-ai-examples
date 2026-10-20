import { afterEach, describe, expect, it, vi } from 'vitest';
import type { MnistIdxDataset } from '../../core/mnist-data.js';

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

async function gzip(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array<ArrayBuffer>> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const TRAIN_LABELS = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
const TEST_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/** Jeu de test de la galerie : 20 exemples par chiffre, dans un ordre entrelacé. */
const GALLERY_TEST_LABELS: number[] = [];
for (let round = 0; round < 20; round += 1) {
  for (let digit = 0; digit < 10; digit += 1) {
    GALLERY_TEST_LABELS.push((digit + round) % 10);
  }
}

async function gzippedAssets(): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
  return new Map([
    ['train-images-idx3-ubyte.gz', await gzip(imageIdx(TRAIN_LABELS.length))],
    ['train-labels-idx1-ubyte.gz', await gzip(labelIdx(TRAIN_LABELS))],
    ['t10k-images-idx3-ubyte.gz', await gzip(imageIdx(TEST_LABELS.length))],
    ['t10k-labels-idx1-ubyte.gz', await gzip(labelIdx(TEST_LABELS))],
  ]);
}

/**
 * Ressources de la galerie : 20 exemples par chiffre dans le jeu de test, contre
 * seulement deux dans le jeu d'entraînement — la galerie lit le jeu de test.
 */
async function gzippedGalleryAssets(): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
  return new Map([
    ['train-images-idx3-ubyte.gz', await gzip(imageIdx(TRAIN_LABELS.length))],
    ['train-labels-idx1-ubyte.gz', await gzip(labelIdx(TRAIN_LABELS))],
    ['t10k-images-idx3-ubyte.gz', await gzip(imageIdx(GALLERY_TEST_LABELS.length))],
    ['t10k-labels-idx1-ubyte.gz', await gzip(labelIdx(GALLERY_TEST_LABELS))],
  ]);
}

/**
 * Oracle indépendant de `selectGalleryIndices` : les `perClass` premiers indices
 * de chaque classe, classe par classe, dans l'ordre naturel du jeu.
 */
function firstOccurrencesPerClass(labels: number[], perClass: number): number[] {
  const expected: number[] = [];
  for (let label = 0; label < 10; label += 1) {
    const indices = labels
      .map((value, index) => (value === label ? index : -1))
      .filter((index) => index >= 0);
    expected.push(...indices.slice(0, perClass));
  }
  return expected;
}

/** Jeu IDX minimal construit à partir de libellés, sans image. */
function labelsDataset(labels: number[]): MnistIdxDataset {
  return {
    count: labels.length,
    images: new Uint8Array(labels.length * 28 * 28),
    labels: Uint8Array.from(labels),
  };
}

function stubFetch(
  assets: Map<string, Uint8Array<ArrayBuffer>>,
  headers?: Record<string, string>,
): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const name = String(input).split('/').pop() ?? '';
    const body = assets.get(name);
    if (body === undefined) {
      return new Response('missing', { status: 404, statusText: 'Not Found' });
    }
    return new Response(body, { status: 200, headers });
  });
  vi.stubGlobal('fetch', mock);
  return mock;
}

async function loadLoader(): Promise<typeof import('./mnist-assets.js')> {
  vi.resetModules();
  return import('./mnist-assets.js');
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('decompressGzip', () => {
  it('round-trips a gzip payload', async () => {
    const { decompressGzip } = await loadLoader();
    const payload = new TextEncoder().encode('mnist payload');
    const compressed = await gzip(payload);
    const result = await decompressGzip(
      compressed.buffer.slice(compressed.byteOffset, compressed.byteOffset + compressed.byteLength),
    );
    expect(new TextDecoder().decode(result)).toBe('mnist payload');
  });

  it('reports a missing DecompressionStream clearly', async () => {
    const original = globalThis.DecompressionStream;
    // @ts-expect-error simulation d'un navigateur sans DecompressionStream
    delete globalThis.DecompressionStream;
    try {
      const { decompressGzip } = await loadLoader();
      await expect(decompressGzip(new ArrayBuffer(0))).rejects.toThrow(/DecompressionStream/);
    } finally {
      globalThis.DecompressionStream = original;
    }
  });
});

describe('isGzipPayload', () => {
  it('returns true only for the 0x1f 0x8b prefix', async () => {
    const { isGzipPayload } = await loadLoader();
    expect(isGzipPayload(new Uint8Array([0x1f, 0x8b, 0x08, 0x00]))).toBe(true);
    expect(isGzipPayload(new Uint8Array([0x00, 0x00, 0x08, 0x03]))).toBe(false);
    expect(isGzipPayload(new Uint8Array([0x1f]))).toBe(false);
    expect(isGzipPayload(new Uint8Array([]))).toBe(false);
  });
});

describe('decodeIdxPayload', () => {
  it('decompresses a gzip payload with no Content-Encoding', async () => {
    const { decodeIdxPayload } = await loadLoader();
    const payload = new TextEncoder().encode('mnist payload');
    const compressed = await gzip(payload);
    const result = await decodeIdxPayload(compressed, null);
    expect(new TextDecoder().decode(result)).toBe('mnist payload');
  });

  it('returns an already-decompressed payload unchanged with Content-Encoding: gzip', async () => {
    const { decodeIdxPayload } = await loadLoader();
    const payload = new Uint8Array(new TextEncoder().encode('mnist payload'));
    const result = await decodeIdxPayload(payload, 'gzip');
    expect(new TextDecoder().decode(result)).toBe('mnist payload');
  });

  it('returns an already-decompressed payload unchanged with no Content-Encoding and no gzip magic', async () => {
    const { decodeIdxPayload } = await loadLoader();
    const payload = new Uint8Array(new TextEncoder().encode('mnist payload'));
    const result = await decodeIdxPayload(payload, null);
    expect(new TextDecoder().decode(result)).toBe('mnist payload');
  });
});

describe('loadMnistSplits', () => {
  it('fetches the four local files once and returns both IDX splits', async () => {
    const assets = await gzippedAssets();
    const fetchMock = stubFetch(assets);
    const { loadMnistSplits } = await loadLoader();

    const [first, second] = await Promise.all([loadMnistSplits(), loadMnistSplits()]);
    expect(first).toBe(second);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(first.train.count).toBe(TRAIN_LABELS.length);
    expect(first.train.images).toBeInstanceOf(Uint8Array);
    expect(first.train.images).toHaveLength(TRAIN_LABELS.length * 28 * 28);
    expect(first.test.count).toBe(TEST_LABELS.length);
    expect(Array.from(first.test.labels)).toEqual(TEST_LABELS);
    for (const call of fetchMock.mock.calls) {
      expect(String(call[0])).toContain('/mnist/');
    }
  });

  it('parses an already-decoded IDX body served with Content-Encoding: gzip', async () => {
    const assets = new Map([
      ['train-images-idx3-ubyte.gz', imageIdx(TRAIN_LABELS.length)],
      ['train-labels-idx1-ubyte.gz', labelIdx(TRAIN_LABELS)],
      ['t10k-images-idx3-ubyte.gz', imageIdx(TEST_LABELS.length)],
      ['t10k-labels-idx1-ubyte.gz', labelIdx(TEST_LABELS)],
    ]);
    stubFetch(assets, { 'content-encoding': 'gzip' });
    const { loadMnistSplits } = await loadLoader();

    const splits = await loadMnistSplits();
    expect(splits.train.count).toBe(TRAIN_LABELS.length);
    expect(splits.test.count).toBe(TEST_LABELS.length);
    expect(Array.from(splits.test.labels)).toEqual(TEST_LABELS);
  });

  it('reports a fetch failure clearly and allows a later retry', async () => {
    const assets = await gzippedAssets();
    let failures = 1;
    const mock = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      if (failures > 0) {
        failures -= 1;
        throw new Error('offline');
      }
      const name = String(input).split('/').pop() ?? '';
      return new Response(assets.get(name) ?? 'missing', { status: 200 });
    });
    vi.stubGlobal('fetch', mock);
    const { loadMnistSplits } = await loadLoader();

    const error = await loadMnistSplits().then(
      () => {
        throw new Error('expected the first load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toMatch(/npm run mnist:data/);
    expect(error.message).toMatch(/train-images-idx3-ubyte\.gz/);
    expect(error.message).toMatch(/offline/);

    const splits = await loadMnistSplits();
    expect(splits.train.count).toBe(TRAIN_LABELS.length);
  });

  it('reports an HTTP error status clearly', async () => {
    stubFetch(new Map());
    const { loadMnistSplits } = await loadLoader();
    await expect(loadMnistSplits()).rejects.toThrow(/HTTP 404/);
    await expect(loadMnistSplits()).rejects.toThrow(/npm run mnist:data/);
  });

  it('raises an actionable error for an HTML error page', async () => {
    const html = new Uint8Array(
      new TextEncoder().encode('<html><body>404 Not Found</body></html>'),
    );
    const mock = vi.fn(
      async (_input: RequestInfo | URL): Promise<Response> => new Response(html, { status: 200 }),
    );
    vi.stubGlobal('fetch', mock);
    const { loadMnistSplits } = await loadLoader();
    const error = await loadMnistSplits().then(
      () => {
        throw new Error('expected the load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain('npm run mnist:data');
    expect(error.message).toMatch(/train-images-idx3-ubyte\.gz|t10k-labels-idx1-ubyte\.gz/);
  });
});

describe('mnistFetchErrorMessage', () => {
  it('names the missing file and points at the download command', async () => {
    const { mnistFetchErrorMessage, mnistFileName } = await loadLoader();
    const message = mnistFetchErrorMessage(
      '/mnist/train-images-idx3-ubyte.gz',
      'network error: offline',
    );
    expect(message).toContain('train-images-idx3-ubyte.gz');
    expect(message).toContain('npm run mnist:data');
    expect(message).toContain('offline');
    expect(mnistFileName('https://example.test/a/b/t10k-labels-idx1-ubyte.gz?v=1')).toBe(
      't10k-labels-idx1-ubyte.gz',
    );
    expect(mnistFileName('t10k-images-idx3-ubyte.gz')).toBe('t10k-images-idx3-ubyte.gz');
  });
});

describe('selectGalleryIndices', () => {
  const dataset = labelsDataset(GALLERY_TEST_LABELS);

  /** Charge le module et retourne `selectGalleryIndices`. */
  async function loader(): Promise<typeof import('./mnist-assets.js')> {
    return loadLoader();
  }

  it('returns the first ten examples of each class, class by class', async () => {
    const { selectGalleryIndices } = await loader();
    const indices = selectGalleryIndices(dataset);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(Array.from(indices)).toHaveLength(100);
    // Ordre classe par classe, ordre naturel du jeu conservé dans chaque classe.
    expect(Array.from(indices)).toEqual(firstOccurrencesPerClass(GALLERY_TEST_LABELS, 10));
    // Les dix premiers indices appartiennent tous à la classe 0.
    expect(Array.from(indices.slice(0, 10))).toEqual([0, 19, 28, 37, 46, 55, 64, 73, 82, 91]);
  });

  it('honours a smaller perClass', async () => {
    const { selectGalleryIndices } = await loader();
    const indices = selectGalleryIndices(dataset, 3);
    expect(indices).toBeInstanceOf(Uint32Array);
    expect(Array.from(indices)).toHaveLength(30);
    expect(Array.from(indices)).toEqual(firstOccurrencesPerClass(GALLERY_TEST_LABELS, 3));
    expect(Array.from(indices.slice(0, 3))).toEqual([0, 19, 28]);
  });

  it('rejects a perClass that is not a positive integer', async () => {
    const { selectGalleryIndices } = await loader();
    for (const perClass of [0, -3, 1.5, Number.NaN]) {
      expect(() => selectGalleryIndices(dataset, perClass)).toThrow(RangeError);
    }
  });

  it('names the class that lacks examples and the count it has', async () => {
    const { selectGalleryIndices } = await loader();
    const sparse = labelsDataset([
      0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9,
    ]);
    expect(() => selectGalleryIndices(sparse, 3)).toThrow(RangeError);
    expect(() => selectGalleryIndices(sparse, 3)).toThrow(/class 9/);
    expect(() => selectGalleryIndices(sparse, 3)).toThrow(/2 example/);
  });
});

describe('loadMnistGallery', () => {
  it('returns ten examples per class from the test split', async () => {
    stubFetch(await gzippedGalleryAssets());
    const { loadMnistGallery } = await loadLoader();
    const gallery = await loadMnistGallery();
    expect(gallery.count).toBe(100);
    expect(gallery.images).toBeInstanceOf(Float32Array);
    expect(gallery.images).toHaveLength(100 * 28 * 28);
    // Ordre classe par classe : dix 0, dix 1, …, dix 9.
    const expectedLabels: number[] = [];
    for (let label = 0; label < 10; label += 1) {
      for (let index = 0; index < 10; index += 1) {
        expectedLabels.push(label);
      }
    }
    expect(Array.from(gallery.labels)).toEqual(expectedLabels);
    // Les deux premiers exemples de la classe 0 sont deux images distinctes :
    // le deuxième ne recopie pas le premier.
    expect(gallery.images[784]).toBeGreaterThan(gallery.images[0]);
  });

  it('returns twenty examples per class when asked', async () => {
    stubFetch(await gzippedGalleryAssets());
    const { loadMnistGallery } = await loadLoader();
    const gallery = await loadMnistGallery(20);
    expect(gallery.count).toBe(200);
    expect(gallery.images).toHaveLength(200 * 28 * 28);
    expect(Array.from(gallery.labels)).toHaveLength(200);
    for (let label = 0; label < 10; label += 1) {
      expect(gallery.labels.filter((value) => value === label)).toHaveLength(20);
    }
  });

  it('rejects a perClass above the test split size and names a class', async () => {
    stubFetch(await gzippedGalleryAssets());
    const { loadMnistGallery } = await loadLoader();
    // Le jeu d'entraînement ne compte que deux exemples par chiffre : un échec
    // ici prouve que la galerie lit bien le jeu de test.
    const error = await loadMnistGallery(21).then(
      () => {
        throw new Error('expected the gallery load to fail');
      },
      (reason: Error) => reason,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toMatch(/class/);
  });
});

/** Manifeste minimal valide : filtres [3, 3, 3, 3], 980 paramètres. */
function minimalManifestJson(): string {
  const layer = (
    kernelLength: number,
    biasLength: number,
  ): { kernel: number[]; bias: number[] } => ({
    kernel: Array.from({ length: kernelLength }, (_unused, index) => index * 0.1),
    bias: new Array<number>(biasLength).fill(0),
  });
  return JSON.stringify({
    schemaVersion: 1,
    architecture: { filters: [3, 3, 3, 3] },
    parameterCount: 980,
    layers: {
      conv1: layer(27, 3),
      conv2: layer(81, 3),
      conv3: layer(81, 3),
      conv4: layer(81, 3),
      dense1: layer(600, 8),
      dense2: layer(80, 10),
    },
  });
}

describe('fetchMnistCnnManifest', () => {
  it('downloads the pretrained manifest under mnist/ and its text parses', async () => {
    const manifest = minimalManifestJson();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL): Promise<Response> =>
        new Response(manifest, { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const { fetchMnistCnnManifest } = await loadLoader();
    const { parseMnistCnnManifestJson } = await import('../../core/mnist-cnn-assets.js');

    const text = await fetchMnistCnnManifest();
    expect(text).toBe(manifest);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/mnist/model-v1.json');
    expect(parseMnistCnnManifestJson(text).config.filters).toEqual([3, 3, 3, 3]);
  });

  it('reports a missing manifest clearly', async () => {
    const fetchMock = vi.fn(
      async (): Promise<Response> =>
        new Response('missing', { status: 404, statusText: 'Not Found' }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const { fetchMnistCnnManifest } = await loadLoader();
    const error = await fetchMnistCnnManifest().then(
      () => {
        throw new Error('expected the fetch to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain('model-v1.json');
    expect(error.message).toContain('npm run mnist:data');
    expect(error.message).toContain('HTTP 404');
  });

  it('reports a network failure clearly', async () => {
    const fetchMock = vi.fn(async (): Promise<Response> => {
      throw new Error('offline');
    });
    vi.stubGlobal('fetch', fetchMock);
    const { fetchMnistCnnManifest } = await loadLoader();
    const error = await fetchMnistCnnManifest().then(
      () => {
        throw new Error('expected the fetch to fail');
      },
      (reason: Error) => reason,
    );
    expect(error.message).toContain('model-v1.json');
    expect(error.message).toContain('offline');
  });
});
