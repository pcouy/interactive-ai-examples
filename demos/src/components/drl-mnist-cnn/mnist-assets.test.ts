import { afterEach, describe, expect, it, vi } from 'vitest';

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

async function gzippedAssets(): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
  return new Map([
    ['train-images-idx3-ubyte.gz', await gzip(imageIdx(TRAIN_LABELS.length))],
    ['train-labels-idx1-ubyte.gz', await gzip(labelIdx(TRAIN_LABELS))],
    ['t10k-images-idx3-ubyte.gz', await gzip(imageIdx(TEST_LABELS.length))],
    ['t10k-labels-idx1-ubyte.gz', await gzip(labelIdx(TEST_LABELS))],
  ]);
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

describe('loadMnistGallery', () => {
  it('returns ten examples, one per class, from the test split', async () => {
    stubFetch(await gzippedAssets());
    const { loadMnistGallery } = await loadLoader();
    const gallery = await loadMnistGallery();
    expect(gallery.count).toBe(10);
    expect(gallery.images).toHaveLength(10 * 28 * 28);
    expect(gallery.images).toBeInstanceOf(Float32Array);
    expect([...gallery.labels].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

/** Manifeste minimal valide : filtres [1, 1, 1, 1], 338 paramètres. */
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
    architecture: { filters: [1, 1, 1, 1] },
    parameterCount: 338,
    layers: {
      conv1: layer(9, 1),
      conv2: layer(9, 1),
      conv3: layer(9, 1),
      conv4: layer(9, 1),
      dense1: layer(200, 8),
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
    expect(parseMnistCnnManifestJson(text).config.filters).toEqual([1, 1, 1, 1]);
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
