import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { DEFAULT_FILTER_COUNTS, forwardMnistCnn, MNIST_INPUT_LENGTH } from './mnist-cnn.js';
import {
  MNIST_CNN_MANIFEST_SCHEMA_VERSION,
  parseMnistCnnManifest,
  parseMnistCnnManifestJson,
} from './mnist-cnn-assets.js';

const MANIFEST_URL = new URL('../../public/mnist/model-v1.json', import.meta.url);
const MANIFEST_TEXT = readFileSync(MANIFEST_URL, 'utf8');

/** Copie modifiable du manifeste exporté, pour les cas d'erreur. */
function manifestClone(): Record<string, unknown> {
  return JSON.parse(MANIFEST_TEXT) as Record<string, unknown>;
}

function layersOf(manifest: Record<string, unknown>): Record<string, Record<string, unknown>> {
  return manifest.layers as Record<string, Record<string, unknown>>;
}

/**
 * Entrées synthétiques déterministes du test de parité, identiques à celles
 * générées par `demos/scripts/export_mnist_cnn_checkpoint.py` : valeur entière
 * exacte / 250, reproductible à l'identique en float32.
 */
function parityInput(image: number): Float32Array {
  const input = new Float32Array(MNIST_INPUT_LENGTH);
  for (let pixel = 0; pixel < MNIST_INPUT_LENGTH; pixel += 1) {
    input[pixel] = ((image * 131 + pixel * 17) % 251) / 250;
  }
  return input;
}

/**
 * Logits de référence PyTorch (float32, CPU) pour `parityInput(0..2)`.
 * Régénérer avec `python export_mnist_cnn_checkpoint.py --parity`.
 */
const PARITY_LOGITS: readonly (readonly number[])[] = [
  [
    4.048971176147461, -6.818856239318848, 1.1084926128387451, -0.6468912363052368,
    -0.424003005027771, -0.31610560417175293, 0.5629981756210327, -5.104931354522705,
    2.9666359424591064, -1.4135956764221191,
  ],
  [
    4.135952472686768, -6.8826141357421875, 1.032541036605835, -0.7455522418022156,
    -0.2619009017944336, -0.27987051010131836, 0.7097632884979248, -5.204697608947754,
    2.9696993827819824, -1.3198918104171753,
  ],
  [
    3.97013521194458, -6.780073642730713, 1.1296665668487549, -0.6580999493598938,
    -0.4311627149581909, -0.3450663089752197, 0.5537043809890747, -5.157440185546875,
    3.021282434463501, -1.4194194078445435,
  ],
];

describe('parseMnistCnnManifest', () => {
  it('parses the exported manifest into config and weights', () => {
    const { config, weights } = parseMnistCnnManifestJson(MANIFEST_TEXT);
    expect(config.filters).toEqual(DEFAULT_FILTER_COUNTS);
    expect(config.convLayers).toBe(4);
    expect(config.denseLayers).toBe(2);
    expect(config.activation).toBe('sigmoid');
    expect(weights.conv).toHaveLength(4);
    expect(weights.dense).toHaveLength(2);
    expect(weights.conv[0].kernel).toBeInstanceOf(Float32Array);
    expect(weights.conv[0].kernel).toHaveLength(27);
    expect(weights.conv[1].kernel).toHaveLength(81);
    expect(weights.conv[2].kernel).toHaveLength(81);
    expect(weights.conv[3].kernel).toHaveLength(81);
    expect(weights.dense[0].kernel).toHaveLength(600);
    expect(weights.dense[0].bias).toHaveLength(8);
    expect(weights.dense[1].kernel).toHaveLength(80);
    expect(weights.dense[1].bias).toHaveLength(10);
  });

  it('rejects a wrong schema version', () => {
    const manifest = manifestClone();
    manifest.schemaVersion = MNIST_CNN_MANIFEST_SCHEMA_VERSION + 1;
    expect(() => parseMnistCnnManifest(manifest)).toThrow(/unsupported schema version 2/);
    manifest.schemaVersion = 'one';
    expect(() => parseMnistCnnManifest(manifest)).toThrow(/unsupported schema version one/);
  });

  it('rejects unknown root-level keys', () => {
    const withExtraKey = manifestClone();
    withExtraKey.exportedAt = '2026-01-01';
    expect(() => parseMnistCnnManifest(withExtraKey)).toThrow(/unsupported manifest keys/);
  });

  it('rejects unsupported architecture metadata', () => {
    const withExtraKey = manifestClone();
    (withExtraKey.architecture as Record<string, unknown>).kernelSize = 3;
    expect(() => parseMnistCnnManifest(withExtraKey)).toThrow(/unsupported architecture metadata/);

    const outOfRange = manifestClone();
    (outOfRange.architecture as Record<string, unknown>).filters = [5, 3, 3, 3];
    expect(() => parseMnistCnnManifest(outOfRange)).toThrow(/unsupported architecture filters/);

    const wrongShape = manifestClone();
    (wrongShape.architecture as Record<string, unknown>).filters = [3, 3, 3];
    expect(() => parseMnistCnnManifest(wrongShape)).toThrow(/unsupported architecture filters/);
  });

  it('rejects a filter tuple that is not the reference architecture', () => {
    const wrongFilters = manifestClone();
    (wrongFilters.architecture as Record<string, unknown>).filters = [1, 2, 3, 4];
    expect(() => parseMnistCnnManifest(wrongFilters)).toThrow(/unsupported architecture filters/);
  });

  it('rejects an extra dense layer', () => {
    const extra = manifestClone();
    layersOf(extra).dense3 = { kernel: [], bias: [] };
    expect(() => parseMnistCnnManifest(extra)).toThrow(/unsupported layers/);
  });

  it('rejects a missing conv layer', () => {
    const missing = manifestClone();
    delete layersOf(missing).conv4;
    expect(() => parseMnistCnnManifest(missing)).toThrow(/layers\.conv4 must be a JSON object/);
  });

  it('rejects a parameter count that disagrees with the architecture', () => {
    const manifest = manifestClone();
    manifest.parameterCount = 979;
    expect(() => parseMnistCnnManifest(manifest)).toThrow(/parameterCount 979 does not match/);
  });

  it('rejects a missing or unknown layer', () => {
    const missing = manifestClone();
    delete layersOf(missing).conv3;
    expect(() => parseMnistCnnManifest(missing)).toThrow(/layers\.conv3 must be a JSON object/);

    const extra = manifestClone();
    layersOf(extra).conv5 = { kernel: [], bias: [] };
    expect(() => parseMnistCnnManifest(extra)).toThrow(/unsupported layers/);
  });

  it('rejects wrong kernel or bias lengths', () => {
    const shortKernel = manifestClone();
    layersOf(shortKernel).conv1.kernel = (layersOf(shortKernel).conv1.kernel as number[]).slice(
      0,
      26,
    );
    expect(() => parseMnistCnnManifest(shortKernel)).toThrow(/conv1 kernel must have 27 values/);

    const longBias = manifestClone();
    layersOf(longBias).dense2.bias = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(() => parseMnistCnnManifest(longBias)).toThrow(/dense2 bias must have 10 values/);
  });

  it('names the layer whose dense bias has the wrong length', () => {
    const wrongBias = manifestClone();
    layersOf(wrongBias).dense1.bias = [0, 0, 0, 0, 0, 0, 0];
    expect(() => parseMnistCnnManifest(wrongBias)).toThrow(/dense1 bias must have 8 values/);
  });

  it('rejects non-finite and non-numeric weight values', () => {
    const notFinite = manifestClone();
    (layersOf(notFinite).conv2.kernel as number[])[3] = Number.NaN;
    expect(() => parseMnistCnnManifest(notFinite)).toThrow(
      /conv2 kernel must contain only finite numbers/,
    );

    const notANumber = manifestClone();
    (layersOf(notANumber).dense1.bias as unknown[])[0] = 'zero';
    expect(() => parseMnistCnnManifest(notANumber)).toThrow(
      /dense1 bias must contain only finite numbers/,
    );
  });

  it('rejects a non-object manifest and invalid JSON text', () => {
    expect(() => parseMnistCnnManifest(null)).toThrow(/manifest must be a JSON object/);
    expect(() => parseMnistCnnManifestJson('{ not json')).toThrow(/invalid JSON manifest/);
  });
});

describe('exported weights parity with PyTorch', () => {
  it('stays within 1e-5 of the PyTorch float32 reference logits', () => {
    const { config, weights } = parseMnistCnnManifestJson(MANIFEST_TEXT);
    let maxError = 0;
    for (let image = 0; image < PARITY_LOGITS.length; image += 1) {
      const { logits } = forwardMnistCnn(parityInput(image), weights, config);
      expect(logits).toHaveLength(PARITY_LOGITS[image].length);
      for (let cls = 0; cls < logits.length; cls += 1) {
        maxError = Math.max(maxError, Math.abs(logits[cls] - PARITY_LOGITS[image][cls]));
      }
    }
    expect(maxError).toBeLessThan(1e-5);
  });
});
