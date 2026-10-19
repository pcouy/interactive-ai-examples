import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import {
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  HIDDEN_DENSE_SIZE,
  LOGIT_COUNT,
  MNIST_INPUT_LENGTH,
  type MnistCnnFilters,
  type MnistCnnWeights,
} from './mnist-cnn.js';
import {
  createMnistCnnTfjsTrainer,
  DEFAULT_TRAINING_SEED,
  MnistCnnTfjsTrainer,
  type MnistTrainingProgress,
} from './mnist-cnn-tfjs.js';
import type { MnistSample } from './mnist-data.js';

const SMALL_FILTERS: MnistCnnFilters = [1, 1, 1, 1];

const disposables: MnistCnnTfjsTrainer[] = [];

beforeAll(async () => {
  await tf.setBackend('cpu');
});

afterEach(async () => {
  while (disposables.length > 0) {
    const trainer = disposables.pop();
    await trainer?.dispose();
  }
});

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Échantillon synthétique : pixels dans [0, 1], libellés cycliques 0 à 9. */
function makeSample(count: number, seed = 7): MnistSample {
  const rng = mulberry32(seed);
  const images = new Float32Array(count * MNIST_INPUT_LENGTH);
  for (let index = 0; index < images.length; index += 1) {
    images[index] = rng();
  }
  const labels = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    labels[index] = index % LOGIT_COUNT;
  }
  return { count, images, labels };
}

function rampInput(): Float32Array {
  const input = new Float32Array(MNIST_INPUT_LENGTH);
  for (let index = 0; index < input.length; index += 1) {
    input[index] = ((index % 9) + 1) / 9;
  }
  return input;
}

async function makeTrainer(
  filters: MnistCnnFilters,
  overrides: Partial<{ learningRate: number; batchSize: number; seed: number }> = {},
): Promise<MnistCnnTfjsTrainer> {
  const trainer = await createMnistCnnTfjsTrainer({
    filters,
    learningRate: overrides.learningRate ?? 3e-3,
    batchSize: overrides.batchSize ?? 2,
    seed: overrides.seed ?? DEFAULT_TRAINING_SEED,
  });
  disposables.push(trainer);
  return trainer;
}

describe('static import graph', () => {
  const source = readFileSync(new URL('./mnist-cnn-tfjs.ts', import.meta.url), 'utf8');
  // Les commentaires contiennent le mot « import » : on les retire avant
  // d'analyser les instructions d'import réelles.
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

  it('does not eagerly import TensorFlow.js', () => {
    const valueImports = [...code.matchAll(/^import\s+(?!type\b)[^;]*?from\s+'([^']+)'/gm)].map(
      (match) => match[1],
    );
    expect(valueImports).not.toContain('@tensorflow/tfjs');
    expect(valueImports).not.toContain('./mnist-cnn.js');
    expect(code).toContain("import('@tensorflow/tfjs')");
    expect(code).toContain("import('./mnist-cnn.js')");
  });

  it('does not eagerly import through side-effect imports or require', () => {
    // Un import à effet de bord (`import '@tensorflow/tfjs';`) n'a pas de clause
    // `from` : il faut le chercher séparément, comme l'appel `require`.
    const sideEffectImports = [...code.matchAll(/^import\s+'([^']+)'/gm)].map((match) => match[1]);
    expect(sideEffectImports).not.toContain('@tensorflow/tfjs');
    expect(sideEffectImports).not.toContain('./mnist-cnn.js');
    expect(code).not.toMatch(/\brequire\s*\(\s*['"]@tensorflow\/tfjs['"]\s*\)/);
    expect(code).not.toMatch(/\brequire\s*\(\s*['"]\.\/mnist-cnn\.js['"]\s*\)/);
  });

  it('never references DOM APIs', () => {
    expect(code).not.toMatch(/\b(document|window|navigator|HTMLElement|localStorage)\b/);
    expect(code).not.toMatch(/\brequestAnimationFrame\b/);
  });
});

describe('model shape', () => {
  it('builds the reference architecture for the default filters', async () => {
    const trainer = await makeTrainer(DEFAULT_FILTER_COUNTS);
    await trainer.exportWeights();
    expect(trainer.graph?.layers.map((layer) => layer.outputShape)).toEqual([
      [null, 26, 26, 3],
      [null, 28, 28, 3],
      [null, 13, 13, 3],
      [null, 11, 11, 3],
      [null, 5, 5, 3],
      [null, 5, 5, 3],
      [null, 75],
      [null, HIDDEN_DENSE_SIZE],
      [null, LOGIT_COUNT],
    ]);
  });

  it('scales the architecture with the filter tuple', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await trainer.exportWeights();
    expect(trainer.graph?.layers.map((layer) => layer.outputShape)).toEqual([
      [null, 26, 26, 1],
      [null, 28, 28, 1],
      [null, 13, 13, 1],
      [null, 11, 11, 1],
      [null, 5, 5, 1],
      [null, 5, 5, 1],
      [null, 25],
      [null, HIDDEN_DENSE_SIZE],
      [null, LOGIT_COUNT],
    ]);
  });

  it('returns softmax probabilities on the simplex', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const probabilities = await trainer.predict(makeSample(3));
    expect(probabilities).toHaveLength(3 * LOGIT_COUNT);
    for (let row = 0; row < 3; row += 1) {
      let sum = 0;
      for (let column = 0; column < LOGIT_COUNT; column += 1) {
        const value = probabilities[row * LOGIT_COUNT + column];
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
        sum += value;
      }
      expect(sum).toBeCloseTo(1, 5);
    }
  });
});

describe('weight export', () => {
  it('exports the default architecture in the MnistCnnWeights layout', async () => {
    const trainer = await makeTrainer(DEFAULT_FILTER_COUNTS);
    const weights: MnistCnnWeights = await trainer.exportWeights();
    expect(weights.conv1.kernel).toHaveLength(9 * 1 * 3);
    expect(weights.conv2.kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv3.kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv4.kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv1.bias).toHaveLength(3);
    expect(weights.conv2.bias).toHaveLength(3);
    expect(weights.conv3.bias).toHaveLength(3);
    expect(weights.conv4.bias).toHaveLength(3);
    expect(weights.dense1.kernel).toHaveLength(5 * 5 * 3 * HIDDEN_DENSE_SIZE);
    expect(weights.dense1.bias).toHaveLength(HIDDEN_DENSE_SIZE);
    expect(weights.dense2.kernel).toHaveLength(HIDDEN_DENSE_SIZE * LOGIT_COUNT);
    expect(weights.dense2.bias).toHaveLength(LOGIT_COUNT);
  });

  it('exports the custom architecture in the MnistCnnWeights layout', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const weights = await trainer.exportWeights();
    expect(weights.conv1.kernel).toHaveLength(9);
    expect(weights.conv4.kernel).toHaveLength(9);
    expect(weights.dense1.kernel).toHaveLength(5 * 5 * 1 * HIDDEN_DENSE_SIZE);
    expect(weights.dense2.kernel).toHaveLength(HIDDEN_DENSE_SIZE * LOGIT_COUNT);
  });

  it('keeps the dense1 input ordering identical to the core NHWC flatten', async () => {
    const trainer = await makeTrainer(DEFAULT_FILTER_COUNTS);
    const weights = await trainer.exportWeights();
    const input = rampInput();
    const probabilities = await trainer.predict({
      count: 1,
      images: input,
      labels: new Uint8Array([0]),
    });
    const reference = forwardMnistCnn(input, weights, { filters: DEFAULT_FILTER_COUNTS });
    for (let index = 0; index < LOGIT_COUNT; index += 1) {
      expect(probabilities[index]).toBeCloseTo(reference.probabilities[index], 4);
    }
  });

  it('is deterministic for an equal seed and differs for another seed', async () => {
    const first = await makeTrainer(SMALL_FILTERS, { seed: 5 });
    const second = await makeTrainer(SMALL_FILTERS, { seed: 5 });
    const third = await makeTrainer(SMALL_FILTERS, { seed: 6 });
    const a = await first.exportWeights();
    const b = await second.exportWeights();
    const c = await third.exportWeights();
    expect(Array.from(a.conv1.kernel)).toEqual(Array.from(b.conv1.kernel));
    expect(Array.from(a.dense2.kernel)).not.toEqual(Array.from(c.dense2.kernel));
  });
});

describe('bounded training', () => {
  it('runs a single bounded update per batch and reports progress', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const initial = await trainer.exportWeights();
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSample(4),
      epochs: 1,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(2);
    expect(Number.isFinite(result.loss)).toBe(true);
    expect(result.loss).toBeGreaterThan(0);
    expect(result.validationLoss).toBeNull();
    expect(result.validationAccuracy).toBeNull();
    expect(events.filter((event) => event.status === 'running')).toHaveLength(2);
    expect(events.at(-1)?.status).toBe('completed');
    const updated = await trainer.exportWeights();
    expect(Array.from(updated.dense1.kernel)).not.toEqual(Array.from(initial.dense1.kernel));
  });

  it('reports validation loss and accuracy at the end of each epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const result = await trainer.train({
      train: makeSample(4, 1),
      validation: makeSample(6, 2),
      epochs: 2,
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(4);
    expect(result.validationLoss).toBeGreaterThan(0);
    expect(result.validationAccuracy).toBeGreaterThanOrEqual(0);
    expect(result.validationAccuracy).toBeLessThanOrEqual(1);
  });

  it('yields between batches instead of running a whole epoch in one tick', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const pending: Array<() => void> = [];
    const events: MnistTrainingProgress[] = [];
    const running = trainer.train({
      train: makeSample(4),
      epochs: 1,
      onProgress: (progress) => events.push(progress),
      yieldControl: () =>
        new Promise<void>((resolve) => {
          pending.push(resolve);
        }),
    });

    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    expect(events.filter((event) => event.status === 'running')).toHaveLength(1);
    expect(pending).toHaveLength(1);

    while (pending.length > 0) {
      pending.shift()?.();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    const result = await running;
    expect(result.status).toBe('completed');
    expect(events.filter((event) => event.status === 'running')).toHaveLength(2);
  });

  it('rejects a concurrent training run', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const pending: Array<() => void> = [];
    const running = trainer.train({
      train: makeSample(4),
      yieldControl: () =>
        new Promise<void>((resolve) => {
          pending.push(resolve);
        }),
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await expect(trainer.train({ train: makeSample(2) })).rejects.toThrow(/already training/i);
    pending.shift()?.();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    pending.shift()?.();
    await running;
  });

  it('stays reusable when a training tensor allocation fails', async () => {
    const core = await import('./mnist-cnn.js');
    const realTensor4d = tf.tensor4d as (...args: unknown[]) => tf.Tensor;
    let shouldFail = true;
    const flakyTf = {
      ...tf,
      tensor4d: (...args: unknown[]): tf.Tensor => {
        if (shouldFail) {
          shouldFail = false;
          throw new Error('MNIST TFJS allocation failed.');
        }
        return realTensor4d(...args);
      },
    } as unknown as typeof tf;
    const trainer = new MnistCnnTfjsTrainer(flakyTf, core, {
      filters: SMALL_FILTERS,
      learningRate: 3e-3,
      batchSize: 2,
      seed: DEFAULT_TRAINING_SEED,
    });
    disposables.push(trainer);
    await expect(trainer.train({ train: makeSample(2) })).rejects.toThrow(/allocation failed/i);
    // La première allocation a échoué : l'entraîneur ne doit pas rester bloqué.
    const result = await trainer.train({ train: makeSample(2), epochs: 1 });
    expect(result.status).toBe('completed');
  });
});

describe('cancellation', () => {
  it('stops before the next batch and releases tensors', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 1 });
    const before = tf.memory().numTensors;
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSample(4),
      epochs: 1,
      onProgress: (progress) => {
        events.push(progress);
        if (progress.batchesProcessed === 1) {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    expect(result.batchesProcessed).toBe(1);
    await trainer.dispose();
    disposables.splice(disposables.indexOf(trainer), 1);
    expect(tf.memory().numTensors).toBe(before);
  });

  it('ignores a cancellation requested after completion', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const first = await trainer.train({ train: makeSample(2), epochs: 1 });
    expect(first.status).toBe('completed');
    await trainer.cancel();
    const second = await trainer.train({ train: makeSample(2), epochs: 1 });
    expect(second.status).toBe('completed');
  });

  it('skips validation when cancellation is requested during the last batch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 4 });
    const result = await trainer.train({
      train: makeSample(4),
      validation: makeSample(4, 2),
      epochs: 1,
      onProgress: (progress) => {
        if (progress.status === 'running') {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    expect(result.validationLoss).toBeNull();
    expect(result.validationAccuracy).toBeNull();
  });
});

describe('reset and disposal', () => {
  it('rebuilds a fresh model after reset', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const initial = await trainer.exportWeights();
    await trainer.train({ train: makeSample(4), epochs: 1 });
    trainer.reset();
    const rebuilt = await trainer.exportWeights();
    expect(trainer.graph).not.toBeNull();
    expect(Array.from(rebuilt.conv1.kernel)).toEqual(Array.from(initial.conv1.kernel));
  });

  it('refuses to run after dispose and is idempotent', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await trainer.dispose();
    expect(trainer.isDisposed).toBe(true);
    await expect(trainer.train({ train: makeSample(2) })).rejects.toThrow(/disposed/i);
    await expect(trainer.exportWeights()).rejects.toThrow(/disposed/i);
    await expect(trainer.predict(makeSample(2))).rejects.toThrow(/disposed/i);
    await expect(trainer.dispose()).resolves.toBeUndefined();
  });
});

describe('error handling', () => {
  it('releases batch tensors when a training update fails', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 1 });
    const before = tf.memory().numTensors;
    const pending: Array<() => void> = [];
    const running = trainer.train({
      train: makeSample(4),
      epochs: 1,
      yieldControl: () =>
        new Promise<void>((resolve) => {
          pending.push(resolve);
        }),
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    // Disposer le modèle en pleine formation fait échouer la mise à jour du lot.
    await trainer.dispose();
    disposables.splice(disposables.indexOf(trainer), 1);
    pending.shift()?.();
    await expect(running).rejects.toThrow();
    expect(tf.memory().numTensors).toBe(before);
  });
});

describe('configuration validation', () => {
  it('rejects invalid filter counts', async () => {
    await expect(makeTrainer([0, 1, 1, 1])).rejects.toThrow(/filter count/i);
    await expect(makeTrainer([1, 1, 1, 9])).rejects.toThrow(/filter count/i);
  });

  it('rejects invalid learning rate, batch size and seed', async () => {
    await expect(makeTrainer(SMALL_FILTERS, { learningRate: 0 })).rejects.toThrow(/learning rate/i);
    await expect(makeTrainer(SMALL_FILTERS, { batchSize: 0 })).rejects.toThrow(/batch size/i);
    await expect(makeTrainer(SMALL_FILTERS, { seed: -1 })).rejects.toThrow(/seed/i);
  });

  it('rejects malformed samples and epochs', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({
        train: { count: 2, images: new Float32Array(10), labels: new Uint8Array(2) },
      }),
    ).rejects.toThrow(/images/i);
    await expect(
      trainer.train({
        train: {
          count: 2,
          images: new Float32Array(2 * MNIST_INPUT_LENGTH),
          labels: new Uint8Array(1),
        },
      }),
    ).rejects.toThrow(/labels/i);
    await expect(trainer.train({ train: makeSample(2), epochs: 0 })).rejects.toThrow(/epochs/i);
  });

  it('rejects out-of-range labels and non-finite pixels', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({
        train: {
          count: 2,
          images: new Float32Array(2 * MNIST_INPUT_LENGTH),
          labels: new Uint8Array([1, 12]),
        },
      }),
    ).rejects.toThrow(/labels/i);

    const notANumber = new Float32Array(2 * MNIST_INPUT_LENGTH);
    notANumber[3] = Number.NaN;
    await expect(
      trainer.train({ train: { count: 2, images: notANumber, labels: new Uint8Array([1, 2]) } }),
    ).rejects.toThrow(/images/i);

    const infinite = new Float32Array(2 * MNIST_INPUT_LENGTH);
    infinite[7] = Number.POSITIVE_INFINITY;
    await expect(
      trainer.predict({ count: 2, images: infinite, labels: new Uint8Array([1, 2]) }),
    ).rejects.toThrow(/images/i);
  });
});
