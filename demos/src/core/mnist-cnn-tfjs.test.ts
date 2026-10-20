import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';
import {
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  HIDDEN_DENSE_SIZE,
  initializeMnistCnnWeights,
  LOGIT_COUNT,
  MNIST_INPUT_LENGTH,
  type MnistCnnActivation,
  type MnistCnnFilters,
  type MnistCnnWeights,
} from './mnist-cnn.js';
import {
  createMnistCnnTfjsTrainer,
  DEFAULT_PREFETCH_DEPTH,
  DEFAULT_TRAINING_SEED,
  MAX_LOSS_SAMPLE_EVERY,
  MAX_PREFETCH_DEPTH,
  MIN_LOSS_SAMPLE_EVERY,
  MIN_PREFETCH_DEPTH,
  MNIST_VALIDATION_CHUNK_SIZE,
  MnistCnnTfjsTrainer,
  type MnistTrainingProgress,
  type MnistTrainSource,
  sampleToTrainSource,
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

/** Source d'entraînement en mémoire, adossée à un échantillon synthétique. */
function makeSource(count: number, seed = 7): MnistTrainSource {
  return sampleToTrainSource(makeSample(count, seed));
}

/** Source dont `readBatch` retourne toujours le même lot, pour tester la validation. */
function sourceOfBatch(batch: MnistSample): MnistTrainSource {
  return { count: batch.count, readBatch: () => batch };
}

/** Source qui découpe un échantillon en tranches et enregistre chaque plage lue. */
function recordingSource(sample: MnistSample, ranges: Array<[number, number]>): MnistTrainSource {
  return {
    count: sample.count,
    readBatch(start, end) {
      ranges.push([start, end]);
      return {
        count: end - start,
        images: sample.images.subarray(start * MNIST_INPUT_LENGTH, end * MNIST_INPUT_LENGTH),
        labels: sample.labels.subarray(start, end),
      };
    },
  };
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
  overrides: Partial<{
    learningRate: number;
    batchSize: number;
    seed: number;
    activation: MnistCnnActivation;
    convLayers: number;
    denseLayers: number;
    denseSizes: readonly number[];
  }> = {},
): Promise<MnistCnnTfjsTrainer> {
  const trainer = await createMnistCnnTfjsTrainer({
    filters,
    learningRate: overrides.learningRate ?? 3e-3,
    batchSize: overrides.batchSize ?? 2,
    activation: overrides.activation ?? 'sigmoid',
    convLayers: overrides.convLayers ?? 4,
    denseLayers: overrides.denseLayers ?? 2,
    denseSizes: overrides.denseSizes,
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
    // Le contrat du core est importé statiquement ; seul TensorFlow.js reste
    // un import dynamique.
    expect(valueImports).toContain('./mnist-cnn.js');
    expect(code).toContain("import('@tensorflow/tfjs')");
    expect(code).not.toContain("import('./mnist-cnn.js')");
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

  it('builds the convolution and hidden layers with the requested activation', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { activation: 'tanh', batchSize: 2 });
    await trainer.exportWeights();
    const graph = trainer.graph;
    if (graph === null) {
      throw new Error('The trainer did not build a model.');
    }
    const layers = graph.layers;
    const convs = layers.filter((layer) => layer.getClassName() === 'Conv2D');
    expect(convs.length).toBe(4);
    for (const layer of convs) {
      expect(layer.getConfig().activation).toBe('tanh');
    }
    const denses = layers.filter((layer) => layer.getClassName() === 'Dense');
    expect(denses.length).toBe(2);
    const hidden = denses.find((layer) => layer.getConfig().units === 8);
    const logits = denses.find((layer) => layer.getConfig().units === 10);
    expect(hidden?.getConfig().activation).toBe('tanh');
    expect(logits?.getConfig().activation).toBe('linear');
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

  it('builds a model for the requested depth', async () => {
    const cases: ReadonlyArray<readonly [number, number, readonly number[]]> = [
      [0, 1, [LOGIT_COUNT]],
      [1, 3, [HIDDEN_DENSE_SIZE, HIDDEN_DENSE_SIZE, LOGIT_COUNT]],
      [3, 2, [HIDDEN_DENSE_SIZE, LOGIT_COUNT]],
    ];
    for (const [convLayers, denseLayers, widths] of cases) {
      const trainer = await makeTrainer(SMALL_FILTERS, { convLayers, denseLayers });
      const weights = await trainer.exportWeights();
      const graph = trainer.graph;
      if (graph === null) {
        throw new Error('The trainer did not build a model.');
      }
      expect(graph.layers.filter((layer) => layer.getClassName() === 'Conv2D')).toHaveLength(
        convLayers,
      );
      expect(
        graph.layers
          .filter((layer) => layer.getClassName() === 'Dense')
          .map((layer) => layer.getConfig().units),
      ).toEqual(widths);
      expect(weights.conv).toHaveLength(convLayers);
      expect(weights.dense).toHaveLength(denseLayers);
    }
  });

  it('builds the hidden dense units from denseSizes', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { denseLayers: 3, denseSizes: [3, 7] });
    await trainer.exportWeights();
    const graph = trainer.graph;
    if (graph === null) {
      throw new Error('The trainer did not build a model.');
    }
    expect(
      graph.layers
        .filter((layer) => layer.getClassName() === 'Dense')
        .map((layer) => layer.getConfig().units),
    ).toEqual([3, 7, LOGIT_COUNT]);
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
    expect(weights.conv).toHaveLength(4);
    expect(weights.dense).toHaveLength(2);
    expect(weights.conv[0].kernel).toHaveLength(9 * 1 * 3);
    expect(weights.conv[1].kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv[2].kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv[3].kernel).toHaveLength(9 * 3 * 3);
    expect(weights.conv[0].bias).toHaveLength(3);
    expect(weights.conv[1].bias).toHaveLength(3);
    expect(weights.conv[2].bias).toHaveLength(3);
    expect(weights.conv[3].bias).toHaveLength(3);
    expect(weights.dense[0].kernel).toHaveLength(5 * 5 * 3 * HIDDEN_DENSE_SIZE);
    expect(weights.dense[0].bias).toHaveLength(HIDDEN_DENSE_SIZE);
    expect(weights.dense[1].kernel).toHaveLength(HIDDEN_DENSE_SIZE * LOGIT_COUNT);
    expect(weights.dense[1].bias).toHaveLength(LOGIT_COUNT);
  });

  it('exports the custom architecture in the MnistCnnWeights layout', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const weights = await trainer.exportWeights();
    expect(weights.conv[0].kernel).toHaveLength(9);
    expect(weights.conv[3].kernel).toHaveLength(9);
    expect(weights.dense[0].kernel).toHaveLength(5 * 5 * 1 * HIDDEN_DENSE_SIZE);
    expect(weights.dense[1].kernel).toHaveLength(HIDDEN_DENSE_SIZE * LOGIT_COUNT);
  });

  it('keeps the dense input ordering identical to the core NHWC flatten', async () => {
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
    expect(Array.from(a.conv[0].kernel)).toEqual(Array.from(b.conv[0].kernel));
    expect(Array.from(a.dense[1].kernel)).not.toEqual(Array.from(c.dense[1].kernel));
  });
});

describe('applyWeights', () => {
  it('round-trips exportWeights through applyWeights and keeps predict identical', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const sample = makeSample(4);
    const before = await trainer.predict(sample);
    const weights = await trainer.exportWeights();
    await trainer.applyWeights(weights);
    const after = await trainer.predict(sample);
    expect(after).toHaveLength(before.length);
    for (let index = 0; index < before.length; index += 1) {
      expect(after[index]).toBeCloseTo(before[index], 5);
    }
  });

  it('makes predict match the core forward pass for the applied weights', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const weights = initializeMnistCnnWeights({ filters: SMALL_FILTERS }, 1234);
    await trainer.applyWeights(weights);
    const input = rampInput();
    const probabilities = await trainer.predict({
      count: 1,
      images: input,
      labels: new Uint8Array([0]),
    });
    const reference = forwardMnistCnn(input, weights, { filters: SMALL_FILTERS });
    for (let index = 0; index < LOGIT_COUNT; index += 1) {
      expect(probabilities[index]).toBeCloseTo(reference.probabilities[index], 4);
    }
  });

  it('round-trips the default multi-channel architecture through applyWeights', async () => {
    const trainer = await makeTrainer(DEFAULT_FILTER_COUNTS);
    const weights = await trainer.exportWeights();
    await trainer.applyWeights(weights);
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

  it('round-trips custom denseSizes through applyWeights and exportWeights', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { denseLayers: 3, denseSizes: [3, 7] });
    const weights = initializeMnistCnnWeights(
      { filters: SMALL_FILTERS, denseLayers: 3, denseSizes: [3, 7] },
      1234,
    );
    await trainer.applyWeights(weights);
    const exported = await trainer.exportWeights();
    expect(exported.dense).toHaveLength(3);
    expect(exported.dense.map((layer) => layer.kernel.length)).toEqual([25 * 3, 3 * 7, 7 * 10]);
    expect(exported.dense.map((layer) => layer.bias.length)).toEqual([3, 7, LOGIT_COUNT]);
    for (let layer = 0; layer < exported.dense.length; layer += 1) {
      expect(Array.from(exported.dense[layer].kernel)).toEqual(
        Array.from(weights.dense[layer].kernel),
      );
      expect(Array.from(exported.dense[layer].bias)).toEqual(Array.from(weights.dense[layer].bias));
    }
  });

  it('rejects a weight set whose filter tuple does not match the config', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const other = await makeTrainer(DEFAULT_FILTER_COUNTS);
    const weights = await other.exportWeights();
    await expect(trainer.applyWeights(weights)).rejects.toThrow(RangeError);
  });

  it('rejects weights with non-finite values', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const weights = initializeMnistCnnWeights({ filters: SMALL_FILTERS }, 7);
    weights.conv[0].kernel[0] = Number.NaN;
    await expect(trainer.applyWeights(weights)).rejects.toThrow(RangeError);
  });

  it('rejects a weight set whose layer counts do not match the config', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { convLayers: 2, denseLayers: 1 });
    const weights = initializeMnistCnnWeights({ filters: SMALL_FILTERS }, 7);
    await expect(trainer.applyWeights(weights)).rejects.toThrow(RangeError);
  });

  it('rejects when the trainer is disposed', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    const weights = await trainer.exportWeights();
    await trainer.dispose();
    await expect(trainer.applyWeights(weights)).rejects.toThrow(/disposed/);
  });

  it('applies and exports weights for a shallow model', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { convLayers: 1, denseLayers: 1 });
    const weights = initializeMnistCnnWeights(
      { filters: SMALL_FILTERS, convLayers: 1, denseLayers: 1 },
      99,
    );
    await trainer.applyWeights(weights);
    const exported = await trainer.exportWeights();
    expect(exported.conv).toHaveLength(1);
    expect(exported.dense).toHaveLength(1);
    expect(Array.from(exported.conv[0].kernel)).toEqual(Array.from(weights.conv[0].kernel));
    expect(Array.from(exported.conv[0].bias)).toEqual(Array.from(weights.conv[0].bias));
    expect(Array.from(exported.dense[0].kernel)).toEqual(Array.from(weights.dense[0].kernel));
    expect(Array.from(exported.dense[0].bias)).toEqual(Array.from(weights.dense[0].bias));
  });

  it('resets Adam moments so a later run resumes from the applied weights', async () => {
    const source = makeSource(16);
    // Premier entraîneur : entraîne d'abord (moments Adam non nuls), puis
    // applique les poids et entraîne à nouveau.
    const first = await makeTrainer(SMALL_FILTERS, { batchSize: 4 });
    await first.train({ train: source, epochs: 1, lossSampleEvery: 1, progressIntervalMs: 0 });
    const applied = initializeMnistCnnWeights({ filters: SMALL_FILTERS }, 1234);
    await first.applyWeights(applied);
    await first.train({ train: source, epochs: 1, lossSampleEvery: 1, progressIntervalMs: 0 });
    const afterFirst = await first.exportWeights();
    // Second entraîneur : applique directement les mêmes poids, mêmes données,
    // même graine. Sans réinitialisation des moments, les deux différeraient.
    const second = await makeTrainer(SMALL_FILTERS, { batchSize: 4 });
    await second.applyWeights(applied);
    await second.train({ train: source, epochs: 1, lossSampleEvery: 1, progressIntervalMs: 0 });
    const afterSecond = await second.exportWeights();
    for (const layer of [afterFirst.conv[0], afterFirst.dense[1]] as const) {
      const other = layer === afterFirst.conv[0] ? afterSecond.conv[0] : afterSecond.dense[1];
      for (let index = 0; index < layer.kernel.length; index += 1) {
        expect(layer.kernel[index]).toBeCloseTo(other.kernel[index], 5);
      }
    }
  });
});

describe('bounded training', () => {
  it('runs a single bounded update per batch and reports progress', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const initial = await trainer.exportWeights();
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4),
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(2);
    expect(Number.isFinite(result.loss)).toBe(true);
    expect(result.loss).toBeGreaterThan(0);
    expect(result.validationLoss).toBeNull();
    expect(result.validationAccuracy).toBeNull();
    // Deux évènements par lot, plus un évènement de fin d'époque.
    expect(events.filter((event) => event.status === 'running')).toHaveLength(3);
    expect(events.at(-1)?.status).toBe('completed');
    const updated = await trainer.exportWeights();
    expect(Array.from(updated.dense[0].kernel)).not.toEqual(Array.from(initial.dense[0].kernel));
  });

  it('reports validation loss and accuracy at the end of each epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const result = await trainer.train({
      train: makeSource(4, 1),
      validation: makeSource(6, 2),
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
      train: makeSource(4),
      epochs: 1,
      progressIntervalMs: 0,
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
    // Deux évènements par lot, plus un évènement de fin d'époque.
    expect(events.filter((event) => event.status === 'running')).toHaveLength(3);
  });

  it('rejects a concurrent training run', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const pending: Array<() => void> = [];
    const running = trainer.train({
      train: makeSource(4),
      yieldControl: () =>
        new Promise<void>((resolve) => {
          pending.push(resolve);
        }),
    });
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    await expect(trainer.train({ train: makeSource(2) })).rejects.toThrow(/already training/i);
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
      activation: 'sigmoid',
      convLayers: 4,
      denseLayers: 2,
      seed: DEFAULT_TRAINING_SEED,
    });
    disposables.push(trainer);
    await expect(trainer.train({ train: makeSource(2) })).rejects.toThrow(/allocation failed/i);
    // La première allocation a échoué : l'entraîneur ne doit pas rester bloqué.
    const result = await trainer.train({ train: makeSource(2), epochs: 1 });
    expect(result.status).toBe('completed');
  });
});

describe('training accuracy', () => {
  it('reports the training accuracy of the completed epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const batch = makeSample(2);
    const result = await trainer.train({
      train: sourceOfBatch(batch),
      epochs: 1,
    });
    expect(result.status).toBe('completed');
    // Référence calculée indépendamment sur les mêmes poids finaux : l'argmax des
    // probabilités softmax est l'argmax des logits.
    const probabilities = await trainer.predict(batch);
    let correct = 0;
    for (let index = 0; index < batch.count; index += 1) {
      let best = 0;
      for (let column = 1; column < LOGIT_COUNT; column += 1) {
        if (
          probabilities[index * LOGIT_COUNT + column] > probabilities[index * LOGIT_COUNT + best]
        ) {
          best = column;
        }
      }
      if (best === batch.labels[index]) {
        correct += 1;
      }
    }
    expect(result.epochTrainAccuracy).toBeCloseTo(correct / batch.count, 6);
  });

  it('keeps the training accuracy null before the first completed epoch and finite at each epoch end', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4),
      epochs: 3,
      progressIntervalMs: 0,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    expect(result.completedEpochs).toBe(3);
    // Avec `progressIntervalMs: 0`, un évènement est aussi émis après chaque lot :
    // il porte `completedEpochs` au numéro de l'époque *précédente*. Seul un
    // évènement de fin d'époque l'incrémente : on ne retient donc que les événements
    // qui font avancer `completedEpochs` par rapport à l'évènement précédent.
    const epochEnds: MnistTrainingProgress[] = [];
    let lastCompletedEpochs = 0;
    for (const event of events) {
      if (event.status !== 'running' || event.completedEpochs <= lastCompletedEpochs) {
        continue;
      }
      epochEnds.push(event);
      lastCompletedEpochs = event.completedEpochs;
    }
    expect(epochEnds.map((event) => event.completedEpochs)).toEqual([1, 2, 3]);
    // Avant la fin de la première époque, l'exactitude d'entraînement n'est pas
    // encore mesurée : elle reste nulle.
    for (const event of events) {
      if (event.status === 'running' && event.completedEpochs === 0) {
        expect(event.epochTrainAccuracy).toBeNull();
      }
    }
    for (const event of epochEnds) {
      expect(Number.isFinite(event.epochTrainAccuracy)).toBe(true);
      expect(event.epochTrainAccuracy).toBeGreaterThanOrEqual(0);
      expect(event.epochTrainAccuracy).toBeLessThanOrEqual(1);
    }
    expect(Number.isFinite(result.epochTrainAccuracy)).toBe(true);
    expect(result.epochTrainAccuracy).toBeGreaterThanOrEqual(0);
    expect(result.epochTrainAccuracy).toBeLessThanOrEqual(1);
  });

  it('disposes the accuracy tensors after completion and cancellation', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    await trainer.exportWeights();
    await trainer.train({ train: makeSource(20, 1), epochs: 1 });
    const baseline = tf.memory().numTensors;

    const completed = await trainer.train({
      train: makeSource(20, 2),
      epochs: 1,
    });
    expect(completed.status).toBe('completed');
    expect(tf.memory().numTensors).toBe(baseline);

    const cancelled = await trainer.train({
      train: makeSource(20, 3),
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        if (progress.batchesProcessed === 1) {
          void trainer.cancel();
        }
      },
    });
    expect(cancelled.status).toBe('cancelled');
    expect(tf.memory().numTensors).toBe(baseline);
  });

  it('keeps the training accuracy of the last completed epoch when a cancel arrives during the read', async () => {
    const base = makeSource(4);
    let trainerRef: MnistCnnTfjsTrainer | null = null;
    // Les lots d'entraînement lisent `[0, 2]` puis `[2, 4]` ; la mesure
    // d'exactitude lit le préfixe `[0, 4]` en un seul bloc. On distingue la
    // mesure par sa taille et on annule pendant sa lecture.
    const source: MnistTrainSource = {
      count: base.count,
      readBatch(start, end) {
        if (end - start === 4) {
          void trainerRef?.cancel();
        }
        return base.readBatch(start, end);
      },
    };
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    trainerRef = trainer;
    await trainer.exportWeights();
    // Premier passage : crée les moments Adam persistants de l'optimiseur, de
    // sorte que le compteur de tenseurs soit stable avant la run mesurée.
    await trainer.train({ train: makeSource(4), epochs: 1 });
    const baseline = tf.memory().numTensors;
    const result = await trainer.train({
      train: source,
      epochs: 1,
    });
    expect(result.status).toBe('cancelled');
    expect(result.completedEpochs).toBe(0);
    expect(result.epochTrainAccuracy).toBeNull();
    expect(tf.memory().numTensors).toBe(baseline);
  });
});

describe('cancellation', () => {
  it('stops before the next batch and releases tensors', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 1 });
    const before = tf.memory().numTensors;
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4),
      epochs: 1,
      progressIntervalMs: 0,
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

  it('a very large epoch count starts and stops on cancel', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(20),
      epochs: Number.MAX_SAFE_INTEGER,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        events.push(progress);
        if (progress.status === 'running') {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    // La boucle a bien démarré : un premier lot a été traité avant l'arrêt.
    expect(events.length).toBeGreaterThan(0);
    expect(result.batchesProcessed).toBe(1);
  });

  it('a cancelled run reports no completed epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(20),
      progressIntervalMs: 0,
      onProgress: (progress) => {
        events.push(progress);
        if (progress.status === 'running' && progress.completedEpochs === 0) {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    expect(result.completedEpochs).toBe(0);
    // Aucun évènement ne rapporte une époque achevée : l'annulation est intervenue
    // avant la fin de la première époque.
    expect(events.some((event) => event.status === 'running' && event.completedEpochs > 0)).toBe(
      false,
    );
  });

  it('ignores a cancellation requested after completion', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const first = await trainer.train({ train: makeSource(2), epochs: 1 });
    expect(first.status).toBe('completed');
    await trainer.cancel();
    const second = await trainer.train({ train: makeSource(2), epochs: 1 });
    expect(second.status).toBe('completed');
  });

  it('skips validation when cancellation is requested during the last batch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 4 });
    // `progressIntervalMs: 0` émet un évènement par lot : l'annulation porte sur le
    // premier lot, avant la fin de l'époque — la validation n'a pas lieu. Un
    // évènement de fin d'époque est émis après la validation : annuler depuis
    // celui-ci ne pourrait pas la sauter.
    const result = await trainer.train({
      train: makeSource(4),
      validation: makeSource(4, 2),
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        if (progress.status === 'running' && progress.completedEpochs === 0) {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    expect(result.validationLoss).toBeNull();
    expect(result.validationAccuracy).toBeNull();
  });

  it('skips the epoch-end flush when cancellation arrives before the end of the epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    // `lossSampleEvery: 3` : la fenêtre n'est jamais pleine pendant le seul lot
    // traité avant l'annulation. Avant la correction, la vidange de fin d'époque
    // remplaçait la perte indéfinie par la moyenne de la fenêtre partielle.
    const result = await trainer.train({
      train: makeSource(10),
      epochs: 1,
      lossSampleEvery: 3,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        if (progress.status === 'running' && progress.batchesProcessed === 1) {
          void trainer.cancel();
        }
      },
    });
    expect(result.status).toBe('cancelled');
    expect(result.batchesProcessed).toBe(1);
    expect(Number.isNaN(result.loss)).toBe(true);
  });
});

describe('reset and disposal', () => {
  it('rebuilds a fresh model after reset', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const initial = await trainer.exportWeights();
    await trainer.train({ train: makeSource(4), epochs: 1 });
    trainer.reset();
    const rebuilt = await trainer.exportWeights();
    expect(trainer.graph).not.toBeNull();
    expect(Array.from(rebuilt.conv[0].kernel)).toEqual(Array.from(initial.conv[0].kernel));
  });

  it('refuses to run after dispose and is idempotent', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await trainer.dispose();
    expect(trainer.isDisposed).toBe(true);
    await expect(trainer.train({ train: makeSource(2) })).rejects.toThrow(/disposed/i);
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
      train: makeSource(4),
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

describe('train source', () => {
  it('reads bounded batches and never the whole split', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 4 });
    const sample = makeSample(10);
    const ranges: Array<[number, number]> = [];
    const result = await trainer.train({
      train: recordingSource(sample, ranges),
      prefetchDepth: 1,
      epochs: 1,
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(3);
    // La mesure d'exactitude d'entraînement lit le préfixe borné de la source en
    // dernier ; les lots d'entraînement restants restent bornés et couvrent le jeu.
    expect(ranges).toHaveLength(4);
    expect(ranges.at(-1)).toEqual([0, sample.count]);
    const batchRanges = ranges.slice(0, -1);
    for (const [start, end] of batchRanges) {
      expect(end - start).toBeGreaterThan(0);
      expect(end - start).toBeLessThanOrEqual(4);
      expect(end - start).toBeLessThan(sample.count);
    }
    const sorted = [...batchRanges].sort((a, b) => a[0] - b[0]);
    let cursor = 0;
    for (const [start, end] of sorted) {
      expect(start).toBe(cursor);
      cursor = end;
    }
    expect(cursor).toBe(sample.count);
  });

  it('rejects a source whose readBatch returns the wrong count', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const wrongCount: MnistTrainSource = {
      count: 4,
      readBatch(start, end) {
        const size = end - start + 1;
        return {
          count: size,
          images: new Float32Array(size * MNIST_INPUT_LENGTH),
          labels: new Uint8Array(size),
        };
      },
    };
    await expect(trainer.train({ train: wrongCount, epochs: 1 })).rejects.toThrow(/readBatch/i);
  });

  it('rejects a source with an invalid count', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({ train: { count: 0, readBatch: () => makeSample(1) } }),
    ).rejects.toThrow(/count/i);
  });
});

describe('validation chunking', () => {
  it('accumulates metrics over chunks larger than the chunk size', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 8 });
    const validationSample = makeSample(MNIST_VALIDATION_CHUNK_SIZE + 8, 3);
    const ranges: Array<[number, number]> = [];
    const result = await trainer.train({
      train: makeSource(8),
      validation: recordingSource(validationSample, ranges),
      epochs: 1,
    });
    expect(result.status).toBe('completed');
    expect(ranges).toHaveLength(2);
    expect(ranges[0][1] - ranges[0][0]).toBe(MNIST_VALIDATION_CHUNK_SIZE);
    expect(ranges[1][1] - ranges[1][0]).toBe(8);

    // Référence calculée indépendamment sur les mêmes poids finaux.
    const probabilities = await trainer.predict(validationSample);
    let correct = 0;
    for (let index = 0; index < validationSample.count; index += 1) {
      let best = 0;
      for (let column = 1; column < LOGIT_COUNT; column += 1) {
        const candidate = probabilities[index * LOGIT_COUNT + column];
        if (candidate > probabilities[index * LOGIT_COUNT + best]) {
          best = column;
        }
      }
      if (best === validationSample.labels[index]) {
        correct += 1;
      }
    }
    expect(result.validationAccuracy).toBeCloseTo(correct / validationSample.count, 6);
    expect(result.validationLoss).toBeGreaterThan(0);
  });
});

describe('prefetch', () => {
  it('materializes at most prefetchDepth batches ahead', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const prefetchDepth = DEFAULT_PREFETCH_DEPTH;
    const sample = makeSample(20);
    const ranges: Array<[number, number]> = [];
    const outstanding: number[] = [];
    let consumed = 0;
    const source: MnistTrainSource = {
      count: sample.count,
      readBatch(start, end) {
        if (ranges.length - consumed >= prefetchDepth) {
          throw new Error('MNIST TFJS source read more than prefetchDepth batches ahead.');
        }
        ranges.push([start, end]);
        outstanding.push(ranges.length - consumed);
        return {
          count: end - start,
          images: sample.images.subarray(start * MNIST_INPUT_LENGTH, end * MNIST_INPUT_LENGTH),
          labels: sample.labels.subarray(start, end),
        };
      },
    };
    const result = await trainer.train({
      train: source,
      prefetchDepth,
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        consumed = progress.batchesProcessed;
      },
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(10);
    // La mesure d'exactitude d'entraînement lit le préfixe borné de la source en
    // dernier, en dehors de la boucle de lots.
    expect(ranges).toHaveLength(11);
    expect(ranges.at(-1)).toEqual([0, sample.count]);
    const batchRanges = ranges.slice(0, -1);
    expect(new Set(batchRanges.map(([start, end]) => `${start}-${end}`)).size).toBe(10);
    expect(Math.max(...outstanding)).toBe(prefetchDepth);
  });

  it('drains and disposes the buffer after completion and cancellation', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    await trainer.exportWeights();
    await trainer.train({ train: makeSource(20, 1), prefetchDepth: 4, epochs: 1 });
    const baseline = tf.memory().numTensors;

    const completed = await trainer.train({
      train: makeSource(20, 2),
      prefetchDepth: 4,
      epochs: 1,
    });
    expect(completed.status).toBe('completed');
    expect(tf.memory().numTensors).toBe(baseline);

    const cancelled = await trainer.train({
      train: makeSource(20, 3),
      prefetchDepth: 4,
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        if (progress.batchesProcessed === 1) {
          void trainer.cancel();
        }
      },
    });
    expect(cancelled.status).toBe('cancelled');
    expect(tf.memory().numTensors).toBe(baseline);
  });

  it('completes and updates the weights with prefetchDepth 1', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const initial = await trainer.exportWeights();
    const result = await trainer.train({ train: makeSource(4), prefetchDepth: 1, epochs: 1 });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(2);
    const updated = await trainer.exportWeights();
    expect(Array.from(updated.dense[0].kernel)).not.toEqual(Array.from(initial.dense[0].kernel));
  });
});

describe('loss sampling', () => {
  it('reads the loss once per window and never synchronously', async () => {
    // On espionne `Tensor.prototype` : la lecture porte sur le scalaire moyen,
    // pas sur les tenseurs de coût individuels.
    const prototype = tf.Tensor.prototype as unknown as {
      data: (...args: unknown[]) => Promise<unknown>;
      dataSync: (...args: unknown[]) => unknown;
    };
    const originalData = prototype.data;
    const originalDataSync = prototype.dataSync;
    let dataCalls = 0;
    let dataSyncCalls = 0;
    prototype.data = function (this: unknown, ...args: unknown[]): Promise<unknown> {
      dataCalls += 1;
      return originalData.apply(this, args);
    };
    prototype.dataSync = function (this: unknown, ...args: unknown[]): unknown {
      dataSyncCalls += 1;
      return originalDataSync.apply(this, args);
    };
    try {
      const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
      const batches = 16;
      const result = await trainer.train({
        train: makeSource(batches * 2),
        lossSampleEvery: 4,
        epochs: 1,
      });
      expect(result.status).toBe('completed');
      expect(result.batchesProcessed).toBe(batches);
      expect(Number.isFinite(result.loss)).toBe(true);
      // Aucune lecture synchrone : la file GPU n'est jamais vidée par lot.
      expect(dataSyncCalls).toBe(0);
      // Une seule lecture par fenêtre : 16 lots / 4 = 4 lectures, plus la lecture
      // de l'exactitude d'entraînement de l'époque (un bloc pour 32 exemples).
      expect(dataCalls).toBe(5);
    } finally {
      prototype.data = originalData;
      prototype.dataSync = originalDataSync;
    }
  });

  it('flushes a partial window at the end of an epoch with the mean loss', async () => {
    const core = await import('./mnist-cnn.js');
    const batchLosses: number[] = [];
    const realAdam = tf.train.adam.bind(tf.train);
    const fakeTf = {
      ...tf,
      train: {
        ...tf.train,
        adam: (learningRate: number) => {
          const optimizer = realAdam(learningRate);
          return new Proxy(optimizer, {
            get(target, property, receiver) {
              if (property !== 'minimize') {
                return Reflect.get(target, property, receiver);
              }
              return (...args: unknown[]) => {
                const cost = (target.minimize as (...inner: unknown[]) => tf.Tensor | null)(
                  ...args,
                );
                if (cost !== null) {
                  batchLosses.push(Number(cost.dataSync()[0]));
                }
                return cost;
              };
            },
          });
        },
      },
    } as unknown as typeof tf;
    const trainer = new MnistCnnTfjsTrainer(fakeTf, core, {
      filters: SMALL_FILTERS,
      learningRate: 3e-3,
      batchSize: 2,
      activation: 'sigmoid',
      convLayers: 4,
      denseLayers: 2,
      seed: DEFAULT_TRAINING_SEED,
    });
    disposables.push(trainer);
    const result = await trainer.train({
      train: makeSource(10),
      lossSampleEvery: 3,
      epochs: 1,
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(5);
    expect(batchLosses).toHaveLength(5);
    // Fenêtre finale partielle : moyenne des deux derniers lots seulement.
    const expected = (batchLosses[3] + batchLosses[4]) / 2;
    expect(result.loss).toBeCloseTo(expected, 4);
    // Perte d'époque : moyenne pondérée par lots de toutes les fenêtres lues,
    // y compris la dernière.
    const epochMean = batchLosses.reduce((total, value) => total + value, 0) / batchLosses.length;
    expect(result.epochLoss).toBeCloseTo(epochMean, 4);
  });

  it('disposes pending cost tensors after completion and cancellation', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    await trainer.exportWeights();
    await trainer.train({ train: makeSource(20, 1), lossSampleEvery: 8, epochs: 1 });
    const baseline = tf.memory().numTensors;

    const completed = await trainer.train({
      train: makeSource(20, 2),
      lossSampleEvery: 8,
      epochs: 1,
    });
    expect(completed.status).toBe('completed');
    expect(tf.memory().numTensors).toBe(baseline);

    const cancelled = await trainer.train({
      train: makeSource(20, 3),
      lossSampleEvery: 8,
      epochs: 1,
      progressIntervalMs: 0,
      onProgress: (progress) => {
        if (progress.batchesProcessed === 1) {
          void trainer.cancel();
        }
      },
    });
    expect(cancelled.status).toBe('cancelled');
    expect(tf.memory().numTensors).toBe(baseline);
  });
});

describe('progress throttling', () => {
  it('emits one running event and one final event for a short run', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 1 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4),
      epochs: 1,
      progressIntervalMs: 60_000,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(4);
    const running = events.filter((event) => event.status === 'running');
    expect(running).toHaveLength(1);
    expect(running[0].batchesProcessed).toBe(4);
    expect(events.at(-1)?.status).toBe('completed');
    expect(events.at(-1)?.batchesProcessed).toBe(4);
  });
});

describe('epoch progress', () => {
  it('emits one epoch-end event per epoch with the epoch loss', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4),
      epochs: 3,
      progressIntervalMs: 0,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    expect(result.completedEpochs).toBe(3);
    // Avec `progressIntervalMs: 0`, un événement est aussi émis après chaque lot :
    // il porte alors `completedEpochs` au numéro de l'époque *précédente*. Seul un
    // événement de fin d'époque l'incrémente : on ne retient donc que les événements
    // qui font avancer `completedEpochs` par rapport à l'événement précédent.
    const epochEnds: MnistTrainingProgress[] = [];
    let lastCompletedEpochs = 0;
    for (const event of events) {
      if (event.status !== 'running' || event.completedEpochs <= lastCompletedEpochs) {
        continue;
      }
      epochEnds.push(event);
      lastCompletedEpochs = event.completedEpochs;
    }
    expect(epochEnds.map((event) => event.completedEpochs)).toEqual([1, 2, 3]);
    for (const event of epochEnds) {
      expect(Number.isFinite(event.epochLoss)).toBe(true);
      expect(event.epochLoss).toBeGreaterThan(0);
    }
    expect(Number.isFinite(result.epochLoss)).toBe(true);
  });

  it('the epoch-end event carries the validation accuracy of the same epoch', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS, { batchSize: 2 });
    const events: MnistTrainingProgress[] = [];
    const result = await trainer.train({
      train: makeSource(4, 1),
      validation: makeSource(6, 2),
      epochs: 2,
      progressIntervalMs: 0,
      onProgress: (progress) => events.push(progress),
    });
    expect(result.status).toBe('completed');
    const firstEpochEnd = events.find(
      (event) => event.status === 'running' && event.completedEpochs === 1,
    );
    expect(firstEpochEnd).toBeDefined();
    if (firstEpochEnd === undefined) {
      throw new Error('expected a running event for the first completed epoch');
    }
    const accuracy: number | null = firstEpochEnd.validationAccuracy;
    expect(accuracy).not.toBeNull();
    if (accuracy === null) {
      throw new Error('expected the first epoch-end event to carry a validation accuracy');
    }
    expect(accuracy).toBeGreaterThanOrEqual(0);
    expect(accuracy).toBeLessThanOrEqual(1);
    expect(result.validationAccuracy).toBe(accuracy);
  });
});

describe('reshuffle', () => {
  it('calls reshuffle once per epoch with deterministic seeds', async () => {
    const sample = makeSample(8);
    const buildSource = (): { seeds: number[]; source: MnistTrainSource } => {
      const seeds: number[] = [];
      return {
        seeds,
        source: {
          count: sample.count,
          readBatch(start, end) {
            return {
              count: end - start,
              images: sample.images.subarray(start * MNIST_INPUT_LENGTH, end * MNIST_INPUT_LENGTH),
              labels: sample.labels.subarray(start, end),
            };
          },
          reshuffle(seed) {
            seeds.push(seed);
          },
        },
      };
    };

    const first = await makeTrainer(SMALL_FILTERS, { batchSize: 2, seed: 11 });
    const firstRun = buildSource();
    const result = await first.train({ train: firstRun.source, epochs: 3 });
    expect(result.status).toBe('completed');
    expect(result.batchesProcessed).toBe(12);
    expect(firstRun.seeds).toHaveLength(3);
    expect(new Set(firstRun.seeds).size).toBe(3);

    // Même graine d'entraînement : mêmes graines d'époque.
    const second = await makeTrainer(SMALL_FILTERS, { batchSize: 2, seed: 11 });
    const secondRun = buildSource();
    await second.train({ train: secondRun.source, epochs: 3 });
    expect(secondRun.seeds).toEqual(firstRun.seeds);
  });
});

describe('configuration validation', () => {
  it('rejects invalid filter counts', async () => {
    await expect(makeTrainer([0, 1, 1, 1])).rejects.toThrow(/filter count/i);
    await expect(makeTrainer([1, 1, 1, 9])).rejects.toThrow(/filter count/i);
  });

  it('rejects invalid denseSizes', async () => {
    await expect(makeTrainer(SMALL_FILTERS, { denseSizes: [129] })).rejects.toThrow(
      /hidden dense size/i,
    );
    await expect(makeTrainer(SMALL_FILTERS, { denseSizes: [0] })).rejects.toThrow(
      /hidden dense size/i,
    );
    await expect(makeTrainer(SMALL_FILTERS, { denseSizes: [1.5] })).rejects.toThrow(
      /hidden dense size/i,
    );
    await expect(makeTrainer(SMALL_FILTERS, { denseSizes: [8, 8, 8] })).rejects.toThrow(
      /at most 2/,
    );
  });

  it('rejects invalid learning rate, batch size and seed', async () => {
    await expect(makeTrainer(SMALL_FILTERS, { learningRate: 0 })).rejects.toThrow(/learning rate/i);
    await expect(makeTrainer(SMALL_FILTERS, { batchSize: 0 })).rejects.toThrow(/batch size/i);
    await expect(makeTrainer(SMALL_FILTERS, { seed: -1 })).rejects.toThrow(/seed/i);
  });

  it('rejects an unknown activation', async () => {
    await expect(makeTrainer(SMALL_FILTERS, { activation: 'relu' as never })).rejects.toThrow(
      /activation/i,
    );
  });

  it('rejects an invalid prefetchDepth', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({ train: makeSource(2), prefetchDepth: MIN_PREFETCH_DEPTH - 1 }),
    ).rejects.toThrow(/prefetchDepth/i);
    await expect(
      trainer.train({ train: makeSource(2), prefetchDepth: MAX_PREFETCH_DEPTH + 1 }),
    ).rejects.toThrow(/prefetchDepth/i);
    await expect(trainer.train({ train: makeSource(2), prefetchDepth: 1.5 })).rejects.toThrow(
      /prefetchDepth/i,
    );
  });

  it('rejects invalid lossSampleEvery and progressIntervalMs', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({ train: makeSource(2), lossSampleEvery: MIN_LOSS_SAMPLE_EVERY - 1 }),
    ).rejects.toThrow(/lossSampleEvery/i);
    await expect(
      trainer.train({ train: makeSource(2), lossSampleEvery: MAX_LOSS_SAMPLE_EVERY + 1 }),
    ).rejects.toThrow(/lossSampleEvery/i);
    await expect(trainer.train({ train: makeSource(2), lossSampleEvery: 2.5 })).rejects.toThrow(
      /lossSampleEvery/i,
    );
    await expect(trainer.train({ train: makeSource(2), progressIntervalMs: -1 })).rejects.toThrow(
      /progressIntervalMs/i,
    );
    await expect(
      trainer.train({ train: makeSource(2), progressIntervalMs: Number.NaN }),
    ).rejects.toThrow(/progressIntervalMs/i);
  });

  it('rejects malformed batches and epochs', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({
        train: sourceOfBatch({ count: 2, images: new Float32Array(10), labels: new Uint8Array(2) }),
      }),
    ).rejects.toThrow(/images/i);
    await expect(
      trainer.train({
        train: sourceOfBatch({
          count: 2,
          images: new Float32Array(2 * MNIST_INPUT_LENGTH),
          labels: new Uint8Array(1),
        }),
      }),
    ).rejects.toThrow(/labels/i);
    await expect(trainer.train({ train: makeSource(2), epochs: 0 })).rejects.toThrow(/epochs/i);
  });

  it('rejects out-of-range labels and non-finite pixels', async () => {
    const trainer = await makeTrainer(SMALL_FILTERS);
    await expect(
      trainer.train({
        train: sourceOfBatch({
          count: 2,
          images: new Float32Array(2 * MNIST_INPUT_LENGTH),
          labels: new Uint8Array([1, 12]),
        }),
      }),
    ).rejects.toThrow(/labels/i);

    const notANumber = new Float32Array(2 * MNIST_INPUT_LENGTH);
    notANumber[3] = Number.NaN;
    await expect(
      trainer.train({
        train: sourceOfBatch({ count: 2, images: notANumber, labels: new Uint8Array([1, 2]) }),
      }),
    ).rejects.toThrow(/images/i);

    const infinite = new Float32Array(2 * MNIST_INPUT_LENGTH);
    infinite[7] = Number.POSITIVE_INFINITY;
    await expect(
      trainer.predict({ count: 2, images: infinite, labels: new Uint8Array([1, 2]) }),
    ).rejects.toThrow(/images/i);
  });
});
