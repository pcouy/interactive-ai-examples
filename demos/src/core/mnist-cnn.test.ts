import { describe, expect, it } from 'vitest';
import {
  activationDomain,
  applyMnistActivation,
  CONV_SCHEDULES,
  contributingTaps,
  convKernelIndex,
  convLayerName,
  convOutputLength,
  DEFAULT_FILTER_COUNTS,
  denseLayerWidths,
  forwardMnistCnn,
  HIDDEN_DENSE_SIZE,
  initializeMnistCnnWeights,
  LOGIT_COUNT,
  MAX_CONV_LAYERS,
  MAX_DENSE_LAYERS,
  MNIST_INPUT_LENGTH,
  MNIST_INPUT_SIZE,
  type MnistCnnConfig,
  type MnistCnnFilters,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
  mnistCnnParameterCount,
  softmax,
  validateDenseSizes,
  withKernelWeight,
} from './mnist-cnn.js';

function sigmoidOf(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function configWith(filters: MnistCnnFilters): MnistCnnConfig {
  return { filters };
}

function constantInput(value: number): Float32Array {
  const input = new Float32Array(MNIST_INPUT_LENGTH);
  input.fill(value);
  return input;
}

function rampInput(): Float32Array {
  const input = new Float32Array(MNIST_INPUT_LENGTH);
  for (let index = 0; index < input.length; index += 1) {
    input[index] = ((index % 7) + 1) / 7;
  }
  return input;
}

/** Copie les poids en remplaçant une couche de convolution. */
function replaceConv(
  weights: MnistCnnWeights,
  layer: number,
  next: MnistCnnLayerWeights,
): MnistCnnWeights {
  const conv = weights.conv.slice();
  conv[layer] = next;
  return { ...weights, conv };
}

/** Copie les poids en remplaçant une couche dense. */
function replaceDense(
  weights: MnistCnnWeights,
  layer: number,
  next: MnistCnnLayerWeights,
): MnistCnnWeights {
  const dense = weights.dense.slice();
  dense[layer] = next;
  return { ...weights, dense };
}

describe('activation', () => {
  it('applies sigmoid or tanh to a pre-activation', () => {
    expect(applyMnistActivation(0, 'sigmoid')).toBe(0.5);
    expect(applyMnistActivation(0.5, 'tanh')).toBe(Math.tanh(0.5));
  });

  it('reports the output domain of each activation', () => {
    expect(activationDomain('sigmoid')).toEqual({ min: 0, max: 1 });
    expect(activationDomain('tanh')).toEqual({ min: -1, max: 1 });
  });
});

describe('convolution schedules', () => {
  it('exposes the expected strides and paddings per layer count', () => {
    expect(CONV_SCHEDULES[0]).toEqual([]);
    expect(CONV_SCHEDULES[1]).toEqual([{ stride: 2, padding: 1 }]);
    expect(CONV_SCHEDULES[4]).toEqual([
      { stride: 1, padding: 0 },
      { stride: 2, padding: 1 },
      { stride: 1, padding: 0 },
      { stride: 2, padding: 0 },
    ]);
    expect(convLayerName(0)).toBe('conv1');
    expect(convLayerName(3)).toBe('conv4');
  });

  it('derives feature-map sizes and names from the layer count', () => {
    const expectedSizes: Record<number, readonly number[]> = {
      0: [],
      1: [14],
      2: [14, 6],
      3: [26, 13, 6],
      4: [26, 13, 11, 5],
    };
    for (const [count, sizes] of Object.entries(expectedSizes)) {
      const config: MnistCnnConfig = {
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: Number(count),
      };
      const weights = initializeMnistCnnWeights(config, 7);
      const output = forwardMnistCnn(rampInput(), weights, config);
      expect(output.featureMaps.map((map) => map.height)).toEqual(sizes);
      expect(output.featureMaps.map((map) => map.name)).toEqual(
        sizes.map((_, index) => `conv${index + 1}`),
      );
    }
  });
});

describe('forwardMnistCnn activation selection', () => {
  it('rejects an unknown activation', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 7);
    expect(() =>
      forwardMnistCnn(rampInput(), weights, {
        filters: DEFAULT_FILTER_COUNTS,
        activation: 'relu' as never,
      }),
    ).toThrow(/activation/i);
  });

  it('runs the tanh forward pass between -1 and 1', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 7);
    const output = forwardMnistCnn(rampInput(), weights, {
      filters: DEFAULT_FILTER_COUNTS,
      activation: 'tanh',
    });
    for (const map of output.featureMaps) {
      for (let index = 0; index < map.values.length; index += 1) {
        expect(map.values[index]).toBeGreaterThanOrEqual(-1);
        expect(map.values[index]).toBeLessThanOrEqual(1);
      }
    }
    const hidden = output.denseActivations[0];
    for (let index = 0; index < HIDDEN_DENSE_SIZE; index += 1) {
      expect(hidden[index]).toBeGreaterThanOrEqual(-1);
      expect(hidden[index]).toBeLessThanOrEqual(1);
    }
    expect(Array.from(output.featureMaps[0].values).some((value) => value < 0)).toBe(true);
    expect(Array.from(hidden).some((value) => value < 0)).toBe(true);
    let sum = 0;
    for (let index = 0; index < output.probabilities.length; index += 1) {
      expect(Number.isFinite(output.probabilities[index])).toBe(true);
      sum += output.probabilities[index];
    }
    expect(sum).toBeCloseTo(1, 5);
  });

  it('keeps sigmoid as the default activation', () => {
    const filters = DEFAULT_FILTER_COUNTS;
    const weights = initializeMnistCnnWeights(configWith(filters), 7);
    const input = rampInput();
    const implicit = forwardMnistCnn(input, weights, { filters });
    const explicit = forwardMnistCnn(input, weights, { filters, activation: 'sigmoid' });
    expect(implicit).toEqual(explicit);
  });
});

describe('convOutputLength', () => {
  it('matches the reference stride and padding formula', () => {
    expect(convOutputLength(28, 1, 0)).toBe(26);
    expect(convOutputLength(26, 2, 1)).toBe(13);
    expect(convOutputLength(13, 1, 0)).toBe(11);
    expect(convOutputLength(11, 2, 0)).toBe(5);
  });
});

describe('initializeMnistCnnWeights', () => {
  it('allocates default-shaped arrays for 980 parameters', () => {
    expect(DEFAULT_FILTER_COUNTS).toEqual([3, 3, 3, 3]);
    const weights = initializeMnistCnnWeights(configWith(DEFAULT_FILTER_COUNTS), 42);
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
    expect(mnistCnnParameterCount()).toBe(980);
  });

  it('sizes each dense layer to its width for custom counts', () => {
    const config: MnistCnnConfig = {
      filters: DEFAULT_FILTER_COUNTS,
      convLayers: 3,
      denseLayers: 3,
    };
    const weights = initializeMnistCnnWeights(config, 42);
    expect(weights.conv).toHaveLength(3);
    expect(weights.dense).toHaveLength(3);
    // Trois convolutions : 28 -> 26 -> 13 -> 6, donc 6·6·3 = 108 caractéristiques.
    expect(weights.dense[0].kernel).toHaveLength(6 * 6 * 3 * HIDDEN_DENSE_SIZE);
    expect(weights.dense[0].bias).toHaveLength(HIDDEN_DENSE_SIZE);
    expect(weights.dense[1].kernel).toHaveLength(HIDDEN_DENSE_SIZE * HIDDEN_DENSE_SIZE);
    expect(weights.dense[1].bias).toHaveLength(HIDDEN_DENSE_SIZE);
    expect(weights.dense[2].kernel).toHaveLength(HIDDEN_DENSE_SIZE * LOGIT_COUNT);
    expect(weights.dense[2].bias).toHaveLength(LOGIT_COUNT);
    expect(denseLayerWidths(config)).toEqual([HIDDEN_DENSE_SIZE, HIDDEN_DENSE_SIZE, LOGIT_COUNT]);
  });

  it('counts parameters for several layer counts', () => {
    // 4 conv + 2 dense : conv 30 + 84 + 84 + 84 = 282 ; dense 75·8+8 + 8·10+10 = 698.
    expect(
      mnistCnnParameterCount({
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: 4,
        denseLayers: 2,
      }),
    ).toBe(980);
    // 0 conv + 1 dense : entrée brute 784, sortie 10 -> 784·10 + 10 = 7850.
    expect(
      mnistCnnParameterCount({
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: 0,
        denseLayers: 1,
      }),
    ).toBe(7850);
    // 2 conv + 3 dense : conv 30 + 84 = 114 ; dense 108·8+8 + 8·8+8 + 8·10+10 = 1034.
    expect(
      mnistCnnParameterCount({
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: 2,
        denseLayers: 3,
      }),
    ).toBe(1148);
    expect(mnistCnnParameterCount({ filters: [1, 2, 3, 4] })).toBe(1097);
  });

  it('reproduces the seed-7 default weights snapshot', () => {
    const weights = initializeMnistCnnWeights(configWith(DEFAULT_FILTER_COUNTS), 7);
    // Valeurs enregistrées avec l'implémentation avant refonte ; la séquence
    // pseudo-aléatoire et les mathématiques d'initialisation sont inchangées.
    expect(Array.from(weights.conv[0].kernel.slice(0, 5))).toEqual([
      -0.6905537843704224, -0.619484543800354, 0.6744492650032043, 0.28146910667419434,
      0.030328189954161644,
    ]);
    expect(Array.from(weights.conv[1].kernel.slice(0, 5))).toEqual([
      0.23422695696353912, -0.38056260347366333, 0.36934539675712585, 0.04804035276174545,
      0.10878455638885498,
    ]);
    expect(Array.from(weights.conv[2].kernel.slice(0, 5))).toEqual([
      -0.32283082604408264, 0.44202467799186707, 0.0061998688615858555, 0.05982815846800804,
      -0.3033161461353302,
    ]);
    expect(Array.from(weights.conv[3].kernel.slice(0, 5))).toEqual([
      -0.30197227001190186, 0.1728944629430771, -0.09766900539398193, 0.09812961518764496,
      0.31844159960746765,
    ]);
    expect(Array.from(weights.dense[0].kernel.slice(0, 5))).toEqual([
      -0.033796075731515884, 0.02380395494401455, -0.17852473258972168, 0.14641030132770538,
      -0.10582263022661209,
    ]);
    expect(Array.from(weights.dense[1].kernel.slice(0, 5))).toEqual([
      0.4957295358181, -0.39072781801223755, 0.07592640072107315, -0.492308109998703,
      -0.1508914828300476,
    ]);
    expect(Array.from(weights.conv[0].bias)).toEqual([0, 0, 0]);
    expect(Array.from(weights.dense[0].bias)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(weights.dense[1].bias)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('is deterministic for equal seeds and differs for different seeds', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    expect(initializeMnistCnnWeights(config, 7)).toEqual(initializeMnistCnnWeights(config, 7));
    const custom: MnistCnnFilters = [1, 2, 3, 4];
    expect(initializeMnistCnnWeights(configWith(custom), 7)).toEqual(
      initializeMnistCnnWeights(configWith(custom), 7),
    );
    const a = initializeMnistCnnWeights(config, 7);
    const b = initializeMnistCnnWeights(config, 8);
    expect(a).not.toEqual(b);
    expect(
      Array.from(a.conv[0].kernel).some((value, index) => value !== b.conv[0].kernel[index]),
    ).toBe(true);
  });

  it('rejects invalid filter counts', () => {
    expect(() => initializeMnistCnnWeights(configWith([0, 3, 3, 3]), 1)).toThrow(/filter count/i);
    expect(() => initializeMnistCnnWeights(configWith([3, 3, 3, 5]), 1)).toThrow(/filter count/i);
    expect(() => initializeMnistCnnWeights(configWith([2.5, 3, 3, 3]), 1)).toThrow(/integer/i);
    expect(() =>
      initializeMnistCnnWeights(configWith([3, 3, 3] as unknown as MnistCnnFilters), 1),
    ).toThrow(/four-item/i);
    expect(() => initializeMnistCnnWeights(undefined as unknown as MnistCnnConfig, 1)).toThrow(
      TypeError,
    );
  });

  it('rejects layer counts outside the supported ranges', () => {
    const base = configWith(DEFAULT_FILTER_COUNTS);
    expect(() =>
      initializeMnistCnnWeights({ ...base, convLayers: MAX_CONV_LAYERS + 1 }, 1),
    ).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, convLayers: -1 }, 1)).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, convLayers: 2.5 }, 1)).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, denseLayers: 0 }, 1)).toThrow(RangeError);
    expect(() =>
      initializeMnistCnnWeights({ ...base, denseLayers: MAX_DENSE_LAYERS + 1 }, 1),
    ).toThrow(RangeError);
    expect(() => mnistCnnParameterCount({ ...base, convLayers: 5 })).toThrow(RangeError);
    expect(() => denseLayerWidths({ ...base, denseLayers: 0 })).toThrow(RangeError);
    expect(() => denseLayerWidths({ ...base, denseLayers: MAX_DENSE_LAYERS + 1 })).toThrow(
      RangeError,
    );
  });

  it('derives the dense widths from denseSizes, with 8 as the padded default', () => {
    const filters = DEFAULT_FILTER_COUNTS;
    expect(denseLayerWidths({ filters, denseLayers: 3, denseSizes: [3, 7] })).toEqual([
      3,
      7,
      LOGIT_COUNT,
    ]);
    expect(denseLayerWidths({ filters, denseLayers: 2 })).toEqual([HIDDEN_DENSE_SIZE, LOGIT_COUNT]);
    expect(denseLayerWidths({ filters, denseLayers: 3, denseSizes: [32] })).toEqual([
      32,
      HIDDEN_DENSE_SIZE,
      LOGIT_COUNT,
    ]);
  });

  it('sizes the dense layers and counts parameters for custom denseSizes', () => {
    const config: MnistCnnConfig = {
      filters: DEFAULT_FILTER_COUNTS,
      convLayers: 4,
      denseLayers: 3,
      denseSizes: [3, 7],
    };
    const weights = initializeMnistCnnWeights(config, 11);
    expect(weights.dense.map((layer) => layer.bias.length)).toEqual([3, 7, LOGIT_COUNT]);
    for (const layer of weights.dense) {
      expect(Array.from(layer.bias)).toEqual(new Array<number>(layer.bias.length).fill(0));
    }
    // 4 conv + 3 dense : conv 30 + 84 + 84 + 84 = 282 ;
    // dense 75·3+3 + 3·7+7 + 7·10+10 = 228 + 28 + 80 = 336 ; total 618.
    expect(mnistCnnParameterCount(config)).toBe(618);
    // 4 conv + 2 dense, hidden 4 : conv 282 ; dense 75·4+4 + 4·10+10 = 354 ; total 636.
    expect(
      mnistCnnParameterCount({
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: 4,
        denseLayers: 2,
        denseSizes: [4],
      }),
    ).toBe(636);
  });

  it('runs forwardMnistCnn with custom denseSizes', () => {
    const config: MnistCnnConfig = {
      filters: DEFAULT_FILTER_COUNTS,
      denseLayers: 3,
      denseSizes: [3, 7],
    };
    const weights = initializeMnistCnnWeights(config, 13);
    const output = forwardMnistCnn(rampInput(), weights, config);
    expect(output.denseActivations.map((activation) => activation.length)).toEqual([
      3,
      7,
      LOGIT_COUNT,
    ]);
    expect(output.logits).toHaveLength(LOGIT_COUNT);
  });

  it('rejects invalid denseSizes', () => {
    const base = { filters: DEFAULT_FILTER_COUNTS };
    expect(() => initializeMnistCnnWeights({ ...base, denseSizes: [0] }, 1)).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, denseSizes: [33] }, 1)).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, denseSizes: [1.5] }, 1)).toThrow(RangeError);
    expect(() => initializeMnistCnnWeights({ ...base, denseSizes: [8, 8, 8] }, 1)).toThrow(
      RangeError,
    );
    expect(() =>
      forwardMnistCnn(rampInput(), initializeMnistCnnWeights({ ...base }, 1), {
        ...base,
        denseSizes: [0],
      }),
    ).toThrow(RangeError);
    expect(() => validateDenseSizes([0])).toThrow(/hidden dense size/i);
    expect(() => validateDenseSizes([33])).toThrow(/hidden dense size/i);
    expect(() => validateDenseSizes([8, 8, 8])).toThrow(/at most 2/);
    expect(() => validateDenseSizes(undefined)).not.toThrow();
  });
});

describe('forwardMnistCnn shapes', () => {
  it('produces the reference shapes for the default architecture', () => {
    const weights = initializeMnistCnnWeights(configWith(DEFAULT_FILTER_COUNTS), 42);
    const output = forwardMnistCnn(rampInput(), weights, configWith(DEFAULT_FILTER_COUNTS));
    expect(
      output.featureMaps.map((map) => [
        map.name,
        map.height,
        map.width,
        map.channels,
        map.values.length,
      ]),
    ).toEqual([
      ['conv1', 26, 26, 3, 26 * 26 * 3],
      ['conv2', 13, 13, 3, 13 * 13 * 3],
      ['conv3', 11, 11, 3, 11 * 11 * 3],
      ['conv4', 5, 5, 3, 5 * 5 * 3],
    ]);
    expect(output.denseActivations).toHaveLength(2);
    expect(output.denseActivations[0]).toHaveLength(8);
    expect(output.logits).toHaveLength(10);
    expect(output.probabilities).toHaveLength(10);
  });

  it('scales channel counts with the filter tuple', () => {
    const filters: MnistCnnFilters = [1, 2, 3, 4];
    const weights = initializeMnistCnnWeights(configWith(filters), 42);
    const output = forwardMnistCnn(constantInput(0.5), weights, configWith(filters));
    expect(
      output.featureMaps.map((map) => [map.height, map.width, map.channels, map.values.length]),
    ).toEqual([
      [26, 26, 1, 26 * 26 * 1],
      [13, 13, 2, 13 * 13 * 2],
      [11, 11, 3, 11 * 11 * 3],
      [5, 5, 4, 5 * 5 * 4],
    ]);
  });

  it('keeps activations in the sigmoid range and probabilities on the simplex', () => {
    const weights = initializeMnistCnnWeights(configWith(DEFAULT_FILTER_COUNTS), 7);
    const output = forwardMnistCnn(rampInput(), weights, configWith(DEFAULT_FILTER_COUNTS));
    for (const map of output.featureMaps) {
      let min = Infinity;
      let max = -Infinity;
      for (let index = 0; index < map.values.length; index += 1) {
        min = Math.min(min, map.values[index]);
        max = Math.max(max, map.values[index]);
      }
      expect(min).toBeGreaterThanOrEqual(0);
      expect(max).toBeLessThanOrEqual(1);
    }
    let sum = 0;
    for (let index = 0; index < output.probabilities.length; index += 1) {
      const probability = output.probabilities[index];
      expect(Number.isFinite(probability)).toBe(true);
      expect(probability).toBeGreaterThanOrEqual(0);
      sum += probability;
    }
    expect(sum).toBeCloseTo(1, 5);
    const direct = softmax(output.logits);
    for (let index = 0; index < LOGIT_COUNT; index += 1) {
      expect(output.probabilities[index]).toBeCloseTo(direct[index], 12);
    }
    for (let index = 0; index < HIDDEN_DENSE_SIZE; index += 1) {
      expect(Number.isFinite(output.denseActivations[0][index])).toBe(true);
      expect(output.denseActivations[0][index]).toBeGreaterThan(0);
      expect(output.denseActivations[0][index]).toBeLessThan(1);
    }
  });

  it('exposes every dense activation with the softmax last', () => {
    const cases = [
      { convLayers: 0, denseLayers: 1, featureMapCount: 0, denseActivationCount: 1 },
      { convLayers: 1, denseLayers: 3, featureMapCount: 1, denseActivationCount: 3 },
      { convLayers: 3, denseLayers: 2, featureMapCount: 3, denseActivationCount: 2 },
    ];
    for (const testCase of cases) {
      const config: MnistCnnConfig = {
        filters: DEFAULT_FILTER_COUNTS,
        convLayers: testCase.convLayers,
        denseLayers: testCase.denseLayers,
      };
      const weights = initializeMnistCnnWeights(config, 11);
      const output = forwardMnistCnn(rampInput(), weights, config);
      expect(output.featureMaps).toHaveLength(testCase.featureMapCount);
      expect(output.denseActivations).toHaveLength(testCase.denseActivationCount);
      expect(output.probabilities).toHaveLength(LOGIT_COUNT);
      let sum = 0;
      for (const probability of output.probabilities) {
        expect(Number.isFinite(probability)).toBe(true);
        sum += probability;
      }
      expect(sum).toBeCloseTo(1, 5);
      const last = output.denseActivations[output.denseActivations.length - 1];
      expect(Array.from(last)).toEqual(Array.from(output.probabilities));
    }
  });
});

describe('convolution semantics', () => {
  it('applies sigmoid to a 3x3 no-padding convolution (conv1 center tap)', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[4] = 1; // [kernelY=1][kernelX=1][cIn=0][cOut=0]
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[0];
    expect(map.name).toBe('conv1');
    for (let y = 0; y < 26; y += 1) {
      for (let x = 0; x < 26; x += 1) {
        // sans remplissage, le tap central de la sortie (y, x) lit l'entrée (y+1, x+1).
        const expected = sigmoidOf(input[(y + 1) * MNIST_INPUT_SIZE + (x + 1)]);
        expect(map.values[y * 26 + x]).toBeCloseTo(expected, 6);
      }
    }
  });

  it('aligns stride-2 conv2 with symmetric zero padding of 1', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[4] = 1;
    weights.conv[1].kernel.fill(0);
    weights.conv[1].kernel[4] = 1;
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[1];
    expect([map.height, map.width]).toEqual([13, 13]);
    for (let oy = 0; oy < 13; oy += 1) {
      for (let ox = 0; ox < 13; ox += 1) {
        // padding 1 : la sortie (oy, ox) échantillonne conv1 en (2·oy, 2·ox),
        // dont le tap central lit l'entrée (2·oy + 1, 2·ox + 1).
        const expected = sigmoidOf(
          sigmoidOf(input[(2 * oy + 1) * MNIST_INPUT_SIZE + (2 * ox + 1)]),
        );
        expect(map.values[oy * 13 + ox]).toBeCloseTo(expected, 6);
      }
    }
  });

  it('counts padded border pixels as zeros in conv2 windows', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[4] = 1;
    weights.conv[1].kernel.fill(1);
    const output = forwardMnistCnn(constantInput(1), weights, configWith(filters));
    const u = sigmoidOf(1);
    const map = output.featureMaps[1];
    expect(map.values[0]).toBeCloseTo(sigmoidOf(4 * u), 6); // coin : 4 pixels valides
    expect(map.values[1]).toBeCloseTo(sigmoidOf(6 * u), 6); // bord (0,1) : 6 pixels valides
    expect(map.values[7 * 13 + 7]).toBeCloseTo(sigmoidOf(9 * u), 6); // centre : 9 pixels valides
  });

  it('keeps stride alignment through all four layers', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    for (const layer of weights.conv) {
      layer.kernel.fill(0);
      layer.kernel[4] = 1;
      layer.bias.fill(0);
    }
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[3];
    expect([map.height, map.width]).toEqual([5, 5]);
    for (let oy = 0; oy < 5; oy += 1) {
      for (let ox = 0; ox < 5; ox += 1) {
        // conv4 (stride 2) -> conv3 -> conv2 (stride 2, padding 1) -> conv1 :
        // la sortie (oy, ox) remonte au pixel d'entrée (4·oy + 5, 4·ox + 5).
        const y = 4 * oy + 5;
        const x = 4 * ox + 5;
        let expected = input[y * MNIST_INPUT_SIZE + x];
        for (let depth = 0; depth < 4; depth += 1) {
          expected = sigmoidOf(expected);
        }
        expect(map.values[oy * 5 + ox]).toBeCloseTo(expected, 6);
      }
    }
  });

  it('reads an off-center tap at the correct input coordinates (ky is the row, kx is the column)', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    // Tap [ky=0][kx=1] (haut droite) : la sortie (y, x) lit l'entrée (y, x+1).
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[1] = 1;
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[0];
    expect(map.values).toHaveLength(26 * 26);
    for (let y = 0; y < 26; y += 1) {
      for (let x = 0; x < 26; x += 1) {
        expect(map.values[y * 26 + x]).toBeCloseTo(
          sigmoidOf(input[y * MNIST_INPUT_SIZE + (x + 1)]),
          6,
        );
      }
    }
  });

  it('treats an off-center tap that falls on the conv2 border as zero', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    // Tap [ky=0][kx=1] sur conv1 et conv2.
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[1] = 1;
    weights.conv[1].kernel.fill(0);
    weights.conv[1].kernel[1] = 1;
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[1];
    expect([map.height, map.width]).toEqual([13, 13]);
    for (let ox = 0; ox < 13; ox += 1) {
      // oy=0 : le tap ky=0 lit la ligne 2·0−1 = −1, hors carte -> zéro.
      expect(map.values[ox]).toBeCloseTo(0.5, 6);
    }
    for (let oy = 1; oy < 13; oy += 1) {
      for (let ox = 0; ox < 13; ox += 1) {
        // conv2 (oy, ox) lit conv1 en (2·oy−1, 2·ox), dont le tap lit l'entrée (2·oy−1, 2·ox+1).
        const expected = sigmoidOf(
          sigmoidOf(input[(2 * oy - 1) * MNIST_INPUT_SIZE + (2 * ox + 1)]),
        );
        expect(map.values[oy * 13 + ox]).toBeCloseTo(expected, 6);
      }
    }
  });

  it('treats a kx=0 tap on the conv2 left border as zero', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    // conv1 : tap central, la sortie (y, x) lit l'entrée (y+1, x+1).
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[4] = 1;
    // conv2 : tap [ky=1][kx=0] (milieu gauche), la sortie (oy, ox) lit conv1 en (2·oy, 2·ox−1).
    weights.conv[1].kernel.fill(0);
    weights.conv[1].kernel[3] = 1;
    const input = rampInput();
    const output = forwardMnistCnn(input, weights, configWith(filters));
    const map = output.featureMaps[1];
    expect([map.height, map.width]).toEqual([13, 13]);
    for (let oy = 0; oy < 13; oy += 1) {
      // ox=0 : le tap kx=0 lit la colonne 2·0−1 = −1, hors carte -> zéro.
      expect(map.values[oy * 13]).toBeCloseTo(0.5, 6);
    }
    for (let oy = 0; oy < 13; oy += 1) {
      for (let ox = 1; ox < 13; ox += 1) {
        // conv2 (oy, ox) lit conv1 en (2·oy, 2·ox−1), dont le tap lit l'entrée (2·oy+1, 2·ox).
        const expected = sigmoidOf(sigmoidOf(input[(2 * oy + 1) * MNIST_INPUT_SIZE + 2 * ox]));
        expect(map.values[oy * 13 + ox]).toBeCloseTo(expected, 6);
      }
    }
  });

  it('reads convolution kernels in HWIO order', () => {
    const filters: MnistCnnFilters = [2, 2, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    // conv1 : tap central identique pour les deux canaux de sortie.
    weights.conv[0].kernel.fill(0);
    weights.conv[0].kernel[8] = 1; // [1][1][cIn=0][cOut=0]
    weights.conv[0].kernel[9] = 1; // [1][1][cIn=0][cOut=1]
    // conv2 : deux taps distincts par couple (cIn, cOut).
    weights.conv[1].kernel.fill(0);
    weights.conv[1].kernel[18] = 2; // ((1·3+1)·2 + 1)·2 + 0 -> [1][1][cIn=1][cOut=0]
    weights.conv[1].kernel[17] = 3; // ((1·3+1)·2 + 0)·2 + 1 -> [1][1][cIn=0][cOut=1]
    const output = forwardMnistCnn(constantInput(1), weights, configWith(filters));
    const u = sigmoidOf(1);
    const map = output.featureMaps[1];
    expect(map.channels).toBe(2);
    expect(map.values[0]).toBeCloseTo(sigmoidOf(2 * u), 6); // cOut=0 reçoit le tap [cIn=1]
    expect(map.values[1]).toBeCloseTo(sigmoidOf(3 * u), 6); // cOut=1 reçoit le tap [cIn=0]
  });

  it('flattens the final 5x5xF4 maps in NHWC order with channels fastest', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 3];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    for (const layer of weights.conv) {
      layer.kernel.fill(0);
      layer.bias.fill(0);
    }
    weights.conv[3].bias[2] = 20; // canal 2 : carte constante ≈ 1 ; canaux 0 et 1 : 0.5
    weights.dense[0].kernel.fill(0);
    weights.dense[0].kernel[2 * HIDDEN_DENSE_SIZE + 0] = 1; // caractéristique NHWC 2 -> (y=0, x=0, c=2)
    weights.dense[0].kernel[26 * HIDDEN_DENSE_SIZE + 2] = 1; // caractéristique NHWC 26 -> (y=1, x=3, c=2)
    const output = forwardMnistCnn(constantInput(1), weights, configWith(filters));
    const nearOne = sigmoidOf(20);
    const hidden = output.denseActivations[0];
    expect(hidden[0]).toBeCloseTo(sigmoidOf(nearOne), 5);
    expect(hidden[2]).toBeCloseTo(sigmoidOf(nearOne), 5);
    for (const index of [1, 3, 4, 5, 6, 7]) {
      expect(hidden[index]).toBe(0.5); // sigmoid(0)
    }
  });
});

describe('convKernelIndex', () => {
  it('returns the expected HWIO flat offset', () => {
    expect(convKernelIndex(1, 3, 0, 0, 0, 0)).toBe(0);
    expect(convKernelIndex(1, 3, 1, 1, 0, 2)).toBe(14);
    expect(convKernelIndex(3, 3, 1, 1, 2, 1)).toBe(43);
    expect(convKernelIndex(2, 4, 2, 2, 1, 3)).toBe(71);
  });

  it('rejects out-of-range indices and non-integer channel counts', () => {
    expect(() => convKernelIndex(1, 3, 3, 0, 0, 0)).toThrow(RangeError);
    expect(() => convKernelIndex(1, 3, 3, 0, 0, 0)).toThrow(/ky/);
    expect(() => convKernelIndex(1, 3, -1, 0, 0, 0)).toThrow(/ky/);
    expect(() => convKernelIndex(1, 3, 1.5, 0, 0, 0)).toThrow(/integer/i);
    expect(() => convKernelIndex(1, 3, 0, 3, 0, 0)).toThrow(/kx/);
    expect(() => convKernelIndex(1, 3, 0, 0, 1, 0)).toThrow(/ic/);
    expect(() => convKernelIndex(1, 3, 0, 0, 0, 3)).toThrow(/oc/);
    expect(() => convKernelIndex(0, 3, 0, 0, 0, 0)).toThrow(/inputChannels/);
    expect(() => convKernelIndex(-1, 3, 0, 0, 0, 0)).toThrow(/inputChannels/);
    expect(() => convKernelIndex(1, 0, 0, 0, 0, 0)).toThrow(/outputChannels/);
    expect(() => convKernelIndex(2.5, 3, 0, 0, 0, 0)).toThrow(/integer/i);
    expect(() => convKernelIndex(undefined as unknown as number, 3, 0, 0, 0, 0)).toThrow(TypeError);
  });
});

describe('withKernelWeight', () => {
  it('writes the value at the right flat index and leaves the input untouched', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 5);
    const before = Float32Array.from(weights.conv[1].kernel);
    const result = withKernelWeight(weights, config, 1, 2, 1, 2, 0, 0.25);
    // ((2·3+0)·3 + 2)·3 + 1 -> 61
    expect(convKernelIndex(3, 3, 2, 0, 2, 1)).toBe(61);
    expect(result.conv[1].kernel[61]).toBe(0.25);
    expect(result.conv[1].kernel).toHaveLength(weights.conv[1].kernel.length);
    for (let index = 0; index < before.length; index += 1) {
      if (index === 61) {
        continue;
      }
      expect(result.conv[1].kernel[index]).toBe(before[index]);
    }
    expect(weights.conv[1].kernel[61]).toBe(before[61]);
    expect(weights.conv[1].kernel).not.toBe(result.conv[1].kernel);
  });

  it('returns a new object and a new kernel only for the touched layer', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 5);
    const result = withKernelWeight(weights, config, 0, 0, 0, 0, 0, 1);
    expect(result).not.toBe(weights);
    expect(result.conv[0]).not.toBe(weights.conv[0]);
    expect(result.conv[0].kernel).not.toBe(weights.conv[0].kernel);
    expect(result.conv[0].bias).toBe(weights.conv[0].bias);
    expect(result.conv[1]).toBe(weights.conv[1]);
    expect(result.conv[1].kernel).toBe(weights.conv[1].kernel);
    expect(result.conv[1].bias).toBe(weights.conv[1].bias);
    expect(result.conv[2]).toBe(weights.conv[2]);
    expect(result.conv[2].bias).toBe(weights.conv[2].bias);
    expect(result.conv[3]).toBe(weights.conv[3]);
    expect(result.conv[3].bias).toBe(weights.conv[3].bias);
    expect(result.dense).toBe(weights.dense);
    expect(result.dense[0]).toBe(weights.dense[0]);
    expect(result.dense[1]).toBe(weights.dense[1]);
    expect(result.dense[1].bias).toBe(weights.dense[1].bias);
  });

  it('rejects a bad layer, bad indices and a non-finite value', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 5);
    expect(() => withKernelWeight(weights, config, 4, 0, 0, 0, 0, 1)).toThrow(/layer/);
    expect(() => withKernelWeight(weights, config, -1, 0, 0, 0, 0, 1)).toThrow(/layer/);
    expect(() =>
      withKernelWeight(weights, config, undefined as unknown as number, 0, 0, 0, 0, 1),
    ).toThrow(TypeError);
    expect(() => withKernelWeight(weights, config, 1.5, 0, 0, 0, 0, 1)).toThrow(/layer/);
    expect(() => withKernelWeight(weights, config, 1, 3, 0, 0, 0, 1)).toThrow(/ic/);
    expect(() => withKernelWeight(weights, config, 1, 0, 3, 0, 0, 1)).toThrow(/oc/);
    expect(() => withKernelWeight(weights, config, 1, 0, 0, 3, 0, 1)).toThrow(/ky/);
    expect(() => withKernelWeight(weights, config, 1, 0, 0, 0, 0, Number.NaN)).toThrow(/finite/i);
    expect(() =>
      withKernelWeight(weights, config, 1, 0, 0, 0, 0, Number.POSITIVE_INFINITY),
    ).toThrow(RangeError);
    expect(() =>
      withKernelWeight(
        weights,
        { filters: [3, 3, 3] as unknown as MnistCnnFilters },
        1,
        0,
        0,
        0,
        0,
        1,
      ),
    ).toThrow(/four-item/);
  });

  it('rejects an out-of-range kx and a filters tuple that disagrees with the weights', () => {
    const config = configWith(DEFAULT_FILTER_COUNTS);
    const weights = initializeMnistCnnWeights(config, 5);
    expect(() => withKernelWeight(weights, config, 1, 0, 0, 0, 3, 1)).toThrow(/kx/);
    // Les poids par défaut attendent 27 valeurs pour conv1, le tuple [1, 1, 1, 1] en attend 9.
    expect(() => withKernelWeight(weights, configWith([1, 1, 1, 1]), 1, 0, 0, 0, 0, 1)).toThrow(
      /conv1 kernel must have 9/,
    );
  });

  it('keeps the edited weights valid for forwardMnistCnn and changes the probabilities', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const config = configWith(filters);
    const base = initializeMnistCnnWeights(config, 3);
    for (const layer of base.conv) {
      layer.kernel.fill(0);
      layer.bias.fill(0);
    }
    const input = rampInput();
    const before = forwardMnistCnn(input, base, config);
    // Chaîne de taps centraux à 1, sauf conv1 réglé à 5 par withKernelWeight :
    // conv4 (oy, ox) remonte à l'entrée (4·oy + 5, 4·ox + 5) après quatre sigmoïdes.
    let edited = base;
    for (const layer of [0, 1, 2, 3]) {
      edited = withKernelWeight(edited, config, layer, 0, 0, 1, 1, 1);
    }
    edited = withKernelWeight(edited, config, 0, 0, 0, 1, 1, 5);
    const after = forwardMnistCnn(input, edited, config);
    const map = after.featureMaps[3];
    expect([map.height, map.width]).toEqual([5, 5]);
    for (let oy = 0; oy < 5; oy += 1) {
      for (let ox = 0; ox < 5; ox += 1) {
        let expected = 5 * input[(4 * oy + 5) * MNIST_INPUT_SIZE + (4 * ox + 5)];
        for (let depth = 0; depth < 4; depth += 1) {
          expected = sigmoidOf(expected);
        }
        expect(map.values[oy * 5 + ox]).toBeCloseTo(expected, 6);
      }
    }
    expect(after.logits).not.toEqual(before.logits);
    expect(after.probabilities).not.toEqual(before.probabilities);
    let sum = 0;
    for (const probability of after.probabilities) {
      expect(Number.isFinite(probability)).toBe(true);
      sum += probability;
    }
    expect(sum).toBeCloseTo(1, 5);
  });
});

describe('contributingTaps', () => {
  it('returns all nine taps for an interior cell of a stride-1 no-padding layer', () => {
    expect(contributingTaps(1, 0, 28, 13, 13)).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 1],
      [1, 2],
      [2, 0],
      [2, 1],
      [2, 2],
    ]);
  });

  it('returns only the [0, 0] tap where the kernel hangs off the map', () => {
    // En stride 1 sans remplissage, le coin haut gauche garde ses neuf taps : le noyau
    // dépasse de la carte par le bas et la droite, jamais par le haut et la gauche.
    expect(contributingTaps(1, 0, 28, 0, 0)).toHaveLength(9);
    // Sur une carte 28×28, la cellule (27, 27) lit les lignes 27 à 29 : une seule ligne
    // reste dans la carte, donc un seul tap.
    expect(contributingTaps(1, 0, 28, 27, 27)).toEqual([[0, 0]]);
  });

  it('returns the three taps of the top row for an edge cell', () => {
    // y = 27 sur une carte 28×28 : seul ky = 0 reste dans la carte.
    expect(contributingTaps(1, 0, 28, 27, 13)).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
    ]);
  });

  it('keeps the taps inside the map for a stride-2 padding-1 layer at (0, 0)', () => {
    // iy = 2·0 − 1 + ky : ky = 0 sort de la carte (−1), ky = 1 et 2 y restent.
    // C'est le coin de conv2 : quatre pixels valides.
    expect(contributingTaps(2, 1, 26, 0, 0)).toEqual([
      [1, 1],
      [1, 2],
      [2, 1],
      [2, 2],
    ]);
    expect(contributingTaps(2, 1, 26, 0, 0)).toHaveLength(4);
  });

  it('truncates only the ky axis for the real conv2 edge cell (0, 12)', () => {
    // conv2 : pas 2, padding 1, entrée 26, sortie 13. Cellule (0, 12) :
    // ky = 0 lit la ligne −1 (hors carte), ky = 1 et 2 restent ; les trois kx restent.
    expect(contributingTaps(2, 1, 26, 0, 12)).toEqual([
      [1, 0],
      [1, 1],
      [1, 2],
      [2, 0],
      [2, 1],
      [2, 2],
    ]);
    expect(contributingTaps(2, 1, 26, 0, 12)).toHaveLength(6);
  });

  it('returns no tap for a cell far outside the map', () => {
    expect(contributingTaps(1, 0, 28, 100, 100)).toEqual([]);
    expect(contributingTaps(2, 1, 26, 500, 500)).toEqual([]);
  });

  it('rejects invalid strides, paddings, sizes and cell coordinates', () => {
    expect(() => contributingTaps(0, 0, 28, 0, 0)).toThrow(/stride/);
    expect(() => contributingTaps(1.5, 0, 28, 0, 0)).toThrow(/stride/);
    expect(() => contributingTaps(1, -1, 28, 0, 0)).toThrow(/padding/);
    expect(() => contributingTaps(1, 0.5, 28, 0, 0)).toThrow(/padding/);
    expect(() => contributingTaps(1, 0, 0, 0, 0)).toThrow(/inputSize/);
    expect(() => contributingTaps(1, 0, 28, -1, 0)).toThrow(/y/);
    expect(() => contributingTaps(1, 0, 28, 0.5, 0)).toThrow(/y/);
    expect(() => contributingTaps(1, 0, 28, 0, -1)).toThrow(/x/);
    expect(() => contributingTaps(1, 0, 28, 0, Number.NaN)).toThrow(TypeError);
    expect(() => contributingTaps(undefined as unknown as number, 0, 28, 0, 0)).toThrow(TypeError);
    expect(() => contributingTaps(1, 0, 28, 0, '0' as unknown as number)).toThrow(TypeError);
  });
});

describe('softmax', () => {
  it('returns the expected probabilities for representative finite logits', () => {
    const probabilities = softmax(new Float32Array([1, 2, 3]));
    expect(probabilities[0]).toBeCloseTo(0.09003057, 6);
    expect(probabilities[1]).toBeCloseTo(0.24472847, 6);
    expect(probabilities[2]).toBeCloseTo(0.66524096, 6);
    expect(probabilities[0] + probabilities[1] + probabilities[2]).toBeCloseTo(1, 5);
  });

  it('stays finite for large-magnitude logits', () => {
    const probabilities = softmax(new Float32Array([1000, 1000.5, -1000]));
    let sum = 0;
    for (const probability of probabilities) {
      expect(Number.isFinite(probability)).toBe(true);
      expect(probability).toBeGreaterThanOrEqual(0);
      sum += probability;
    }
    expect(sum).toBeCloseTo(1, 5);
  });

  it('rejects non-finite logits', () => {
    expect(() => softmax(new Float32Array([Number.NaN, 0]))).toThrow(RangeError);
    expect(() => softmax(new Float32Array([Number.POSITIVE_INFINITY, 0]))).toThrow(/finite/i);
  });
});

describe('validation', () => {
  const filters = DEFAULT_FILTER_COUNTS;
  const config = configWith(filters);
  const weights: MnistCnnWeights = initializeMnistCnnWeights(config, 42);

  it('rejects inputs with the wrong length or type', () => {
    expect(() => forwardMnistCnn(new Float32Array(783), weights, config)).toThrow(/784/);
    expect(() => forwardMnistCnn(new Float32Array(785), weights, config)).toThrow(/784/);
    const fake = new Array<number>(MNIST_INPUT_LENGTH).fill(0) as unknown as Float32Array;
    expect(() => forwardMnistCnn(fake, weights, config)).toThrow(/Float32Array/);
  });

  it('rejects non-finite input values', () => {
    const input = rampInput();
    input[100] = Number.NaN;
    expect(() => forwardMnistCnn(input, weights, config)).toThrow(/finite/i);
  });

  it('rejects weight arrays with wrong lengths', () => {
    const input = constantInput(0.5);
    const badKernel = replaceConv(weights, 1, {
      kernel: new Float32Array(5),
      bias: weights.conv[1].bias,
    });
    expect(() => forwardMnistCnn(input, badKernel, config)).toThrow(/conv2 kernel must have 81/);
    const badBias = replaceConv(weights, 0, {
      kernel: weights.conv[0].kernel,
      bias: new Float32Array(2),
    });
    expect(() => forwardMnistCnn(input, badBias, config)).toThrow(/conv1 bias must have 3/);
    const badDense = replaceDense(weights, 0, {
      kernel: new Float32Array(10),
      bias: weights.dense[0].bias,
    });
    expect(() => forwardMnistCnn(input, badDense, config)).toThrow(/dense1 kernel must have 600/);
    const badOutputBias = replaceDense(weights, 1, {
      kernel: weights.dense[1].kernel,
      bias: new Float32Array(9),
    });
    expect(() => forwardMnistCnn(input, badOutputBias, config)).toThrow(/dense2 bias must have 10/);
  });

  it('rejects kernel or bias arrays that are not Float32Arrays', () => {
    const input = constantInput(0.5);
    const plainArrayKernel = replaceConv(weights, 0, {
      kernel: new Array<number>(27).fill(0) as unknown as Float32Array,
      bias: weights.conv[0].bias,
    });
    expect(() => forwardMnistCnn(input, plainArrayKernel, config)).toThrow(/Float32Array/i);

    const missingKernel = replaceConv(weights, 1, {
      kernel: undefined as unknown as Float32Array,
      bias: weights.conv[1].bias,
    });
    expect(() => forwardMnistCnn(input, missingKernel, config)).toThrow(/Float32Array/i);

    const missingBias = replaceDense(weights, 1, {
      kernel: weights.dense[1].kernel,
      bias: undefined as unknown as Float32Array,
    });
    expect(() => forwardMnistCnn(input, missingBias, config)).toThrow(/Float32Array/i);
  });

  it('rejects a missing layer object', () => {
    const input = constantInput(0.5);
    const missingLayer = replaceConv(weights, 2, undefined as unknown as MnistCnnLayerWeights);
    expect(() => forwardMnistCnn(input, missingLayer, config)).toThrow(TypeError);
    expect(() => forwardMnistCnn(input, missingLayer, config)).toThrow(/conv3 weights must expose/);
  });

  it('rejects non-finite kernel or bias values with the layer name', () => {
    const input = constantInput(0.5);
    const nanKernel: MnistCnnLayerWeights = {
      ...weights.conv[0],
      kernel: weights.conv[0].kernel.slice(),
    };
    nanKernel.kernel[3] = Number.NaN;
    expect(() => forwardMnistCnn(input, replaceConv(weights, 0, nanKernel), config)).toThrow(
      /conv1 kernel must contain only finite values/,
    );

    const infBias: MnistCnnLayerWeights = {
      ...weights.dense[1],
      bias: weights.dense[1].bias.slice(),
    };
    infBias.bias[5] = Number.POSITIVE_INFINITY;
    expect(() => forwardMnistCnn(input, replaceDense(weights, 1, infBias), config)).toThrow(
      /dense2 bias must contain only finite values/,
    );

    const negInfKernel: MnistCnnLayerWeights = {
      ...weights.conv[3],
      kernel: weights.conv[3].kernel.slice(),
    };
    negInfKernel.kernel[0] = Number.NEGATIVE_INFINITY;
    expect(() => forwardMnistCnn(input, replaceConv(weights, 3, negInfKernel), config)).toThrow(
      /conv4 kernel must contain only finite values/,
    );
  });

  it('rejects weight layer arrays that disagree with the layer counts', () => {
    const input = constantInput(0.5);
    const shortConv: MnistCnnWeights = { ...weights, conv: weights.conv.slice(0, 3) };
    expect(() => forwardMnistCnn(input, shortConv, config)).toThrow(/conv weights must hold 4/);
    const shortDense: MnistCnnWeights = { ...weights, dense: weights.dense.slice(0, 1) };
    expect(() => forwardMnistCnn(input, shortDense, config)).toThrow(/dense weights must hold 2/);
    expect(() => forwardMnistCnn(input, weights, { ...config, convLayers: 3 })).toThrow(
      /conv weights must hold 3/,
    );
  });

  it('rejects configs whose filter counts disagree with the weights', () => {
    const mismatched = configWith([1, 1, 1, 1]);
    expect(() => forwardMnistCnn(constantInput(0.5), weights, mismatched)).toThrow(
      /conv1 kernel must have 9/,
    );
    expect(() => forwardMnistCnn(constantInput(0.5), weights, configWith([3, 3, 3, 5]))).toThrow(
      /filter count/i,
    );
    expect(() =>
      forwardMnistCnn(constantInput(0.5), weights, undefined as unknown as MnistCnnConfig),
    ).toThrow(TypeError);
  });
});
