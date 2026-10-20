import { describe, expect, it } from 'vitest';
import {
  convOutputLength,
  DEFAULT_FILTER_COUNTS,
  forwardMnistCnn,
  HIDDEN_DENSE_SIZE,
  initializeMnistCnnWeights,
  LOGIT_COUNT,
  MNIST_INPUT_LENGTH,
  MNIST_INPUT_SIZE,
  type MnistCnnConfig,
  type MnistCnnFilters,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
  mnistCnnParameterCount,
  softmax,
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
    expect(mnistCnnParameterCount()).toBe(980);
  });

  it('counts parameters for a custom filter tuple', () => {
    expect(mnistCnnParameterCount([1, 2, 3, 4])).toBe(1097);
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
    expect(Array.from(a.conv1.kernel).some((value, index) => value !== b.conv1.kernel[index])).toBe(
      true,
    );
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
    expect(output.hidden).toHaveLength(8);
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
      expect(Number.isFinite(output.hidden[index])).toBe(true);
      expect(output.hidden[index]).toBeGreaterThan(0);
      expect(output.hidden[index]).toBeLessThan(1);
    }
  });
});

describe('convolution semantics', () => {
  it('applies sigmoid to a 3x3 no-padding convolution (conv1 center tap)', () => {
    const filters: MnistCnnFilters = [1, 1, 1, 1];
    const weights = initializeMnistCnnWeights(configWith(filters), 1);
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[4] = 1; // [kernelY=1][kernelX=1][cIn=0][cOut=0]
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[4] = 1;
    weights.conv2.kernel.fill(0);
    weights.conv2.kernel[4] = 1;
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[4] = 1;
    weights.conv2.kernel.fill(1);
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
    for (const layer of [weights.conv1, weights.conv2, weights.conv3, weights.conv4]) {
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[1] = 1;
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[1] = 1;
    weights.conv2.kernel.fill(0);
    weights.conv2.kernel[1] = 1;
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[4] = 1;
    // conv2 : tap [ky=1][kx=0] (milieu gauche), la sortie (oy, ox) lit conv1 en (2·oy, 2·ox−1).
    weights.conv2.kernel.fill(0);
    weights.conv2.kernel[3] = 1;
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
    weights.conv1.kernel.fill(0);
    weights.conv1.kernel[8] = 1; // [1][1][cIn=0][cOut=0]
    weights.conv1.kernel[9] = 1; // [1][1][cIn=0][cOut=1]
    // conv2 : deux taps distincts par couple (cIn, cOut).
    weights.conv2.kernel.fill(0);
    weights.conv2.kernel[18] = 2; // ((1·3+1)·2 + 1)·2 + 0 -> [1][1][cIn=1][cOut=0]
    weights.conv2.kernel[17] = 3; // ((1·3+1)·2 + 0)·2 + 1 -> [1][1][cIn=0][cOut=1]
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
    for (const layer of [weights.conv1, weights.conv2, weights.conv3, weights.conv4]) {
      layer.kernel.fill(0);
      layer.bias.fill(0);
    }
    weights.conv4.bias[2] = 20; // canal 2 : carte constante ≈ 1 ; canaux 0 et 1 : 0.5
    weights.dense1.kernel.fill(0);
    weights.dense1.kernel[2 * HIDDEN_DENSE_SIZE + 0] = 1; // caractéristique NHWC 2 -> (y=0, x=0, c=2)
    weights.dense1.kernel[26 * HIDDEN_DENSE_SIZE + 2] = 1; // caractéristique NHWC 26 -> (y=1, x=3, c=2)
    const output = forwardMnistCnn(constantInput(1), weights, configWith(filters));
    const nearOne = sigmoidOf(20);
    expect(output.hidden[0]).toBeCloseTo(sigmoidOf(nearOne), 5);
    expect(output.hidden[2]).toBeCloseTo(sigmoidOf(nearOne), 5);
    for (const index of [1, 3, 4, 5, 6, 7]) {
      expect(output.hidden[index]).toBe(0.5); // sigmoid(0)
    }
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
    const badKernel: MnistCnnWeights = {
      ...weights,
      conv2: { kernel: new Float32Array(5), bias: weights.conv2.bias },
    };
    expect(() => forwardMnistCnn(input, badKernel, config)).toThrow(/conv2 kernel must have 81/);
    const badBias: MnistCnnWeights = {
      ...weights,
      conv1: { kernel: weights.conv1.kernel, bias: new Float32Array(2) },
    };
    expect(() => forwardMnistCnn(input, badBias, config)).toThrow(/conv1 bias must have 3/);
    const badDense: MnistCnnWeights = {
      ...weights,
      dense1: { kernel: new Float32Array(10), bias: weights.dense1.bias },
    };
    expect(() => forwardMnistCnn(input, badDense, config)).toThrow(/dense1 kernel must have 600/);
    const badOutputBias: MnistCnnWeights = {
      ...weights,
      dense2: { kernel: weights.dense2.kernel, bias: new Float32Array(9) },
    };
    expect(() => forwardMnistCnn(input, badOutputBias, config)).toThrow(/dense2 bias must have 10/);
  });

  it('rejects kernel or bias arrays that are not Float32Arrays', () => {
    const input = constantInput(0.5);
    const plainArrayKernel: MnistCnnWeights = {
      ...weights,
      conv1: {
        kernel: new Array<number>(27).fill(0) as unknown as Float32Array,
        bias: weights.conv1.bias,
      },
    };
    expect(() => forwardMnistCnn(input, plainArrayKernel, config)).toThrow(/Float32Array/i);

    const missingKernel: MnistCnnWeights = {
      ...weights,
      conv2: {
        kernel: undefined as unknown as Float32Array,
        bias: weights.conv2.bias,
      },
    };
    expect(() => forwardMnistCnn(input, missingKernel, config)).toThrow(/Float32Array/i);

    const missingBias: MnistCnnWeights = {
      ...weights,
      dense2: {
        kernel: weights.dense2.kernel,
        bias: undefined as unknown as Float32Array,
      },
    };
    expect(() => forwardMnistCnn(input, missingBias, config)).toThrow(/Float32Array/i);
  });

  it('rejects a missing layer object', () => {
    const input = constantInput(0.5);
    const missingLayer: MnistCnnWeights = {
      ...weights,
      conv3: undefined as unknown as MnistCnnLayerWeights,
    };
    expect(() => forwardMnistCnn(input, missingLayer, config)).toThrow(TypeError);
    expect(() => forwardMnistCnn(input, missingLayer, config)).toThrow(/conv3 weights must expose/);
  });

  it('rejects non-finite kernel or bias values with the layer name', () => {
    const input = constantInput(0.5);
    const nanKernel = {
      ...weights.conv1,
      kernel: weights.conv1.kernel.slice(),
    };
    nanKernel.kernel[3] = Number.NaN;
    expect(() => forwardMnistCnn(input, { ...weights, conv1: nanKernel }, config)).toThrow(
      /conv1 kernel must contain only finite values/,
    );

    const infBias = {
      ...weights.dense2,
      bias: weights.dense2.bias.slice(),
    };
    infBias.bias[5] = Number.POSITIVE_INFINITY;
    expect(() => forwardMnistCnn(input, { ...weights, dense2: infBias }, config)).toThrow(
      /dense2 bias must contain only finite values/,
    );

    const negInfKernel = {
      ...weights.conv4,
      kernel: weights.conv4.kernel.slice(),
    };
    negInfKernel.kernel[0] = Number.NEGATIVE_INFINITY;
    expect(() => forwardMnistCnn(input, { ...weights, conv4: negInfKernel }, config)).toThrow(
      /conv4 kernel must contain only finite values/,
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
