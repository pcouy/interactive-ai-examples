/**
 * Passage forward pur TypeScript du CNN de référence MNIST décrit dans
 * `tf-playground/mnist/train.py` : quatre convolutions 3×3 suivies d'une
 * sigmoïde, aplatissement, couche dense de taille 8 + sigmoïde, puis dix
 * logits bruts. Le dropout est désactivé en inférence. Ce module n'importe
 * rien : aucune dépendance DOM, Lit, D3 ou TensorFlow.js.
 *
 * Conventions de stockage :
 * - activations en NHWC (le canal est l'indice le plus rapide) ;
 * - noyaux de convolution à plat en HWIO : [kernelY][kernelX][canalEntrée][canalSortie] ;
 * - noyaux denses à plat en [caractéristiqueEntrée][caractéristiqueSortie].
 */

export type MnistCnnFilters = readonly [number, number, number, number];

export interface MnistCnnConfig {
  readonly filters: MnistCnnFilters;
}

/** Poids d'une couche : noyau à plat plus biais. */
export interface MnistCnnLayerWeights {
  readonly kernel: Float32Array;
  readonly bias: Float32Array;
}

export interface MnistCnnWeights {
  readonly conv1: MnistCnnLayerWeights;
  readonly conv2: MnistCnnLayerWeights;
  readonly conv3: MnistCnnLayerWeights;
  readonly conv4: MnistCnnLayerWeights;
  readonly dense1: MnistCnnLayerWeights;
  readonly dense2: MnistCnnLayerWeights;
}

export interface MnistCnnFeatureMap {
  readonly name: string;
  readonly height: number;
  readonly width: number;
  readonly channels: number;
  /** Activations NHWC, longueur height * width * channels. */
  readonly values: Float32Array;
}

export interface MnistCnnForwardResult {
  readonly featureMaps: readonly MnistCnnFeatureMap[];
  /** Activations de la couche dense cachée (après sigmoïde), longueur 8. */
  readonly hidden: Float32Array;
  readonly logits: Float32Array;
  readonly probabilities: Float32Array;
}

export const MNIST_INPUT_SIZE = 28;
export const MNIST_INPUT_LENGTH = MNIST_INPUT_SIZE * MNIST_INPUT_SIZE;
export const CONV_KERNEL_SIZE = 3;
export const HIDDEN_DENSE_SIZE = 8;
export const LOGIT_COUNT = 10;

export const MIN_FILTER_COUNT = 1;
export const MAX_FILTER_COUNT = 4;

export const DEFAULT_FILTER_COUNTS: MnistCnnFilters = [3, 3, 3, 3];
export const CONV_STRIDES: readonly number[] = [1, 2, 1, 2];
/** Seule conv2 utilise un remplissage zéro symétrique de 1. */
export const CONV_PADDINGS: readonly number[] = [0, 1, 0, 0];
export const CONV_LAYER_NAMES: readonly string[] = ['conv1', 'conv2', 'conv3', 'conv4'];

const DEFAULT_SEED = 42;

interface ConvLayerPlan {
  readonly name: string;
  readonly inputSize: number;
  readonly outputSize: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly stride: number;
  readonly padding: number;
  readonly kernelLength: number;
}

/** Générateur pseudo-aléatoire déterministe (même implantation que perceptron.ts). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/** Taille de sortie d'une convolution : floor((n + 2·p − k) / s) + 1. */
export function convOutputLength(inputLength: number, stride: number, padding: number): number {
  return Math.floor((inputLength + 2 * padding - CONV_KERNEL_SIZE) / stride) + 1;
}

export function validateFilterCounts(filters: MnistCnnFilters): void {
  if (!Array.isArray(filters) || filters.length !== 4) {
    throw new RangeError(
      `Filter counts must be a four-item tuple, received ${JSON.stringify(filters)}.`,
    );
  }
  for (const count of filters) {
    if (!Number.isInteger(count) || count < MIN_FILTER_COUNT || count > MAX_FILTER_COUNT) {
      throw new RangeError(
        `Each filter count must be an integer from ${MIN_FILTER_COUNT} to ${MAX_FILTER_COUNT}, received ${String(count)}.`,
      );
    }
  }
}

function validateConfig(config: MnistCnnConfig): void {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('MNIST CNN config with a filters tuple is required.');
  }
}

function convLayerPlans(filters: MnistCnnFilters): ConvLayerPlan[] {
  validateFilterCounts(filters);
  const plans: ConvLayerPlan[] = [];
  let inputSize = MNIST_INPUT_SIZE;
  let inputChannels = 1;
  for (let layer = 0; layer < CONV_LAYER_NAMES.length; layer += 1) {
    const outputChannels = filters[layer];
    const stride = CONV_STRIDES[layer];
    const padding = CONV_PADDINGS[layer];
    const outputSize = convOutputLength(inputSize, stride, padding);
    if (outputSize < 1) {
      throw new RangeError(
        `${CONV_LAYER_NAMES[layer]}: output size ${outputSize} is invalid for input size ${inputSize}.`,
      );
    }
    plans.push({
      name: CONV_LAYER_NAMES[layer],
      inputSize,
      outputSize,
      inputChannels,
      outputChannels,
      stride,
      padding,
      kernelLength: CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels * outputChannels,
    });
    inputSize = outputSize;
    inputChannels = outputChannels;
  }
  return plans;
}

function validateInput(input: Float32Array): void {
  if (!(input instanceof Float32Array)) {
    throw new TypeError(
      `MNIST input must be a Float32Array of ${MNIST_INPUT_LENGTH} values in row-major order.`,
    );
  }
  if (input.length !== MNIST_INPUT_LENGTH) {
    throw new RangeError(
      `MNIST input must have ${MNIST_INPUT_LENGTH} values (${MNIST_INPUT_SIZE}x${MNIST_INPUT_SIZE} row-major), received ${input.length}.`,
    );
  }
  for (let index = 0; index < input.length; index += 1) {
    if (!Number.isFinite(input[index])) {
      throw new RangeError(
        `MNIST input must contain only finite values, received ${input[index]} at index ${index}.`,
      );
    }
  }
}

/** Type effectif d'un tableau de poids, pour les messages d'erreur. */
function arrayKind(value: unknown): string {
  if (value instanceof Float32Array) {
    return 'Float32Array';
  }
  if (Array.isArray(value)) {
    return 'Array';
  }
  if (value === undefined) {
    return 'undefined';
  }
  return typeof value;
}

function validateLayerWeights(
  layer: MnistCnnLayerWeights,
  name: string,
  expectedKernelLength: number,
  expectedBiasLength: number,
): void {
  if (layer === null || typeof layer !== 'object') {
    throw new TypeError(`${name} weights must expose kernel and bias Float32Arrays.`);
  }
  if (!(layer.kernel instanceof Float32Array) || !(layer.bias instanceof Float32Array)) {
    throw new TypeError(
      `${name} kernel and bias must be Float32Arrays, received ${arrayKind(layer.kernel)} and ${arrayKind(layer.bias)}.`,
    );
  }
  if (layer.kernel.length !== expectedKernelLength) {
    const order = name.startsWith('conv')
      ? ' in HWIO order [kernelY][kernelX][inputChannel][outputChannel]'
      : ' in [inputFeature][outputFeature] order';
    throw new RangeError(
      `${name} kernel must have ${expectedKernelLength} values${order}, received ${layer.kernel.length}.`,
    );
  }
  if (layer.bias.length !== expectedBiasLength) {
    throw new RangeError(
      `${name} bias must have ${expectedBiasLength} values, received ${layer.bias.length}.`,
    );
  }
  for (let index = 0; index < layer.kernel.length; index += 1) {
    if (!Number.isFinite(layer.kernel[index])) {
      throw new RangeError(
        `${name} kernel must contain only finite values, received ${layer.kernel[index]} at index ${index}.`,
      );
    }
  }
  for (let index = 0; index < layer.bias.length; index += 1) {
    if (!Number.isFinite(layer.bias[index])) {
      throw new RangeError(
        `${name} bias must contain only finite values, received ${layer.bias[index]} at index ${index}.`,
      );
    }
  }
}

function validateWeights(weights: MnistCnnWeights, filters: MnistCnnFilters): void {
  if (weights === null || typeof weights !== 'object') {
    throw new TypeError('MNIST CNN weights object is required.');
  }
  const plans = convLayerPlans(filters);
  validateLayerWeights(weights.conv1, 'conv1', plans[0].kernelLength, plans[0].outputChannels);
  validateLayerWeights(weights.conv2, 'conv2', plans[1].kernelLength, plans[1].outputChannels);
  validateLayerWeights(weights.conv3, 'conv3', plans[2].kernelLength, plans[2].outputChannels);
  validateLayerWeights(weights.conv4, 'conv4', plans[3].kernelLength, plans[3].outputChannels);
  const denseInputLength = plans[3].outputSize * plans[3].outputSize * plans[3].outputChannels;
  validateLayerWeights(
    weights.dense1,
    'dense1',
    denseInputLength * HIDDEN_DENSE_SIZE,
    HIDDEN_DENSE_SIZE,
  );
  validateLayerWeights(weights.dense2, 'dense2', HIDDEN_DENSE_SIZE * LOGIT_COUNT, LOGIT_COUNT);
}

/**
 * Convolution 3×3 + sigmoïde sur une carte carrée NHWC. Le remplissage est
 * traité implicitement : les coordonnées hors carte contribuent zéro.
 */
function convolve(
  input: Float32Array,
  plan: ConvLayerPlan,
  layer: MnistCnnLayerWeights,
): Float32Array {
  const { inputSize, outputSize, inputChannels, outputChannels, stride, padding } = plan;
  const values = new Float32Array(outputSize * outputSize * outputChannels);
  for (let oy = 0; oy < outputSize; oy += 1) {
    for (let ox = 0; ox < outputSize; ox += 1) {
      for (let oc = 0; oc < outputChannels; oc += 1) {
        let sum = layer.bias[oc];
        for (let ky = 0; ky < CONV_KERNEL_SIZE; ky += 1) {
          const iy = oy * stride - padding + ky;
          if (iy < 0 || iy >= inputSize) {
            continue;
          }
          for (let kx = 0; kx < CONV_KERNEL_SIZE; kx += 1) {
            const ix = ox * stride - padding + kx;
            if (ix < 0 || ix >= inputSize) {
              continue;
            }
            const inputBase = (iy * inputSize + ix) * inputChannels;
            const kernelBase = (ky * CONV_KERNEL_SIZE + kx) * inputChannels * outputChannels;
            for (let ic = 0; ic < inputChannels; ic += 1) {
              sum += layer.kernel[kernelBase + ic * outputChannels + oc] * input[inputBase + ic];
            }
          }
        }
        values[(oy * outputSize + ox) * outputChannels + oc] = sigmoid(sum);
      }
    }
  }
  return values;
}

/** Couche entièrement connectée à plat [inputFeature][outputFeature]. */
function denseLayer(
  input: Float32Array,
  layer: MnistCnnLayerWeights,
  outputCount: number,
  applySigmoid: boolean,
): Float32Array {
  const output = new Float32Array(outputCount);
  for (let o = 0; o < outputCount; o += 1) {
    let sum = layer.bias[o];
    for (let i = 0; i < input.length; i += 1) {
      sum += layer.kernel[i * outputCount + o] * input[i];
    }
    output[o] = applySigmoid ? sigmoid(sum) : sum;
  }
  return output;
}

/** Softmax numériquement stable (soustraction du maximum) ; exige des logits finis. */
export function softmax(logits: Float32Array): Float32Array {
  for (let index = 0; index < logits.length; index += 1) {
    if (!Number.isFinite(logits[index])) {
      throw new RangeError(
        `Softmax requires finite logits, received ${logits[index]} at index ${index}.`,
      );
    }
  }
  let max = -Infinity;
  for (let index = 0; index < logits.length; index += 1) {
    if (logits[index] > max) {
      max = logits[index];
    }
  }
  const probabilities = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] = Math.exp(logits[index] - max);
    sum += probabilities[index];
  }
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] /= sum;
  }
  return probabilities;
}

function randomUniform(rng: () => number, length: number, limit: number): Float32Array {
  const values = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    values[index] = (rng() * 2 - 1) * limit;
  }
  return values;
}

function glorotLimit(fanIn: number, fanOut: number): number {
  return Math.sqrt(6 / (fanIn + fanOut));
}

function randomConvLayer(rng: () => number, plan: ConvLayerPlan): MnistCnnLayerWeights {
  const fanIn = CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * plan.inputChannels;
  return {
    kernel: randomUniform(rng, plan.kernelLength, glorotLimit(fanIn, plan.outputChannels)),
    bias: new Float32Array(plan.outputChannels),
  };
}

/**
 * Initialisation déterministe (Glorot uniforme, biais à zéro) : deux appels
 * avec la même graine et les mêmes filtres produisent exactement les mêmes
 * poids, des graines différentes produisent des poids différents.
 */
export function initializeMnistCnnWeights(
  config: MnistCnnConfig,
  seed: number = DEFAULT_SEED,
): MnistCnnWeights {
  validateConfig(config);
  const plans = convLayerPlans(config.filters);
  const rng = mulberry32(seed);
  const denseInputLength = plans[3].outputSize * plans[3].outputSize * plans[3].outputChannels;
  return {
    conv1: randomConvLayer(rng, plans[0]),
    conv2: randomConvLayer(rng, plans[1]),
    conv3: randomConvLayer(rng, plans[2]),
    conv4: randomConvLayer(rng, plans[3]),
    dense1: {
      kernel: randomUniform(
        rng,
        denseInputLength * HIDDEN_DENSE_SIZE,
        glorotLimit(denseInputLength, HIDDEN_DENSE_SIZE),
      ),
      bias: new Float32Array(HIDDEN_DENSE_SIZE),
    },
    dense2: {
      kernel: randomUniform(
        rng,
        HIDDEN_DENSE_SIZE * LOGIT_COUNT,
        glorotLimit(HIDDEN_DENSE_SIZE, LOGIT_COUNT),
      ),
      bias: new Float32Array(LOGIT_COUNT),
    },
  };
}

/** Nombre total de paramètres (poids et biais) pour le tuple de filtres donné. */
export function mnistCnnParameterCount(filters: MnistCnnFilters = DEFAULT_FILTER_COUNTS): number {
  const plans = convLayerPlans(filters);
  let total = 0;
  for (const plan of plans) {
    total += plan.kernelLength + plan.outputChannels;
  }
  const denseInputLength = plans[3].outputSize * plans[3].outputSize * plans[3].outputChannels;
  total += denseInputLength * HIDDEN_DENSE_SIZE + HIDDEN_DENSE_SIZE;
  total += HIDDEN_DENSE_SIZE * LOGIT_COUNT + LOGIT_COUNT;
  return total;
}

/**
 * Passage forward complet sur une image 28×28×1 (Float32Array de 784 valeurs
 * en ordre row-major, normalisée). Retourne les cartes de caractéristiques de
 * chaque convolution (après sigmoïde), les activations cachées (8), les logits
 * (10) et les probabilités softmax (10).
 */
export function forwardMnistCnn(
  input: Float32Array,
  weights: MnistCnnWeights,
  config: MnistCnnConfig,
): MnistCnnForwardResult {
  validateConfig(config);
  validateInput(input);
  validateWeights(weights, config.filters);
  const plans = convLayerPlans(config.filters);
  const convLayers: readonly MnistCnnLayerWeights[] = [
    weights.conv1,
    weights.conv2,
    weights.conv3,
    weights.conv4,
  ];
  const featureMaps: MnistCnnFeatureMap[] = [];
  let activation: Float32Array = input;
  for (let layer = 0; layer < plans.length; layer += 1) {
    const plan = plans[layer];
    const values = convolve(activation, plan, convLayers[layer]);
    featureMaps.push({
      name: plan.name,
      height: plan.outputSize,
      width: plan.outputSize,
      channels: plan.outputChannels,
      values,
    });
    activation = values;
  }
  const hidden = denseLayer(activation, weights.dense1, HIDDEN_DENSE_SIZE, true);
  const logits = denseLayer(hidden, weights.dense2, LOGIT_COUNT, false);
  return { featureMaps, hidden, logits, probabilities: softmax(logits) };
}
