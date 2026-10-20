/**
 * Passage forward pur TypeScript du CNN de référence MNIST décrit dans
 * `tf-playground/mnist/train.py` : convolutions 3×3 suivies d'une fonction
 * d'activation, aplatissement, couches denses, puis dix logits bruts. Le
 * dropout est désactivé en inférence. Ce module n'importe rien : aucune
 * dépendance DOM, Lit, D3 ou TensorFlow.js.
 *
 * Le nombre de couches est paramétrable : 0 à 4 convolutions et 1 à 3 couches
 * denses, couche de sortie comprise. L'architecture par défaut reste celle de
 * référence (quatre convolutions et deux couches denses, 980 paramètres).
 *
 * Conventions de stockage :
 * - activations en NHWC (le canal est l'indice le plus rapide) ;
 * - noyaux de convolution à plat en HWIO : [kernelY][kernelX][canalEntrée][canalSortie] ;
 * - noyaux denses à plat en [caractéristiqueEntrée][caractéristiqueSortie].
 */

export type MnistCnnFilters = readonly [number, number, number, number];

/** Fonction d'activation des couches de convolution et des couches denses cachées. */
export type MnistCnnActivation = 'sigmoid' | 'tanh';

export interface MnistCnnConfig {
  readonly filters: MnistCnnFilters;
  /** Nombre de couches de convolution, de 0 à 4. Défaut : 4. */
  readonly convLayers?: number;
  /** Nombre de couches denses, couche de sortie comprise, de 1 à 3. Défaut : 2. */
  readonly denseLayers?: number;
  /**
   * Largeurs des couches denses cachées, dans l'ordre, couche de sortie exclue.
   * Défaut : `[8, 8]`. Une entrée manquante est remplacée par 8 ; les entrées
   * au-delà du nombre de couches cachées sont ignorées.
   */
  readonly denseSizes?: readonly number[];
  /**
   * Nombre de classes de sortie, de 2 à `MAX_OUTPUT_CLASSES`. Défaut :
   * `LOGIT_COUNT` (10).
   */
  readonly outputClasses?: number;
  /**
   * Fonction d'activation des convolutions et des couches denses cachées.
   * La valeur par défaut est `DEFAULT_MNIST_CNN_ACTIVATION`. La couche de
   * logits reste linéaire dans les deux cas.
   */
  readonly activation?: MnistCnnActivation;
}

/** Poids d'une couche : noyau à plat plus biais. */
export interface MnistCnnLayerWeights {
  readonly kernel: Float32Array;
  readonly bias: Float32Array;
}

export interface MnistCnnWeights {
  /** Poids des convolutions, dans l'ordre : longueur `convLayers`. */
  readonly conv: readonly MnistCnnLayerWeights[];
  /** Poids des couches denses, sortie comprise : longueur `denseLayers`. */
  readonly dense: readonly MnistCnnLayerWeights[];
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
  /** Activation de chaque couche dense ; la dernière entrée est le softmax. */
  readonly denseActivations: readonly Float32Array[];
  readonly logits: Float32Array;
  readonly probabilities: Float32Array;
}

export const MNIST_INPUT_SIZE = 28;
export const MNIST_INPUT_LENGTH = MNIST_INPUT_SIZE * MNIST_INPUT_SIZE;
export const CONV_KERNEL_SIZE = 3;
export const HIDDEN_DENSE_SIZE = 8;
export const LOGIT_COUNT = 10;
/** Nombre maximal de classes de sortie (QuickDraw : vingt catégories). */
export const MAX_OUTPUT_CLASSES = 20;

/** Bornes des couches denses cachées. */
export const MIN_HIDDEN_DENSE_SIZE = 1;
export const MAX_HIDDEN_DENSE_SIZE = 128;

/** Largeurs par défaut des couches denses cachées. */
export const DEFAULT_DENSE_SIZES: readonly [number, number] = [
  HIDDEN_DENSE_SIZE,
  HIDDEN_DENSE_SIZE,
];

export const MIN_FILTER_COUNT = 1;
export const MAX_FILTER_COUNT = 8;

export const DEFAULT_FILTER_COUNTS: MnistCnnFilters = [3, 3, 3, 3];

/** Nombre maximal de couches de convolution. */
export const MAX_CONV_LAYERS = 4;
/** Nombre maximal de couches denses, couche de sortie comprise. */
export const MAX_DENSE_LAYERS = 3;
/** Architecture par défaut : quatre convolutions et deux couches denses. */
export const DEFAULT_CONV_LAYERS = 4;
export const DEFAULT_DENSE_LAYERS = 2;

export interface ConvLayerSpec {
  readonly stride: number;
  readonly padding: number;
}

/**
 * Pas et remplissage par nombre de couches de convolution. Les couches sans
 * pas sont retirées en premier quand le nombre baisse, puis la couche sans
 * remplissage ; la couche avec remplissage part en dernier.
 */
export const CONV_SCHEDULES: Readonly<Record<number, readonly ConvLayerSpec[]>> = {
  0: [],
  1: [{ stride: 2, padding: 1 }],
  2: [
    { stride: 2, padding: 1 },
    { stride: 2, padding: 0 },
  ],
  3: [
    { stride: 1, padding: 0 },
    { stride: 2, padding: 1 },
    { stride: 2, padding: 0 },
  ],
  4: [
    { stride: 1, padding: 0 },
    { stride: 2, padding: 1 },
    { stride: 1, padding: 0 },
    { stride: 2, padding: 0 },
  ],
};

/** Nom affiché d'une couche de convolution, à partir de 0. */
export function convLayerName(layer: number): string {
  return `conv${layer + 1}`;
}

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

/** Fonction d'activation par défaut, celle des poids pré-entraînés du manifeste. */
export const DEFAULT_MNIST_CNN_ACTIVATION: MnistCnnActivation = 'sigmoid';

/** Bornes de sortie d'une fonction d'activation. */
export function activationDomain(activation: MnistCnnActivation): {
  readonly min: number;
  readonly max: number;
} {
  return activation === 'tanh' ? { min: -1, max: 1 } : { min: 0, max: 1 };
}

/** Applique la fonction d'activation choisie à une pré-activation. */
export function applyMnistActivation(value: number, activation: MnistCnnActivation): number {
  return activation === 'tanh' ? Math.tanh(value) : sigmoid(value);
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

/** Nombre de couches de convolution demandé, avec sa valeur par défaut. */
function resolveConvLayers(config: MnistCnnConfig): number {
  const value = config.convLayers ?? DEFAULT_CONV_LAYERS;
  if (!Number.isInteger(value) || value < 0 || value > MAX_CONV_LAYERS) {
    throw new RangeError(
      `convLayers must be an integer from 0 to ${MAX_CONV_LAYERS}, received ${String(value)}.`,
    );
  }
  return value;
}

/** Nombre de couches denses demandé, avec sa valeur par défaut. */
function resolveDenseLayers(config: MnistCnnConfig): number {
  const value = config.denseLayers ?? DEFAULT_DENSE_LAYERS;
  if (!Number.isInteger(value) || value < 1 || value > MAX_DENSE_LAYERS) {
    throw new RangeError(
      `denseLayers must be an integer from 1 to ${MAX_DENSE_LAYERS}, received ${String(value)}.`,
    );
  }
  return value;
}

/** Résout et valide le nombre de classes de sortie. */
export function resolveOutputClasses(config: MnistCnnConfig): number {
  const value = config.outputClasses ?? LOGIT_COUNT;
  if (!Number.isInteger(value) || value < 2 || value > MAX_OUTPUT_CLASSES) {
    throw new RangeError(
      `outputClasses must be an integer from 2 to ${MAX_OUTPUT_CLASSES}, received ${String(value)}.`,
    );
  }
  return value;
}

/** Valide les largeurs cachées : au plus deux entrées, entières, de 1 à 128. */
export function validateDenseSizes(sizes: readonly number[] | undefined): void {
  if (sizes === undefined) {
    return;
  }
  if (!Array.isArray(sizes) || sizes.length > MAX_DENSE_LAYERS - 1) {
    throw new RangeError(
      `Hidden dense sizes must be an array of at most ${MAX_DENSE_LAYERS - 1} integer(s) from ${MIN_HIDDEN_DENSE_SIZE} to ${MAX_HIDDEN_DENSE_SIZE}, received ${JSON.stringify(sizes)}.`,
    );
  }
  for (const size of sizes) {
    if (!Number.isInteger(size) || size < MIN_HIDDEN_DENSE_SIZE || size > MAX_HIDDEN_DENSE_SIZE) {
      throw new RangeError(
        `Each hidden dense size must be an integer from ${MIN_HIDDEN_DENSE_SIZE} to ${MAX_HIDDEN_DENSE_SIZE}, received ${String(size)}.`,
      );
    }
  }
}

function validateConfig(config: MnistCnnConfig): void {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('MNIST CNN config with a filters tuple is required.');
  }
  if (
    config.activation !== undefined &&
    config.activation !== 'sigmoid' &&
    config.activation !== 'tanh'
  ) {
    throw new RangeError(
      `MNIST CNN activation must be "sigmoid" or "tanh", received ${String(config.activation)}.`,
    );
  }
  resolveConvLayers(config);
  resolveDenseLayers(config);
  resolveOutputClasses(config);
  validateDenseSizes(config.denseSizes);
}

/** Nom affiché d'une couche dense, à partir de 0. */
function denseLayerName(layer: number): string {
  return `dense${layer + 1}`;
}

function convLayerPlans(config: MnistCnnConfig): ConvLayerPlan[] {
  validateFilterCounts(config.filters);
  const convLayers = resolveConvLayers(config);
  const schedule = CONV_SCHEDULES[convLayers];
  const plans: ConvLayerPlan[] = [];
  let inputSize = MNIST_INPUT_SIZE;
  let inputChannels = 1;
  for (let layer = 0; layer < convLayers; layer += 1) {
    const spec = schedule[layer];
    const outputChannels = config.filters[layer];
    const outputSize = convOutputLength(inputSize, spec.stride, spec.padding);
    if (outputSize < 1) {
      throw new RangeError(
        `${convLayerName(layer)}: output size ${outputSize} is invalid for input size ${inputSize}.`,
      );
    }
    plans.push({
      name: convLayerName(layer),
      inputSize,
      outputSize,
      inputChannels,
      outputChannels,
      stride: spec.stride,
      padding: spec.padding,
      kernelLength: CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels * outputChannels,
    });
    inputSize = outputSize;
    inputChannels = outputChannels;
  }
  return plans;
}

/** Largeur d'entrée de la première couche dense : carte finale ou image brute. */
function denseInputWidth(plans: readonly ConvLayerPlan[]): number {
  if (plans.length === 0) {
    return MNIST_INPUT_LENGTH;
  }
  const last = plans[plans.length - 1];
  return last.outputSize * last.outputSize * last.outputChannels;
}

/**
 * Largeurs des couches denses : celles demandées par `config.denseSizes` (8 par
 * défaut et pour une entrée manquante), puis le nombre de classes de sortie.
 */
export function denseLayerWidths(config: MnistCnnConfig): readonly number[] {
  const denseLayers = resolveDenseLayers(config);
  validateDenseSizes(config.denseSizes);
  const requested = config.denseSizes ?? DEFAULT_DENSE_SIZES;
  const widths = new Array<number>(denseLayers - 1);
  for (let index = 0; index < denseLayers - 1; index += 1) {
    widths[index] = requested[index] ?? HIDDEN_DENSE_SIZE;
  }
  widths.push(resolveOutputClasses(config));
  return widths;
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

/** Longueur d'un tableau de couches, ou mention explicite si ce n'est pas un tableau. */
function layerArrayLength(value: unknown): string {
  return Array.isArray(value) ? String(value.length) : 'not an array';
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

function validateWeights(weights: MnistCnnWeights, config: MnistCnnConfig): void {
  if (weights === null || typeof weights !== 'object') {
    throw new TypeError('MNIST CNN weights object is required.');
  }
  const convLayers = resolveConvLayers(config);
  const denseLayers = resolveDenseLayers(config);
  const plans = convLayerPlans(config);
  if (!Array.isArray(weights.conv) || weights.conv.length !== convLayers) {
    throw new RangeError(
      `conv weights must hold ${convLayers} layer(s), received ${layerArrayLength(weights.conv)}.`,
    );
  }
  for (let layer = 0; layer < convLayers; layer += 1) {
    validateLayerWeights(
      weights.conv[layer],
      convLayerName(layer),
      plans[layer].kernelLength,
      plans[layer].outputChannels,
    );
  }
  const widths = denseLayerWidths(config);
  if (!Array.isArray(weights.dense) || weights.dense.length !== denseLayers) {
    throw new RangeError(
      `dense weights must hold ${denseLayers} layer(s), received ${layerArrayLength(weights.dense)}.`,
    );
  }
  let inputWidth = denseInputWidth(plans);
  for (let layer = 0; layer < denseLayers; layer += 1) {
    const width = widths[layer];
    validateLayerWeights(weights.dense[layer], denseLayerName(layer), inputWidth * width, width);
    inputWidth = width;
  }
}

/** Exige un nombre fini, utilisé pour les paramètres qui doivent être des entiers. */
function requireFiniteNumber(value: number, name: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`${name} must be a finite number, received ${String(value)}.`);
  }
}

/** Exige un entier ≥ 1 (pas, taille de carte, compte de canaux). */
function requirePositiveInteger(value: number, name: string): void {
  requireFiniteNumber(value, name);
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive integer, received ${String(value)}.`);
  }
}

/** Exige un entier ≥ 0, sans borne haute. */
function requireNonNegativeInteger(value: number, name: string): void {
  requireFiniteNumber(value, name);
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer, received ${String(value)}.`);
  }
}

/** Exige un indice entier dans [0, limit). */
function requireIndexWithin(value: number, limit: number, name: string): void {
  requireFiniteNumber(value, name);
  if (!Number.isInteger(value) || value < 0 || value >= limit) {
    throw new RangeError(
      `${name} must be an integer from 0 to ${limit - 1}, received ${String(value)}.`,
    );
  }
}

/**
 * Indice à plat HWIO d'un poids de noyau de convolution :
 * (ky · CONV_KERNEL_SIZE + kx) · inputChannels · outputChannels
 * + ic · outputChannels + oc, le même ordre que `convolve`.
 */
export function convKernelIndex(
  inputChannels: number,
  outputChannels: number,
  ky: number,
  kx: number,
  ic: number,
  oc: number,
): number {
  requirePositiveInteger(inputChannels, 'inputChannels');
  requirePositiveInteger(outputChannels, 'outputChannels');
  requireIndexWithin(ky, CONV_KERNEL_SIZE, 'ky');
  requireIndexWithin(kx, CONV_KERNEL_SIZE, 'kx');
  requireIndexWithin(ic, inputChannels, 'ic');
  requireIndexWithin(oc, outputChannels, 'oc');
  const kernelBase = (ky * CONV_KERNEL_SIZE + kx) * inputChannels * outputChannels;
  return kernelBase + ic * outputChannels + oc;
}

/**
 * Retourne une copie des poids où un seul poids de noyau de convolution est
 * remplacé : le noyau de la couche touchée est un nouveau Float32Array, toutes
 * les autres couches et tous les biais gardent leur référence d'origine.
 * `weights` n'est jamais modifié.
 */
export function withKernelWeight(
  weights: MnistCnnWeights,
  config: MnistCnnConfig,
  layer: number,
  ic: number,
  oc: number,
  ky: number,
  kx: number,
  value: number,
): MnistCnnWeights {
  validateConfig(config);
  const convLayers = resolveConvLayers(config);
  requireFiniteNumber(layer, 'layer');
  if (!Number.isInteger(layer) || layer < 0 || layer >= convLayers) {
    throw new RangeError(
      `layer must be an integer from 0 to ${convLayers - 1}, received ${String(layer)}.`,
    );
  }
  validateWeights(weights, config);
  if (!Number.isFinite(value)) {
    throw new RangeError(`value must be a finite number, received ${String(value)}.`);
  }
  const plans = convLayerPlans(config);
  const plan = plans[layer];
  const previous = weights.conv[layer];
  const kernel = previous.kernel.slice();
  kernel[convKernelIndex(plan.inputChannels, plan.outputChannels, ky, kx, ic, oc)] = value;
  const conv = weights.conv.slice();
  conv[layer] = { kernel, bias: previous.bias };
  return { ...weights, conv };
}

/**
 * Taps `[ky, kx]` d'une convolution 3×3 qui contribuent à la cellule de sortie
 * `(y, x)` : un tap compte quand ses coordonnées d'entrée `y·stride − padding + ky`
 * et `x·stride − padding + kx` restent dans la carte d'entrée `[0, inputSize)`.
 * Les taps sont renvoyés en ordre ligne-major ; une cellule trop éloignée de la
 * carte n'en reçoit aucun. `convolve` applique exactement la même condition.
 */
export function contributingTaps(
  stride: number,
  padding: number,
  inputSize: number,
  y: number,
  x: number,
): ReadonlyArray<readonly [number, number]> {
  requirePositiveInteger(stride, 'stride');
  requireNonNegativeInteger(padding, 'padding');
  requirePositiveInteger(inputSize, 'inputSize');
  requireNonNegativeInteger(y, 'y');
  requireNonNegativeInteger(x, 'x');
  const taps: [number, number][] = [];
  for (let ky = 0; ky < CONV_KERNEL_SIZE; ky += 1) {
    const iy = y * stride - padding + ky;
    if (iy < 0 || iy >= inputSize) {
      continue;
    }
    for (let kx = 0; kx < CONV_KERNEL_SIZE; kx += 1) {
      const ix = x * stride - padding + kx;
      if (ix < 0 || ix >= inputSize) {
        continue;
      }
      taps.push([ky, kx]);
    }
  }
  return taps;
}

/**
 * Convolution 3×3 + fonction d'activation choisie sur une carte carrée NHWC.
 * Le remplissage est traité implicitement : les coordonnées hors carte
 * contribuent zéro.
 */
function convolve(
  input: Float32Array,
  plan: ConvLayerPlan,
  layer: MnistCnnLayerWeights,
  activation: MnistCnnActivation,
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
        values[(oy * outputSize + ox) * outputChannels + oc] = applyMnistActivation(
          sum,
          activation,
        );
      }
    }
  }
  return values;
}

/**
 * Couche entièrement connectée à plat [inputFeature][outputFeature].
 * `activation` nul laisse la sortie linéaire (couche de logits).
 */
function denseLayer(
  input: Float32Array,
  layer: MnistCnnLayerWeights,
  outputCount: number,
  activation: MnistCnnActivation | null,
): Float32Array {
  const output = new Float32Array(outputCount);
  for (let o = 0; o < outputCount; o += 1) {
    let sum = layer.bias[o];
    for (let i = 0; i < input.length; i += 1) {
      sum += layer.kernel[i * outputCount + o] * input[i];
    }
    output[o] = activation === null ? sum : applyMnistActivation(sum, activation);
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
 * poids, des graines différentes produisent des poids différents. La séquence
 * pseudo-aléatoire ne dépend que de l'ordre des couches.
 */
export function initializeMnistCnnWeights(
  config: MnistCnnConfig,
  seed: number = DEFAULT_SEED,
): MnistCnnWeights {
  validateConfig(config);
  const plans = convLayerPlans(config);
  const denseLayers = resolveDenseLayers(config);
  const widths = denseLayerWidths(config);
  const rng = mulberry32(seed);
  const conv = plans.map((plan) => randomConvLayer(rng, plan));
  let inputWidth = denseInputWidth(plans);
  const dense: MnistCnnLayerWeights[] = [];
  for (let layer = 0; layer < denseLayers; layer += 1) {
    const width = widths[layer];
    dense.push({
      kernel: randomUniform(rng, inputWidth * width, glorotLimit(inputWidth, width)),
      bias: new Float32Array(width),
    });
    inputWidth = width;
  }
  return { conv, dense };
}

/** Nombre total de paramètres (poids et biais) pour la configuration donnée. */
export function mnistCnnParameterCount(
  config: MnistCnnConfig = { filters: DEFAULT_FILTER_COUNTS },
): number {
  const plans = convLayerPlans(config);
  const widths = denseLayerWidths(config);
  let total = 0;
  for (const plan of plans) {
    total += plan.kernelLength + plan.outputChannels;
  }
  let inputWidth = denseInputWidth(plans);
  for (const width of widths) {
    total += inputWidth * width + width;
    inputWidth = width;
  }
  return total;
}

/**
 * Passage forward complet sur une image 28×28×1 (Float32Array de 784 valeurs
 * en ordre row-major, normalisée). Retourne les cartes de caractéristiques de
 * chaque convolution, l'activation de chaque couche dense (la dernière entrée
 * est le softmax), les logits linéaires et les probabilités softmax (autant que
 * de classes de sortie).
 */
export function forwardMnistCnn(
  input: Float32Array,
  weights: MnistCnnWeights,
  config: MnistCnnConfig,
): MnistCnnForwardResult {
  validateConfig(config);
  validateInput(input);
  validateWeights(weights, config);
  const selectedActivation = config.activation ?? DEFAULT_MNIST_CNN_ACTIVATION;
  const convLayers = resolveConvLayers(config);
  const denseLayers = resolveDenseLayers(config);
  const plans = convLayerPlans(config);
  const widths = denseLayerWidths(config);
  const featureMaps: MnistCnnFeatureMap[] = [];
  let activation: Float32Array = input;
  for (let layer = 0; layer < convLayers; layer += 1) {
    const plan = plans[layer];
    const values = convolve(activation, plan, weights.conv[layer], selectedActivation);
    featureMaps.push({
      name: plan.name,
      height: plan.outputSize,
      width: plan.outputSize,
      channels: plan.outputChannels,
      values,
    });
    activation = values;
  }
  const denseOutputs: Float32Array[] = [];
  for (let layer = 0; layer < denseLayers; layer += 1) {
    const isOutput = layer === denseLayers - 1;
    const values = denseLayer(
      activation,
      weights.dense[layer],
      widths[layer],
      isOutput ? null : selectedActivation,
    );
    denseOutputs.push(values);
    activation = values;
  }
  const logits = denseOutputs[denseLayers - 1];
  const probabilities = softmax(logits);
  const denseActivations = denseOutputs.slice(0, denseLayers - 1);
  denseActivations.push(probabilities);
  return { featureMaps, denseActivations, logits, probabilities };
}
