/**
 * Chargement du manifeste de poids du CNN MNIST exporté par
 * `demos/scripts/export_mnist_cnn_checkpoint.py`.
 *
 * Le manifeste JSON est versionné (`schemaVersion`) et décrit à la fois
 * l'architecture (`filters`) et les poids. Ce module le valide entièrement
 * avant de le convertir en `MnistCnnConfig` + `MnistCnnWeights`, afin que le
 * core d'inférence ne reçoive jamais de données douteuses. Il n'importe rien
 * du DOM : il accepte un objet déjà désérialisé ou une chaîne JSON.
 *
 * Seule l'architecture de référence est acceptée : quatre convolutions avec
 * trois filtres, deux couches denses et une activation sigmoïde.
 *
 * Disposition des poids attendue dans le manifeste (voir `mnist-cnn.ts`) :
 * - convolutions en HWIO [kernelY][kernelX][entrée][sortie] ;
 * - denses en [entrée][sortie] ;
 * - première dense déjà réordonnée en aplatissement NHWC.
 */

import {
  CONV_KERNEL_SIZE,
  CONV_SCHEDULES,
  convOutputLength,
  DEFAULT_CONV_LAYERS,
  DEFAULT_DENSE_LAYERS,
  DEFAULT_FILTER_COUNTS,
  DEFAULT_MNIST_CNN_ACTIVATION,
  HIDDEN_DENSE_SIZE,
  LOGIT_COUNT,
  MNIST_INPUT_SIZE,
  type MnistCnnConfig,
  type MnistCnnFilters,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
  mnistCnnParameterCount,
} from './mnist-cnn.js';

/** Version de schéma acceptée par ce lecteur. */
export const MNIST_CNN_MANIFEST_SCHEMA_VERSION = 1;

/** Manifeste validé, prêt pour `forwardMnistCnn`. */
export interface MnistCnnAssets {
  readonly config: MnistCnnConfig;
  readonly weights: MnistCnnWeights;
}

/** Clés des couches de convolution du manifeste, dans l'ordre. */
const CONV_LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4'] as const;
/** Clés des couches denses du manifeste, sortie comprise, dans l'ordre. */
const DENSE_LAYER_NAMES = ['dense1', 'dense2'] as const;

/** Clés de couches acceptées par le manifeste versionné. */
const LAYER_NAMES: readonly string[] = [...CONV_LAYER_NAMES, ...DENSE_LAYER_NAMES];

/** Clés racine acceptées par le manifeste versionné. */
const MANIFEST_ROOT_KEYS = ['schemaVersion', 'architecture', 'parameterCount', 'layers'] as const;

interface LayerShape {
  readonly kernel: number;
  readonly bias: number;
}

function requireObject(value: unknown, context: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`mnist-cnn-assets: ${context} must be a JSON object.`);
  }
  return value as Record<string, unknown>;
}

function readNumberArray(value: unknown, context: string, expectedLength: number): Float32Array {
  if (!Array.isArray(value)) {
    throw new TypeError(`mnist-cnn-assets: ${context} must be an array of numbers.`);
  }
  if (value.length !== expectedLength) {
    throw new RangeError(
      `mnist-cnn-assets: ${context} must have ${expectedLength} values, received ${value.length}.`,
    );
  }
  const result = new Float32Array(value.length);
  for (let index = 0; index < value.length; index += 1) {
    const entry = value[index];
    if (typeof entry !== 'number' || !Number.isFinite(entry)) {
      throw new RangeError(
        `mnist-cnn-assets: ${context} must contain only finite numbers, received ${String(entry)} at index ${index}.`,
      );
    }
    result[index] = entry;
  }
  return result;
}

/** Longueurs attendues de chaque noyau et biais pour l'architecture de référence. */
function layerShapes(): {
  readonly conv: readonly LayerShape[];
  readonly dense: readonly LayerShape[];
} {
  const schedule = CONV_SCHEDULES[DEFAULT_CONV_LAYERS];
  const conv: LayerShape[] = [];
  let inputSize = MNIST_INPUT_SIZE;
  let inputChannels = 1;
  for (let layer = 0; layer < schedule.length; layer += 1) {
    const spec = schedule[layer];
    const outputSize = convOutputLength(inputSize, spec.stride, spec.padding);
    const outputChannels = DEFAULT_FILTER_COUNTS[layer];
    conv.push({
      kernel: CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels * outputChannels,
      bias: outputChannels,
    });
    inputSize = outputSize;
    inputChannels = outputChannels;
  }
  const denseInputLength = inputSize * inputSize * inputChannels;
  return {
    conv,
    dense: [
      { kernel: denseInputLength * HIDDEN_DENSE_SIZE, bias: HIDDEN_DENSE_SIZE },
      { kernel: HIDDEN_DENSE_SIZE * LOGIT_COUNT, bias: LOGIT_COUNT },
    ],
  };
}

/**
 * Lit `architecture.filters` et exige l'architecture de référence : un tuple
 * de quatre filtres égal à `DEFAULT_FILTER_COUNTS`, comparé entrée par entrée.
 */
function readFilters(architecture: Record<string, unknown>): MnistCnnFilters {
  const extraKeys = Object.keys(architecture).filter((key) => key !== 'filters');
  if (extraKeys.length > 0) {
    throw new RangeError(
      `mnist-cnn-assets: unsupported architecture metadata ${JSON.stringify(extraKeys)}, only "filters" is supported.`,
    );
  }
  const filters = architecture.filters;
  if (!Array.isArray(filters)) {
    throw new TypeError('mnist-cnn-assets: architecture.filters must be an array of four counts.');
  }
  if (filters.length !== DEFAULT_FILTER_COUNTS.length) {
    throw new RangeError(
      `mnist-cnn-assets: unsupported architecture filters ${JSON.stringify(filters)}, expected ${JSON.stringify(DEFAULT_FILTER_COUNTS)}.`,
    );
  }
  for (let index = 0; index < filters.length; index += 1) {
    if (filters[index] !== DEFAULT_FILTER_COUNTS[index]) {
      throw new RangeError(
        `mnist-cnn-assets: unsupported architecture filters ${JSON.stringify(filters)}, expected ${JSON.stringify(DEFAULT_FILTER_COUNTS)}.`,
      );
    }
  }
  return DEFAULT_FILTER_COUNTS;
}

/**
 * Valide un manifeste déjà désérialisé et le convertit en configuration et
 * poids exploitables par `forwardMnistCnn`. Rejette une version de schéma
 * inconnue, des clés racine inconnues, une architecture non prise en charge,
 * des couches manquantes ou en trop, des longueurs de noyau ou de biais
 * incorrectes, un nombre de paramètres incohérent et toute valeur non finie.
 */
export function parseMnistCnnManifest(raw: unknown): MnistCnnAssets {
  const root = requireObject(raw, 'manifest');
  const version = root.schemaVersion;
  if (version !== MNIST_CNN_MANIFEST_SCHEMA_VERSION) {
    throw new RangeError(
      `mnist-cnn-assets: unsupported schema version ${String(version)}, expected ${MNIST_CNN_MANIFEST_SCHEMA_VERSION}.`,
    );
  }
  const extraRootKeys = Object.keys(root).filter(
    (key) => !(MANIFEST_ROOT_KEYS as readonly string[]).includes(key),
  );
  if (extraRootKeys.length > 0) {
    throw new RangeError(
      `mnist-cnn-assets: unsupported manifest keys ${JSON.stringify(extraRootKeys)}, only "schemaVersion", "architecture", "parameterCount" and "layers" are supported.`,
    );
  }
  const filters = readFilters(requireObject(root.architecture, 'architecture'));

  const config: MnistCnnConfig = {
    filters,
    convLayers: DEFAULT_CONV_LAYERS,
    denseLayers: DEFAULT_DENSE_LAYERS,
    activation: DEFAULT_MNIST_CNN_ACTIVATION,
  };

  const expectedParameters = mnistCnnParameterCount(config);
  if (root.parameterCount !== expectedParameters) {
    throw new RangeError(
      `mnist-cnn-assets: parameterCount ${String(root.parameterCount)} does not match the ${expectedParameters} parameters implied by the architecture.`,
    );
  }

  const layers = requireObject(root.layers, 'layers');
  const extraLayers = Object.keys(layers).filter((key) => !LAYER_NAMES.includes(key));
  if (extraLayers.length > 0) {
    throw new RangeError(`mnist-cnn-assets: unsupported layers ${JSON.stringify(extraLayers)}.`);
  }

  const shapes = layerShapes();
  const conv: MnistCnnLayerWeights[] = [];
  for (let layer = 0; layer < CONV_LAYER_NAMES.length; layer += 1) {
    const name = CONV_LAYER_NAMES[layer];
    const layerData = requireObject(layers[name], `layers.${name}`);
    conv.push({
      kernel: readNumberArray(layerData.kernel, `${name} kernel`, shapes.conv[layer].kernel),
      bias: readNumberArray(layerData.bias, `${name} bias`, shapes.conv[layer].bias),
    });
  }
  const dense: MnistCnnLayerWeights[] = [];
  for (let layer = 0; layer < DENSE_LAYER_NAMES.length; layer += 1) {
    const name = DENSE_LAYER_NAMES[layer];
    const layerData = requireObject(layers[name], `layers.${name}`);
    dense.push({
      kernel: readNumberArray(layerData.kernel, `${name} kernel`, shapes.dense[layer].kernel),
      bias: readNumberArray(layerData.bias, `${name} bias`, shapes.dense[layer].bias),
    });
  }

  return { config, weights: { conv, dense } };
}

/** Variante de `parseMnistCnnManifest` qui accepte une chaîne JSON. */
export function parseMnistCnnManifestJson(text: string): MnistCnnAssets {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new SyntaxError(`mnist-cnn-assets: invalid JSON manifest, ${(error as Error).message}`);
  }
  return parseMnistCnnManifest(raw);
}
