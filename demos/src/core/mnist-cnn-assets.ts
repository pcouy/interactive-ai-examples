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
 * Disposition des poids attendue dans le manifeste (voir `mnist-cnn.ts`) :
 * - convolutions en HWIO [kernelY][kernelX][entrée][sortie] ;
 * - denses en [entrée][sortie] ;
 * - première dense déjà réordonnée en aplatissement NHWC.
 */

import {
  CONV_KERNEL_SIZE,
  CONV_PADDINGS,
  CONV_STRIDES,
  convOutputLength,
  HIDDEN_DENSE_SIZE,
  LOGIT_COUNT,
  MNIST_INPUT_SIZE,
  type MnistCnnConfig,
  type MnistCnnFilters,
  type MnistCnnLayerWeights,
  type MnistCnnWeights,
  mnistCnnParameterCount,
  validateFilterCounts,
} from './mnist-cnn.js';

/** Version de schéma acceptée par ce lecteur. */
export const MNIST_CNN_MANIFEST_SCHEMA_VERSION = 1;

/** Manifeste validé, prêt pour `forwardMnistCnn`. */
export interface MnistCnnAssets {
  readonly config: MnistCnnConfig;
  readonly weights: MnistCnnWeights;
}

const LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4', 'dense1', 'dense2'] as const;
type LayerName = (typeof LAYER_NAMES)[number];

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

/** Longueurs attendues de chaque noyau et biais pour un tuple de filtres. */
function layerShapes(filters: MnistCnnFilters): Record<LayerName, LayerShape> {
  const convShapes: LayerShape[] = [];
  let inputSize = MNIST_INPUT_SIZE;
  let inputChannels = 1;
  for (let layer = 0; layer < filters.length; layer += 1) {
    const outputSize = convOutputLength(inputSize, CONV_STRIDES[layer], CONV_PADDINGS[layer]);
    const outputChannels = filters[layer];
    convShapes.push({
      kernel: CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * inputChannels * outputChannels,
      bias: outputChannels,
    });
    inputSize = outputSize;
    inputChannels = outputChannels;
  }
  const denseInputLength = inputSize * inputSize * inputChannels;
  return {
    conv1: convShapes[0],
    conv2: convShapes[1],
    conv3: convShapes[2],
    conv4: convShapes[3],
    dense1: { kernel: denseInputLength * HIDDEN_DENSE_SIZE, bias: HIDDEN_DENSE_SIZE },
    dense2: { kernel: HIDDEN_DENSE_SIZE * LOGIT_COUNT, bias: LOGIT_COUNT },
  };
}

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
  try {
    validateFilterCounts(filters as unknown as MnistCnnFilters);
  } catch (error) {
    throw new RangeError(
      `mnist-cnn-assets: unsupported architecture filters, ${(error as Error).message}`,
    );
  }
  return filters as unknown as MnistCnnFilters;
}

/**
 * Valide un manifeste déjà désérialisé et le convertit en configuration et
 * poids exploitables par `forwardMnistCnn`. Rejette une version de schéma
 * inconnue, des clés racine inconnues, une architecture non prise en charge,
 * des longueurs de noyau ou de biais incorrectes, un nombre de paramètres
 * incohérent et toute valeur non finie.
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

  const expectedParameters = mnistCnnParameterCount(filters);
  if (root.parameterCount !== expectedParameters) {
    throw new RangeError(
      `mnist-cnn-assets: parameterCount ${String(root.parameterCount)} does not match the ${expectedParameters} parameters implied by the architecture.`,
    );
  }

  const layers = requireObject(root.layers, 'layers');
  const extraLayers = Object.keys(layers).filter(
    (key) => !(LAYER_NAMES as readonly string[]).includes(key),
  );
  if (extraLayers.length > 0) {
    throw new RangeError(`mnist-cnn-assets: unsupported layers ${JSON.stringify(extraLayers)}.`);
  }

  const shapes = layerShapes(filters);
  const weights = {} as Record<LayerName, MnistCnnLayerWeights>;
  for (const name of LAYER_NAMES) {
    const layer = requireObject(layers[name], `layers.${name}`);
    weights[name] = {
      kernel: readNumberArray(layer.kernel, `${name} kernel`, shapes[name].kernel),
      bias: readNumberArray(layer.bias, `${name} bias`, shapes[name].bias),
    };
  }

  return {
    config: { filters },
    weights: {
      conv1: weights.conv1,
      conv2: weights.conv2,
      conv3: weights.conv3,
      conv4: weights.conv4,
      dense1: weights.dense1,
      dense2: weights.dense2,
    },
  };
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
