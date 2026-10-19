/**
 * Adaptateur d'entraînement TensorFlow.js du CNN MNIST de référence décrit
 * dans `tf-playground/mnist/train.py` : quatre convolutions 3×3 suivies d'une
 * sigmoïde (pas 1, 2, 1, 2 ; remplissage zéro symétrique de 1 sur la deuxième
 * seulement), dropout 0.2, aplatissement, dense 8 + sigmoïde, puis dix logits.
 * Perte d'entropie croisée softmax, optimiseur Adam au pas 3e-3.
 *
 * Ce module n'importe TensorFlow.js ni le DOM au niveau statique : les imports
 * de `@tensorflow/tfjs` et du contrat `mnist-cnn.ts` sont des imports de types,
 * effacés à la compilation. Seul `createMnistCnnTfjsTrainer` charge réellement
 * TensorFlow.js, par import dynamique. Vite peut donc isoler la bibliothèque
 * dans un chunk paresseux distinct.
 *
 * L'entraînement est asynchrone et borné : la boucle traite un lot, rend la
 * main à la boucle d'évènements, puis vérifie le drapeau d'annulation avant le
 * lot suivant. Aucune époque entière ne s'exécute dans un seul tick.
 */

import type { AdamOptimizer, Sequential, Tensor, Variable } from '@tensorflow/tfjs';
import type { MnistCnnFilters, MnistCnnLayerWeights, MnistCnnWeights } from './mnist-cnn.js';
import type { MnistSample } from './mnist-data.js';

type TfModule = typeof import('@tensorflow/tfjs');
type CoreModule = typeof import('./mnist-cnn.js');
type TfLayer = Sequential['layers'][number];

/** Fraction d'unités désactivées pendant l'entraînement (`Dropout(p=0.2)`). */
export const MNIST_DROPOUT_RATE = 0.2;

/** Graine par défaut, même convention que les autres modules du dossier `core`. */
export const DEFAULT_TRAINING_SEED = 42;

/** Configuration figée d'un entraînement MNIST. */
export interface MnistCnnTrainingConfig {
  /** Nombre de filtres des quatre convolutions. */
  readonly filters: MnistCnnFilters;
  /** Pas d'apprentissage Adam. */
  readonly learningRate: number;
  /** Nombre d'exemples par lot. */
  readonly batchSize: number;
  /** Graine optionnelle : même graine et même configuration donnent les mêmes poids. */
  readonly seed?: number;
}

/** État d'un entraînement, tel que rapporté par `MnistTrainingProgress`. */
export type MnistTrainingStatus = 'running' | 'completed' | 'cancelled';

/** Instantané de progression émis entre deux lots et à la fin de l'entraînement. */
export interface MnistTrainingProgress {
  readonly status: MnistTrainingStatus;
  /** Nombre cumulé de lots traités depuis le début de l'entraînement. */
  readonly batchesProcessed: number;
  /** Perte d'entraînement du dernier lot traité. */
  readonly loss: number;
  /** Perte de validation, ou `null` si aucun jeu de validation n'est fourni. */
  readonly validationLoss: number | null;
  /** Exactitude de validation dans `[0, 1]`, ou `null` sans jeu de validation. */
  readonly validationAccuracy: number | null;
  /** Millisecondes écoulées depuis le début de l'entraînement. */
  readonly elapsedMs: number;
}

/** Options d'un appel à `MnistCnnTfjsTrainer.train`. */
export interface MnistTrainOptions {
  /** Exemples d'entraînement (images aplaties 28×28, libellés 0 à 9). */
  readonly train: MnistSample;
  /** Exemples de validation évalués à la fin de chaque époque. */
  readonly validation?: MnistSample;
  /** Nombre d'époques, au moins 1. Défaut : 1. */
  readonly epochs?: number;
  /** Rappel appelé après chaque lot et une dernière fois en fin d'entraînement. */
  readonly onProgress?: (progress: MnistTrainingProgress) => void;
  /**
   * Cession de la boucle d'évènements entre deux lots. Défaut : un macrotâche
   * `setTimeout(0)`. Injectable pour rendre les tests déterministes.
   */
  readonly yieldControl?: () => Promise<void>;
}

/** Cède la main à la boucle d'évènements sans dépendre d'une API DOM. */
function defaultYieldControl(): Promise<void> {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
}

/** Générateur pseudo-aléatoire déterministe, même implantation que les autres modules. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Mélange de Fisher-Yates, même convention que `mnist-data.ts`. */
function shuffledIndices(length: number, rng: () => number): number[] {
  const indices: number[] = new Array<number>(length);
  for (let index = 0; index < length; index += 1) {
    indices[index] = index;
  }
  for (let index = length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1));
    const temporary = indices[index];
    indices[index] = indices[swap];
    indices[swap] = temporary;
  }
  return indices;
}

function assertConfig(config: MnistCnnTrainingConfig, core: CoreModule): void {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('MNIST TFJS training config is required.');
  }
  core.validateFilterCounts(config.filters);
  if (!Number.isFinite(config.learningRate) || config.learningRate <= 0) {
    throw new RangeError(
      `MNIST TFJS learning rate must be a positive finite number, received ${String(config.learningRate)}.`,
    );
  }
  if (!Number.isInteger(config.batchSize) || config.batchSize < 1) {
    throw new RangeError(
      `MNIST TFJS batch size must be a positive integer, received ${String(config.batchSize)}.`,
    );
  }
  if (config.seed !== undefined && (!Number.isInteger(config.seed) || config.seed < 0)) {
    throw new RangeError(
      `MNIST TFJS seed must be a non-negative integer, received ${String(config.seed)}.`,
    );
  }
}

function assertSample(sample: MnistSample, label: string, core: CoreModule): void {
  if (sample === null || typeof sample !== 'object') {
    throw new TypeError(`MNIST TFJS ${label} sample is required.`);
  }
  if (!Number.isInteger(sample.count) || sample.count < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} sample count must be a positive integer, received ${String(sample.count)}.`,
    );
  }
  const expectedImages = sample.count * core.MNIST_INPUT_LENGTH;
  if (sample.images.length !== expectedImages) {
    throw new RangeError(
      `MNIST TFJS ${label} images must have ${expectedImages} values, received ${sample.images.length}.`,
    );
  }
  for (let index = 0; index < sample.images.length; index += 1) {
    if (!Number.isFinite(sample.images[index])) {
      throw new RangeError(
        `MNIST TFJS ${label} images must contain only finite values, received ${sample.images[index]} at index ${index}.`,
      );
    }
  }
  if (sample.labels.length !== sample.count) {
    throw new RangeError(
      `MNIST TFJS ${label} labels must have ${sample.count} values, received ${sample.labels.length}.`,
    );
  }
  for (let index = 0; index < sample.labels.length; index += 1) {
    const value = sample.labels[index];
    if (!Number.isInteger(value) || value < 0 || value > core.LOGIT_COUNT - 1) {
      throw new RangeError(
        `MNIST TFJS ${label} labels must be integers from 0 to ${core.LOGIT_COUNT - 1}, received ${value} at index ${index}.`,
      );
    }
  }
}

/**
 * Construit le modèle séquentiel de référence. Le remplissage zéro de conv2 est
 * posé explicitement avant la convolution : le mode `same` de TensorFlow.js
 * ajoute un remplissage asymétrique (0 en haut, 1 en bas) qui décalerait les
 * cartes par rapport au remplissage symétrique de PyTorch.
 */
function buildModel(tf: TfModule, core: CoreModule, config: MnistCnnTrainingConfig): Sequential {
  const seed = config.seed ?? DEFAULT_TRAINING_SEED;
  const kernelInitializer = tf.initializers.glorotUniform({ seed });
  const biasInitializer = tf.initializers.zeros();
  const model = tf.sequential();
  let inputShape: number[] | undefined = [core.MNIST_INPUT_SIZE, core.MNIST_INPUT_SIZE, 1];
  for (let index = 0; index < config.filters.length; index += 1) {
    const stride = core.CONV_STRIDES[index];
    const padding = core.CONV_PADDINGS[index];
    if (padding > 0) {
      model.add(
        tf.layers.zeroPadding2d({
          padding: [
            [padding, padding],
            [padding, padding],
          ],
          inputShape,
        }),
      );
      inputShape = undefined;
    }
    model.add(
      tf.layers.conv2d({
        filters: config.filters[index],
        kernelSize: core.CONV_KERNEL_SIZE,
        strides: stride,
        padding: 'valid',
        activation: 'sigmoid',
        kernelInitializer,
        biasInitializer,
        inputShape,
      }),
    );
    inputShape = undefined;
  }
  model.add(tf.layers.dropout({ rate: MNIST_DROPOUT_RATE, seed }));
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: core.HIDDEN_DENSE_SIZE,
      activation: 'sigmoid',
      kernelInitializer,
      biasInitializer,
    }),
  );
  model.add(tf.layers.dense({ units: core.LOGIT_COUNT, kernelInitializer, biasInitializer }));
  return model;
}

/**
 * Lit les poids d'une couche dans la disposition attendue par `mnist-cnn.ts`.
 *
 * TensorFlow.js stocke les noyaux de convolution en HWIO
 * `[ky][kx][canalEntrée][canalSortie]` et les noyaux denses en
 * `[caractéristiqueEntrée][caractéristiqueSortie]` : l'aplatissement ligne-major
 * donne donc exactement la disposition de référence, sans transposition. De
 * plus, le `flatten` TensorFlow.js d'un tenseur NHWC (canal le plus rapide)
 * produit le même ordre que l'aplatissement de `mnist-cnn.ts`, donc la ligne
 * `i` du noyau dense1 correspond à la caractéristique NHWC `i`.
 */
function readLayerWeights(layer: TfLayer): MnistCnnLayerWeights {
  const [kernel, bias] = layer.getWeights();
  return {
    kernel: new Float32Array(kernel.dataSync()),
    bias: new Float32Array(bias.dataSync()),
  };
}

/**
 * Entraîneur asynchrone du CNN MNIST sur TensorFlow.js.
 *
 * Le modèle est construit paresseusement au premier usage. `reset` libère le
 * modèle et l'optimiseur (poids et variables d'Adam), `dispose` ferme
 * définitivement l'instance.
 */
export class MnistCnnTfjsTrainer {
  private readonly tf: TfModule;

  private readonly core: CoreModule;

  private readonly config: MnistCnnTrainingConfig;

  private readonly seed: number;

  private model: Sequential | null = null;

  private optimizer: AdamOptimizer | null = null;

  private running = false;

  private cancelled = false;

  private disposed = false;

  /**
   * Construit un entraîneur à partir des modules déjà chargés. Préférez
   * `createMnistCnnTfjsTrainer`, qui importe TensorFlow.js dynamiquement.
   */
  constructor(tf: TfModule, core: CoreModule, config: MnistCnnTrainingConfig) {
    assertConfig(config, core);
    this.tf = tf;
    this.core = core;
    this.config = config;
    this.seed = config.seed ?? DEFAULT_TRAINING_SEED;
  }

  /** Modèle TensorFlow.js sous-jacent, pour inspection (formes, débogage). */
  get graph(): Sequential | null {
    return this.model;
  }

  /** Vrai après `dispose`. */
  get isDisposed(): boolean {
    return this.disposed;
  }

  /**
   * Entraîne le modèle par lots bornés. La boucle rend la main entre deux lots
   * et vérifie le drapeau d'annulation avant chaque lot.
   */
  async train(options: MnistTrainOptions): Promise<MnistTrainingProgress> {
    this.assertUsable();
    if (this.running) {
      throw new Error('MNIST TFJS trainer is already training.');
    }
    if (options === null || typeof options !== 'object') {
      throw new TypeError('MNIST TFJS train options are required.');
    }
    const epochs = options.epochs ?? 1;
    if (!Number.isInteger(epochs) || epochs < 1) {
      throw new RangeError(`MNIST TFJS epochs must be a positive integer, received ${epochs}.`);
    }
    assertSample(options.train, 'train', this.core);
    if (options.validation !== undefined) {
      assertSample(options.validation, 'validation', this.core);
    }

    const yieldControl = options.yieldControl ?? defaultYieldControl;
    const model = this.ensureModel();
    const optimizer = this.optimizer;
    if (optimizer === null) {
      throw new Error('MNIST TFJS optimizer was not initialised.');
    }
    // `getWeights(true)` renvoie les variables d'entraînement elles-mêmes
    // (LayerVariable.read() retourne `val`) ; le type déclaré les expose comme
    // des `Tensor`, d'où la conversion vers `Variable`.
    const trainable = model.getWeights(true) as unknown as Variable[];
    const { batchSize } = this.config;
    const count = options.train.count;
    const rng = mulberry32(this.seed);
    const startedAt = Date.now();

    // L'allocation précède la bascule de `running` : si un tenseur ne peut pas
    // être créé, l'entraîneur ne reste pas bloqué sur « already training ».
    const images = this.tf.tensor4d(options.train.images, [
      count,
      this.core.MNIST_INPUT_SIZE,
      this.core.MNIST_INPUT_SIZE,
      1,
    ]);
    const labels = this.tf.tensor1d(options.train.labels, 'int32');

    this.running = true;
    this.cancelled = false;

    let batchesProcessed = 0;
    let loss = Number.NaN;
    let validationLoss: number | null = null;
    let validationAccuracy: number | null = null;
    let status: MnistTrainingStatus = 'completed';

    try {
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        const order = shuffledIndices(count, rng);
        for (let start = 0; start < count; start += batchSize) {
          if (this.cancelled) {
            status = 'cancelled';
            break;
          }
          const end = Math.min(start + batchSize, count);
          loss = this.runBatch(
            model,
            optimizer,
            trainable,
            images,
            labels,
            order.slice(start, end),
          );
          batchesProcessed += 1;
          this.emit(options.onProgress, {
            status: 'running',
            batchesProcessed,
            loss,
            validationLoss,
            validationAccuracy,
            elapsedMs: Date.now() - startedAt,
          });
          await yieldControl();
        }
        if (this.cancelled) {
          status = 'cancelled';
          break;
        }
        if (options.validation !== undefined) {
          const metrics = this.evaluate(model, options.validation);
          validationLoss = metrics.loss;
          validationAccuracy = metrics.accuracy;
        }
      }
    } finally {
      images.dispose();
      labels.dispose();
      this.running = false;
    }

    const progress: MnistTrainingProgress = {
      status,
      batchesProcessed,
      loss,
      validationLoss,
      validationAccuracy,
      elapsedMs: Date.now() - startedAt,
    };
    this.emit(options.onProgress, progress);
    return progress;
  }

  /** Demande l'arrêt : effectif avant le lot suivant. */
  async cancel(): Promise<void> {
    this.cancelled = true;
  }

  /** Exporte les poids dans la disposition `MnistCnnWeights` de `mnist-cnn.ts`. */
  async exportWeights(): Promise<MnistCnnWeights> {
    this.assertUsable();
    const model = this.ensureModel();
    const convolutions = model.layers.filter((layer) => layer.getClassName() === 'Conv2D');
    const denses = model.layers.filter((layer) => layer.getClassName() === 'Dense');
    if (convolutions.length !== 4 || denses.length !== 2) {
      throw new Error(
        `MNIST TFJS model exposes ${convolutions.length} convolutions and ${denses.length} dense layers, expected 4 and 2.`,
      );
    }
    return {
      conv1: readLayerWeights(convolutions[0]),
      conv2: readLayerWeights(convolutions[1]),
      conv3: readLayerWeights(convolutions[2]),
      conv4: readLayerWeights(convolutions[3]),
      dense1: readLayerWeights(denses[0]),
      dense2: readLayerWeights(denses[1]),
    };
  }

  /**
   * Retourne les probabilités softmax (count × 10, ligne-major) pour
   * l'affichage. Le dropout est désactivé.
   */
  async predict(sample: MnistSample): Promise<Float32Array> {
    this.assertUsable();
    assertSample(sample, 'predict', this.core);
    const model = this.ensureModel();
    const images = this.tf.tensor4d(sample.images, [
      sample.count,
      this.core.MNIST_INPUT_SIZE,
      this.core.MNIST_INPUT_SIZE,
      1,
    ]);
    let probabilities = new Float32Array(0);
    this.tf.tidy(() => {
      const logits = model.apply(images, { training: false }) as Tensor;
      probabilities = new Float32Array(logits.softmax().dataSync());
    });
    images.dispose();
    return probabilities;
  }

  /** Libère le modèle et l'optimiseur ; l'instance reste réutilisable. */
  reset(): void {
    if (this.optimizer !== null) {
      this.optimizer.dispose();
      this.optimizer = null;
    }
    if (this.model !== null) {
      this.model.dispose();
      this.model = null;
    }
    this.cancelled = false;
  }

  /** Libère le modèle, l'optimiseur et marque l'instance comme inutilisable. */
  async dispose(): Promise<void> {
    this.reset();
    this.disposed = true;
  }

  private ensureModel(): Sequential {
    if (this.model === null) {
      this.model = buildModel(this.tf, this.core, this.config);
      this.optimizer = this.tf.train.adam(this.config.learningRate);
    }
    return this.model;
  }

  /** Un pas d'Adam sur un lot ; retourne la perte du lot. */
  private runBatch(
    model: Sequential,
    optimizer: AdamOptimizer,
    trainable: readonly Variable[],
    images: Tensor,
    labels: Tensor,
    batchIndices: readonly number[],
  ): number {
    const tf = this.tf;
    const indexTensor = tf.tensor1d(batchIndices as number[], 'int32');
    const labelBatch = tf.gather(labels, indexTensor);
    const xBatch = tf.gather(images, indexTensor);
    const yBatch = tf.oneHot(labelBatch, this.core.LOGIT_COUNT);
    // Le `finally` libère les tenseurs du lot même si la mise à jour échoue.
    try {
      const cost = optimizer.minimize(
        () => {
          const logits = model.apply(xBatch, { training: true }) as Tensor;
          return tf.losses.softmaxCrossEntropy(yBatch, logits).mean();
        },
        true,
        trainable as Variable[],
      );
      if (cost === null) {
        return Number.NaN;
      }
      try {
        return cost.dataSync()[0];
      } finally {
        cost.dispose();
      }
    } finally {
      indexTensor.dispose();
      labelBatch.dispose();
      xBatch.dispose();
      yBatch.dispose();
    }
  }

  /** Évalue perte et exactitude sur un jeu de validation, dropout désactivé. */
  private evaluate(model: Sequential, sample: MnistSample): { loss: number; accuracy: number } {
    const tf = this.tf;
    const images = tf.tensor4d(sample.images, [
      sample.count,
      this.core.MNIST_INPUT_SIZE,
      this.core.MNIST_INPUT_SIZE,
      1,
    ]);
    const labels = tf.tensor1d(sample.labels, 'int32');
    let loss = Number.NaN;
    let correct = 0;
    tf.tidy(() => {
      const logits = model.apply(images, { training: false }) as Tensor;
      const oneHot = tf.oneHot(labels, this.core.LOGIT_COUNT);
      loss = tf.losses.softmaxCrossEntropy(oneHot, logits).mean().dataSync()[0];
      const predictions = logits.argMax(1);
      correct = predictions.equal(labels).sum().dataSync()[0];
    });
    images.dispose();
    labels.dispose();
    return { loss, accuracy: correct / sample.count };
  }

  private emit(
    onProgress: ((progress: MnistTrainingProgress) => void) | undefined,
    progress: MnistTrainingProgress,
  ): void {
    if (onProgress !== undefined) {
      onProgress(progress);
    }
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error('MNIST TFJS trainer is disposed.');
    }
  }
}

/**
 * Charge TensorFlow.js et le contrat `mnist-cnn.ts` dynamiquement, puis
 * construit un entraîneur. C'est le seul point d'entrée qui charge
 * TensorFlow.js : le graphe d'imports statique du module reste léger.
 */
export async function createMnistCnnTfjsTrainer(
  config: MnistCnnTrainingConfig,
): Promise<MnistCnnTfjsTrainer> {
  const [tf, core] = await Promise.all([import('@tensorflow/tfjs'), import('./mnist-cnn.js')]);
  return new MnistCnnTfjsTrainer(tf, core, config);
}
