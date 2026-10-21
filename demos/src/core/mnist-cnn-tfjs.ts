/**
 * Adaptateur d'entraînement TensorFlow.js du CNN MNIST de référence décrit
 * dans `tf-playground/mnist/train.py`. Le nombre de couches est paramétrable :
 * 0 à 4 convolutions 3×3 suivies de la fonction d'activation choisie
 * (sigmoïde par défaut, `tanh` possible ; pas et remplissage tirés de
 * `CONV_SCHEDULES`), puis dropout 0.2, aplatissement, puis 1 à 3 couches
 * denses (8 unités pour chaque couche cachée, la dernière donnant le nombre de
 * classes de sortie demandé, dix par défaut).
 * L'architecture par défaut reste celle de référence : quatre convolutions et
 * deux couches denses. Perte d'entropie croisée softmax, optimiseur Adam au
 * pas 3e-3.
 *
 * Ce module n'importe ni TensorFlow.js ni le DOM au niveau statique : seul
 * `@tensorflow/tfjs` est importé dynamiquement, par `createMnistCnnTfjsTrainer`.
 * Le contrat `mnist-cnn.ts` est importé statiquement (le composant l'importe
 * déjà statiquement) : Vite isole TensorFlow.js dans un chunk paresseux distinct.
 *
 * L'entraînement est asynchrone et borné : un tampon circulaire garde au plus
 * `prefetchDepth` lots matérialisés d'avance. La boucle prend le plus ancien,
 * traite une mise à jour, recharge le tampon pendant la fenêtre de cession,
 * puis vérifie le drapeau d'annulation avant le lot suivant. Aucune époque
 * entière ne s'exécute dans un seul tick et les images ne sont jamais
 * matérialisées en entier : l'entraînement lit des lots bornés et la validation
 * est lue par blocs.
 *
 * La matérialisation des lots (CPU) et le calcul (GPU) partagent un seul fil
 * JavaScript : le préchargement retire l'étape de lecture du chemin critique
 * d'une itération, mais ne la recouvre pas avec le calcul. De même, la perte
 * n'est jamais lue par `dataSync` : les tenseurs de coût d'une fenêtre de
 * `lossSampleEvery` lots sont moyennés en tenseurs (`tf.stack` puis moyenne),
 * et un seul scalaire est ramené par `await mean.data()`. Une seule lecture par
 * fenêtre laisse la file GPU se remplir au lieu de la vider à chaque lot.
 */

import type { AdamOptimizer, Sequential, Tensor, Variable } from '@tensorflow/tfjs';
import type {
  MnistCnnActivation,
  MnistCnnFilters,
  MnistCnnLayerWeights,
  MnistCnnWeights,
} from './mnist-cnn.js';
import * as core from './mnist-cnn.js';
import type {
  RoundEntry,
  RoundRef,
  SegmentLoader,
  SegmentPlan,
  SegmentRound,
} from './mnist-cnn-segmented.js';
import type { MnistSample } from './mnist-data.js';

type TfModule = typeof import('@tensorflow/tfjs');
type CoreModule = typeof core;
type TfLayer = Sequential['layers'][number];

/** Fraction d'unités désactivées pendant l'entraînement (`Dropout(p=0.2)`). */
export const MNIST_DROPOUT_RATE = 0.2;

/** Graine par défaut, même convention que les autres modules du dossier `core`. */
export const DEFAULT_TRAINING_SEED = 42;

/** Taille des blocs lus sur la source de validation (accumulation par blocs). */
export const MNIST_VALIDATION_CHUNK_SIZE = 512;

/** Nombre maximal d'exemples lus pour mesurer l'exactitude d'entraînement d'une époque. */
export const MNIST_TRAIN_ACCURACY_SAMPLE_SIZE = 1024;

/** Profondeur de préchargement par défaut (lots matérialisés d'avance). */
export const DEFAULT_PREFETCH_DEPTH = 3;

/** Profondeur de préchargement minimale : `1` lit un lot à la fois, sans recouvrement. */
export const MIN_PREFETCH_DEPTH = 1;

/** Profondeur de préchargement maximale acceptée. */
export const MAX_PREFETCH_DEPTH = 16;

/** Intervalle minimal entre deux évènements `running`, par défaut, en millisecondes. */
export const DEFAULT_PROGRESS_INTERVAL_MS = 100;

/** Nombre de lots moyennés avant chaque lecture de perte, par défaut. */
export const DEFAULT_LOSS_SAMPLE_EVERY = 4;

/** Nombre minimal de lots moyennés avant chaque lecture de perte. */
export const MIN_LOSS_SAMPLE_EVERY = 1;

/** Nombre maximal de lots moyennés avant chaque lecture de perte. */
export const MAX_LOSS_SAMPLE_EVERY = 64;

/** Configuration figée d'un entraînement MNIST. */
export interface MnistCnnTrainingConfig {
  /** Nombre de filtres des quatre convolutions. */
  readonly filters: MnistCnnFilters;
  /** Pas d'apprentissage Adam. */
  readonly learningRate: number;
  /** Nombre d'exemples par lot. */
  readonly batchSize: number;
  /** Fonction d'activation des convolutions et de la couche dense cachée. */
  readonly activation: MnistCnnActivation;
  /** Nombre de couches de convolution, de 0 à 4. */
  readonly convLayers: number;
  /** Nombre de couches denses, sortie comprise, de 1 à 3. */
  readonly denseLayers: number;
  /**
   * Largeurs des couches denses cachées, dans l'ordre, couche de sortie exclue.
   * Défaut : `[8, 8]`. Une entrée manquante est remplacée par 8 ; les entrées
   * au-delà du nombre de couches cachées sont ignorées.
   */
  readonly denseSizes?: readonly number[];
  /**
   * Nombre de classes de sortie, de 2 à `MAX_OUTPUT_CLASSES`. Défaut :
   * `core.LOGIT_COUNT` (10).
   */
  readonly outputClasses?: number;
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
  /** Perte d'entraînement moyenne de la dernière fenêtre lue. */
  readonly loss: number;
  /** Perte d'entraînement moyenne de la dernière époque achevée, ou `null`. */
  readonly epochLoss: number | null;
  /**
   * Exactitude d'entraînement de la dernière époque achevée dans `[0, 1]`, ou
   * `null` avant la première époque achevée.
   */
  readonly epochTrainAccuracy: number | null;
  /** Nombre d'époques achevées depuis le début de l'entraînement. */
  readonly completedEpochs: number;
  /** Perte de validation, ou `null` si aucun jeu de validation n'est fourni. */
  readonly validationLoss: number | null;
  /** Exactitude de validation dans `[0, 1]`, ou `null` sans jeu de validation. */
  readonly validationAccuracy: number | null;
  /** Millisecondes écoulées depuis le début de l'entraînement. */
  readonly elapsedMs: number;
  /** Indice du round en cours dans l'époque, de 0 à `roundsPerEpoch - 1`. */
  readonly roundIndex?: number;
  /** Nombre de rounds par époque du plan segmenté. */
  readonly roundsPerEpoch?: number;
  /** Nombre cumulé de rounds achevés depuis le début de l'entraînement. */
  readonly completedRounds?: number;
  /** Nombre de fragments couverts par la dernière fenêtre de perte lue. */
  readonly windowShards?: number;
}

/**
 * Source d'exemples lue par lots, sans matérialiser tout le jeu. `readBatch`
 * reçoit une plage semi-ouverte `[start, end)` et retourne exactement
 * `end - start` exemples.
 */
export interface MnistTrainSource {
  /** Nombre total d'exemples disponibles. */
  readonly count: number;
  /** Retourne les exemples de la plage semi-ouverte `[start, end)`. */
  readBatch(start: number, end: number): MnistSample;
  /**
   * Optionnel : réordonne les exemples de la source. Appelé une fois au début
   * de chaque époque avec une graine déterministe dérivée de la graine
   * d'entraînement et de l'indice d'époque. Quand il est absent, l'adaptateur
   * mélange lui-même l'ordre des lots.
   */
  reshuffle?(seed: number): void;
}

/** Options d'un appel à `MnistCnnTfjsTrainer.train`. */
export interface MnistTrainOptions {
  /** Source d'entraînement lue par lots (voir `MnistTrainSource`). */
  readonly train: MnistTrainSource;
  /** Source de validation lue par blocs à la fin de chaque époque. */
  readonly validation?: MnistTrainSource;
  /** Nombre d'époques, au moins 1. Défaut : 1. */
  readonly epochs?: number;
  /**
   * Nombre de lots matérialisés d'avance (tampon circulaire borné). Défaut :
   * `DEFAULT_PREFETCH_DEPTH` (3), borné entre `MIN_PREFETCH_DEPTH` (1) et
   * `MAX_PREFETCH_DEPTH` (16). `1` désactive le recouvrement.
   */
  readonly prefetchDepth?: number;
  /**
   * Nombre de lots dont la perte est moyennée avant chaque lecture. Défaut :
   * `DEFAULT_LOSS_SAMPLE_EVERY` (1), borné entre `MIN_LOSS_SAMPLE_EVERY` (1) et
   * `MAX_LOSS_SAMPLE_EVERY` (64). Une valeur plus grande réduit le nombre de
   * lectures et laisse la file GPU se remplir.
   */
  readonly lossSampleEvery?: number;
  /**
   * Intervalle minimal entre deux évènements `running`, en millisecondes.
   * Défaut : `DEFAULT_PROGRESS_INTERVAL_MS` (100). Un évènement est toujours
   * émis à la fin de chaque époque, puis à la fin de l'entraînement.
   */
  readonly progressIntervalMs?: number;
  /** Rappel appelé au plus une fois par intervalle, et toujours en fin d'entraînement. */
  readonly onProgress?: (progress: MnistTrainingProgress) => void;
  /**
   * Cession de la boucle d'évènements entre deux lots. Défaut : un macrotâche
   * `setTimeout(0)`. Injectable pour rendre les tests déterministes.
   */
  readonly yieldControl?: () => Promise<void>;
}

/** Options d'un appel à `MnistCnnTfjsTrainer.trainSegmented`. */
export interface MnistSegmentedTrainOptions {
  /** Plan des fragments d'entraînement. */
  readonly trainPlan: SegmentPlan;
  /** Chargeur des rounds d'entraînement. */
  readonly trainLoader: SegmentLoader;
  /** Jeu de validation déjà en mémoire, utilisé sans plan segmenté. */
  readonly validation?: MnistSample;
  /** Plan des fragments de validation, à fournir avec `validationLoader`. */
  readonly validationPlan?: SegmentPlan;
  /** Chargeur des rounds de validation, à fournir avec `validationPlan`. */
  readonly validationLoader?: SegmentLoader;
  /** Nombre d'époques, au moins 1. */
  readonly epochs: number;
  /** Rappel de progression, appelé au plus une fois par intervalle et en fin d'époque. */
  readonly onProgress: (progress: MnistTrainingProgress) => void;
  /** Nombre de lots dont la perte est moyennée avant chaque lecture. */
  readonly lossSampleEvery?: number;
  /** Intervalle minimal entre deux évènements `running`, en millisecondes. */
  readonly progressIntervalMs?: number;
  /**
   * Nombre maximal d'exemples du dernier round lus pour mesurer l'exactitude
   * d'entraînement. Défaut : `MNIST_TRAIN_ACCURACY_SAMPLE_SIZE` (1024).
   */
  readonly trainAccuracySampleSize?: number;
}

/** Un lot matérialisé : ses propres tenseurs, possédés par le tampon de préchargement. */
interface MaterializedBatch {
  readonly images: Tensor;
  readonly labels: Tensor;
}

/** Libère les tenseurs d'un lot matérialisé. */
function disposeBatch(batch: MaterializedBatch): void {
  batch.images.dispose();
  batch.labels.dispose();
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

/**
 * Graine déterministe d'époque, dérivée de la graine d'entraînement et de
 * l'indice d'époque : deux entraînements de même graine donnent les mêmes
 * graines d'époque.
 */
function deriveEpochSeed(baseSeed: number, epoch: number): number {
  return (baseSeed + Math.imul(epoch + 1, 0x9e3779b1)) >>> 0;
}

function assertConfig(config: MnistCnnTrainingConfig, core: CoreModule): void {
  if (config === null || typeof config !== 'object') {
    throw new TypeError('MNIST TFJS training config is required.');
  }
  core.validateFilterCounts(config.filters);
  if (
    !Number.isInteger(config.convLayers) ||
    config.convLayers < 0 ||
    config.convLayers > core.MAX_CONV_LAYERS
  ) {
    throw new RangeError(
      `MNIST TFJS convLayers must be an integer from 0 to ${core.MAX_CONV_LAYERS}, received ${String(config.convLayers)}.`,
    );
  }
  if (
    !Number.isInteger(config.denseLayers) ||
    config.denseLayers < 1 ||
    config.denseLayers > core.MAX_DENSE_LAYERS
  ) {
    throw new RangeError(
      `MNIST TFJS denseLayers must be an integer from 1 to ${core.MAX_DENSE_LAYERS}, received ${String(config.denseLayers)}.`,
    );
  }
  core.validateDenseSizes(config.denseSizes);
  core.resolveOutputClasses(config);
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
  if (config.activation !== 'sigmoid' && config.activation !== 'tanh') {
    throw new RangeError(
      `MNIST TFJS activation must be "sigmoid" or "tanh", received ${String(config.activation)}.`,
    );
  }
  if (config.seed !== undefined && (!Number.isInteger(config.seed) || config.seed < 0)) {
    throw new RangeError(
      `MNIST TFJS seed must be a non-negative integer, received ${String(config.seed)}.`,
    );
  }
}

function assertSample(
  sample: MnistSample,
  label: string,
  core: CoreModule,
  classCount: number,
): void {
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
    if (!Number.isInteger(value) || value < 0 || value >= classCount) {
      throw new RangeError(
        `MNIST TFJS ${label} labels must be integers from 0 to ${classCount - 1}, received ${value} at index ${index}.`,
      );
    }
  }
}

/**
 * Adapte un échantillon déjà en mémoire en `MnistTrainSource`. Les plages
 * retournées sont des vues `subarray` sur les tampons de l'échantillon : aucune
 * copie n'est faite et le tenseur n'est créé qu'au moment où le lot est lu.
 */
export function sampleToTrainSource(sample: MnistSample): MnistTrainSource {
  if (sample === null || typeof sample !== 'object') {
    throw new TypeError('MNIST TFJS sample is required to build a train source.');
  }
  if (!Number.isInteger(sample.count) || sample.count < 1) {
    throw new RangeError(
      `MNIST TFJS sample count must be a positive integer, received ${String(sample.count)}.`,
    );
  }
  const stride = sample.images.length / sample.count;
  if (!Number.isInteger(stride) || stride < 1) {
    throw new RangeError(
      `MNIST TFJS sample images must split evenly into ${sample.count} examples, received ${sample.images.length} values.`,
    );
  }
  return {
    count: sample.count,
    readBatch(start: number, end: number): MnistSample {
      return {
        count: end - start,
        images: sample.images.subarray(start * stride, end * stride),
        labels: sample.labels.subarray(start, end),
      };
    },
  };
}

function assertTrainSource(source: MnistTrainSource, label: string): void {
  if (source === null || typeof source !== 'object') {
    throw new TypeError(`MNIST TFJS ${label} source is required.`);
  }
  if (!Number.isInteger(source.count) || source.count < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} source count must be a positive integer, received ${String(source.count)}.`,
    );
  }
  if (typeof source.readBatch !== 'function') {
    throw new TypeError(`MNIST TFJS ${label} source must expose a readBatch function.`);
  }
}

/** Lit un lot sur la source et vérifie qu'il contient exactement `end - start` exemples. */
function readSourceBatch(
  source: MnistTrainSource,
  start: number,
  end: number,
  label: string,
  core: CoreModule,
  classCount: number,
): MnistSample {
  const batch = source.readBatch(start, end);
  assertSample(batch, label, core, classCount);
  if (batch.count !== end - start) {
    throw new RangeError(
      `MNIST TFJS ${label} source readBatch(${start}, ${end}) returned ${batch.count} examples, expected ${end - start}.`,
    );
  }
  return batch;
}

/**
 * Fragments couverts par la plage `[start, end)` d'un round : le chargeur
 * concatène les fragments dans l'ordre de `entries`. Repli quand le round ne
 * fournit pas `shardsInRange`, par exemple un round contigu en mémoire.
 */
function shardsForRange(entries: readonly RoundEntry[], start: number, end: number): string[] {
  const files: string[] = [];
  let offset = 0;
  for (const entry of entries) {
    const next = offset + entry.count;
    if (start < next && end > offset) {
      files.push(entry.file);
    }
    offset = next;
    if (offset >= end) {
      break;
    }
  }
  return files;
}

/**
 * Nombre de fragments distincts couverts par `[start, end)`. Un round équilibré
 * répond lui-même ; un round contigu passe par `shardsForRange`.
 */
function shardsInRoundRange(
  round: SegmentRound,
  entries: readonly RoundEntry[],
  start: number,
  end: number,
): number {
  return round.shardsInRange?.(start, end) ?? shardsForRange(entries, start, end).length;
}

/** Adapte un round segmenté en source lue par lots. */
function roundToTrainSource(round: SegmentRound): MnistTrainSource {
  return {
    count: round.count,
    readBatch(start: number, end: number): MnistSample {
      return round.readBatch(start, end);
    },
  };
}

function assertSegmentPlan(plan: SegmentPlan, label: string): void {
  if (plan === null || typeof plan !== 'object') {
    throw new TypeError(`MNIST TFJS ${label} segment plan is required.`);
  }
  if (typeof plan.roundsForEpoch !== 'function') {
    throw new TypeError(`MNIST TFJS ${label} segment plan must expose roundsForEpoch.`);
  }
  if (!Number.isInteger(plan.classes) || plan.classes < 2 || plan.classes > 20) {
    throw new RangeError(
      `MNIST TFJS ${label} segment plan classes must be an integer from 2 to 20, received ${String(plan.classes)}.`,
    );
  }
  if (!Number.isInteger(plan.shardSize) || plan.shardSize < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} segment plan shardSize must be a positive integer, received ${String(plan.shardSize)}.`,
    );
  }
  if (!Number.isInteger(plan.totalCount) || plan.totalCount < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} segment plan totalCount must be a positive integer, received ${String(plan.totalCount)}.`,
    );
  }
  if (!Number.isInteger(plan.roundsPerEpoch) || plan.roundsPerEpoch < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} segment plan roundsPerEpoch must be a positive integer, received ${String(plan.roundsPerEpoch)}.`,
    );
  }
}

function assertSegmentLoader(loader: SegmentLoader, label: string): void {
  if (loader === null || typeof loader !== 'object') {
    throw new TypeError(`MNIST TFJS ${label} segment loader is required.`);
  }
  if (!Number.isInteger(loader.totalCount) || loader.totalCount < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} segment loader totalCount must be a positive integer, received ${String(loader.totalCount)}.`,
    );
  }
  for (const method of ['loadRound', 'prefetchRound', 'releaseRound'] as const) {
    if (typeof loader[method] !== 'function') {
      throw new TypeError(`MNIST TFJS ${label} segment loader must expose ${method}.`);
    }
  }
}

/**
 * Vérifie les champs d'un round segmenté sans parcourir ses pixels : chaque lot
 * est validé plus tard par `readSourceBatch`, comme pour l'entraînement borné.
 */
function assertSegmentRound(
  round: SegmentRound,
  label: string,
  core: CoreModule,
  classCount: number,
): void {
  if (round === null || typeof round !== 'object') {
    throw new TypeError(`MNIST TFJS ${label} round is required.`);
  }
  if (!Number.isInteger(round.count) || round.count < 1) {
    throw new RangeError(
      `MNIST TFJS ${label} round count must be a positive integer, received ${String(round.count)}.`,
    );
  }
  const expectedImages = round.count * core.MNIST_INPUT_LENGTH;
  if (!(round.images instanceof Float32Array) || round.images.length !== expectedImages) {
    throw new RangeError(
      `MNIST TFJS ${label} round images must be a Float32Array of ${expectedImages} values.`,
    );
  }
  if (!(round.labels instanceof Uint8Array) || round.labels.length !== round.count) {
    throw new RangeError(
      `MNIST TFJS ${label} round labels must be a Uint8Array of ${round.count} values.`,
    );
  }
  for (let index = 0; index < round.labels.length; index += 1) {
    const value = round.labels[index];
    if (!Number.isInteger(value) || value < 0 || value >= classCount) {
      throw new RangeError(
        `MNIST TFJS ${label} round labels must be integers from 0 to ${classCount - 1}, received ${value} at index ${index}.`,
      );
    }
  }
  if (typeof round.readBatch !== 'function') {
    throw new TypeError(`MNIST TFJS ${label} round must expose a readBatch function.`);
  }
}

/**
 * Construit le modèle séquentiel. Le remplissage zéro est posé explicitement
 * avant la convolution quand le `CONV_SCHEDULES` le demande : le mode `same` de
 * TensorFlow.js ajoute un remplissage asymétrique (0 en haut, 1 en bas) qui
 * décalerait les cartes par rapport au remplissage symétrique de PyTorch.
 *
 * La forme d'entrée n'est portée que par la première couche ajoutée : le
 * remplissage ou la convolution qui commence la pile, sinon le dropout quand il
 * n'y a aucune convolution. Les couches suivantes l'héritent de leur
 * prédécesseur.
 */
function buildModel(tf: TfModule, core: CoreModule, config: MnistCnnTrainingConfig): Sequential {
  const seed = config.seed ?? DEFAULT_TRAINING_SEED;
  const kernelInitializer = tf.initializers.glorotUniform({ seed });
  const biasInitializer = tf.initializers.zeros();
  const model = tf.sequential();
  let inputShape: number[] | undefined = [core.MNIST_INPUT_SIZE, core.MNIST_INPUT_SIZE, 1];
  const schedule = core.CONV_SCHEDULES[config.convLayers];
  for (let index = 0; index < config.convLayers; index += 1) {
    const spec = schedule[index];
    if (spec.padding > 0) {
      model.add(
        tf.layers.zeroPadding2d({
          padding: [
            [spec.padding, spec.padding],
            [spec.padding, spec.padding],
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
        strides: spec.stride,
        padding: 'valid',
        activation: config.activation,
        kernelInitializer,
        biasInitializer,
        inputShape,
      }),
    );
    inputShape = undefined;
  }
  model.add(tf.layers.dropout({ rate: MNIST_DROPOUT_RATE, seed, inputShape }));
  model.add(tf.layers.flatten());
  const widths = core.denseLayerWidths(config);
  for (let index = 0; index < config.denseLayers - 1; index += 1) {
    model.add(
      tf.layers.dense({
        units: widths[index],
        activation: config.activation,
        kernelInitializer,
        biasInitializer,
      }),
    );
  }
  model.add(
    tf.layers.dense({
      units: core.resolveOutputClasses(config),
      kernelInitializer,
      biasInitializer,
    }),
  );
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
 * `i` du noyau d'une couche dense correspond à la caractéristique NHWC `i`.
 */
function readLayerWeights(layer: TfLayer): MnistCnnLayerWeights {
  const [kernel, bias] = layer.getWeights();
  return {
    kernel: new Float32Array(kernel.dataSync()),
    bias: new Float32Array(bias.dataSync()),
  };
}

interface WeightLayerShape {
  readonly kernel: number;
  readonly bias: number;
}

/** Formes attendues des couches de convolution et des couches denses. */
interface WeightLayerShapes {
  readonly conv: readonly WeightLayerShape[];
  readonly dense: readonly WeightLayerShape[];
}

/**
 * Longueurs attendues de chaque noyau et biais pour la configuration donnée :
 * les pas et remplissages viennent de `CONV_SCHEDULES`, les largeurs denses de
 * `denseLayerWidths`.
 */
function weightLayerShapes(core: CoreModule, config: MnistCnnTrainingConfig): WeightLayerShapes {
  const schedule = core.CONV_SCHEDULES[config.convLayers];
  const conv: WeightLayerShape[] = [];
  let inputSize = core.MNIST_INPUT_SIZE;
  let inputChannels = 1;
  for (let layer = 0; layer < config.convLayers; layer += 1) {
    const spec = schedule[layer];
    const outputSize = core.convOutputLength(inputSize, spec.stride, spec.padding);
    const outputChannels = config.filters[layer];
    conv.push({
      kernel: core.CONV_KERNEL_SIZE * core.CONV_KERNEL_SIZE * inputChannels * outputChannels,
      bias: outputChannels,
    });
    inputSize = outputSize;
    inputChannels = outputChannels;
  }
  const widths = core.denseLayerWidths(config);
  let inputLength =
    conv.length === 0 ? core.MNIST_INPUT_LENGTH : inputSize * inputSize * inputChannels;
  const dense: WeightLayerShape[] = [];
  for (let layer = 0; layer < config.denseLayers; layer += 1) {
    const width = widths[layer];
    dense.push({ kernel: inputLength * width, bias: width });
    inputLength = width;
  }
  return { conv, dense };
}

/**
 * Rejette un jeu de poids qui ne correspond pas aux nombres de couches de la
 * configuration : nombre de couches, longueurs de noyau ou de biais
 * incorrectes, ou valeur non finie. Les formes sont celles de
 * `weightLayerShapes`.
 */
function assertWeightsForConfig(
  weights: MnistCnnWeights,
  core: CoreModule,
  config: MnistCnnTrainingConfig,
): void {
  const shapes = weightLayerShapes(core, config);
  if (!Array.isArray(weights.conv) || weights.conv.length !== shapes.conv.length) {
    throw new RangeError(
      `MNIST TFJS conv weights must hold ${shapes.conv.length} layer(s), received ${
        Array.isArray(weights.conv) ? weights.conv.length : 'not an array'
      }.`,
    );
  }
  if (!Array.isArray(weights.dense) || weights.dense.length !== shapes.dense.length) {
    throw new RangeError(
      `MNIST TFJS dense weights must hold ${shapes.dense.length} layer(s), received ${
        Array.isArray(weights.dense) ? weights.dense.length : 'not an array'
      }.`,
    );
  }
  const layers: Array<readonly [string, MnistCnnLayerWeights, WeightLayerShape]> = [];
  for (let layer = 0; layer < shapes.conv.length; layer += 1) {
    layers.push([core.convLayerName(layer), weights.conv[layer], shapes.conv[layer]]);
  }
  for (let layer = 0; layer < shapes.dense.length; layer += 1) {
    layers.push([`dense${layer + 1}`, weights.dense[layer], shapes.dense[layer]]);
  }
  for (const [name, layer, shape] of layers) {
    if (!(layer.kernel instanceof Float32Array) || !(layer.bias instanceof Float32Array)) {
      throw new TypeError(`MNIST TFJS ${name} weights must expose kernel and bias Float32Arrays.`);
    }
    if (layer.kernel.length !== shape.kernel || layer.bias.length !== shape.bias) {
      throw new RangeError(
        `MNIST TFJS ${name} weights do not match the ${JSON.stringify(config.filters)} filter tuple and layer counts.`,
      );
    }
    for (const value of layer.kernel) {
      if (!Number.isFinite(value)) {
        throw new RangeError(`MNIST TFJS ${name} kernel must contain only finite values.`);
      }
    }
    for (const value of layer.bias) {
      if (!Number.isFinite(value)) {
        throw new RangeError(`MNIST TFJS ${name} bias must contain only finite values.`);
      }
    }
  }
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

  private readonly outputClasses: number;

  private readonly seed: number;

  private model: Sequential | null = null;

  private optimizer: AdamOptimizer | null = null;

  private running = false;

  private cancelled = false;

  private disposed = false;

  /** Tampon circulaire des lots matérialisés d'avance (au plus `prefetchDepth`). */
  private prefetched: MaterializedBatch[] = [];

  /** Tenseurs de coût conservés pour la fenêtre de perte en cours. */
  private pendingCosts: Tensor[] = [];

  /**
   * Construit un entraîneur à partir des modules déjà chargés. Préférez
   * `createMnistCnnTfjsTrainer`, qui importe TensorFlow.js dynamiquement.
   */
  constructor(tf: TfModule, core: CoreModule, config: MnistCnnTrainingConfig) {
    assertConfig(config, core);
    this.tf = tf;
    this.core = core;
    this.config = config;
    this.outputClasses = core.resolveOutputClasses(config);
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
   * Entraîne le modèle par lots bornés lus sur la source. Un tampon circulaire
   * garde au plus `prefetchDepth` lots matérialisés d'avance ; la boucle prend
   * le plus ancien, fait une mise à jour, puis recharge le tampon pendant la
   * fenêtre de cession et vérifie le drapeau d'annulation avant chaque lot.
   *
   * La perte est lue par fenêtre de `lossSampleEvery` lots, jamais par lot, et
   * les évènements `running` sont limités à un par `progressIntervalMs`, plus
   * un par époque. Un évènement final est toujours émis.
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
    const prefetchDepth = options.prefetchDepth ?? DEFAULT_PREFETCH_DEPTH;
    if (
      !Number.isInteger(prefetchDepth) ||
      prefetchDepth < MIN_PREFETCH_DEPTH ||
      prefetchDepth > MAX_PREFETCH_DEPTH
    ) {
      throw new RangeError(
        `MNIST TFJS prefetchDepth must be an integer from ${MIN_PREFETCH_DEPTH} to ${MAX_PREFETCH_DEPTH}, received ${String(options.prefetchDepth)}.`,
      );
    }
    const lossSampleEvery = options.lossSampleEvery ?? DEFAULT_LOSS_SAMPLE_EVERY;
    if (
      !Number.isInteger(lossSampleEvery) ||
      lossSampleEvery < MIN_LOSS_SAMPLE_EVERY ||
      lossSampleEvery > MAX_LOSS_SAMPLE_EVERY
    ) {
      throw new RangeError(
        `MNIST TFJS lossSampleEvery must be an integer from ${MIN_LOSS_SAMPLE_EVERY} to ${MAX_LOSS_SAMPLE_EVERY}, received ${String(options.lossSampleEvery)}.`,
      );
    }
    const progressIntervalMs = options.progressIntervalMs ?? DEFAULT_PROGRESS_INTERVAL_MS;
    if (!Number.isFinite(progressIntervalMs) || progressIntervalMs < 0) {
      throw new RangeError(
        `MNIST TFJS progressIntervalMs must be a non-negative finite number, received ${String(options.progressIntervalMs)}.`,
      );
    }
    assertTrainSource(options.train, 'train');
    if (options.validation !== undefined) {
      assertTrainSource(options.validation, 'validation');
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

    // Seules les bornes des lots sont calculées d'avance (des entiers, jamais
    // les images). L'ordre des lots est mélangé à chaque époque : le caractère
    // stochastique est conservé sans matérialiser tout le jeu.
    const ranges: Array<readonly [number, number]> = [];
    for (let start = 0; start < count; start += batchSize) {
      ranges.push([start, Math.min(start + batchSize, count)]);
    }

    this.running = true;
    this.cancelled = false;

    let batchesProcessed = 0;
    let loss = Number.NaN;
    let epochLoss: number | null = null;
    let completedEpochs = 0;
    let validationLoss: number | null = null;
    let validationAccuracy: number | null = null;
    let epochTrainAccuracy: number | null = null;
    let status: MnistTrainingStatus = 'completed';
    let lastProgressAt = startedAt;

    const emitRunning = (now: number): void => {
      this.emit(options.onProgress, {
        status: 'running',
        batchesProcessed,
        loss,
        epochLoss,
        epochTrainAccuracy,
        completedEpochs,
        validationLoss,
        validationAccuracy,
        elapsedMs: now - startedAt,
      });
    };

    try {
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        // Accumulateurs de l'époque : la perte moyenne de l'époque est la moyenne
        // pondérée par lots de toutes les fenêtres lues, y compris la dernière.
        let epochLossSum = 0;
        let epochBatches = 0;
        const reshuffle = options.train.reshuffle;
        if (reshuffle !== undefined) {
          reshuffle(deriveEpochSeed(this.seed, epoch));
        }
        // Avec `reshuffle`, la source ordonne elle-même les exemples et les lots
        // sont lus dans l'ordre naturel. Sinon l'adaptateur mélange les lots.
        const order =
          reshuffle === undefined ? shuffledIndices(ranges.length, rng) : Array.from(ranges.keys());
        this.disposePrefetched();
        // Préremplissage : le tampon ne dépasse jamais `prefetchDepth` lots.
        let next = 0;
        while (this.prefetched.length < prefetchDepth && next < order.length) {
          const [start, end] = ranges[order[next]];
          this.prefetched.push(this.materializeBatch(options.train, start, end));
          next += 1;
        }
        for (let step = 0; step < order.length; step += 1) {
          if (this.cancelled) {
            status = 'cancelled';
            break;
          }
          const current = this.prefetched.shift();
          if (current === undefined) {
            throw new Error('MNIST TFJS prefetch buffer is empty during training.');
          }
          let cost: Tensor | null = null;
          try {
            cost = this.runBatch(model, optimizer, trainable, current);
          } finally {
            disposeBatch(current);
          }
          if (cost !== null) {
            this.pendingCosts.push(cost);
          }
          batchesProcessed += 1;
          // La fenêtre de perte n'est lue qu'à intervalle : aucune
          // synchronisation GPU par lot.
          if (this.pendingCosts.length >= lossSampleEvery) {
            const windowBatches = this.pendingCosts.length;
            const windowLoss = await this.readPendingLoss();
            if (windowLoss !== null) {
              loss = windowLoss;
              epochLossSum += windowLoss * windowBatches;
              epochBatches += windowBatches;
            }
          }
          const now = Date.now();
          if (now - lastProgressAt >= progressIntervalMs) {
            emitRunning(now);
            lastProgressAt = now;
          }
          // Recharge pendant la fenêtre de cession, après la mise à jour.
          await yieldControl();
          if (!this.cancelled && next < order.length) {
            const [start, end] = ranges[order[next]];
            this.prefetched.push(this.materializeBatch(options.train, start, end));
            next += 1;
          }
        }
        // Une annulation pendant l'époque saute la vidange de la fenêtre de perte,
        // la validation, l'incrément d'époque et l'évènement de fin d'époque : le
        // bloc `finally` libère la fenêtre en attente. Une annulation pendant
        // `await this.readPendingLoss()` de la vidange est rattrapée par la
        // vérification qui suit celle-ci.
        if (this.cancelled) {
          status = 'cancelled';
          break;
        }
        // Fin d'époque : on vide la fenêtre de perte, on valide sur le sous-ensemble
        // tenu à l'écart, puis on émet un évènement qui porte les trois mesures de
        // l'époque — même si l'intervalle n'est pas écoulé.
        const windowBatches = this.pendingCosts.length;
        const epochEndLoss = await this.readPendingLoss();
        if (epochEndLoss !== null) {
          loss = epochEndLoss;
          epochLossSum += epochEndLoss * windowBatches;
          epochBatches += windowBatches;
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
        const measuredTrainAccuracy = await this.measureAccuracy(
          model,
          options.train,
          MNIST_TRAIN_ACCURACY_SAMPLE_SIZE,
        );
        if (this.cancelled) {
          status = 'cancelled';
          break;
        }
        epochTrainAccuracy = measuredTrainAccuracy;
        epochLoss = epochBatches > 0 ? epochLossSum / epochBatches : null;
        completedEpochs += 1;
        const epochEndedAt = Date.now();
        emitRunning(epochEndedAt);
        lastProgressAt = epochEndedAt;
      }
    } finally {
      this.disposePendingCosts();
      this.disposePrefetched();
      this.running = false;
    }

    const progress: MnistTrainingProgress = {
      status,
      batchesProcessed,
      loss,
      epochLoss,
      epochTrainAccuracy,
      completedEpochs,
      validationLoss,
      validationAccuracy,
      elapsedMs: Date.now() - startedAt,
    };
    this.emit(options.onProgress, progress);
    return progress;
  }

  /**
   * Entraîne le modèle par rounds segmentés. Un round contient un fragment par
   * classe ; la boucle le charge, précharge le suivant, traite ses lots bornés,
   * puis le libère. La perte d'époque est la moyenne pondérée par le nombre de
   * lots de chaque round, et l'exactitude d'entraînement porte sur le dernier
   * round de l'époque, plafonnée à `trainAccuracySampleSize` exemples.
   *
   * Une annulation est vérifiée avant chaque round et chaque lot : le round en
   * cours et le round préchargé sont libérés, l'époque n'est pas comptée et le
   * résultat porte `cancelled`. Un seul optimiseur sert à toute la course.
   */
  async trainSegmented(options: MnistSegmentedTrainOptions): Promise<MnistTrainingProgress> {
    this.assertUsable();
    if (this.running) {
      throw new Error('MNIST TFJS trainer is already training.');
    }
    if (options === null || typeof options !== 'object') {
      throw new TypeError('MNIST TFJS segmented train options are required.');
    }
    const epochs = options.epochs;
    if (!Number.isInteger(epochs) || epochs < 1) {
      throw new RangeError(`MNIST TFJS epochs must be a positive integer, received ${epochs}.`);
    }
    const lossSampleEvery = options.lossSampleEvery ?? DEFAULT_LOSS_SAMPLE_EVERY;
    if (
      !Number.isInteger(lossSampleEvery) ||
      lossSampleEvery < MIN_LOSS_SAMPLE_EVERY ||
      lossSampleEvery > MAX_LOSS_SAMPLE_EVERY
    ) {
      throw new RangeError(
        `MNIST TFJS lossSampleEvery must be an integer from ${MIN_LOSS_SAMPLE_EVERY} to ${MAX_LOSS_SAMPLE_EVERY}, received ${String(options.lossSampleEvery)}.`,
      );
    }
    const progressIntervalMs = options.progressIntervalMs ?? DEFAULT_PROGRESS_INTERVAL_MS;
    if (!Number.isFinite(progressIntervalMs) || progressIntervalMs < 0) {
      throw new RangeError(
        `MNIST TFJS progressIntervalMs must be a non-negative finite number, received ${String(options.progressIntervalMs)}.`,
      );
    }
    const trainAccuracySampleSize =
      options.trainAccuracySampleSize ?? MNIST_TRAIN_ACCURACY_SAMPLE_SIZE;
    if (!Number.isInteger(trainAccuracySampleSize) || trainAccuracySampleSize < 1) {
      throw new RangeError(
        `MNIST TFJS trainAccuracySampleSize must be a positive integer, received ${String(options.trainAccuracySampleSize)}.`,
      );
    }
    assertSegmentPlan(options.trainPlan, 'train');
    assertSegmentLoader(options.trainLoader, 'train');
    if (options.trainPlan.totalCount !== options.trainLoader.totalCount) {
      throw new RangeError(
        `MNIST TFJS train segment plan totalCount ${options.trainPlan.totalCount} does not match loader totalCount ${options.trainLoader.totalCount}.`,
      );
    }

    const validationPlan = options.validationPlan;
    const validationLoader = options.validationLoader;
    const segmentedValidation = validationPlan !== undefined || validationLoader !== undefined;
    if (segmentedValidation && (validationPlan === undefined || validationLoader === undefined)) {
      throw new TypeError(
        'MNIST TFJS validationPlan and validationLoader must be provided together.',
      );
    }
    if (validationPlan !== undefined && validationLoader !== undefined) {
      assertSegmentPlan(validationPlan, 'validation');
      assertSegmentLoader(validationLoader, 'validation');
      if (validationPlan.totalCount !== validationLoader.totalCount) {
        throw new RangeError(
          `MNIST TFJS validation segment plan totalCount ${validationPlan.totalCount} does not match loader totalCount ${validationLoader.totalCount}.`,
        );
      }
    } else if (options.validation !== undefined) {
      assertSample(options.validation, 'validation', this.core, this.outputClasses);
    }

    const trainPlan = options.trainPlan;
    const trainLoader = options.trainLoader;
    const model = this.ensureModel();
    const optimizer = this.optimizer;
    if (optimizer === null) {
      throw new Error('MNIST TFJS optimizer was not initialised.');
    }
    const trainable = model.getWeights(true) as unknown as Variable[];
    const { batchSize } = this.config;
    const startedAt = Date.now();

    this.running = true;
    this.cancelled = false;

    let batchesProcessed = 0;
    let loss = Number.NaN;
    let epochLoss: number | null = null;
    let epochTrainAccuracy: number | null = null;
    let completedEpochs = 0;
    let validationLoss: number | null = null;
    let validationAccuracy: number | null = null;
    let status: MnistTrainingStatus = 'completed';
    let lastProgressAt = startedAt;
    let roundIndex = 0;
    let completedRounds = 0;
    let windowShards = 0;
    let lastRound: SegmentRound | null = null;
    let lastRoundRef: RoundRef | null = null;
    let prefetchedRef: RoundRef | null = null;
    let stopped = false;
    /** Premier exemple de la fenêtre de perte en cours. */
    let pendingWindowStart = 0;

    const emitRunning = (now: number): void => {
      this.emit(options.onProgress, {
        status: 'running',
        batchesProcessed,
        loss,
        epochLoss,
        epochTrainAccuracy,
        completedEpochs,
        validationLoss,
        validationAccuracy,
        elapsedMs: now - startedAt,
        roundIndex,
        roundsPerEpoch: trainPlan.roundsPerEpoch,
        completedRounds,
        windowShards,
      });
    };

    try {
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        const rounds = trainPlan.roundsForEpoch(epoch);
        // Accumulateurs de l'époque : chaque round apporte sa perte moyenne
        // pondérée par son nombre de lots, y compris une fenêtre finale partielle.
        let epochLossSum = 0;
        let epochBatches = 0;
        for (let position = 0; position < rounds.length; position += 1) {
          if (this.cancelled) {
            status = 'cancelled';
            stopped = true;
            break;
          }
          const ref = rounds[position];
          const round = await trainLoader.loadRound(ref);
          if (prefetchedRef === ref) {
            prefetchedRef = null;
          }
          let keepRound = false;
          let roundLossSum = 0;
          let roundBatches = 0;
          try {
            assertSegmentRound(round, 'train', this.core, this.outputClasses);
            roundIndex = ref.index;
            if (position + 1 < rounds.length) {
              const nextRef = rounds[position + 1];
              trainLoader.prefetchRound(nextRef);
              prefetchedRef = nextRef;
            }
            const source = roundToTrainSource(round);
            for (let start = 0; start < round.count; start += batchSize) {
              if (this.cancelled) {
                status = 'cancelled';
                stopped = true;
                break;
              }
              const end = Math.min(start + batchSize, round.count);
              const windowWasEmpty = this.pendingCosts.length === 0;
              const batch = this.materializeBatch(source, start, end);
              let cost: Tensor | null = null;
              try {
                cost = this.runBatch(model, optimizer, trainable, batch);
              } finally {
                disposeBatch(batch);
              }
              if (cost !== null) {
                this.pendingCosts.push(cost);
                if (windowWasEmpty) {
                  pendingWindowStart = start;
                }
              }
              batchesProcessed += 1;
              // La fenêtre de perte n'est lue qu'à intervalle, jamais par lot.
              if (this.pendingCosts.length >= lossSampleEvery) {
                const windowBatchCount = this.pendingCosts.length;
                const windowShardCount = shardsInRoundRange(
                  round,
                  ref.entries,
                  pendingWindowStart,
                  end,
                );
                const windowLoss = await this.readPendingLoss();
                pendingWindowStart = end;
                if (windowLoss !== null) {
                  loss = windowLoss;
                  roundLossSum += windowLoss * windowBatchCount;
                  roundBatches += windowBatchCount;
                  windowShards = windowShardCount;
                }
              }
              const now = Date.now();
              if (now - lastProgressAt >= progressIntervalMs) {
                emitRunning(now);
                lastProgressAt = now;
              }
              await defaultYieldControl();
            }
            if (stopped) {
              break;
            }
            // Fin du round : on vide la fenêtre partielle avant de passer au
            // round suivant, pour que sa perte reste dans ce round.
            const finalWindowBatches = this.pendingCosts.length;
            const finalWindowShards = shardsInRoundRange(
              round,
              ref.entries,
              pendingWindowStart,
              round.count,
            );
            const roundEndLoss = await this.readPendingLoss();
            pendingWindowStart = round.count;
            if (roundEndLoss !== null) {
              loss = roundEndLoss;
              roundLossSum += roundEndLoss * finalWindowBatches;
              roundBatches += finalWindowBatches;
              windowShards = finalWindowShards;
            }
            if (this.cancelled) {
              status = 'cancelled';
              stopped = true;
              break;
            }
            if (roundBatches > 0) {
              const roundLoss = roundLossSum / roundBatches;
              epochLossSum += roundLoss * roundBatches;
              epochBatches += roundBatches;
            }
            completedRounds += 1;
            // Le dernier round porte la mesure d'exactitude de l'époque : il
            // reste chargé jusqu'après la validation.
            if (position === rounds.length - 1) {
              keepRound = true;
              lastRound = round;
              lastRoundRef = ref;
            }
          } finally {
            if (!keepRound) {
              trainLoader.releaseRound(ref);
              if (lastRound === round) {
                lastRound = null;
                lastRoundRef = null;
              }
            }
          }
        }
        if (stopped) {
          break;
        }

        if (segmentedValidation && validationPlan !== undefined && validationLoader !== undefined) {
          const metrics = await this.evaluateSegmentedRounds(
            model,
            validationPlan,
            validationLoader,
            epoch,
          );
          if (this.cancelled) {
            status = 'cancelled';
            stopped = true;
            break;
          }
          validationLoss = metrics.loss;
          validationAccuracy = metrics.accuracy;
        } else if (options.validation !== undefined) {
          const metrics = this.evaluate(model, sampleToTrainSource(options.validation));
          if (this.cancelled) {
            status = 'cancelled';
            stopped = true;
            break;
          }
          validationLoss = metrics.loss;
          validationAccuracy = metrics.accuracy;
        }

        let measuredAccuracy: number | null = null;
        if (lastRound !== null) {
          measuredAccuracy = await this.measureAccuracy(
            model,
            roundToTrainSource(lastRound),
            trainAccuracySampleSize,
          );
        }
        if (this.cancelled) {
          status = 'cancelled';
          stopped = true;
          break;
        }
        epochTrainAccuracy = measuredAccuracy;
        epochLoss = epochBatches > 0 ? epochLossSum / epochBatches : null;
        completedEpochs += 1;
        if (lastRoundRef !== null) {
          trainLoader.releaseRound(lastRoundRef);
          lastRound = null;
          lastRoundRef = null;
        }
        const epochEndedAt = Date.now();
        emitRunning(epochEndedAt);
        lastProgressAt = epochEndedAt;
      }
    } finally {
      if (prefetchedRef !== null) {
        trainLoader.releaseRound(prefetchedRef);
      }
      if (lastRoundRef !== null) {
        trainLoader.releaseRound(lastRoundRef);
      }
      this.disposePendingCosts();
      this.running = false;
    }

    const progress: MnistTrainingProgress = {
      status,
      batchesProcessed,
      loss,
      epochLoss,
      epochTrainAccuracy,
      completedEpochs,
      validationLoss,
      validationAccuracy,
      elapsedMs: Date.now() - startedAt,
      roundIndex,
      roundsPerEpoch: trainPlan.roundsPerEpoch,
      completedRounds,
      windowShards,
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
    if (
      convolutions.length !== this.config.convLayers ||
      denses.length !== this.config.denseLayers
    ) {
      throw new Error(
        `MNIST TFJS model exposes ${convolutions.length} convolutions and ${denses.length} dense layers, expected ${this.config.convLayers} and ${this.config.denseLayers}.`,
      );
    }
    return {
      conv: convolutions.map(readLayerWeights),
      dense: denses.map(readLayerWeights),
    };
  }

  /**
   * Remplace les poids du modèle par un jeu `MnistCnnWeights` du core, puis
   * recrée l'optimiseur pour repartir de moments Adam à zéro.
   *
   * Les noyaux de convolution du core sont déjà en HWIO
   * `[kernelY][kernelX][canalEntrée][canalSortie]` et les noyaux denses déjà en
   * `[caractéristiqueEntrée][caractéristiqueSortie]` : aucune conversion de
   * disposition n'est nécessaire. `setWeights` recopie les valeurs dans les
   * variables du modèle, donc les tenseurs temporaires sont disposés ici.
   *
   * Les couches sont lues dans l'ordre du modèle, sans indice fixe : chaque
   * `Conv2D` prend l'entrée suivante de `weights.conv` et chaque `Dense` celle
   * de `weights.dense`, les formes venant des propres tenseurs de la couche.
   *
   * Rejette un entraîneur disposé et un jeu de poids dont les nombres de
   * couches ou le tuple de filtres ne correspond pas à la configuration.
   */
  async applyWeights(weights: MnistCnnWeights): Promise<void> {
    this.assertUsable();
    assertWeightsForConfig(weights, this.core, this.config);
    const model = this.ensureModel();
    const convWeights = weights.conv;
    const denseWeights = weights.dense;
    let convIndex = 0;
    let denseIndex = 0;
    const values: Tensor[] = [];
    try {
      for (const layer of model.layers) {
        const className = layer.getClassName();
        if (className !== 'Conv2D' && className !== 'Dense') {
          continue;
        }
        const layerWeights =
          className === 'Conv2D' ? convWeights[convIndex++] : denseWeights[denseIndex++];
        const [kernel, bias] = layer.getWeights();
        values.push(
          this.tf.tensor(layerWeights.kernel, kernel.shape),
          this.tf.tensor(layerWeights.bias, bias.shape),
        );
      }
      model.setWeights(values);
    } finally {
      for (const value of values) {
        value.dispose();
      }
    }
    // Les moments Adam de l'ancien jeu de poids sont périmés : on recrée
    // l'optimiseur pour repartir de moments à zéro.
    if (this.optimizer !== null) {
      this.optimizer.dispose();
    }
    this.optimizer = this.tf.train.adam(this.config.learningRate);
  }

  /**
   * Retourne les probabilités softmax (count × classes de sortie, ligne-major)
   * pour l'affichage. Le dropout est désactivé.
   */
  async predict(sample: MnistSample): Promise<Float32Array> {
    this.assertUsable();
    assertSample(sample, 'predict', this.core, this.outputClasses);
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

  /** Libère le modèle, l'optimiseur et les tampons ; l'instance reste réutilisable. */
  reset(): void {
    this.disposePendingCosts();
    this.disposePrefetched();
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

  /** Lit et matérialise un lot de la source en tenseurs prêts pour la mise à jour. */
  private materializeBatch(
    source: MnistTrainSource,
    start: number,
    end: number,
  ): MaterializedBatch {
    const sample = readSourceBatch(source, start, end, 'train', this.core, this.outputClasses);
    const images = this.tf.tensor4d(sample.images, [
      sample.count,
      this.core.MNIST_INPUT_SIZE,
      this.core.MNIST_INPUT_SIZE,
      1,
    ]);
    const labels = this.tf.tensor1d(sample.labels, 'int32');
    return { images, labels };
  }

  /** Libère tous les lots du tampon de préchargement. */
  private disposePrefetched(): void {
    for (const batch of this.prefetched) {
      disposeBatch(batch);
    }
    this.prefetched = [];
  }

  /**
   * Un pas d'Adam sur un lot déjà matérialisé ; retourne le tenseur de coût
   * sans le synchroniser. L'appelant possède le tenseur retourné.
   */
  private runBatch(
    model: Sequential,
    optimizer: AdamOptimizer,
    trainable: readonly Variable[],
    batch: MaterializedBatch,
  ): Tensor | null {
    const tf = this.tf;
    const yBatch = tf.oneHot(batch.labels, this.outputClasses);
    // Le `finally` libère le one-hot du lot même si la mise à jour échoue.
    try {
      return optimizer.minimize(
        () => {
          const logits = model.apply(batch.images, { training: true }) as Tensor;
          return tf.losses.softmaxCrossEntropy(yBatch, logits).mean();
        },
        true,
        trainable as Variable[],
      );
    } finally {
      yBatch.dispose();
    }
  }

  /**
   * Moyenne la fenêtre de pertes en tenseurs, puis lit le seul scalaire obtenu
   * par `await mean.data()`. Aucun `dataSync` et une seule lecture par fenêtre.
   * Le tenseur empilé et chaque tenseur de coût sont libérés dans un `finally`.
   * Retourne `null` quand la fenêtre est vide.
   */
  private async readPendingLoss(): Promise<number | null> {
    if (this.pendingCosts.length === 0) {
      return null;
    }
    const costs = this.pendingCosts;
    this.pendingCosts = [];
    let stacked: Tensor | null = null;
    try {
      stacked = this.tf.stack(costs);
      const mean = stacked.mean();
      try {
        const value = await mean.data();
        return Number((value as Float32Array)[0]);
      } finally {
        mean.dispose();
      }
    } finally {
      stacked?.dispose();
      for (const cost of costs) {
        cost.dispose();
      }
    }
  }

  /** Libère les tenseurs de coût de la fenêtre en cours. */
  private disposePendingCosts(): void {
    for (const cost of this.pendingCosts) {
      cost.dispose();
    }
    this.pendingCosts = [];
  }

  /**
   * Exactitude d'entraînement sur un préfixe borné de la source, dropout
   * désactivé. Une seule lecture asynchrone par bloc, jamais `dataSync` : la
   * mesure ne vide pas la file GPU. Retourne `NaN` sans exemple.
   */
  private async measureAccuracy(
    model: Sequential,
    source: MnistTrainSource,
    limit: number,
  ): Promise<number> {
    const tf = this.tf;
    const total = Math.min(source.count, limit);
    if (total <= 0) {
      return Number.NaN;
    }
    let correct = 0;
    for (let start = 0; start < total; start += MNIST_VALIDATION_CHUNK_SIZE) {
      const end = Math.min(start + MNIST_VALIDATION_CHUNK_SIZE, total);
      const chunk = readSourceBatch(
        source,
        start,
        end,
        'training accuracy',
        this.core,
        this.outputClasses,
      );
      const images = tf.tensor4d(chunk.images, [
        chunk.count,
        this.core.MNIST_INPUT_SIZE,
        this.core.MNIST_INPUT_SIZE,
        1,
      ]);
      const labels = tf.tensor1d(chunk.labels, 'int32');
      try {
        const correctTensor = tf.tidy(() => {
          const logits = model.apply(images, { training: false }) as Tensor;
          return logits.argMax(1).equal(labels).sum();
        }) as Tensor;
        try {
          correct += (await correctTensor.data())[0];
        } finally {
          correctTensor.dispose();
        }
      } finally {
        images.dispose();
        labels.dispose();
      }
      // Une annulation pendant la lecture arrête la mesure : l'époque reste
      // inachevée et l'appelant ignore la valeur retournée.
      if (this.cancelled) {
        break;
      }
    }
    return correct / total;
  }

  /**
   * Évalue perte et exactitude par blocs bornés sur la source de validation,
   * dropout désactivé. Les blocs sont accumulés : le jeu complet n'est jamais
   * matérialisé d'un coup.
   */
  private evaluate(
    model: Sequential,
    source: MnistTrainSource,
  ): { loss: number; accuracy: number } {
    const tf = this.tf;
    let lossSum = 0;
    let correct = 0;
    for (let start = 0; start < source.count; start += MNIST_VALIDATION_CHUNK_SIZE) {
      const end = Math.min(start + MNIST_VALIDATION_CHUNK_SIZE, source.count);
      const chunk = readSourceBatch(
        source,
        start,
        end,
        'validation',
        this.core,
        this.outputClasses,
      );
      const images = tf.tensor4d(chunk.images, [
        chunk.count,
        this.core.MNIST_INPUT_SIZE,
        this.core.MNIST_INPUT_SIZE,
        1,
      ]);
      const labels = tf.tensor1d(chunk.labels, 'int32');
      try {
        tf.tidy(() => {
          const logits = model.apply(images, { training: false }) as Tensor;
          const oneHot = tf.oneHot(labels, this.outputClasses);
          lossSum +=
            tf.losses.softmaxCrossEntropy(oneHot, logits).mean().dataSync()[0] * chunk.count;
          const predictions = logits.argMax(1);
          correct += predictions.equal(labels).sum().dataSync()[0];
        });
      } finally {
        images.dispose();
        labels.dispose();
      }
    }
    return { loss: lossSum / source.count, accuracy: correct / source.count };
  }

  /**
   * Évalue la validation round par round : chaque round est lu comme une source
   * bornée, puis sa perte et son exactitude sont pondérées par son effectif.
   * Les rounds préchargés sont libérés, y compris quand une annulation ou une
   * erreur interrompt la lecture.
   */
  private async evaluateSegmentedRounds(
    model: Sequential,
    plan: SegmentPlan,
    loader: SegmentLoader,
    epoch: number,
  ): Promise<{ loss: number; accuracy: number }> {
    const rounds = plan.roundsForEpoch(epoch);
    let lossSum = 0;
    let correct = 0;
    let total = 0;
    let prefetchedRef: RoundRef | null = null;
    try {
      for (let position = 0; position < rounds.length; position += 1) {
        if (this.cancelled) {
          break;
        }
        const ref = rounds[position];
        const round = await loader.loadRound(ref);
        if (prefetchedRef === ref) {
          prefetchedRef = null;
        }
        try {
          assertSegmentRound(round, 'validation', this.core, this.outputClasses);
          if (position + 1 < rounds.length) {
            const nextRef = rounds[position + 1];
            loader.prefetchRound(nextRef);
            prefetchedRef = nextRef;
          }
          const metrics = this.evaluate(model, roundToTrainSource(round));
          lossSum += metrics.loss * round.count;
          correct += metrics.accuracy * round.count;
          total += round.count;
        } finally {
          loader.releaseRound(ref);
        }
      }
    } finally {
      if (prefetchedRef !== null) {
        loader.releaseRound(prefetchedRef);
      }
    }
    if (total === 0) {
      return { loss: Number.NaN, accuracy: Number.NaN };
    }
    return { loss: lossSum / total, accuracy: correct / total };
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
 * Charge TensorFlow.js dynamiquement, puis construit un entraîneur. C'est le
 * seul point d'entrée qui charge TensorFlow.js : le graphe d'imports statique
 * du module reste léger (le contrat `mnist-cnn.ts` est importé statiquement).
 */
export async function createMnistCnnTfjsTrainer(
  config: MnistCnnTrainingConfig,
): Promise<MnistCnnTfjsTrainer> {
  const tf = await import('@tensorflow/tfjs');
  return new MnistCnnTfjsTrainer(tf, core, config);
}
