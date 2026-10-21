/**
 * Planificateur pur de l'entraînement segmenté du CNN MNIST.
 *
 * Un jeu segmenté range les images par classe et par fragment (`shard`). Le
 * planificateur ne lit aucune image : il ordonne les fragments d'une époque en
 * « rounds », chaque round contenant exactement un fragment par classe. Le
 * nombre de rounds d'une époque est le nombre maximal de fragments d'une
 * classe ; une classe qui a moins de fragments répète les siens en cycle
 * (`r % length`), donc chaque fragment sélectionné apparaît au moins une fois
 * par époque.
 *
 * L'ordre des rounds et l'ordre des fragments de chaque classe sont mélangés
 * séparément à partir d'une graine dérivée de la graine du plan et de l'indice
 * d'époque : deux entraînements de même graine suivent exactement le même
 * ordre, et une autre époque change l'ordre sans changer la couverture.
 */

import type { MnistSample } from './mnist-data.js';

/** Nombre d'images sélectionnées à partir duquel l'entraînement est segmenté. */
export const SEGMENT_IMAGE_THRESHOLD = 60_000;

/** Nombre minimal de classes accepté par un plan segmenté. */
const MIN_SEGMENT_CLASSES = 2;

/** Nombre maximal de classes accepté par un plan segmenté. */
const MAX_SEGMENT_CLASSES = 20;

/** Vrai au-delà de `SEGMENT_IMAGE_THRESHOLD`, faux à la valeur exacte. */
export function shouldUseSegments(selectedImageCount: number): boolean {
  return selectedImageCount > SEGMENT_IMAGE_THRESHOLD;
}

/** Un fragment d'images d'une classe, tel que décrit par l'index du jeu. */
export interface RoundEntry {
  readonly file: string;
  readonly classIndex: number;
  readonly start: number;
  readonly count: number;
}

/** Un round : son indice dans l'époque et un fragment par classe. */
export interface RoundRef {
  readonly index: number;
  readonly entries: readonly RoundEntry[];
}

/** Entrée de `buildSegmentPlan`. */
export interface SegmentPlanInput {
  /** Fragments des classes sélectionnées, remappés de `0` à `C - 1`. */
  readonly shardsByClass: readonly (readonly RoundEntry[])[];
  /** Taille nominale d'un fragment, strictement positive. */
  readonly shardSize: number;
  /** Graine entière non négative. */
  readonly seed: number;
}

/** Plan segmenté immuable, interrogeable époque par époque. */
export interface SegmentPlan {
  /** Nombre de classes sélectionnées. */
  readonly classes: number;
  /** Taille nominale d'un fragment. */
  readonly shardSize: number;
  /** Somme des effectifs de tous les fragments du plan. */
  readonly totalCount: number;
  /** Nombre de rounds d'une époque : maximum de fragments par classe. */
  readonly roundsPerEpoch: number;
  /** Rounds de l'époque demandée, dans un ordre mélangé déterministe. */
  roundsForEpoch(epoch: number): readonly RoundRef[];
}

/**
 * Un round matérialisé : images normalisées (`count * 784`) et libellés, lus
 * par plages semi-ouvertes `[start, end)`.
 */
export interface SegmentRound {
  readonly count: number;
  /** Pixels normalisés dans `[0, 1]`, ligne-major : `image * 784 + pixel`. */
  readonly images: Float32Array;
  /** Un libellé par image. */
  readonly labels: Uint8Array;
  /** Retourne exactement `end - start` exemples de la plage `[start, end)`. */
  readBatch(start: number, end: number): MnistSample;
}

/**
 * Chargeur de rounds segmentés. `prefetchRound` et `releaseRound` équilibrent
 * les ressources d'un round : le chargeur peut garder le round préchargé
 * jusqu'à son `loadRound`, puis le libérer après usage.
 */
export interface SegmentLoader {
  readonly totalCount: number;
  loadRound(ref: RoundRef): Promise<SegmentRound>;
  prefetchRound(ref: RoundRef): void;
  releaseRound(ref: RoundRef): void;
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

/** Graine d'époque dérivée de la graine du plan, comme `deriveEpochSeed`. */
function deriveSegmentSeed(baseSeed: number, epoch: number): number {
  return (baseSeed + Math.imul(epoch + 1, 0x9e3779b1)) >>> 0;
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(
      `mnist-cnn-segmented: ${label} must be a positive integer, received ${String(value)}.`,
    );
  }
}

function assertSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      `mnist-cnn-segmented: seed must be a non-negative integer, received ${String(seed)}.`,
    );
  }
}

/**
 * Construit un plan segmenté à partir des fragments sélectionnés. Rejette un
 * nombre de classes hors de `[2, 20]`, une classe sans fragment, un `shardSize`
 * non entier ou non positif, un effectif de fragment non entier ou non positif
 * et une graine invalide.
 *
 * Les fragments sont recopiés : le plan reste stable même si l'appelant modifie
 * ensuite ses tableaux.
 */
export function buildSegmentPlan(input: SegmentPlanInput): SegmentPlan {
  if (input === null || typeof input !== 'object') {
    throw new TypeError('mnist-cnn-segmented: a segment plan input object is required.');
  }
  const { shardsByClass, shardSize, seed } = input;
  if (!Array.isArray(shardsByClass)) {
    throw new TypeError('mnist-cnn-segmented: shardsByClass must be an array of classes.');
  }
  const classes = shardsByClass.length;
  if (classes < MIN_SEGMENT_CLASSES || classes > MAX_SEGMENT_CLASSES) {
    throw new RangeError(
      `mnist-cnn-segmented: classes must be an integer from ${MIN_SEGMENT_CLASSES} to ${MAX_SEGMENT_CLASSES}, received ${classes}.`,
    );
  }
  assertPositiveInteger(shardSize, 'shardSize');
  assertSeed(seed);

  const classShards: RoundEntry[][] = [];
  let totalCount = 0;
  let roundsPerEpoch = 0;
  for (let classIndex = 0; classIndex < classes; classIndex += 1) {
    const shards = shardsByClass[classIndex];
    if (!Array.isArray(shards) || shards.length === 0) {
      throw new RangeError(
        `mnist-cnn-segmented: class ${classIndex} must have at least one shard.`,
      );
    }
    roundsPerEpoch = Math.max(roundsPerEpoch, shards.length);
    const copied: RoundEntry[] = [];
    for (const shard of shards) {
      if (shard === null || typeof shard !== 'object') {
        throw new TypeError(
          `mnist-cnn-segmented: class ${classIndex} contains a shard that is not an object.`,
        );
      }
      if (typeof shard.file !== 'string' || shard.file.length === 0) {
        throw new TypeError(
          `mnist-cnn-segmented: class ${classIndex} contains a shard without a file name.`,
        );
      }
      assertPositiveInteger(shard.count, `class ${classIndex} shard count`);
      copied.push({
        file: shard.file,
        classIndex: shard.classIndex,
        start: shard.start,
        count: shard.count,
      });
      totalCount += shard.count;
    }
    classShards.push(copied);
  }

  return {
    classes,
    shardSize,
    totalCount,
    roundsPerEpoch,
    roundsForEpoch(epoch: number): readonly RoundRef[] {
      if (!Number.isInteger(epoch) || epoch < 0 || epoch > Number.MAX_SAFE_INTEGER) {
        throw new RangeError(
          `mnist-cnn-segmented: epoch must be a non-negative integer, received ${String(epoch)}.`,
        );
      }
      const rng = mulberry32(deriveSegmentSeed(seed, epoch));
      const roundOrder = shuffledIndices(roundsPerEpoch, rng);
      const shardOrders = classShards.map((shards) => shuffledIndices(shards.length, rng));
      const rounds: RoundRef[] = [];
      for (const roundIndex of roundOrder) {
        const entries: RoundEntry[] = [];
        for (let classIndex = 0; classIndex < classes; classIndex += 1) {
          const shards = classShards[classIndex];
          const order = shardOrders[classIndex];
          entries.push(shards[order[roundIndex % shards.length]]);
        }
        rounds.push({ index: roundIndex, entries });
      }
      return rounds;
    },
  };
}
