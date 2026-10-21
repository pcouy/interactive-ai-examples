/**
 * Chargeur navigateur des jeux segmentés de la démo `drl-mnist-cnn`, par
 * exemple QuickDraw.
 *
 * Le registre décrit un jeu segmenté par un fichier d'index JSON (`index.json`)
 * qui nomme, pour chaque fraction, un fichier de libellés IDX global et une
 * liste de fragments d'images IDX (un fragment par classe et par tranche). Les
 * fragments sont des fichiers IDX d'images seuls ; les libellés du fragment
 * sont extraits du fichier de libellés global à l'aide de `start` et `count`.
 *
 * Ce module ne télécharge que les fragments sélectionnés par la fraction
 * demandée : la sélection regroupe les fragments par classe, garde les `N`
 * premiers de chaque classe dans l'ordre de l'index et les retourne dans
 * l'ordre des classes. `N` vaut `max(1, round(percent / 100 * shardsPerClass))`,
 * plafonné au nombre de fragments disponibles. La sélection ne dépend jamais du
 * nom du fichier. Une seconde sélection, `classIndices`, restreint les classes
 * téléchargées et remappe leurs libellés vers `0..N-1`.
 *
 * Les octets décodés sont mis en cache par identifiant de jeu et chemin de
 * fichier ; seuls les téléchargements réussis sont conservés, ce qui permet une
 * nouvelle tentative après un échec. Les téléchargements sont limités à quatre
 * simultanément. Les erreurs nomment le fichier concerné.
 *
 * `SegmentStore` sert un entraînement ou une validation segmentés : il
 * matérialise un round de la fraction choisie en entrelaçant les exemples des
 * classes, remappe les classes sélectionnées vers `0..N-1`, garde les fragments
 * décompressés dans une fenêtre bornée et s'appuie sur le cache Cache API pour
 * éviter de retélécharger un fragment déjà vu. Les lots consécutifs mélangent
 * donc les classes au lieu de les regrouper.
 *
 * Ce module est réservé au navigateur : il utilise `fetch` et `Response`. Le
 * core ne l'importe jamais.
 */

import type { ShardedDatasetDefinition } from '../../core/dataset-registry.js';
import type {
  RoundEntry,
  RoundRef,
  SegmentLoader,
  SegmentRound,
} from '../../core/mnist-cnn-segmented.js';
import {
  loadIdxDataset,
  type MnistIdxDataset,
  type MnistSample,
  materializeSample,
} from '../../core/mnist-data.js';
import { type GallerySplit, galleryCacheKey } from './gallery.js';
import {
  assetBaseUrl,
  decodeIdxPayload,
  fetchGzip,
  type MnistSplits,
  mnistFetchErrorMessage,
} from './mnist-assets.js';

/** Magics IDX gros-boutiste : `2051` pour les images, `2049` pour les libellés. */
const IDX_IMAGE_MAGIC = 2051;
const IDX_LABEL_MAGIC = 2049;

/** Tailles d'en-tête IDX : 16 octets pour les images, 8 pour les libellés. */
const IMAGE_HEADER_BYTES = 16;
const LABEL_HEADER_BYTES = 8;

/** Nombre de pixels d'une image MNIST (28 × 28). */
const PIXELS_PER_IMAGE = 784;

/** Nombre minimal et maximal de classes accepté par l'index. */
const MIN_CLASS_COUNT = 2;
const MAX_CLASS_COUNT = 20;

/** Nombre maximal de téléchargements simultanés. */
const FETCH_CONCURRENCY = 4;

/** Répertoire public des ressources des jeux, relatif à la base Vite. */
const DATASET_ASSET_DIRECTORY = 'datasets';

/** Un fragment d'images d'une fraction, décrit par l'index JSON. */
export interface ShardEntry {
  readonly file: string;
  readonly classIndex: number;
  readonly start: number;
  readonly count: number;
}

/** Description d'une fraction : effectif, fichier de libellés et fragments. */
export interface ShardedSplitIndex {
  readonly count: number;
  readonly labelsFile: string;
  readonly shards: readonly ShardEntry[];
}

/** Galerie d'une fraction : fichiers IDX et nombre d'exemples par classe. */
export interface ShardedGalleryInfo {
  readonly imagesFile: string;
  readonly labelsFile: string;
  readonly perClass: number;
}

/** Index JSON d'un jeu segmenté. */
export interface ShardedIndex {
  readonly version: number;
  readonly dataset: string;
  readonly classNames: readonly string[];
  readonly shardSize: number;
  readonly splits: { readonly train: ShardedSplitIndex; readonly test: ShardedSplitIndex };
  readonly galleries: { readonly train: ShardedGalleryInfo; readonly test: ShardedGalleryInfo };
}

/** Url publique d'un fichier d'un jeu, sous `datasets/<id>/`. */
function datasetAssetUrl(dataset: ShardedDatasetDefinition, file: string): string {
  return `${assetBaseUrl()}${DATASET_ASSET_DIRECTORY}/${dataset.id}/${file}`;
}

/** Clé de cache : identifiant du jeu et chemin du fichier. */
function cacheKey(dataset: ShardedDatasetDefinition, file: string): string {
  return `${dataset.id}/${file}`;
}

/** Lit un entier non signé 32 bits gros-boutiste à l'offset donné. */
function readU32At(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] * 0x1000000 +
      bytes[offset + 1] * 0x10000 +
      bytes[offset + 2] * 0x100 +
      bytes[offset + 3]) >>>
    0
  );
}

/** Rejette une valeur qui n'est pas une chaîne non vide, en nommant le champ. */
function requireString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`sharded-assets: index field "${field}" must be a non-empty string.`);
  }
  return value;
}

/** Rejette un entier strictement positif, en nommant le champ. */
function requirePositiveInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(
      `sharded-assets: index field "${field}" must be a positive integer, received ${String(value)}.`,
    );
  }
  return value;
}

/** Rejette un entier non négatif, en nommant le champ. */
function requireNonNegativeInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new Error(
      `sharded-assets: index field "${field}" must be a non-negative integer, received ${String(value)}.`,
    );
  }
  return value;
}

/** Valide et convertit une fraction décrite par l'index JSON. */
function parseSplit(
  value: unknown,
  name: string,
  classNames: readonly string[],
): ShardedSplitIndex {
  if (value === null || typeof value !== 'object') {
    throw new Error(`sharded-assets: index splits.${name} must be an object.`);
  }
  const record = value as Record<string, unknown>;
  const count = requirePositiveInteger(record.count, `splits.${name}.count`);
  const labelsFile = requireString(record.labelsFile, `splits.${name}.labelsFile`);
  if (!Array.isArray(record.shards)) {
    throw new Error(`sharded-assets: index splits.${name}.shards must be an array.`);
  }

  const shards: ShardEntry[] = [];
  for (let index = 0; index < record.shards.length; index += 1) {
    const raw = record.shards[index];
    if (raw === null || typeof raw !== 'object') {
      throw new Error(`sharded-assets: index splits.${name}.shards[${index}] must be an object.`);
    }
    const shard = raw as Record<string, unknown>;
    const file = requireString(shard.file, `splits.${name}.shards[${index}].file`);
    const classIndex = requireNonNegativeInteger(
      shard.classIndex,
      `splits.${name}.shards[${index}].classIndex`,
    );
    if (classIndex >= classNames.length) {
      throw new RangeError(
        `sharded-assets: index splits.${name}.shards[${index}] "${file}" has classIndex ${classIndex}, which is not below the ${classNames.length} classes.`,
      );
    }
    const start = requireNonNegativeInteger(shard.start, `splits.${name}.shards[${index}].start`);
    const shardCount = requirePositiveInteger(shard.count, `splits.${name}.shards[${index}].count`);
    if (start + shardCount > count) {
      throw new Error(
        `sharded-assets: index splits.${name}.shards[${index}] "${file}" ends at ${start + shardCount}, beyond the split count ${count}.`,
      );
    }
    shards.push({ file, classIndex, start, count: shardCount });
  }
  return { count, labelsFile, shards };
}

/**
 * Valide la structure d'un index JSON (version 2) et retourne un `ShardedIndex`
 * figé par le typage. Rejette une version inconnue, un nombre de classes hors de
 * `[2, 20]`, une taille de fragment non positive, une fraction sans effectif
 * positif, un fragment dont l'intervalle dépasse l'effectif de la fraction, et
 * un fragment dont `classIndex` dépasse le nombre de classes. Un index version 1
 * est refusé avec la commande qui le régénère. Quand `expectedDataset` est
 * fourni, l'index doit déclarer ce même jeu de données.
 */
export function parseIndexJson(text: string, expectedDataset?: string): ShardedIndex {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new Error(`sharded-assets: the index is not valid JSON (${(error as Error).message}).`);
  }
  if (raw === null || typeof raw !== 'object') {
    throw new Error('sharded-assets: the index must be a JSON object.');
  }
  const record = raw as Record<string, unknown>;

  if (record.version === 1) {
    throw new RangeError(
      'sharded-assets: index version 1 is obsolete and has no per-split galleries. Rerun "npm run quickdraw:data" (QuickDraw) or "npm run medmnist:data" (MedMNIST) to regenerate a version 2 index.',
    );
  }
  if (record.version !== 2) {
    throw new Error(
      `sharded-assets: index version ${String(record.version)} is not supported, expected 2.`,
    );
  }
  const dataset = requireString(record.dataset, 'dataset');
  if (expectedDataset !== undefined && dataset !== expectedDataset) {
    throw new RangeError(
      `sharded-assets: index dataset "${dataset}" does not match the expected dataset "${expectedDataset}".`,
    );
  }
  if (!Array.isArray(record.classNames)) {
    throw new Error('sharded-assets: index field "classNames" must be an array.');
  }
  const classNames: string[] = [];
  for (let index = 0; index < record.classNames.length; index += 1) {
    classNames.push(requireString(record.classNames[index], `classNames[${index}]`));
  }
  if (classNames.length < MIN_CLASS_COUNT || classNames.length > MAX_CLASS_COUNT) {
    throw new Error(
      `sharded-assets: index field "classNames" must contain from ${MIN_CLASS_COUNT} to ${MAX_CLASS_COUNT} classes, received ${classNames.length}.`,
    );
  }
  const shardSize = requirePositiveInteger(record.shardSize, 'shardSize');

  if (record.splits === null || typeof record.splits !== 'object') {
    throw new Error('sharded-assets: index field "splits" must be an object.');
  }
  const splitsRecord = record.splits as Record<string, unknown>;
  const train = parseSplit(splitsRecord.train, 'train', classNames);
  const test = parseSplit(splitsRecord.test, 'test', classNames);

  if (record.galleries === null || typeof record.galleries !== 'object') {
    throw new Error('sharded-assets: index field "galleries" must be an object.');
  }
  const galleriesRecord = record.galleries as Record<string, unknown>;
  const trainGallery = parseGallery(galleriesRecord.train, 'galleries.train');
  const testGallery = parseGallery(galleriesRecord.test, 'galleries.test');

  // Les fragments d'une classe sont contigus : tout fragment qui n'est pas le
  // dernier de sa classe doit remplir la taille annoncée.
  for (const split of [train, test]) {
    validateShardSizes(split, shardSize);
  }

  return {
    version: 2,
    dataset,
    classNames,
    shardSize,
    splits: { train, test },
    galleries: { train: trainGallery, test: testGallery },
  };
}

/** Valide et convertit une galerie de fraction décrite par l'index JSON. */
function parseGallery(value: unknown, name: string): ShardedGalleryInfo {
  if (value === null || typeof value !== 'object') {
    throw new Error(`sharded-assets: index ${name} must be an object.`);
  }
  const record = value as Record<string, unknown>;
  return {
    imagesFile: requireString(record.imagesFile, `${name}.imagesFile`),
    labelsFile: requireString(record.labelsFile, `${name}.labelsFile`),
    perClass: requirePositiveInteger(record.perClass, `${name}.perClass`),
  };
}

/** Vérifie que chaque fragment respecte `shardSize`, sauf le dernier de sa classe. */
function validateShardSizes(split: ShardedSplitIndex, shardSize: number): void {
  const lastByClass = new Map<number, number>();
  for (let index = 0; index < split.shards.length; index += 1) {
    lastByClass.set(split.shards[index].classIndex, index);
  }
  for (let index = 0; index < split.shards.length; index += 1) {
    const shard = split.shards[index];
    if (shard.count > shardSize) {
      throw new Error(
        `sharded-assets: shard "${shard.file}" has ${shard.count} images, above shardSize ${shardSize}.`,
      );
    }
    if (lastByClass.get(shard.classIndex) !== index && shard.count !== shardSize) {
      throw new Error(
        `sharded-assets: shard "${shard.file}" has ${shard.count} images, expected the full shardSize ${shardSize} (only the last shard of a class may be smaller).`,
      );
    }
  }
}

/**
 * Retourne le nombre de fragments à garder par classe pour `percent` pourcent
 * des `shardsPerClass` fragments disponibles : `max(1, round(percent / 100 *
 * shardsPerClass))`, plafonné à `shardsPerClass`. `percent` doit être un nombre
 * fini de 1 à 100.
 */
export function shardCountForPercent(percent: number, shardsPerClass: number): number {
  if (!Number.isFinite(percent) || percent < 1 || percent > 100) {
    throw new RangeError(
      `sharded-assets: percent must be a finite number from 1 to 100, received ${String(percent)}.`,
    );
  }
  if (!Number.isInteger(shardsPerClass) || shardsPerClass < 1) {
    throw new RangeError(
      `sharded-assets: shardsPerClass must be a positive integer, received ${String(shardsPerClass)}.`,
    );
  }
  const wanted = Math.max(1, Math.round((percent / 100) * shardsPerClass));
  return Math.min(wanted, shardsPerClass);
}

/**
 * Ordre trié et sans doublon des classes demandées, ou `null` pour toutes les
 * classes. Rejette un indice qui n'est pas un entier non négatif ou une classe
 * sans fragment dans la fraction.
 */
function requestedClassOrder(
  classIndices: readonly number[] | null | undefined,
  available: ReadonlySet<number>,
): number[] | null {
  if (classIndices === null || classIndices === undefined || classIndices.length === 0) {
    return null;
  }
  const selected = new Set<number>();
  for (const value of classIndices) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(
        `sharded-assets: classIndices must contain non-negative integers, received ${String(value)}.`,
      );
    }
    if (!available.has(value)) {
      throw new RangeError(`sharded-assets: selected class ${value} has no shard in this split.`);
    }
    selected.add(value);
  }
  return Array.from(selected).sort((left, right) => left - right);
}

/**
 * Sélectionne les fragments d'une fraction pour `percent` pourcent : les
 * fragments sont regroupés par classe, les `N` premiers de chaque classe sont
 * gardés dans l'ordre de l'index, et le résultat est retourné dans l'ordre trié
 * des classes. Quand `classIndices` est fourni, seules ces classes sont gardées,
 * dans l'ordre trié de la sélection ; `null` ou une liste vide garde toutes les
 * classes. La sélection est déterministe et ne dépend pas du nom des fichiers.
 */
export function selectShardEntries(
  split: ShardedSplitIndex,
  percent: number,
  classIndices?: readonly number[] | null,
): readonly ShardEntry[] {
  const groups = new Map<number, ShardEntry[]>();
  for (const shard of split.shards) {
    const group = groups.get(shard.classIndex);
    if (group === undefined) {
      groups.set(shard.classIndex, [shard]);
    } else {
      group.push(shard);
    }
  }
  if (groups.size === 0) {
    return [];
  }
  const order = requestedClassOrder(classIndices, new Set(groups.keys()));
  const classes = order ?? Array.from(groups.keys()).sort((a, b) => a - b);

  let shardsPerClass = 0;
  for (const classIndex of classes) {
    shardsPerClass = Math.max(shardsPerClass, (groups.get(classIndex) as ShardEntry[]).length);
  }
  const wanted = shardCountForPercent(percent, shardsPerClass);

  const selected: ShardEntry[] = [];
  for (const classIndex of classes) {
    const group = groups.get(classIndex) as ShardEntry[];
    selected.push(...group.slice(0, wanted));
  }
  return selected;
}

/**
 * Concatène les fragments sélectionnés en un jeu IDX brut. Chaque tableau
 * d'images doit contenir exactement `count * 784` octets de pixels, et la
 * tranche de libellés `[start, start + count)` doit exister. Les tableaux
 * retournés sont des copies fraîches ; les tableaux d'entrée ne sont jamais
 * modifiés. Quand `labelRemap` est fourni, chaque libellé tranché passe par
 * cette fonction ; le résultat doit être un entier dans `[0, classCount)`, où
 * `classCount` est le nombre de classes distinctes des entrées. Par défaut, les
 * libellés sont copiés tels quels.
 */
export function mergeShards(
  entries: readonly ShardEntry[],
  shardImages: readonly Uint8Array[],
  labels: Uint8Array,
  labelRemap?: (classIndex: number) => number,
): MnistIdxDataset {
  if (entries.length !== shardImages.length) {
    throw new RangeError(
      `sharded-assets: ${entries.length} shard entries but ${shardImages.length} image arrays.`,
    );
  }
  let totalCount = 0;
  for (const entry of entries) {
    totalCount += entry.count;
  }
  const classCount =
    labelRemap === undefined ? 0 : new Set(entries.map((entry) => entry.classIndex)).size;
  const images = new Uint8Array(totalCount * PIXELS_PER_IMAGE);
  const mergedLabels = new Uint8Array(totalCount);
  let imageOffset = 0;
  let labelOffset = 0;
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const image = shardImages[index];
    const expectedBytes = entry.count * PIXELS_PER_IMAGE;
    if (image.length !== expectedBytes) {
      throw new RangeError(
        `sharded-assets: shard "${entry.file}" has ${image.length} image bytes, expected ${expectedBytes}.`,
      );
    }
    if (entry.start + entry.count > labels.length) {
      throw new RangeError(
        `sharded-assets: shard "${entry.file}" needs labels ${entry.start} to ${entry.start + entry.count}, but the split has ${labels.length}.`,
      );
    }
    images.set(image, imageOffset);
    if (labelRemap === undefined) {
      mergedLabels.set(labels.subarray(entry.start, entry.start + entry.count), labelOffset);
    } else {
      for (let offset = 0; offset < entry.count; offset += 1) {
        const original = labels[entry.start + offset];
        const mapped = labelRemap(original);
        if (!Number.isInteger(mapped) || mapped < 0 || mapped >= classCount) {
          throw new RangeError(
            `sharded-assets: shard "${entry.file}" remaps label ${original} to ${String(mapped)}, outside [0, ${classCount}).`,
          );
        }
        mergedLabels[labelOffset + offset] = mapped;
      }
    }
    imageOffset += expectedBytes;
    labelOffset += entry.count;
  }
  return { count: totalCount, images, labels: mergedLabels };
}

/** Index analysé par jeu et fichier : un second appel ne télécharge rien. */
const indexCache = new Map<string, ShardedIndex>();
const inFlightIndex = new Map<string, Promise<ShardedIndex>>();

/** Octets décodés par jeu et fichier : seuls les succès sont conservés. */
const bytesCache = new Map<string, Uint8Array>();
const inFlightBytes = new Map<string, Promise<Uint8Array>>();

/** Télécharge le texte d'une URL, avec une erreur qui nomme le fichier. */
async function fetchText(url: string, downloadCommand: string): Promise<string> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(
      mnistFetchErrorMessage(url, `network error: ${(error as Error).message}`, downloadCommand),
    );
  }
  if (!response.ok) {
    throw new Error(
      mnistFetchErrorMessage(
        url,
        `HTTP ${response.status} ${response.statusText}`,
        downloadCommand,
      ),
    );
  }
  return response.text();
}

/**
 * Télécharge et décode un fichier IDX d'un jeu, avec un cache par jeu et
 * chemin. Un échec n'est pas mis en cache : une nouvelle tentative relance le
 * téléchargement. Les appels concurrents partagent la même promesse.
 */
function fetchCachedBytes(
  dataset: ShardedDatasetDefinition,
  file: string,
  expectedMagic: number,
): Promise<Uint8Array> {
  const key = cacheKey(dataset, file);
  const cached = bytesCache.get(key);
  if (cached !== undefined) {
    return Promise.resolve(cached);
  }
  const pending = inFlightBytes.get(key);
  if (pending !== undefined) {
    return pending;
  }
  const promise = fetchGzip(
    datasetAssetUrl(dataset, file),
    expectedMagic,
    dataset.downloadCommand,
  ).then(
    (bytes) => {
      bytesCache.set(key, bytes);
      inFlightBytes.delete(key);
      return bytes;
    },
    (error: unknown) => {
      inFlightBytes.delete(key);
      throw error;
    },
  );
  inFlightBytes.set(key, promise);
  return promise;
}

/**
 * Télécharge et analyse l'index JSON d'un jeu segmenté. L'index analysé est mis
 * en cache par jeu ; un échec n'est pas conservé.
 */
export function loadShardedIndex(dataset: ShardedDatasetDefinition): Promise<ShardedIndex> {
  const key = cacheKey(dataset, dataset.indexFile);
  const cached = indexCache.get(key);
  if (cached !== undefined) {
    return Promise.resolve(cached);
  }
  const pending = inFlightIndex.get(key);
  if (pending !== undefined) {
    return pending;
  }
  const promise = fetchText(
    datasetAssetUrl(dataset, dataset.indexFile),
    dataset.downloadCommand,
  ).then(
    (text) => {
      const index = parseIndexJson(text, dataset.id);
      indexCache.set(key, index);
      inFlightIndex.delete(key);
      return index;
    },
    (error: unknown) => {
      inFlightIndex.delete(key);
      throw error;
    },
  );
  inFlightIndex.set(key, promise);
  return promise;
}

/** Retire l'en-tête IDX de 16 octets d'un fragment d'images et le valide. */
function imagePayload(bytes: Uint8Array, file: string, expectedCount: number): Uint8Array {
  if (bytes.length < IMAGE_HEADER_BYTES) {
    throw new Error(
      `sharded-assets: shard "${file}" is truncated, expected at least ${IMAGE_HEADER_BYTES} bytes.`,
    );
  }
  const count = readU32At(bytes, 4);
  if (count !== expectedCount) {
    throw new Error(
      `sharded-assets: shard "${file}" declares ${count} images, expected ${expectedCount}.`,
    );
  }
  const rows = readU32At(bytes, 8);
  const cols = readU32At(bytes, 12);
  if (rows !== 28 || cols !== 28) {
    throw new Error(`sharded-assets: shard "${file}" has ${rows}x${cols} images, expected 28x28.`);
  }
  return bytes.subarray(IMAGE_HEADER_BYTES);
}

/**
 * Retire l'en-tête IDX de 8 octets d'un fichier de libellés et vérifie que le
 * nombre de libellés annoncé vaut `expectedCount`.
 */
function labelPayload(bytes: Uint8Array, file: string, expectedCount: number): Uint8Array {
  if (bytes.length < LABEL_HEADER_BYTES) {
    throw new Error(
      `sharded-assets: "${file}" is truncated, expected at least ${LABEL_HEADER_BYTES} bytes.`,
    );
  }
  const count = readU32At(bytes, 4);
  if (count !== expectedCount) {
    throw new RangeError(
      `sharded-assets: "${file}" declares ${count} labels, expected ${expectedCount}.`,
    );
  }
  return bytes.subarray(LABEL_HEADER_BYTES);
}

/** Sélection de classes : ordre trié et remap des libellés vers `0..N-1`. */
interface ClassSelection {
  readonly order: readonly number[];
  readonly remap: (classIndex: number) => number;
}

/**
 * Construit la sélection de classes à partir de `classIndices`. Retourne
 * `undefined` quand aucune sélection n'est demandée (`null` ou liste vide), pour
 * garder le chemin par défaut. Rejette un indice hors du nombre de classes et
 * tout libellé qui n'appartient pas à la sélection.
 */
function classSelection(
  classCount: number,
  classIndices: readonly number[] | null | undefined,
): ClassSelection | undefined {
  if (classIndices === null || classIndices === undefined || classIndices.length === 0) {
    return undefined;
  }
  const selected = new Set<number>();
  for (const value of classIndices) {
    if (!Number.isInteger(value) || value < 0 || value >= classCount) {
      throw new RangeError(
        `sharded-assets: classIndices must contain integers from 0 to ${classCount - 1}, received ${String(value)}.`,
      );
    }
    selected.add(value);
  }
  const order = Array.from(selected).sort((left, right) => left - right);
  const positions = new Map(order.map((classIndex, position) => [classIndex, position]));
  return {
    order,
    remap: (classIndex: number): number => {
      const position = positions.get(classIndex);
      if (position === undefined) {
        throw new RangeError(
          `sharded-assets: label ${classIndex} is not one of the selected classes.`,
        );
      }
      return position;
    },
  };
}

/** Construit une fraction IDX brute à partir des octets téléchargés. */
function buildSplit(
  split: ShardedSplitIndex,
  entries: readonly ShardEntry[],
  byFile: ReadonlyMap<string, Uint8Array>,
  labelRemap?: (classIndex: number) => number,
): MnistIdxDataset {
  const labelsBytes = byFile.get(split.labelsFile);
  if (labelsBytes === undefined) {
    throw new Error(`sharded-assets: labels file "${split.labelsFile}" was not downloaded.`);
  }
  const labels = labelPayload(labelsBytes, split.labelsFile, split.count);
  const shardImages = entries.map((entry) => {
    const bytes = byFile.get(entry.file);
    if (bytes === undefined) {
      throw new Error(`sharded-assets: shard "${entry.file}" was not downloaded.`);
    }
    return imagePayload(bytes, entry.file, entry.count);
  });
  return mergeShards(entries, shardImages, labels, labelRemap);
}

/** Applique `fn` à chaque élément avec au plus `limit` appels simultanés. */
async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (;;) {
      const index = next;
      next += 1;
      if (index >= items.length) {
        return;
      }
      results[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

/**
 * Télécharge et décode les deux fractions d'un jeu segmenté. L'index, les deux
 * fichiers de libellés et les fragments sélectionnés sont téléchargés avec un
 * cache par jeu et chemin, au plus quatre fichiers à la fois. `onProgress`
 * reçoit `(filesLoaded, totalFiles)`, l'index comptant pour un fichier. Quand
 * `classIndices` est fourni, seules ces classes sont téléchargées et leurs
 * libellés sont remappés vers `0..N-1` dans l'ordre trié de la sélection ;
 * `null` ou une liste vide garde toutes les classes.
 */
export async function loadShardedSplits(
  dataset: ShardedDatasetDefinition,
  trainPercent: number,
  validationPercent: number,
  onProgress?: (loaded: number, total: number) => void,
  classIndices?: readonly number[] | null,
): Promise<MnistSplits> {
  const index = await loadShardedIndex(dataset);
  const trainSplit = index.splits.train;
  const testSplit = index.splits.test;
  const selection = classSelection(index.classNames.length, classIndices);
  const trainEntries = selectShardEntries(trainSplit, trainPercent, selection?.order);
  const testEntries = selectShardEntries(testSplit, validationPercent, selection?.order);

  const totalFiles = 1 + 2 + trainEntries.length + testEntries.length;
  let loaded = 1;
  onProgress?.(loaded, totalFiles);

  const tasks: { file: string; magic: number }[] = [
    { file: trainSplit.labelsFile, magic: IDX_LABEL_MAGIC },
    { file: testSplit.labelsFile, magic: IDX_LABEL_MAGIC },
    ...trainEntries.map((entry) => ({ file: entry.file, magic: IDX_IMAGE_MAGIC })),
    ...testEntries.map((entry) => ({ file: entry.file, magic: IDX_IMAGE_MAGIC })),
  ];

  const results = await mapWithConcurrency(tasks, FETCH_CONCURRENCY, async (task) => {
    const bytes = await fetchCachedBytes(dataset, task.file, task.magic);
    loaded += 1;
    onProgress?.(loaded, totalFiles);
    return { file: task.file, bytes };
  });

  const byFile = new Map(results.map((result) => [result.file, result.bytes]));
  return {
    train: buildSplit(trainSplit, trainEntries, byFile, selection?.remap),
    test: buildSplit(testSplit, testEntries, byFile, selection?.remap),
  };
}

/** Cache des galeries décodées par jeu, fraction et nombre d'exemples par classe. */
const galleryCache = new Map<string, Promise<MnistSample>>();

/**
 * Télécharge et décode la galerie d'une fraction d'un jeu segmenté et la
 * retourne sous forme de `MnistSample` normalisé (`Float32` dans `[0, 1]`), via
 * `materializeSample`. Les noms des fichiers viennent de `galleries[split]` de
 * l'index, qui en est la source autoritative. La galerie doit contenir
 * `perClass * classCount` exemples. Le résultat est mis en cache par jeu,
 * fraction et `perClass` ; un échec n'est pas conservé. Aucun fragment n'est
 * téléchargé.
 */
export async function loadShardedGallery(
  dataset: ShardedDatasetDefinition,
  split: GallerySplit = 'test',
): Promise<MnistSample> {
  const index = await loadShardedIndex(dataset);
  const galleryIndex = index.galleries[split];
  const key = galleryCacheKey(dataset.id, split, galleryIndex.perClass);
  const cached = galleryCache.get(key);
  if (cached !== undefined) {
    return cached;
  }
  const pending = downloadGallery(dataset, index, galleryIndex).catch((error: unknown) => {
    galleryCache.delete(key);
    throw error;
  });
  galleryCache.set(key, pending);
  return pending;
}

/** Nombre maximal de fragments décompressés gardés en mémoire par défaut. */
const DEFAULT_MAX_SHARDS = 60;

/** Options de `SegmentStore`. */
export interface SegmentStoreOptions {
  readonly dataset: ShardedDatasetDefinition;
  readonly index: ShardedIndex;
  /** Indices de classes d'origine, dans l'ordre de sortie. */
  readonly classIndices: readonly number[];
  /** Nom du cache Cache API, par exemple `drl-datasets-v2-quickdraw`. */
  readonly cacheName: string;
  /** Fraction servie par le magasin (`train` par défaut). */
  readonly split?: 'train' | 'test';
  /**
   * Fragments exacts retenus pour cette fraction, par exemple le résultat de
   * `selectShardEntries`. Par défaut, tous les fragments des classes
   * sélectionnées.
   */
  readonly entries?: readonly RoundEntry[];
  /**
   * Borne dure du nombre de fragments décompressés résidents (défaut 60). La
   * protection des rounds est au mieux quand l'ensemble de travail dépasse
   * cette borne.
   */
  readonly maxShards?: number;
  /** Implémentation de `fetch`, remplaçable pour les tests. */
  readonly fetchImpl?: typeof fetch;
}

/** Décode un corps téléchargé en payload IDX validé. */
type PayloadDecoder = (bytes: Uint8Array, contentEncoding: string | null) => Promise<Uint8Array>;

/** Fragment décodé résident : payload d'images sans en-tête IDX. */
interface ResidentShard {
  readonly file: string;
  readonly classIndex: number;
  readonly payload: Uint8Array;
  lastUsed: number;
}

/** Libellés décodés d'une fraction, sans en-tête IDX. */
interface ResidentLabels {
  readonly file: string;
  readonly payload: Uint8Array;
}

/** Clé d'un round : ses fragments suffisent à identifier ses données. */
function roundKey(ref: RoundRef): string {
  return ref.entries.map((entry) => `${entry.file}:${entry.start}:${entry.count}`).join('|');
}

/** Fichiers distincts référencés par un round. */
function roundFiles(ref: RoundRef): ReadonlySet<string> {
  return new Set(ref.entries.map((entry) => entry.file));
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

/** Hash FNV-1a 32 bits d'une chaîne, stable entre exécutions. */
function hashString(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** Graine du mélange équilibré d'un round, dérivée de son indice et de ses fichiers. */
function roundShuffleSeed(ref: RoundRef): number {
  const key = ref.entries.map((entry) => entry.file).join('|');
  return (Math.imul(ref.index + 1, 0x9e3779b1) ^ hashString(key)) >>> 0;
}

/** Limiteur de téléchargements partagé par tous les appels du magasin. */
class FetchLimiter {
  private active = 0;

  private readonly waiting: (() => void)[] = [];

  constructor(private readonly limit: number) {}

  async run<T>(task: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await task();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    if (this.active < this.limit) {
      this.active += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  private release(): void {
    const next = this.waiting.shift();
    if (next !== undefined) {
      // Le créneau passe directement au premier appelant en attente.
      next();
      return;
    }
    this.active -= 1;
  }
}

/** Rejette une plage `[start, end)` hors de `[0, count)`. */
function assertBatchRange(count: number, start: number, end: number): void {
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end < start ||
    end > count
  ) {
    throw new RangeError(
      `sharded-assets: invalid batch range [${String(start)}, ${String(end)}) for ${count} images.`,
    );
  }
}

/** Lit la tranche `[start, end)` d'un round en copiant images et libellés. */
function readRoundBatch(
  images: Float32Array,
  labels: Uint8Array,
  count: number,
  start: number,
  end: number,
): MnistSample {
  assertBatchRange(count, start, end);
  return {
    count: end - start,
    images: images.slice(start * PIXELS_PER_IMAGE, end * PIXELS_PER_IMAGE),
    labels: labels.slice(start, end),
  };
}

/** Nombre de fichiers distincts dans la tranche `[start, end)` d'un round. */
function countDistinctShards(
  fileByPosition: readonly string[],
  count: number,
  start: number,
  end: number,
): number {
  assertBatchRange(count, start, end);
  const files = new Set<string>();
  for (let position = start; position < end; position += 1) {
    files.add(fileByPosition[position]);
  }
  return files.size;
}

/**
 * Magasin navigateur des rounds segmentés d'un jeu.
 *
 * Le magasin sert une fraction (`train` par défaut, ou `test`) et un
 * sous-ensemble de classes : `classIndices` liste les indices d'origine dans
 * l'ordre de sortie, et les libellés sont remappés vers `0..N-1`. Un round est
 * matérialisé en `Float32Array` normalisé et `Uint8Array` de libellés, avec les
 * classes entrelacées, puis gardé jusqu'à `releaseRound`.
 *
 * Les fragments compressés passent par trois niveaux : la mémoire du magasin,
 * le cache Cache API (`cacheName`, clé = URL du fragment), puis le réseau avec
 * au plus quatre téléchargements simultanés. Un corps n'est mis en cache
 * qu'après un décodage réussi. Une panne du cache (absence de `caches`, quota
 * dépassé) ne fait jamais échouer un chargement.
 *
 * Les fragments décompressés résidents sont bornés par `maxShards` (60 par
 * défaut). `maxShards` est une borne dure. La protection des rounds est au
 * mieux quand l'ensemble de travail dépasse cette borne. L'éviction choisit le
 * fragment le moins récemment utilisé qui n'est ni chargé, ni référencé par le
 * round courant, un round préchargé ou un round résident, et elle garde un
 * fragment par classe quand la borne le permet.
 */
export class SegmentStore implements SegmentLoader {
  readonly totalCount: number;

  private readonly dataset: ShardedDatasetDefinition;

  private readonly cacheName: string;

  private readonly split: 'train' | 'test';

  private readonly maxShards: number;

  private readonly fetchImpl: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

  private readonly classPositions: ReadonlyMap<number, number>;

  private readonly splitByShard: ReadonlyMap<string, ShardedSplitIndex>;

  private readonly selectedEntries: ReadonlyMap<string, RoundEntry>;

  private readonly limiter = new FetchLimiter(FETCH_CONCURRENCY);

  private readonly shards = new Map<string, ResidentShard>();

  private readonly labels = new Map<string, ResidentLabels>();

  private readonly roundShards = new Map<string, ReadonlySet<string>>();

  private readonly rounds = new Map<string, SegmentRound>();

  private readonly prefetchedRounds = new Map<string, Promise<SegmentRound>>();

  private readonly prefetchedShards = new Map<string, ReadonlySet<string>>();

  private readonly shardLoads = new Map<string, Promise<ResidentShard>>();

  private readonly labelLoads = new Map<string, Promise<ResidentLabels>>();

  private readonly roundLoads = new Map<string, Promise<SegmentRound>>();

  private readonly downloads = new Map<string, Promise<Uint8Array>>();

  private readonly memoryPayloads = new Map<string, Uint8Array>();

  private readonly loadingShards = new Map<string, number>();

  private readonly discardedRounds = new Set<string>();

  private cachePromise: Promise<Cache | null> | undefined;

  private useCounter = 0;

  constructor(options: SegmentStoreOptions) {
    if (options === null || typeof options !== 'object') {
      throw new TypeError('sharded-assets: SegmentStore options are required.');
    }
    const { dataset, index, classIndices } = options;
    if (dataset === null || typeof dataset !== 'object') {
      throw new TypeError('sharded-assets: SegmentStore dataset is required.');
    }
    if (index === null || typeof index !== 'object') {
      throw new TypeError('sharded-assets: SegmentStore index is required.');
    }
    if (typeof options.cacheName !== 'string' || options.cacheName.length === 0) {
      throw new TypeError('sharded-assets: SegmentStore cacheName must be a non-empty string.');
    }
    if (!Array.isArray(classIndices) || classIndices.length === 0) {
      throw new RangeError('sharded-assets: SegmentStore needs at least one class index.');
    }

    const maxShards = options.maxShards ?? DEFAULT_MAX_SHARDS;
    if (!Number.isInteger(maxShards) || maxShards < 1) {
      throw new RangeError(
        `sharded-assets: maxShards must be a positive integer, received ${String(maxShards)}.`,
      );
    }

    const split = options.split ?? 'train';
    if (split !== 'train' && split !== 'test') {
      throw new RangeError(
        `sharded-assets: split must be "train" or "test", received ${String(split)}.`,
      );
    }
    const splitIndex = index.splits[split];

    const positions = new Map<number, number>();
    for (let position = 0; position < classIndices.length; position += 1) {
      const classIndex = classIndices[position];
      if (
        !Number.isInteger(classIndex) ||
        classIndex < 0 ||
        classIndex >= index.classNames.length
      ) {
        throw new RangeError(
          `sharded-assets: classIndices[${position}] must be an integer from 0 to ${index.classNames.length - 1}, received ${String(classIndex)}.`,
        );
      }
      if (positions.has(classIndex)) {
        throw new RangeError(
          `sharded-assets: classIndices contains class ${classIndex} more than once.`,
        );
      }
      positions.set(classIndex, position);
    }

    const splitByShard = new Map<string, ShardedSplitIndex>();
    for (const shard of splitIndex.shards) {
      splitByShard.set(shard.file, splitIndex);
    }

    const indexedShards = new Map(splitIndex.shards.map((shard) => [shard.file, shard] as const));
    const selectedShards =
      options.entries === undefined
        ? splitIndex.shards.filter((shard) => positions.has(shard.classIndex))
        : [...options.entries];
    const selectedEntries = new Map<string, RoundEntry>();
    for (const entry of selectedShards) {
      const indexed = indexedShards.get(entry.file);
      if (indexed === undefined) {
        throw new RangeError(
          `sharded-assets: entry "${entry.file}" is not in the ${split} split of the dataset index.`,
        );
      }
      if (
        indexed.classIndex !== entry.classIndex ||
        indexed.start !== entry.start ||
        indexed.count !== entry.count
      ) {
        throw new RangeError(
          `sharded-assets: entry "${entry.file}" does not match the ${split} split of the dataset index.`,
        );
      }
      if (!positions.has(entry.classIndex)) {
        throw new RangeError(
          `sharded-assets: entry "${entry.file}" has classIndex ${entry.classIndex}, which is not selected.`,
        );
      }
      if (selectedEntries.has(entry.file)) {
        throw new RangeError(
          `sharded-assets: entry "${entry.file}" appears more than once in the selected shards.`,
        );
      }
      selectedEntries.set(entry.file, {
        file: entry.file,
        classIndex: entry.classIndex,
        start: entry.start,
        count: entry.count,
      });
    }
    for (const classIndex of classIndices) {
      if (![...selectedEntries.values()].some((entry) => entry.classIndex === classIndex)) {
        throw new RangeError(
          `sharded-assets: selected class ${classIndex} has no shard in the selected entries for the ${split} split.`,
        );
      }
    }

    let totalCount = 0;
    for (const entry of selectedEntries.values()) {
      totalCount += entry.count;
    }
    if (totalCount < 1) {
      throw new RangeError('sharded-assets: SegmentStore selected no images.');
    }

    this.dataset = dataset;
    this.cacheName = options.cacheName;
    this.split = split;
    this.maxShards = maxShards;
    // `fetch` doit être appelé comme une fonction nue : dans un navigateur,
    // `this.fetchImpl(url)` lierait le magasin comme récepteur et `window.fetch`
    // lèverait un TypeError d'invocation illégale.
    this.fetchImpl = (input, init) => (options.fetchImpl ?? fetch)(input, init);
    this.classPositions = positions;
    this.splitByShard = splitByShard;
    this.selectedEntries = selectedEntries;
    this.totalCount = totalCount;
  }

  /** Nombre de fragments décompressés actuellement résidents. */
  get residentShards(): number {
    return this.shards.size;
  }

  /**
   * Retourne le round demandé, en réutilisant un round résident ou préchargé.
   * Les appels concurrents pour le même round partagent la même promesse.
   */
  async loadRound(ref: RoundRef): Promise<SegmentRound> {
    const key = roundKey(ref);
    const resident = this.rounds.get(key);
    if (resident !== undefined) {
      return resident;
    }
    const prefetched = this.prefetchedRounds.get(key);
    if (prefetched !== undefined) {
      this.prefetchedRounds.delete(key);
      this.prefetchedShards.delete(key);
      return prefetched;
    }
    return this.startRound(ref);
  }

  /**
   * Lance le chargement d'un round sans l'attendre. Le résultat reste disponible
   * pour le prochain `loadRound`, et une erreur est ignorée jusqu'à cet appel.
   */
  prefetchRound(ref: RoundRef): void {
    const key = roundKey(ref);
    if (this.rounds.has(key) || this.prefetchedRounds.has(key) || this.roundLoads.has(key)) {
      return;
    }
    this.prefetchedShards.set(key, roundFiles(ref));
    const promise = this.startRound(ref).catch((error: unknown) => {
      this.prefetchedRounds.delete(key);
      this.prefetchedShards.delete(key);
      throw error;
    });
    this.prefetchedRounds.set(key, promise);
    // Sans consommateur, l'échec ne doit pas produire de rejet non géré.
    promise.catch(() => undefined);
  }

  /**
   * Libère les images et libellés d'un round, puis évince les fragments que
   * plus aucun round résident, préchargé ou en cours de chargement n'utilise.
   */
  releaseRound(ref: RoundRef): void {
    const key = roundKey(ref);
    const files = this.roundShards.get(key);
    this.rounds.delete(key);
    this.roundShards.delete(key);
    this.prefetchedRounds.delete(key);
    this.prefetchedShards.delete(key);
    if (this.roundLoads.has(key)) {
      this.discardedRounds.add(key);
    }
    if (files === undefined) {
      return;
    }
    for (const file of files) {
      if (!this.isNeeded(file)) {
        this.evictShard(file);
      }
    }
  }

  /** Enregistre un round en cours de chargement et le rend résident. */
  private startRound(ref: RoundRef): Promise<SegmentRound> {
    const key = roundKey(ref);
    this.discardedRounds.delete(key);
    const pending = this.roundLoads.get(key);
    if (pending !== undefined) {
      return pending;
    }
    const files = roundFiles(ref);
    this.retainLoading(files);
    let retained = false;
    const promise = this.buildRound(ref)
      .then((round) => {
        if (this.discardedRounds.delete(key)) {
          return round;
        }
        retained = true;
        this.rounds.set(key, round);
        this.roundShards.set(key, files);
        return round;
      })
      .finally(() => {
        this.releaseLoading(files);
        this.discardedRounds.delete(key);
        if (!retained) {
          for (const file of files) {
            if (!this.isNeeded(file)) {
              this.evictShard(file);
            }
          }
        }
        if (this.roundLoads.get(key) === promise) {
          this.roundLoads.delete(key);
        }
      });
    this.roundLoads.set(key, promise);
    return promise;
  }

  /** Charge les fragments et les libellés, puis assemble le round en ordre équilibré. */
  private async buildRound(ref: RoundRef): Promise<SegmentRound> {
    if (ref.entries.length === 0) {
      throw new RangeError('sharded-assets: a round must contain at least one shard entry.');
    }

    const splits = new Map<string, ShardedSplitIndex>();
    const parts = await Promise.all(
      ref.entries.map(async (entry) => {
        const split = this.splitByShard.get(entry.file);
        if (split === undefined) {
          throw new Error(
            `sharded-assets: shard "${entry.file}" is not in the ${this.split} split of the dataset index.`,
          );
        }
        const selected = this.selectedEntries.get(entry.file);
        if (selected === undefined) {
          throw new RangeError(
            `sharded-assets: shard "${entry.file}" is not in the selected shard set for the ${this.split} split.`,
          );
        }
        if (
          selected.classIndex !== entry.classIndex ||
          selected.start !== entry.start ||
          selected.count !== entry.count
        ) {
          throw new RangeError(
            `sharded-assets: shard "${entry.file}" does not match its selected entry (classIndex ${entry.classIndex}, start ${entry.start}, count ${entry.count}).`,
          );
        }
        splits.set(split.labelsFile, split);
        const shard = await this.loadShard(entry);
        return { entry, split, payload: shard.payload };
      }),
    );

    const labels = new Map<string, Uint8Array>();
    await Promise.all(
      [...splits.values()].map(async (split) => {
        const resident = await this.loadLabels(split);
        labels.set(split.labelsFile, resident.payload);
      }),
    );

    let count = 0;
    for (const entry of ref.entries) {
      count += entry.count;
    }
    const images = new Float32Array(count * PIXELS_PER_IMAGE);
    const roundLabels = new Uint8Array(count);
    const fileByPosition = new Array<string>(count);
    const mappedLabels: Uint8Array[] = [];

    for (const part of parts) {
      const source = labels.get(part.split.labelsFile);
      if (source === undefined) {
        throw new Error(
          `sharded-assets: labels file "${part.split.labelsFile}" was not downloaded.`,
        );
      }
      const entry = part.entry;
      const position = this.classPositions.get(entry.classIndex);
      if (position === undefined) {
        throw new RangeError(
          `sharded-assets: shard "${entry.file}" has classIndex ${entry.classIndex}, which is not selected.`,
        );
      }
      if (entry.start + entry.count > source.length) {
        throw new RangeError(
          `sharded-assets: shard "${entry.file}" needs labels ${entry.start} to ${entry.start + entry.count}, but "${part.split.labelsFile}" has ${source.length}.`,
        );
      }
      const entryLabels = new Uint8Array(entry.count);
      for (let index = 0; index < entry.count; index += 1) {
        const original = source[entry.start + index];
        // Le fichier global porte les index d'origine. Un fichier déjà remappé
        // (libellé égal à la position de sortie) est aussi accepté.
        const mapped =
          this.classPositions.get(original) ?? (original === position ? position : undefined);
        if (mapped === undefined) {
          throw new RangeError(
            `sharded-assets: "${part.split.labelsFile}" has label ${original} at index ${entry.start + index}, which is not one of the selected classes.`,
          );
        }
        entryLabels[index] = mapped;
      }
      mappedLabels.push(entryLabels);
    }

    // Ordre équilibré : chaque cycle prend un exemple de plus par entrée, dans
    // un ordre de classes mélangé, jusqu'à épuisement. Les lots consécutifs
    // alternent donc les classes au lieu de les regrouper.
    const rng = mulberry32(roundShuffleSeed(ref));
    const entryOrder = shuffledIndices(parts.length, rng);
    const localOrders = parts.map((part) => shuffledIndices(part.entry.count, rng));
    const cursors = new Array<number>(parts.length).fill(0);
    let position = 0;
    while (position < count) {
      for (const partIndex of entryOrder) {
        const part = parts[partIndex];
        const cursor = cursors[partIndex];
        if (cursor >= part.entry.count) {
          continue;
        }
        cursors[partIndex] = cursor + 1;
        const local = localOrders[partIndex][cursor];
        const imageStart = local * PIXELS_PER_IMAGE;
        const imageOffset = position * PIXELS_PER_IMAGE;
        for (let pixel = 0; pixel < PIXELS_PER_IMAGE; pixel += 1) {
          images[imageOffset + pixel] = part.payload[imageStart + pixel] / 255;
        }
        roundLabels[position] = mappedLabels[partIndex][local];
        fileByPosition[position] = part.entry.file;
        position += 1;
      }
    }

    return {
      count,
      images,
      labels: roundLabels,
      readBatch: (start: number, end: number): MnistSample =>
        readRoundBatch(images, roundLabels, count, start, end),
      shardsInRange: (start: number, end: number): number =>
        countDistinctShards(fileByPosition, count, start, end),
    };
  }

  /** Charge et décode un fragment, en partageant les chargements en vol. */
  private loadShard(entry: RoundEntry): Promise<ResidentShard> {
    const resident = this.shards.get(entry.file);
    if (resident !== undefined) {
      this.useCounter += 1;
      resident.lastUsed = this.useCounter;
      return Promise.resolve(resident);
    }
    const pending = this.shardLoads.get(entry.file);
    if (pending !== undefined) {
      return pending;
    }
    const promise = this.decodeShardSource(entry)
      .then((shard) => {
        this.shards.set(entry.file, shard);
        this.enforceWindow();
        return shard;
      })
      .finally(() => {
        if (this.shardLoads.get(entry.file) === promise) {
          this.shardLoads.delete(entry.file);
        }
      });
    this.shardLoads.set(entry.file, promise);
    return promise;
  }

  /** Décompresse un fragment, retire son en-tête et valide sa longueur. */
  private async decodeShardSource(entry: RoundEntry): Promise<ResidentShard> {
    const url = datasetAssetUrl(this.dataset, entry.file);
    const payload = await this.readDecoded(url, (bytes, contentEncoding) =>
      this.decodeShardPayload(entry, bytes, contentEncoding),
    );
    this.useCounter += 1;
    return {
      file: entry.file,
      classIndex: entry.classIndex,
      payload,
      lastUsed: this.useCounter,
    };
  }

  /** Décode et valide le payload d'un fragment. */
  private async decodeShardPayload(
    entry: RoundEntry,
    bytes: Uint8Array,
    contentEncoding: string | null,
  ): Promise<Uint8Array> {
    let decoded: Uint8Array;
    try {
      decoded = await decodeIdxPayload(bytes, contentEncoding);
    } catch (error) {
      throw new Error(
        `sharded-assets: could not decode shard "${entry.file}". (${(error as Error).message})`,
      );
    }
    const payload = imagePayload(decoded, entry.file, entry.count);
    const expectedBytes = entry.count * PIXELS_PER_IMAGE;
    if (payload.length !== expectedBytes) {
      throw new RangeError(
        `sharded-assets: shard "${entry.file}" has ${payload.length} image bytes, expected ${expectedBytes}.`,
      );
    }
    return payload;
  }

  /** Charge et décode les libellés globaux d'une fraction, une seule fois. */
  private loadLabels(split: ShardedSplitIndex): Promise<ResidentLabels> {
    const resident = this.labels.get(split.labelsFile);
    if (resident !== undefined) {
      return Promise.resolve(resident);
    }
    const pending = this.labelLoads.get(split.labelsFile);
    if (pending !== undefined) {
      return pending;
    }
    const promise = this.decodeLabels(split).finally(() => {
      if (this.labelLoads.get(split.labelsFile) === promise) {
        this.labelLoads.delete(split.labelsFile);
      }
    });
    this.labelLoads.set(split.labelsFile, promise);
    return promise;
  }

  private async decodeLabels(split: ShardedSplitIndex): Promise<ResidentLabels> {
    const url = datasetAssetUrl(this.dataset, split.labelsFile);
    const payload = await this.readDecoded(url, (bytes, contentEncoding) =>
      this.decodeLabelsPayload(split, bytes, contentEncoding),
    );
    const resident = { file: split.labelsFile, payload };
    this.labels.set(split.labelsFile, resident);
    return resident;
  }

  /** Décode et valide le payload d'un fichier de libellés. */
  private async decodeLabelsPayload(
    split: ShardedSplitIndex,
    bytes: Uint8Array,
    contentEncoding: string | null,
  ): Promise<Uint8Array> {
    let decoded: Uint8Array;
    try {
      decoded = await decodeIdxPayload(bytes, contentEncoding);
    } catch (error) {
      throw new Error(
        `sharded-assets: could not decode labels "${split.labelsFile}". (${(error as Error).message})`,
      );
    }
    return labelPayload(decoded, split.labelsFile, split.count);
  }

  /**
   * Lit et décode un fichier depuis la mémoire, puis le cache Cache API, puis le
   * réseau. Un corps qui ne se décode pas n'est jamais mis en cache. Un corps
   * de cache invalide est supprimé, puis retéléchargé une fois. Un échec réseau
   * rejette la promesse et sera retenté au prochain appel.
   */
  private readDecoded(url: string, decode: PayloadDecoder): Promise<Uint8Array> {
    const cached = this.memoryPayloads.get(url);
    if (cached !== undefined) {
      return Promise.resolve(cached);
    }
    const pending = this.downloads.get(url);
    if (pending !== undefined) {
      return pending;
    }
    const promise = this.downloadDecoded(url, decode).finally(() => {
      if (this.downloads.get(url) === promise) {
        this.downloads.delete(url);
      }
    });
    this.downloads.set(url, promise);
    return promise;
  }

  private async downloadDecoded(url: string, decode: PayloadDecoder): Promise<Uint8Array> {
    const cache = await this.openCache();
    if (cache !== null) {
      try {
        const hit = await cache.match(url);
        if (hit !== undefined) {
          try {
            const decoded = await decode(
              new Uint8Array(await hit.arrayBuffer()),
              hit.headers.get('content-encoding'),
            );
            this.memoryPayloads.set(url, decoded);
            return decoded;
          } catch {
            // Le corps du cache est invalide : il ne doit pas bloquer un retrait.
            try {
              await cache.delete(url);
            } catch {
              // Un cache qui refuse la suppression ne doit pas bloquer le réseau.
            }
          }
        }
      } catch {
        // Une lecture de cache en échec ne doit pas empêcher le téléchargement.
      }
    }
    return this.limiter.run(() => this.fetchDecoded(url, cache, decode));
  }

  private async fetchDecoded(
    url: string,
    cache: Cache | null,
    decode: PayloadDecoder,
  ): Promise<Uint8Array> {
    let response: Response;
    try {
      response = await this.fetchImpl(url);
    } catch (error) {
      throw new Error(
        mnistFetchErrorMessage(
          url,
          `network error: ${(error as Error).message}`,
          this.dataset.downloadCommand,
        ),
      );
    }
    if (!response.ok) {
      throw new Error(
        mnistFetchErrorMessage(
          url,
          `HTTP ${response.status} ${response.statusText}`,
          this.dataset.downloadCommand,
        ),
      );
    }

    let copy: Response | undefined;
    if (cache !== null) {
      try {
        copy = response.clone();
      } catch {
        copy = undefined;
      }
    }
    let bytes: Uint8Array;
    try {
      bytes = new Uint8Array(await response.arrayBuffer());
    } catch (error) {
      throw new Error(
        mnistFetchErrorMessage(
          url,
          `could not read the body: ${(error as Error).message}`,
          this.dataset.downloadCommand,
        ),
      );
    }
    const decoded = await decode(bytes, response.headers.get('content-encoding'));
    if (cache !== null && copy !== undefined) {
      try {
        await cache.put(url, copy);
      } catch {
        // Quota dépassé ou cache indisponible : la mémoire reste la seule copie.
      }
    }
    this.memoryPayloads.set(url, decoded);
    return decoded;
  }

  /** Ouvre le cache nommé, ou retourne `null` si Cache API est indisponible. */
  private openCache(): Promise<Cache | null> {
    if (this.cachePromise === undefined) {
      if (typeof caches === 'undefined') {
        this.cachePromise = Promise.resolve(null);
      } else {
        try {
          this.cachePromise = caches.open(this.cacheName).then(
            (cache) => cache,
            () => null,
          );
        } catch {
          this.cachePromise = Promise.resolve(null);
        }
      }
    }
    return this.cachePromise;
  }

  /**
   * Évince les fragments les moins récemment utilisés jusqu'à respecter
   * `maxShards`. Les fragments protégés passent en dernier, et une classe qui a
   * plusieurs fragments résidents perd le sien en premier.
   */
  private enforceWindow(): void {
    while (this.shards.size > this.maxShards) {
      const protectedFiles = new Set<string>();
      for (const files of this.roundShards.values()) {
        for (const file of files) {
          protectedFiles.add(file);
        }
      }
      for (const files of this.prefetchedShards.values()) {
        for (const file of files) {
          protectedFiles.add(file);
        }
      }
      for (const file of this.loadingShards.keys()) {
        protectedFiles.add(file);
      }

      const classCounts = new Map<number, number>();
      for (const shard of this.shards.values()) {
        classCounts.set(shard.classIndex, (classCounts.get(shard.classIndex) ?? 0) + 1);
      }
      const ordered = [...this.shards.values()].sort((a, b) => a.lastUsed - b.lastUsed);
      const victim =
        ordered.find(
          (shard) =>
            !protectedFiles.has(shard.file) && (classCounts.get(shard.classIndex) ?? 0) > 1,
        ) ??
        ordered.find(
          (shard) => protectedFiles.has(shard.file) && (classCounts.get(shard.classIndex) ?? 0) > 1,
        ) ??
        ordered.find((shard) => !protectedFiles.has(shard.file)) ??
        ordered[0];
      if (victim === undefined) {
        return;
      }
      this.evictShard(victim.file);
    }
  }

  /** Retire un fragment résident et son payload décodé en mémoire. */
  private evictShard(file: string): void {
    this.shards.delete(file);
    this.memoryPayloads.delete(datasetAssetUrl(this.dataset, file));
  }

  /** Vrai si un round résident, préchargé ou en cours utilise ce fragment. */
  private isNeeded(file: string): boolean {
    if (this.loadingShards.has(file)) {
      return true;
    }
    for (const files of this.roundShards.values()) {
      if (files.has(file)) {
        return true;
      }
    }
    for (const files of this.prefetchedShards.values()) {
      if (files.has(file)) {
        return true;
      }
    }
    return false;
  }

  private retainLoading(files: ReadonlySet<string>): void {
    for (const file of files) {
      this.loadingShards.set(file, (this.loadingShards.get(file) ?? 0) + 1);
    }
  }

  private releaseLoading(files: ReadonlySet<string>): void {
    for (const file of files) {
      const count = this.loadingShards.get(file);
      if (count === undefined) {
        continue;
      }
      if (count <= 1) {
        this.loadingShards.delete(file);
      } else {
        this.loadingShards.set(file, count - 1);
      }
    }
  }
}

/** Télécharge, décode et valide la galerie d'une fraction. */
async function downloadGallery(
  dataset: ShardedDatasetDefinition,
  index: ShardedIndex,
  galleryIndex: ShardedGalleryInfo,
): Promise<MnistSample> {
  const [images, labels] = await Promise.all([
    fetchCachedBytes(dataset, galleryIndex.imagesFile, IDX_IMAGE_MAGIC),
    fetchCachedBytes(dataset, galleryIndex.labelsFile, IDX_LABEL_MAGIC),
  ]);
  const gallery = loadIdxDataset(images, labels, index.classNames.length);
  const expectedCount = galleryIndex.perClass * index.classNames.length;
  if (gallery.count !== expectedCount) {
    throw new RangeError(
      `sharded-assets: the gallery holds ${gallery.count} images, expected ${expectedCount} (perClass ${galleryIndex.perClass} times ${index.classNames.length} classes).`,
    );
  }
  const indices = new Uint32Array(gallery.count);
  for (let index = 0; index < gallery.count; index += 1) {
    indices[index] = index;
  }
  return materializeSample(gallery, indices);
}
