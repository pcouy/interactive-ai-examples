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
 * nom du fichier.
 *
 * Les octets décodés sont mis en cache par identifiant de jeu et chemin de
 * fichier ; seuls les téléchargements réussis sont conservés, ce qui permet une
 * nouvelle tentative après un échec. Les téléchargements sont limités à quatre
 * simultanément. Les erreurs nomment le fichier concerné.
 *
 * Ce module est réservé au navigateur : il utilise `fetch` et `Response`. Le
 * core ne l'importe jamais.
 */

import type { ShardedDatasetDefinition } from '../../core/dataset-registry.js';
import {
  loadIdxDataset,
  type MnistIdxDataset,
  type MnistSample,
  materializeSample,
} from '../../core/mnist-data.js';
import {
  assetBaseUrl,
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

/** Index JSON d'un jeu segmenté. */
export interface ShardedIndex {
  readonly version: number;
  readonly dataset: string;
  readonly classNames: readonly string[];
  readonly shardSize: number;
  readonly splits: { readonly train: ShardedSplitIndex; readonly test: ShardedSplitIndex };
  readonly gallery: {
    readonly imagesFile: string;
    readonly labelsFile: string;
    readonly perClass: number;
  };
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
function parseSplit(value: unknown, name: string): ShardedSplitIndex {
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
 * Valide la structure d'un index JSON et retourne un `ShardedIndex` figé par
 * le typage. Rejette une version inconnue, un nombre de classes hors de
 * `[2, 20]`, une taille de fragment non positive, une fraction sans effectif
 * positif, et un fragment dont l'intervalle dépasse l'effectif de la fraction.
 */
export function parseIndexJson(text: string): ShardedIndex {
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

  if (record.version !== 1) {
    throw new Error(
      `sharded-assets: index version ${String(record.version)} is not supported, expected 1.`,
    );
  }
  const dataset = requireString(record.dataset, 'dataset');
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
  const train = parseSplit(splitsRecord.train, 'train');
  const test = parseSplit(splitsRecord.test, 'test');

  if (record.gallery === null || typeof record.gallery !== 'object') {
    throw new Error('sharded-assets: index field "gallery" must be an object.');
  }
  const galleryRecord = record.gallery as Record<string, unknown>;
  const gallery = {
    imagesFile: requireString(galleryRecord.imagesFile, 'gallery.imagesFile'),
    labelsFile: requireString(galleryRecord.labelsFile, 'gallery.labelsFile'),
    perClass: requirePositiveInteger(galleryRecord.perClass, 'gallery.perClass'),
  };

  // Les fragments d'une classe sont contigus : tout fragment qui n'est pas le
  // dernier de sa classe doit remplir la taille annoncée.
  for (const split of [train, test]) {
    validateShardSizes(split, shardSize);
  }

  return { version: 1, dataset, classNames, shardSize, splits: { train, test }, gallery };
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
 * Sélectionne les fragments d'une fraction pour `percent` pourcent : les
 * fragments sont regroupés par classe, les `N` premiers de chaque classe sont
 * gardés dans l'ordre de l'index, et le résultat est retourné dans l'ordre des
 * classes. La sélection est déterministe et ne dépend pas du nom des fichiers.
 */
export function selectShardEntries(
  split: ShardedSplitIndex,
  percent: number,
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
  let shardsPerClass = 0;
  for (const group of groups.values()) {
    shardsPerClass = Math.max(shardsPerClass, group.length);
  }
  const wanted = shardCountForPercent(percent, shardsPerClass);

  const classIndices = Array.from(groups.keys()).sort((a, b) => a - b);
  const selected: ShardEntry[] = [];
  for (const classIndex of classIndices) {
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
 * modifiés.
 */
export function mergeShards(
  entries: readonly ShardEntry[],
  shardImages: readonly Uint8Array[],
  labels: Uint8Array,
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
    mergedLabels.set(labels.subarray(entry.start, entry.start + entry.count), labelOffset);
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
      const index = parseIndexJson(text);
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

/** Construit une fraction IDX brute à partir des octets téléchargés. */
function buildSplit(
  split: ShardedSplitIndex,
  entries: readonly ShardEntry[],
  byFile: ReadonlyMap<string, Uint8Array>,
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
  return mergeShards(entries, shardImages, labels);
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
 * reçoit `(filesLoaded, totalFiles)`, l'index comptant pour un fichier.
 */
export async function loadShardedSplits(
  dataset: ShardedDatasetDefinition,
  trainPercent: number,
  validationPercent: number,
  onProgress?: (loaded: number, total: number) => void,
): Promise<MnistSplits> {
  const index = await loadShardedIndex(dataset);
  const trainSplit = index.splits.train;
  const testSplit = index.splits.test;
  const trainEntries = selectShardEntries(trainSplit, trainPercent);
  const testEntries = selectShardEntries(testSplit, validationPercent);

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
    train: buildSplit(trainSplit, trainEntries, byFile),
    test: buildSplit(testSplit, testEntries, byFile),
  };
}

/**
 * Télécharge et décode la galerie d'un jeu segmenté et la retourne sous forme
 * de `MnistSample` normalisé (`Float32` dans `[0, 1]`), via `materializeSample`.
 * Les noms des fichiers viennent de l'index, qui en est la source autoritative.
 * La galerie doit contenir `perClass * classCount` exemples.
 */
export async function loadShardedGallery(dataset: ShardedDatasetDefinition): Promise<MnistSample> {
  const index = await loadShardedIndex(dataset);
  const galleryIndex = index.gallery;
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
