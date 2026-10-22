#!/usr/bin/env node
/**
 * Convertit les sept archives `.npz` MedMNIST en morceaux IDX d'images gzip.
 *
 * Chaque archive est un zip : six membres `.npy`, `train_*`, `val_*` et
 * `test_*`, chacun en version 1.0. Les images sont `(N, 28, 28, 1)` octets non
 * signés, les libellés `(N, 1)`. Le script lit le zip sans dépendance (annuaire
 * central, en-têtes locaux, `node:zlib` pour l'inflation), puis écrit sous
 * `demos/public/datasets/<id>/` le même agencement que QuickDraw :
 *
 * - `train-labels-idx1-ubyte.gz` et `test-labels-idx1-ubyte.gz` : les libellés
 *   complets de chaque fraction, classes dans l'ordre ;
 * - `train-c<C>-s<NN>.idx.gz` et `test-c<C>-s<NN>.idx.gz` : les images par
 *   classe et par tranche de 1 000, pour ne télécharger que la fraction choisie ;
 * - `train-gallery.idx.gz` et `train-gallery-labels-idx1-ubyte.gz` : dix images
 *   d'apprentissage par classe, prises au début de chaque classe ;
 * - `gallery.idx.gz` et `gallery-labels-idx1-ubyte.gz` : dix images de test par
 *   classe, prises au début de chaque classe de test ;
 * - `index.json` : le descripteur lu par le navigateur (écrit en dernier).
 *
 * Le script supprime ensuite chaque `.npz` traité de `demos/.cache/medmnist/`,
 * sauf avec `--keep-raw`. Si un `.npz` manque mais que `index.json` et toutes
 * les sorties qu'il décrit existent, le jeu est ignoré sans téléchargement.
 *
 * La division d'apprentissage fusionne `train` et `val` ; la division de test
 * est `test`. Chaque fraction est triée par classe (stable) : les morceaux sont
 * contigus et `start` est le décalage de la classe dans le fichier de libellés.
 *
 * Options :
 * - `--only <id>` : ne traite que le jeu donné ; répétable ou séparée par des
 *   virgules. La sortie ne décrit alors que les jeux sélectionnés.
 * - `--dry-run` : affiche le plan sans télécharger ni rien écrire.
 * - `--force` : retélécharge les `.npz` et réécrit les morceaux existants.
 * - `--keep-raw` : conserve les `.npz` téléchargés dans le cache.
 * - `--help` : affiche cette aide.
 *
 * Chaque écriture passe par un temporaire `<nom>.tmp` puis un `rename` : un
 * morceau n'est jamais laissé tronqué sous son nom final. Un morceau déjà
 * présent est conservé sauf `--force`. Le script sort avec un code non nul si
 * un téléchargement ou une vérification échoue. Les fonctions pures sont
 * exportées pour les tests ; la CLI ne s'exécute que si ce fichier est le point
 * d'entrée du processus.
 */

import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync as gzipRaw, inflateRawSync } from 'node:zlib';
import { writeCacheDirTag } from './cachedir-tag.mjs';
import { MEDMNIST_BASE_URL, MEDMNIST_FILES } from './dataset-sources.mjs';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(SCRIPT_DIRECTORY, '..');

/** Répertoire de sortie par défaut : `demos/public/datasets/`. */
export const PUBLIC_MEDMNIST_DIR = join(PROJECT_ROOT, 'public', 'datasets');

/** Répertoire de cache des `.npz` : `demos/.cache/medmnist/`. */
export const CACHE_MEDMNIST_DIR = join(PROJECT_ROOT, '.cache', 'medmnist');

/** Nombre d'images par morceau. */
export const SHARD_SIZE = 1000;

/** Nombre d'images de galerie par classe. */
export const GALLERY_PER_CLASS = 10;

/** Nombre d'octets d'une image 28 × 28. */
export const IMAGE_SIZE = 784;

/** Dimensions des images MedMNIST. */
export const IMAGE_ROWS = 28;
export const IMAGE_COLS = 28;

/** Membres `.npy` attendus dans chaque archive MedMNIST. */
export const MEDMNIST_NPY_KEYS = [
  'train_images',
  'val_images',
  'test_images',
  'train_labels',
  'val_labels',
  'test_labels',
];

/** Noms de classes, par jeu, dans l'ordre des indices de classe. */
export const MEDMNIST_CLASS_NAMES = {
  octmnist: ['choroidal neovascularization', 'diabetic macular edema', 'drusen', 'normal'],
  pneumoniamnist: ['normal', 'pneumonia'],
  breastmnist: ['malignant', 'normal benign'],
  tissuemnist: [
    'Collecting Duct, Connecting Tubule',
    'Distal Convoluted Tubule',
    'Glomerular endothelial cells',
    'Interstitial endothelial cells',
    'Leukocytes',
    'Podocytes',
    'Proximal Tubule Segments',
    'Thick Ascending Limb',
  ],
  organamnist: [
    'bladder',
    'femur-left',
    'femur-right',
    'heart',
    'kidney-left',
    'kidney-right',
    'liver',
    'lung-left',
    'lung-right',
    'pancreas',
    'spleen',
  ],
  organcmnist: [
    'bladder',
    'femur-left',
    'femur-right',
    'heart',
    'kidney-left',
    'kidney-right',
    'liver',
    'lung-left',
    'lung-right',
    'pancreas',
    'spleen',
  ],
  organsmnist: [
    'bladder',
    'femur-left',
    'femur-right',
    'heart',
    'kidney-left',
    'kidney-right',
    'liver',
    'lung-left',
    'lung-right',
    'pancreas',
    'spleen',
  ],
};

/**
 * Agent de téléchargement utilisé par défaut. Zenodo refuse l'agent de Node et
 * les agents de navigateur ; il accepte un agent `curl`.
 */
const DEFAULT_USER_AGENT = 'curl/8.5.0';

const NPY_MAGIC = '\x93NUMPY';
const IDX_IMAGE_MAGIC = 2051;
const IDX_LABEL_MAGIC = 2049;

const USAGE = `Usage: node scripts/convert-medmnist.mjs [options]

Convertit les sept archives .npz MedMNIST en morceaux IDX d'images gzip sous
public/datasets/<id>/.

Options:
  --only <id>   Ne traite que le jeu donné (répétable ou séparé par des virgules).
                Jeux connus : ${MEDMNIST_FILES.map((file) => file.dataset).join(', ')}.
  --dry-run     Affiche le plan sans télécharger ni écrire.
  --force       Retélécharge et réécrit même les fichiers déjà présents.
  --keep-raw    Conserve les .npz téléchargés dans le cache.
  --help        Affiche cette aide.`;

/** Convertit une entrée `ArrayBuffer` ou vue en `Uint8Array`. */
function toBytes(input) {
  if (input instanceof Uint8Array) return input;
  return new Uint8Array(input);
}

/** Entier 32 bits non signé, petit-boutiste, lu à `offset`. */
function readU32LE(bytes, offset) {
  return (
    (bytes[offset] |
      (bytes[offset + 1] << 8) |
      (bytes[offset + 2] << 16) |
      (bytes[offset + 3] << 24)) >>>
    0
  );
}

/** Entier 32 bits non signé, gros-boutiste, lu à `offset`. */
function readU32BE(bytes, offset) {
  return (
    (bytes[offset] * 0x1000000 +
      bytes[offset + 1] * 0x10000 +
      bytes[offset + 2] * 0x100 +
      bytes[offset + 3]) >>>
    0
  );
}

/** Entier 16 bits non signé, petit-boutiste, lu à `offset`. */
function readU16LE(bytes, offset) {
  return bytes[offset] | (bytes[offset + 1] << 8);
}

/** Valeur d'un champ de l'en-tête `.npy`, ou `undefined` si absent. */
function readHeaderField(headerText, pattern) {
  const match = headerText.match(pattern);
  return match === null ? undefined : match[1];
}

/**
 * Analyse l'annuaire central d'un zip et renvoie les entrées par nom.
 *
 * L'annuaire central finit par un enregistrement de fin (`0x06054b50`) ; son
 * en-tête donne le nombre d'entrées et l'octet où commence l'annuaire. Chaque
 * entrée donne le nom, la méthode (0 stocké, 8 déflé), les tailles compressée et
 * réelle, et l'octet de l'en-tête local. Les en-têtes locaux donnent le nom et
 * l'octet où commence la charge utile.
 *
 * Rejette un archive trop courte, un magic d'en-tête local inattendu, une
 * longueur de nom incohérente et une charge utile qui dépasse l'archive.
 */
export function parseZipEntries(buffer) {
  const bytes = toBytes(buffer);
  const END_SIGNATURE = 0x06054b50;
  let endOffset = -1;
  const searchStart = Math.max(0, bytes.length - 66_000);
  for (let offset = bytes.length - 22; offset >= searchStart; offset -= 1) {
    if (readU32LE(bytes, offset) === END_SIGNATURE) {
      endOffset = offset;
      break;
    }
  }
  if (endOffset < 0) {
    throw new RangeError('convert-medmnist: zip central directory not found');
  }
  const entryCount = readU16LE(bytes, endOffset + 10);
  let cursor = readU32LE(bytes, endOffset + 16);
  const entries = new Map();
  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > bytes.length || readU32LE(bytes, cursor) !== 0x02014b50) {
      throw new RangeError(`convert-medmnist: zip central entry ${index} is malformed`);
    }
    const method = readU16LE(bytes, cursor + 10);
    const compressedSize = readU32LE(bytes, cursor + 20);
    const nameLength = readU16LE(bytes, cursor + 28);
    const extraLength = readU16LE(bytes, cursor + 30);
    const commentLength = readU16LE(bytes, cursor + 32);
    const localHeaderOffset = readU32LE(bytes, cursor + 42);
    const nameBytes = bytes.subarray(cursor + 46, cursor + 46 + nameLength);
    const name = new TextDecoder().decode(nameBytes);
    if (
      localHeaderOffset + 30 > bytes.length ||
      readU32LE(bytes, localHeaderOffset) !== 0x04034b50
    ) {
      throw new RangeError(`convert-medmnist: zip local header for "${name}" is missing`);
    }
    const localNameLength = readU16LE(bytes, localHeaderOffset + 26);
    const localExtraLength = readU16LE(bytes, localHeaderOffset + 28);
    const dataOffset = localHeaderOffset + 30 + localNameLength + localExtraLength;
    const compressed = bytes.subarray(dataOffset, dataOffset + compressedSize);
    if (compressed.length !== compressedSize) {
      throw new RangeError(`convert-medmnist: zip entry "${name}" is truncated`);
    }
    let data;
    if (method === 0) {
      data = new Uint8Array(compressed);
    } else if (method === 8) {
      data = new Uint8Array(inflateRawSync(compressed));
    } else {
      throw new RangeError(
        `convert-medmnist: zip entry "${name}" uses unsupported method ${method}`,
      );
    }
    entries.set(name, data);
    cursor += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

/**
 * Valide et lit un membre `.npy` : en-tête puis charge utile normalisée.
 *
 * Accepte les versions 1.0, 2.0 et 3.0, un `descr` `u1` ou `i8` et un
 * `fortran_order` faux. Rejette un magic absent, un `fortran_order` vrai et une
 * charge utile tronquée. Renvoie `{ descr, shape, dataOffset, payload }` où
 * `payload` est `N` octets pour une forme `(N, 1)` et `N * 784` pour
 * `(N, 28, 28, 1)` ; un libellé `i8` est ramené à un octet non signé.
 */
export function parseNpy(buffer, fileName = '<memory>', key = '<member>') {
  const bytes = toBytes(buffer);
  if (bytes.length < 10) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" is truncated, expected at least 10 bytes, got ${bytes.length}`,
    );
  }
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
  if (magic !== NPY_MAGIC) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" is not a .npy file (bad magic ${JSON.stringify(magic)})`,
    );
  }
  const major = bytes[6];
  let headerLength;
  let dataOffset;
  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLength;
  } else if (major === 2 || major === 3) {
    headerLength = readU32LE(bytes, 8);
    dataOffset = 12 + headerLength;
  } else {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" uses unsupported .npy version ${major}.${bytes[7]}`,
    );
  }
  const headerText = new TextDecoder().decode(
    bytes.subarray(dataOffset - headerLength, dataOffset),
  );
  const descr = readHeaderField(headerText, /'descr':\s*'([^']*)'/);
  const fortran = readHeaderField(headerText, /'fortran_order':\s*(True|False)/);
  const shapeText = readHeaderField(headerText, /'shape':\s*\(([^)]*)\)/);
  if (descr === undefined || fortran === undefined || shapeText === undefined) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has an incomplete .npy header`,
    );
  }
  if (fortran === 'True') {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has fortran_order True, expected False`,
    );
  }
  const shape = shapeText
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part !== '')
    .map(Number);
  if (shape.some((part) => !Number.isInteger(part) || part < 0)) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has an unsupported .npy shape (${shapeText})`,
    );
  }
  const itemBytes = descr.endsWith('u1') ? 1 : 8;
  let items = 1;
  for (const dimension of shape) items *= dimension;
  const byteLength = items * itemBytes;
  const available = bytes.length - dataOffset;
  if (available < byteLength) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" is truncated, expected ${byteLength} payload bytes, got ${available}`,
    );
  }
  const raw = bytes.subarray(dataOffset, dataOffset + byteLength);
  let payload;
  if (itemBytes === 1) {
    payload = new Uint8Array(raw);
  } else {
    payload = new Uint8Array(items);
    for (let index = 0; index < items; index += 1) {
      payload[index] = readU32LE(raw, index * 8);
    }
  }
  return { descr, shape, dataOffset, payload };
}

/**
 * Valide et lit un membre `.npy` d'images MedMNIST : `N` images de 784 octets.
 *
 * Le `descr` doit être `<u1` ou `|u1` (un octet non signé) et la forme
 * `(N, 28, 28)` ou `(N, 28, 28, 1)`.
 */
export function parseNpyImages(buffer, fileName, key) {
  const parsed = parseNpy(buffer, fileName, key);
  const { descr, shape } = parsed;
  if (descr !== '<u1' && descr !== '|u1') {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has dtype "${descr}", expected "<u1" or "|u1"`,
    );
  }
  // Les archives MedMNIST écrivent `(N, 28, 28)` ; d'autres écrivent
  // `(N, 28, 28, 1)`. Les deux donnent 784 octets par image.
  const isImageShape =
    (shape.length === 3 && shape[1] === IMAGE_ROWS && shape[2] === IMAGE_COLS) ||
    (shape.length === 4 && shape[1] === IMAGE_ROWS && shape[2] === IMAGE_COLS && shape[3] === 1);
  if (!isImageShape) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has shape (${shape.join(', ')}), expected (N, ${IMAGE_ROWS}, ${IMAGE_COLS}) or (N, ${IMAGE_ROWS}, ${IMAGE_COLS}, 1)`,
    );
  }
  return parsed;
}

/**
 * Valide et lit un membre `.npy` de libellés MedMNIST : `N` entiers.
 *
 * Le `descr` doit être `<u1`, `|u1`, `<i8` ou `|i8` et la forme `(N, 1)`.
 */
export function parseNpyLabels(buffer, fileName, key) {
  const parsed = parseNpy(buffer, fileName, key);
  const { descr, shape } = parsed;
  const allowed = ['<u1', '|u1', '<i8', '|i8'];
  if (!allowed.includes(descr)) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has dtype "${descr}", expected one of ${allowed.join(', ')}`,
    );
  }
  if (shape.length !== 2 || shape[1] !== 1) {
    throw new RangeError(
      `convert-medmnist: ${fileName}: member "${key}" has shape (${shape.join(', ')}), expected (N, 1)`,
    );
  }
  return parsed;
}

/** Fusionne les divisions `train` et `val` d'une archive. */
export function mergeTrainVal(train, val) {
  const trainLabels = toBytes(train.labels);
  const valLabels = toBytes(val.labels);
  const images = new Uint8Array(train.images.length + val.images.length);
  images.set(train.images, 0);
  images.set(val.images, train.images.length);
  const labels = new Uint8Array(trainLabels.length + valLabels.length);
  labels.set(trainLabels, 0);
  labels.set(valLabels, trainLabels.length);
  return {
    images,
    labels,
    trainCount: trainLabels.length,
    valCount: valLabels.length,
    count: trainLabels.length + valLabels.length,
  };
}

/**
 * Réordonne des images et des libellés par classe, de façon stable.
 *
 * Les images d'une classe restent dans leur ordre d'origine ; les classes sont
 * groupées dans l'ordre croissant de libellé. Renvoie `{ images, labels, counts }`
 * où `counts` donne l'effectif de chaque classe.
 */
export function groupByClass(images, labels) {
  const bytes = toBytes(images);
  const labelBytes = toBytes(labels);
  if (bytes.length !== labelBytes.length * IMAGE_SIZE) {
    throw new RangeError(
      `convert-medmnist: ${bytes.length} image bytes for ${labelBytes.length} labels, expected ${labelBytes.length * IMAGE_SIZE}`,
    );
  }
  let maxLabel = 0;
  for (const label of labelBytes) {
    if (label > maxLabel) maxLabel = label;
  }
  const classCount = maxLabel + 1;
  const counts = new Array(classCount).fill(0);
  for (const label of labelBytes) counts[label] += 1;

  // `ordered` liste les indices d'origine dans l'ordre des classes : chaque
  // classe occupe un intervalle contigu, rempli dans l'ordre d'origine.
  const classStart = new Array(classCount).fill(0);
  for (let classIndex = 1; classIndex < classCount; classIndex += 1) {
    classStart[classIndex] = classStart[classIndex - 1] + counts[classIndex - 1];
  }
  const cursor = classStart.slice();
  const ordered = new Array(labelBytes.length);
  for (let index = 0; index < labelBytes.length; index += 1) {
    const label = labelBytes[index];
    ordered[cursor[label]] = index;
    cursor[label] += 1;
  }

  const orderedImages = new Uint8Array(bytes.length);
  const orderedLabels = new Uint8Array(labelBytes.length);
  let imageOffset = 0;
  let labelOffset = 0;
  for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
    const count = counts[classIndex];
    for (let position = 0; position < count; position += 1) {
      const source = ordered[labelOffset + position] * IMAGE_SIZE;
      orderedImages.set(bytes.subarray(source, source + IMAGE_SIZE), imageOffset);
      orderedLabels[labelOffset + position] = classIndex;
      imageOffset += IMAGE_SIZE;
    }
    labelOffset += count;
  }
  return { images: orderedImages, labels: orderedLabels, counts };
}

/**
 * Découpe `count` images en tranches de `shardSize` : `[{ start, count }, …]`.
 *
 * `start` est l'indice de la première image de la tranche, `count` son nombre
 * d'images (la dernière tranche peut être plus courte).
 */
export function shardRanges(count, shardSize) {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(
      `convert-medmnist: shardRanges expects a non-negative integer, got ${count}`,
    );
  }
  if (!Number.isInteger(shardSize) || shardSize <= 0) {
    throw new RangeError(
      `convert-medmnist: shardRanges expects a positive shard size, got ${shardSize}`,
    );
  }
  const ranges = [];
  for (let start = 0; start < count; start += shardSize) {
    ranges.push({ start, count: Math.min(shardSize, count - start) });
  }
  return ranges;
}

/** Encode des images en en-tête IDX d'images : magic `2051`, puis `count`, `rows`, `cols`. */
export function writeIdxImagesBytes(images) {
  const bytes = toBytes(images);
  if (bytes.length % IMAGE_SIZE !== 0) {
    throw new RangeError(
      `convert-medmnist: image payload length ${bytes.length} is not a multiple of ${IMAGE_SIZE}`,
    );
  }
  const count = bytes.length / IMAGE_SIZE;
  const out = new Uint8Array(16 + bytes.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, IDX_IMAGE_MAGIC);
  view.setUint32(4, count);
  view.setUint32(8, IMAGE_ROWS);
  view.setUint32(12, IMAGE_COLS);
  out.set(bytes, 16);
  return out;
}

/** Encode des libellés en en-tête IDX : magic `2049`, puis `count` et les octets. */
export function writeIdxLabelsBytes(labels) {
  const bytes = toBytes(labels);
  const out = new Uint8Array(8 + bytes.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, IDX_LABEL_MAGIC);
  view.setUint32(4, bytes.length);
  out.set(bytes, 8);
  return out;
}

/** Compresse en gzip. Node écrit un en-tête déterministe (mtime nul). */
export function gzipBytes(bytes) {
  return new Uint8Array(gzipRaw(toBytes(bytes)));
}

/**
 * Construit le descripteur `index.json` d'un jeu MedMNIST.
 *
 * `trainCounts` et `testCounts` donnent l'effectif par classe, dans l'ordre de
 * `classNames`. Chaque morceau est décrit par son fichier, sa classe, son
 * `start` (décalage de la classe dans le fichier de libellés) et son `count`.
 * `galleries` décrit la galerie d'apprentissage et celle de test.
 */
export function buildShardedIndex({
  dataset,
  classNames,
  trainCounts,
  testCounts,
  shardSize,
  galleryPerClass = GALLERY_PER_CLASS,
  trainLabelsFile = 'train-labels-idx1-ubyte.gz',
  testLabelsFile = 'test-labels-idx1-ubyte.gz',
  trainGalleryImagesFile = 'train-gallery.idx.gz',
  trainGalleryLabelsFile = 'train-gallery-labels-idx1-ubyte.gz',
  testGalleryImagesFile = 'gallery.idx.gz',
  testGalleryLabelsFile = 'gallery-labels-idx1-ubyte.gz',
}) {
  const buildShards = (split, counts) => {
    const shards = [];
    let classStart = 0;
    counts.forEach((count, classIndex) => {
      let shardIndex = 0;
      for (const range of shardRanges(count, shardSize)) {
        shards.push({
          file: `${split}-c${classIndex}-s${String(shardIndex).padStart(2, '0')}.idx.gz`,
          classIndex,
          start: classStart + range.start,
          count: range.count,
        });
        shardIndex += 1;
      }
      classStart += count;
    });
    return shards;
  };
  const total = (counts) => counts.reduce((sum, count) => sum + count, 0);
  return {
    version: 2,
    dataset,
    classNames: [...classNames],
    shardSize,
    splits: {
      train: {
        count: total(trainCounts),
        labelsFile: trainLabelsFile,
        shards: buildShards('train', trainCounts),
      },
      test: {
        count: total(testCounts),
        labelsFile: testLabelsFile,
        shards: buildShards('test', testCounts),
      },
    },
    galleries: {
      train: {
        imagesFile: trainGalleryImagesFile,
        labelsFile: trainGalleryLabelsFile,
        perClass: galleryPerClass,
      },
      test: {
        imagesFile: testGalleryImagesFile,
        labelsFile: testGalleryLabelsFile,
        perClass: galleryPerClass,
      },
    },
  };
}

/** Analyse les arguments de la ligne de commande. */
export function parseArgs(argv, files = MEDMNIST_FILES) {
  const options = { only: [], dryRun: false, force: false, keepRaw: false, help: false };
  const ids = new Set();

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    switch (argument) {
      case '--help':
      case '-h':
        options.help = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--force':
        options.force = true;
        break;
      case '--keep-raw':
        options.keepRaw = true;
        break;
      case '--only': {
        const value = argv[index + 1];
        if (value === undefined) {
          throw new RangeError('convert-medmnist: --only requires a dataset id');
        }
        index += 1;
        for (const id of value.split(',')) {
          if (id !== '') ids.add(id);
        }
        break;
      }
      default:
        throw new RangeError(
          `convert-medmnist: unknown argument "${argument}", see --help for usage`,
        );
    }
  }

  const known = files.map((file) => file.dataset);
  for (const id of ids) {
    if (!known.includes(id)) {
      throw new RangeError(
        `convert-medmnist: unknown dataset "${id}", expected one of ${known.join(', ')}`,
      );
    }
  }
  options.only = [...ids];
  return options;
}

/** Décrit le plan d'exécution sans rien écrire. */
export function describePlan(selected, targetRoot = PUBLIC_MEDMNIST_DIR) {
  const lines = [];
  for (const file of selected) {
    lines.push(`# ${file.dataset} (dry-run)`);
    lines.push(`  ${file.name} <- ${MEDMNIST_BASE_URL}${file.name}?download=1`);
    lines.push(`  -> ${join(targetRoot, file.dataset)}/ (${file.size} bytes, MD5 ${file.md5})`);
  }
  return lines.join('\n');
}

/** Empreinte MD5 hexadécimale d'un buffer. */
function md5(bytes) {
  return createHash('md5').update(bytes).digest('hex');
}

/** URL d'un fichier MedMNIST, avec le paramètre `download=1` de Zenodo. */
function fileUrl(file) {
  return `${MEDMNIST_BASE_URL}${file.name}?download=1`;
}

/** Télécharge avec un en-tête `User-Agent` : Zenodo bloque l'agent de Node. */
async function fetchWithUserAgent(fetchImpl, url) {
  try {
    return await fetchImpl(url, { headers: { 'User-Agent': DEFAULT_USER_AGENT } });
  } catch {
    return await fetchImpl(url);
  }
}

/**
 * Télécharge un fichier `.npz` si le cache est absent ou invalide, puis renvoie
 * ses octets. Un fichier déjà présent avec la bonne taille et le bon MD5 est
 * conservé. L'écriture passe par un temporaire `.tmp` supprimé en cas d'échec.
 */
async function fetchRawFile(file, targetPath, { fetchImpl, force }) {
  if (!force) {
    try {
      const existing = await readFile(targetPath);
      if (existing.length === file.size && md5(existing) === file.md5) {
        console.log(`  ${file.dataset}/${file.name}: cached (${file.size} bytes, MD5 ${file.md5})`);
        return existing;
      }
    } catch {
      // pas d'entrée de cache valide : on télécharge
    }
  }

  const url = fileUrl(file);
  const tempPath = `${targetPath}.tmp`;
  try {
    const response = await fetchWithUserAgent(fetchImpl, url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.length !== file.size) {
      throw new Error(`size mismatch: expected ${file.size} bytes, received ${bytes.length}`);
    }
    const digest = md5(bytes);
    if (digest !== file.md5) {
      throw new Error(`MD5 mismatch: expected ${file.md5}, received ${digest}`);
    }
    await writeFile(tempPath, bytes);
    await rename(tempPath, targetPath);
    console.log(`  ${file.dataset}/${file.name}: downloaded (${file.size} bytes, MD5 ${file.md5})`);
    return bytes;
  } catch (error) {
    await unlink(tempPath).catch(() => {});
    throw new Error(`could not fetch ${url}: ${error.message}`);
  }
}

/**
 * Vérifie que `index.json` décrit la version 2 et que chaque fichier qu'il
 * référence existe sous `outputDir`.
 */
async function outputsAreComplete(indexPath, outputDir) {
  let index;
  try {
    index = JSON.parse(await readFile(indexPath, 'utf8'));
  } catch {
    return false;
  }
  if (index?.version !== 2) return false;
  const referenced = [];
  for (const split of ['train', 'test']) {
    const part = index.splits?.[split];
    if (!part || typeof part.labelsFile !== 'string' || !Array.isArray(part.shards)) return false;
    referenced.push(part.labelsFile);
    for (const shard of part.shards) {
      if (typeof shard?.file !== 'string') return false;
      referenced.push(shard.file);
    }
    const gallery = index.galleries?.[split];
    if (
      !gallery ||
      typeof gallery.imagesFile !== 'string' ||
      typeof gallery.labelsFile !== 'string'
    ) {
      return false;
    }
    referenced.push(gallery.imagesFile, gallery.labelsFile);
  }
  return referenced.every((name) => existsSync(join(outputDir, name)));
}

/** Lit les six membres d'une archive `.npz` et renvoie images et libellés. */
function readArchive(buffer, file) {
  const fileName = file.name;
  const entries = parseZipEntries(buffer);
  const member = (key) => {
    const memberName = `${key}.npy`;
    const data = entries.get(memberName);
    if (data === undefined) {
      throw new RangeError(`convert-medmnist: ${fileName}: zip member "${memberName}" is missing`);
    }
    return data;
  };

  const train = {
    images: parseNpyImages(member('train_images'), fileName, 'train_images').payload,
    labels: parseNpyLabels(member('train_labels'), fileName, 'train_labels').payload,
  };
  const val = {
    images: parseNpyImages(member('val_images'), fileName, 'val_images').payload,
    labels: parseNpyLabels(member('val_labels'), fileName, 'val_labels').payload,
  };
  const test = {
    images: parseNpyImages(member('test_images'), fileName, 'test_images').payload,
    labels: parseNpyLabels(member('test_labels'), fileName, 'test_labels').payload,
  };

  const merged = mergeTrainVal(train, val);
  const trainGrouped = groupByClass(merged.images, merged.labels);
  const testGrouped = groupByClass(test.images, test.labels);

  const classCount = MEDMNIST_CLASS_NAMES[file.dataset].length;
  for (const group of [trainGrouped, testGrouped]) {
    for (const label of group.labels) {
      if (label >= classCount) {
        throw new RangeError(
          `convert-medmnist: ${fileName}: label ${label} is outside the ${classCount} expected classes`,
        );
      }
    }
    // Une classe sans exemple est complétée par un effectif nul : l'index doit
    // rester aligné sur `classNames`.
    while (group.counts.length < classCount) group.counts.push(0);
    if (group.counts.length > classCount) {
      throw new RangeError(
        `convert-medmnist: ${fileName}: more than ${classCount} classes are present`,
      );
    }
  }

  return {
    train: {
      images: trainGrouped.images,
      labels: trainGrouped.labels,
      counts: trainGrouped.counts,
    },
    test: { images: testGrouped.images, labels: testGrouped.labels, counts: testGrouped.counts },
    trainCount: trainGrouped.labels.length,
    testCount: testGrouped.labels.length,
    mergedCount: merged.count,
    trainRawCount: merged.trainCount,
    valRawCount: merged.valCount,
  };
}

/** Copie `count` images `offset … offset + count` hors du tableau de classe. */
function collectImages(images, offset, count) {
  const start = offset * IMAGE_SIZE;
  return new Uint8Array(images.subarray(start, start + count * IMAGE_SIZE));
}

/** Écrit un fichier IDX gzip et renvoie sa taille en octets. */
async function writeGzipIdx(outputDir, name, idxBytes) {
  const path = join(outputDir, name);
  const compressed = gzipBytes(idxBytes);
  await writeFile(`${path}.tmp`, compressed);
  await rename(`${path}.tmp`, path);
  return compressed.length;
}

/**
 * Écrit les morceaux IDX d'une fraction, classe par classe.
 *
 * `counts` donne l'effectif de chaque classe, dans l'ordre des indices ; les
 * classes sont contiguës dans `images`. Un morceau déjà présent est conservé
 * sauf `force`. Renvoie le nombre total d'octets gzip écrits ou déjà présents,
 * et le nombre de morceaux.
 */
async function writeSplitShards({ split, counts, images, outputDir, force }) {
  let totalBytes = 0;
  let shardCount = 0;
  let classOffset = 0;
  for (let classIndex = 0; classIndex < counts.length; classIndex += 1) {
    const count = counts[classIndex];
    let shardIndex = 0;
    for (const range of shardRanges(count, SHARD_SIZE)) {
      const name = `${split}-c${classIndex}-s${String(shardIndex).padStart(2, '0')}.idx.gz`;
      const path = join(outputDir, name);
      if (!force && existsSync(path)) {
        // morceau existant conservé
      } else {
        const payload = collectImages(images, classOffset + range.start, range.count);
        const compressed = gzipBytes(writeIdxImagesBytes(payload));
        await writeFile(`${path}.tmp`, compressed);
        await rename(`${path}.tmp`, path);
      }
      const size = (await stat(path)).size;
      totalBytes += size;
      shardCount += 1;
      shardIndex += 1;
      console.log(`    ${name}: ${size} bytes`);
    }
    classOffset += count;
  }
  return { totalBytes, shardCount };
}

/**
 * Point d'entrée de la CLI. Renvoie le code de sortie du processus.
 *
 * `sources`, `targetRoot`, `cacheRoot` et `fetchImpl` peuvent être injectés pour
 * les tests ; sans injection, les défauts actuels sont conservés.
 */
export async function main(argv = process.argv.slice(2), options = {}) {
  const files = options.sources ?? MEDMNIST_FILES;
  const targetRoot = options.targetRoot ?? PUBLIC_MEDMNIST_DIR;
  const cacheRoot = options.cacheRoot ?? CACHE_MEDMNIST_DIR;
  const fetchImpl = options.fetchImpl ?? fetch;

  let parsed;
  try {
    parsed = parseArgs(argv, files);
  } catch (error) {
    console.error(error.message);
    console.error(USAGE);
    return 1;
  }
  if (parsed.help) {
    console.log(USAGE);
    return 0;
  }

  const selected =
    parsed.only.length > 0 ? files.filter((file) => parsed.only.includes(file.dataset)) : files;

  if (parsed.dryRun) {
    console.log(describePlan(selected, targetRoot));
    return 0;
  }

  await mkdir(cacheRoot, { recursive: true });
  let totalBytes = 0;
  const reports = [];

  try {
    for (const file of selected) {
      const outputDir = join(targetRoot, file.dataset);
      const rawPath = join(cacheRoot, file.name);
      const indexPath = join(outputDir, 'index.json');
      await writeCacheDirTag(outputDir);
      if (
        !parsed.force &&
        !existsSync(rawPath) &&
        (await outputsAreComplete(indexPath, outputDir))
      ) {
        console.log(`  ${file.dataset}: raw absent, outputs complete (skipped)`);
        continue;
      }
      await mkdir(outputDir, { recursive: true });
      const bytes = await fetchRawFile(file, rawPath, { fetchImpl, force: parsed.force });
      const archive = readArchive(bytes, file);
      const classNames = MEDMNIST_CLASS_NAMES[file.dataset];
      for (const [classIndex, count] of archive.test.counts.entries()) {
        if (count < GALLERY_PER_CLASS) {
          throw new RangeError(
            `convert-medmnist: ${file.name}: class ${classIndex} has ${count} test images, below the ${GALLERY_PER_CLASS} needed for the test gallery`,
          );
        }
      }
      const trainResult = await writeSplitShards({
        split: 'train',
        counts: archive.train.counts,
        images: archive.train.images,
        outputDir,
        force: parsed.force,
      });
      const testResult = await writeSplitShards({
        split: 'test',
        counts: archive.test.counts,
        images: archive.test.images,
        outputDir,
        force: parsed.force,
      });
      totalBytes += trainResult.totalBytes + testResult.totalBytes;

      // Une classe absente de la division d'apprentissage ne peut pas remplir
      // une galerie de dix images : elle est omise au lieu de lire un décalage
      // indéfini. Les classes suivantes gardent leur décalage réel.
      const trainGalleryClasses = archive.train.counts.filter(
        (count) => count >= GALLERY_PER_CLASS,
      ).length;
      const trainGalleryImages = new Uint8Array(
        GALLERY_PER_CLASS * trainGalleryClasses * IMAGE_SIZE,
      );
      const trainGalleryLabels = new Uint8Array(GALLERY_PER_CLASS * trainGalleryClasses);
      let trainGalleryOffset = 0;
      let trainGalleryLabelOffset = 0;
      archive.train.counts.forEach((count, classIndex) => {
        if (count >= GALLERY_PER_CLASS) {
          trainGalleryImages.set(
            archive.train.images.subarray(
              trainGalleryLabelOffset * IMAGE_SIZE,
              (trainGalleryLabelOffset + GALLERY_PER_CLASS) * IMAGE_SIZE,
            ),
            trainGalleryOffset * IMAGE_SIZE,
          );
          trainGalleryLabels.fill(
            classIndex,
            trainGalleryOffset,
            trainGalleryOffset + GALLERY_PER_CLASS,
          );
          trainGalleryOffset += GALLERY_PER_CLASS;
        }
        trainGalleryLabelOffset += count;
      });

      const galleryImages = new Uint8Array(GALLERY_PER_CLASS * classNames.length * IMAGE_SIZE);
      const galleryLabels = new Uint8Array(GALLERY_PER_CLASS * classNames.length);
      let galleryOffset = 0;
      let galleryLabelOffset = 0;
      archive.test.counts.forEach((count, classIndex) => {
        galleryImages.set(
          archive.test.images.subarray(
            galleryLabelOffset * IMAGE_SIZE,
            (galleryLabelOffset + GALLERY_PER_CLASS) * IMAGE_SIZE,
          ),
          galleryOffset * IMAGE_SIZE,
        );
        galleryLabels.fill(classIndex, galleryOffset, galleryOffset + GALLERY_PER_CLASS);
        galleryOffset += GALLERY_PER_CLASS;
        galleryLabelOffset += count;
      });

      totalBytes += await writeGzipIdx(
        outputDir,
        'train-labels-idx1-ubyte.gz',
        writeIdxLabelsBytes(archive.train.labels),
      );
      totalBytes += await writeGzipIdx(
        outputDir,
        'test-labels-idx1-ubyte.gz',
        writeIdxLabelsBytes(archive.test.labels),
      );
      totalBytes += await writeGzipIdx(
        outputDir,
        'train-gallery.idx.gz',
        writeIdxImagesBytes(trainGalleryImages),
      );
      totalBytes += await writeGzipIdx(
        outputDir,
        'train-gallery-labels-idx1-ubyte.gz',
        writeIdxLabelsBytes(trainGalleryLabels),
      );
      totalBytes += await writeGzipIdx(
        outputDir,
        'gallery.idx.gz',
        writeIdxImagesBytes(galleryImages),
      );
      totalBytes += await writeGzipIdx(
        outputDir,
        'gallery-labels-idx1-ubyte.gz',
        writeIdxLabelsBytes(galleryLabels),
      );

      const index = buildShardedIndex({
        dataset: file.dataset,
        classNames,
        trainCounts: archive.train.counts,
        testCounts: archive.test.counts,
        shardSize: SHARD_SIZE,
        galleryPerClass: GALLERY_PER_CLASS,
      });
      const indexText = `${JSON.stringify(index, null, 2)}\n`;
      await writeFile(`${indexPath}.tmp`, indexText);
      await rename(`${indexPath}.tmp`, indexPath);
      totalBytes += Buffer.byteLength(indexText, 'utf8');

      if (parsed.keepRaw) {
        console.log(`  ${file.name}: kept in cache (--keep-raw)`);
      } else {
        const rawBytes = (await stat(rawPath)).size;
        await unlink(rawPath);
        console.log(`  ${file.name}: deleted from cache (${rawBytes} bytes)`);
      }

      reports.push({
        dataset: file.dataset,
        trainCount: archive.trainCount,
        testCount: archive.testCount,
        mergedCount: archive.mergedCount,
        trainRawCount: archive.trainRawCount,
        valRawCount: archive.valRawCount,
        trainShards: trainResult.shardCount,
        testShards: testResult.shardCount,
        bytes: trainResult.totalBytes + testResult.totalBytes,
      });
      console.log(
        `  ${file.dataset}: ${archive.trainCount} train (train ${archive.trainRawCount} + val ${archive.valRawCount}), ${archive.testCount} test, ${trainResult.shardCount}+${testResult.shardCount} shards`,
      );
    }
  } catch (error) {
    console.error(`convert-medmnist: FAILED - ${error.message}`);
    return 1;
  }

  console.log('convert-medmnist: conversion complete');
  for (const report of reports) {
    console.log(
      `  ${report.dataset}: ${report.trainCount} train, ${report.testCount} test, ${report.trainShards}+${report.testShards} shards`,
    );
  }
  console.log(`  total output: ${totalBytes} bytes`);
  return 0;
}

/** Lit le nombre d'images déclaré par l'en-tête IDX d'un fragment. */
export function readIdxImageCount(bytes) {
  const data = toBytes(bytes);
  if (data.length < 16) {
    throw new RangeError('convert-medmnist: shard is truncated, expected at least 16 bytes');
  }
  if (readU32BE(data, 0) !== IDX_IMAGE_MAGIC) {
    throw new RangeError('convert-medmnist: shard does not start with the IDX image magic 2051');
  }
  return readU32BE(data, 4);
}

/** Lit le nombre de libellés déclaré par l'en-tête IDX d'un fichier de libellés. */
export function readIdxLabelCount(bytes) {
  const data = toBytes(bytes);
  if (data.length < 8) {
    throw new RangeError('convert-medmnist: label file is truncated, expected at least 8 bytes');
  }
  if (readU32BE(data, 0) !== IDX_LABEL_MAGIC) {
    throw new RangeError(
      'convert-medmnist: label file does not start with the IDX label magic 2049',
    );
  }
  return readU32BE(data, 4);
}

const entryPoint = process.argv[1] === undefined ? '' : pathToFileURL(process.argv[1]).href;
const isEntryPoint = import.meta.url === entryPoint;
if (isEntryPoint) {
  process.exitCode = await main();
}
