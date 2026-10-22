#!/usr/bin/env node
/**
 * Convertit quinze fichiers `.npy` QuickDraw en morceaux IDX d'images gzip.
 *
 * Chaque fichier `.npy` contient `N` images 28 × 28 (784 octets) non signés,
 * rangées en ligne-major. Le script télécharge les quinze classes dans
 * `demos/.cache/quickdraw/`, les vérifie (taille, MD5, en-tête `.npy`), les
 * répartit de façon déterministe en 80 % d'apprentissage et 20 % de test, puis
 * écrit sous `demos/public/datasets/quickdraw/` :
 *
 * - `train-labels-idx1-ubyte.gz` et `test-labels-idx1-ubyte.gz` : les libellés
 *   complets de chaque partition, classes dans l'ordre ;
 * - `train-c<C>-s<NN>.idx.gz` et `test-c<C>-s<NN>.idx.gz` : les images par
 *   classe et par tranche de 1 000, pour ne télécharger que la fraction choisie ;
 * - `train-gallery.idx.gz` et `train-gallery-labels-idx1-ubyte.gz` : dix images
 *   d'apprentissage par classe, prises au début de chaque classe ;
 * - `gallery.idx.gz` et `gallery-labels-idx1-ubyte.gz` : dix images de test par
 *   classe, prises au début de chaque classe de test ;
 * - `index.json` : le descripteur lu par le navigateur (écrit en dernier).
 *
 * Le script supprime ensuite chaque `.npy` traité de `demos/.cache/quickdraw/`,
 * sauf avec `--keep-raw`. Si un `.npy` manque mais que `index.json` et toutes
 * les sorties qu'il décrit existent, la conversion est ignorée sans
 * téléchargement.
 *
 * La répartition est déterministe : les indices de chaque classe sont mélangés
 * avec la graine `SPLIT_SEED` (42) par un mélange de Fisher-Yates alimenté par
 * un générateur `mulberry32`, puis les 80 % premiers indices forment
 * l'apprentissage et les 20 % derniers le test. Le même script sur les mêmes
 * fichiers produit donc exactement les mêmes partitions.
 *
 * Options :
 * - `--only <classe>` : ne traite que la classe donnée ; répétable ou séparée
 *   par des virgules. La sortie (libellés, galeries, `index.json`) ne décrit
 *   alors que les classes sélectionnées.
 * - `--dry-run` : affiche le plan sans télécharger ni rien écrire.
 * - `--force` : retélécharge les `.npy` et réécrit les morceaux existants.
 * - `--keep-raw` : conserve les `.npy` téléchargés dans le cache.
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
import { gzipSync as gzipSyncZlib } from 'node:zlib';
import { writeCacheDirTag } from './cachedir-tag.mjs';
import { QUICKDRAW_BASE_URL, QUICKDRAW_FILES } from './dataset-sources.mjs';

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(SCRIPT_DIRECTORY, '..');

/** Répertoire de sortie par défaut : `demos/public/datasets/quickdraw/`. */
export const PUBLIC_QUICKDRAW_DIR = join(PROJECT_ROOT, 'public', 'datasets', 'quickdraw');

/** Répertoire de cache des `.npy` téléchargés : `demos/.cache/quickdraw/`. */
export const CACHE_QUICKDRAW_DIR = join(PROJECT_ROOT, '.cache', 'quickdraw');

/** Graine du mélange 80/20 : la même graine donne toujours la même partition. */
export const SPLIT_SEED = 42;

/** Fraction d'apprentissage, le reste forme le test. */
export const TRAIN_FRACTION = 0.8;

/** Nombre d'images par morceau. */
export const SHARD_SIZE = 1000;

/** Nombre d'images de galerie par classe. */
export const GALLERY_PER_CLASS = 10;

/** Nombre d'octets d'une image 28 × 28. */
export const IMAGE_SIZE = 784;

/** Dimensions des images QuickDraw. */
export const IMAGE_ROWS = 28;
export const IMAGE_COLS = 28;

/**
 * Noms de classes, dans l'ordre des fichiers `QUICKDRAW_FILES` (livre, lune,
 * chat, parapluie, gâteau, chien, poisson, fleur, étoile, maison, arbre,
 * soleil, vélo, pizza, pomme) : l'indice de classe est l'indice dans ce tableau.
 */
export const QUICKDRAW_CLASS_NAMES = [
  'livre',
  'lune',
  'chat',
  'parapluie',
  'gâteau',
  'chien',
  'poisson',
  'fleur',
  'étoile',
  'maison',
  'arbre',
  'soleil',
  'vélo',
  'pizza',
  'pomme',
];

const NPY_MAGIC = '\x93NUMPY';
const IDX_IMAGE_MAGIC = 2051;
const IDX_LABEL_MAGIC = 2049;

const USAGE = `Usage: node scripts/convert-quickdraw.mjs [options]

Convertit les quinze fichiers .npy QuickDraw en morceaux IDX d'images gzip sous
public/datasets/quickdraw/.

Options:
  --only <class>  Ne traite que la classe donnée (répétable ou séparée par des
                  virgules). Classes : ${QUICKDRAW_FILES.map((file) => file.name.replace(/\.npy$/, '')).join(', ')}.
  --dry-run       Affiche les URL sans télécharger ni écrire.
  --force         Retélécharge et réécrit même les fichiers déjà présents.
  --keep-raw      Conserve les .npy téléchargés dans le cache.
  --help          Affiche cette aide.`;

/** Normalise un identifiant de classe : `book.npy` et `book` donnent `book`. */
function classId(fileName) {
  return fileName.replace(/\.npy$/, '');
}

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

/** Valeur d'un champ de l'en-tête `.npy`, ou `undefined` si absent. */
function readHeaderField(headerText, pattern) {
  const match = headerText.match(pattern);
  return match === null ? undefined : match[1];
}

/**
 * Valide et lit l'en-tête d'un fichier `.npy` non compressé.
 *
 * Accepte les versions 1.0, 2.0 et 3.0 de NumPy. Rejette un magic absent, un
 * `descr` qui n'est pas `<u1` ou `|u1` (les deux codent un octet non signé), un
 * `fortran_order` vrai et une forme qui n'est pas `(N, 784)`. Renvoie
 * `{ descr, fortranOrder, shape, dataOffset }` où `shape` est `[N, 784]` et
 * `dataOffset` l'octet où commence la charge utile.
 */
export function parseNpyHeader(buffer) {
  const bytes = toBytes(buffer);
  if (bytes.length < 10) {
    throw new RangeError(
      `convert-quickdraw: truncated .npy header, expected at least 10 bytes, got ${bytes.length}`,
    );
  }
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
  if (magic !== NPY_MAGIC) {
    throw new RangeError(`convert-quickdraw: not a .npy file (bad magic ${JSON.stringify(magic)})`);
  }
  const major = bytes[6];
  const minor = bytes[7];
  let headerLength;
  let dataOffset;
  if (major === 1) {
    headerLength = bytes[8] | (bytes[9] << 8);
    dataOffset = 10 + headerLength;
  } else if (major === 2 || major === 3) {
    headerLength = readU32LE(bytes, 8);
    dataOffset = 12 + headerLength;
  } else {
    throw new RangeError(`convert-quickdraw: unsupported .npy version ${major}.${minor}`);
  }
  const headerText = new TextDecoder().decode(
    bytes.subarray(dataOffset - headerLength, dataOffset),
  );

  const descr = readHeaderField(headerText, /'descr':\s*'([^']*)'/);
  const fortran = readHeaderField(headerText, /'fortran_order':\s*(True|False)/);
  const shapeText = readHeaderField(headerText, /'shape':\s*\(([^)]*)\)/);
  if (descr === undefined) {
    throw new RangeError('convert-quickdraw: .npy header has no descr field');
  }
  if (fortran === undefined) {
    throw new RangeError('convert-quickdraw: .npy header has no fortran_order field');
  }
  if (shapeText === undefined) {
    throw new RangeError('convert-quickdraw: .npy header has no shape field');
  }
  if (descr !== '<u1' && descr !== '|u1') {
    throw new RangeError(
      `convert-quickdraw: unsupported .npy dtype "${descr}", expected "<u1" or "|u1"`,
    );
  }
  if (fortran === 'True') {
    throw new RangeError('convert-quickdraw: .npy fortran_order is True, expected False');
  }
  const shapeParts = shapeText
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part !== '');
  const rows = shapeParts.length === 2 ? Number(shapeParts[1]) : Number.NaN;
  if (shapeParts.length !== 2 || !Number.isInteger(Number(shapeParts[0])) || rows !== IMAGE_SIZE) {
    throw new RangeError(
      `convert-quickdraw: unsupported .npy shape (${shapeParts.join(', ')}), expected (N, ${IMAGE_SIZE})`,
    );
  }
  return {
    descr,
    fortranOrder: fortran === 'True',
    shape: [Number(shapeParts[0]), rows],
    dataOffset,
  };
}

/**
 * Lit la charge utile d'un fichier `.npy` : `N` images de 784 octets.
 *
 * Renvoie une vue sur le buffer d'entrée (le buffer n'est ni modifié ni copié).
 */
export function readNpyImages(buffer) {
  const bytes = toBytes(buffer);
  const { shape, dataOffset } = parseNpyHeader(bytes);
  const byteLength = shape[0] * IMAGE_SIZE;
  const available = bytes.length - dataOffset;
  if (available < byteLength) {
    throw new RangeError(
      `convert-quickdraw: truncated .npy payload, expected ${byteLength} bytes, got ${available}`,
    );
  }
  return bytes.subarray(dataOffset, dataOffset + byteLength);
}

/** Générateur pseudo-aléatoire `mulberry32`, déterministe et portable. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), 1 | value);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Répartit les indices `0` à `count - 1` en apprentissage (80 %) et test (20 %).
 *
 * Les indices sont mélangés une fois par un Fisher-Yates alimenté par la graine
 * `seed`, puis les `floor(count * 0.8)` premiers forment `train` et les autres
 * `test`. La fonction est déterministe : la même graine donne la même liste.
 */
export function splitIndices(count, seed) {
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(
      `convert-quickdraw: splitIndices expects a non-negative integer, got ${count}`,
    );
  }
  if (!Number.isInteger(seed)) {
    throw new RangeError(`convert-quickdraw: splitIndices expects an integer seed, got ${seed}`);
  }
  const indices = new Uint32Array(count);
  for (let index = 0; index < count; index += 1) indices[index] = index;
  const random = mulberry32(seed);
  for (let index = count - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const temporary = indices[index];
    indices[index] = indices[swap];
    indices[swap] = temporary;
  }
  const trainCount = Math.floor(count * TRAIN_FRACTION);
  return { train: indices.slice(0, trainCount), test: indices.slice(trainCount) };
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
      `convert-quickdraw: shardRanges expects a non-negative integer, got ${count}`,
    );
  }
  if (!Number.isInteger(shardSize) || shardSize <= 0) {
    throw new RangeError(
      `convert-quickdraw: shardRanges expects a positive shard size, got ${shardSize}`,
    );
  }
  const ranges = [];
  for (let start = 0; start < count; start += shardSize) {
    ranges.push({ start, count: Math.min(shardSize, count - start) });
  }
  return ranges;
}

/**
 * Encode des images en en-tête IDX d'images : magic `2051`, puis `count`,
 * `rows`, `cols` (u32 gros-boutiste) et les octets de pixels.
 */
export function writeIdxImagesBytes(images) {
  const bytes = toBytes(images);
  if (bytes.length % IMAGE_SIZE !== 0) {
    throw new RangeError(
      `convert-quickdraw: image payload length ${bytes.length} is not a multiple of ${IMAGE_SIZE}`,
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
export function gzipSync(bytes) {
  return new Uint8Array(gzipSyncZlib(toBytes(bytes)));
}

/**
 * Construit le descripteur `index.json` d'un jeu QuickDraw.
 *
 * `trainCounts` et `testCounts` donnent l'effectif par classe, dans l'ordre de
 * `classNames`. Chaque morceau est décrit par son fichier, sa classe, son
 * `start` (indice dans le tableau de libellés de la partition, classes
 * séquentielles) et son `count`. Les effectifs totaux sont la somme par classe.
 * `galleries` décrit la galerie d'apprentissage et celle de test.
 */
export function buildIndex({
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
    dataset: 'quickdraw',
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
export function parseArgs(argv, files = QUICKDRAW_FILES) {
  const options = { only: [], dryRun: false, force: false, keepRaw: false, help: false };
  const names = new Set();

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
          throw new RangeError('convert-quickdraw: --only requires a class name');
        }
        index += 1;
        for (const name of value.split(',')) {
          if (name !== '') names.add(name);
        }
        break;
      }
      default:
        throw new RangeError(
          `convert-quickdraw: unknown argument "${argument}", see --help for usage`,
        );
    }
  }

  const known = files.map((file) => classId(file.name));
  options.only = [...names].map((name) => classId(name));
  for (const name of options.only) {
    if (!known.includes(name)) {
      throw new RangeError(
        `convert-quickdraw: unknown class "${name}", expected one of ${known.join(', ')}`,
      );
    }
  }
  return options;
}

/** Décrit le plan d'exécution sans rien écrire. */
export function describePlan(selected) {
  const lines = ['# quickdraw (dry-run)'];
  for (const file of selected) {
    lines.push(`  ${file.name} <- ${QUICKDRAW_BASE_URL}${file.name}`);
    lines.push(`  -> ${file.size} bytes, MD5 ${file.md5}`);
  }
  return lines.join('\n');
}

/** Empreinte MD5 hexadécimale d'un buffer. */
function md5(bytes) {
  return createHash('md5').update(bytes).digest('hex');
}

/**
 * Télécharge un fichier `.npy` si le cache est absent ou invalide, puis renvoie
 * ses octets. Un fichier déjà présent avec la bonne taille et le bon MD5 est
 * conservé. L'écriture passe par un temporaire `.tmp` supprimé en cas d'échec.
 */
async function fetchRawFile(file, targetPath, { fetchImpl, force }) {
  if (!force) {
    try {
      const existing = await readFile(targetPath);
      if (existing.length === file.size && md5(existing) === file.md5) {
        console.log(`  ${file.name}: cached (${file.size} bytes, MD5 ${file.md5})`);
        return existing;
      }
    } catch {
      // pas d'entrée de cache valide : on télécharge
    }
  }

  const url = `${QUICKDRAW_BASE_URL}${file.name}`;
  const tempPath = `${targetPath}.tmp`;
  try {
    const response = await fetchImpl(url);
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
    console.log(`  ${file.name}: downloaded (${file.size} bytes, MD5 ${file.md5})`);
    return bytes;
  } catch (error) {
    await unlink(tempPath).catch(() => {});
    throw new Error(`could not fetch ${url}: ${error.message}`);
  }
}

/**
 * Vérifie que `index.json` décrit la version 2 et que chaque fichier qu'il
 * référence existe sous `outputDir`. Si `classIndex` est fourni, exige aussi un
 * morceau d'apprentissage et un morceau de test pour cette classe.
 */
async function outputsAreComplete(indexPath, outputDir, classIndex) {
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
  if (classIndex !== undefined) {
    for (const split of ['train', 'test']) {
      const hasClass = index.splits[split].shards.some((shard) => shard.classIndex === classIndex);
      if (!hasClass) return false;
    }
  }
  return referenced.every((name) => existsSync(join(outputDir, name)));
}

/** Copie les `count` images `indices[start … start + count)` hors de la classe. */
function collectImages(payload, indices, start, count) {
  const out = new Uint8Array(count * IMAGE_SIZE);
  for (let offset = 0; offset < count; offset += 1) {
    const source = indices[start + offset] * IMAGE_SIZE;
    out.set(payload.subarray(source, source + IMAGE_SIZE), offset * IMAGE_SIZE);
  }
  return out;
}

/** Concatène des vues en un seul `Uint8Array`. */
function concatBytes(parts) {
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/** Construit les libellés d'une partition : chaque classe répète son indice. */
function buildLabels(counts) {
  const labels = new Uint8Array(counts.reduce((sum, count) => sum + count, 0));
  let offset = 0;
  counts.forEach((count, classIndex) => {
    labels.fill(classIndex, offset, offset + count);
    offset += count;
  });
  return labels;
}

/**
 * Écrit les morceaux IDX d'une partition pour une classe.
 *
 * Un morceau déjà présent est conservé sauf `force`. Renvoie le nombre total
 * d'octets gzip écrits ou déjà présents, et le nombre de morceaux.
 */
async function writeSplitShards({ split, classIndex, indices, count, payload, outputDir, force }) {
  let totalBytes = 0;
  let shardCount = 0;
  let shardIndex = 0;
  for (const range of shardRanges(count, SHARD_SIZE)) {
    const name = `${split}-c${classIndex}-s${String(shardIndex).padStart(2, '0')}.idx.gz`;
    const path = join(outputDir, name);
    if (!force && existsSync(path)) {
      // morceau existant conservé
    } else {
      const images = collectImages(payload, indices, range.start, range.count);
      const compressed = gzipSync(writeIdxImagesBytes(images));
      await writeFile(`${path}.tmp`, compressed);
      await rename(`${path}.tmp`, path);
    }
    const size = (await stat(path)).size;
    totalBytes += size;
    shardCount += 1;
    shardIndex += 1;
    console.log(`    ${name}: ${size} bytes`);
  }
  return { totalBytes, shardCount };
}

/** Écrit un fichier IDX gzip et renvoie sa taille en octets. */
async function writeGzipIdx(outputDir, name, idxBytes) {
  const path = join(outputDir, name);
  const compressed = gzipSync(idxBytes);
  await writeFile(`${path}.tmp`, compressed);
  await rename(`${path}.tmp`, path);
  return compressed.length;
}

/**
 * Point d'entrée de la CLI. Renvoie le code de sortie du processus.
 *
 * `outputDir`, `cacheDir` et `fetchImpl` peuvent être injectés pour les tests ;
 * sans injection, les défauts actuels sont conservés.
 */
export async function main(argv = process.argv.slice(2), options = {}) {
  const files = options.files ?? QUICKDRAW_FILES;
  const outputDir = options.outputDir ?? PUBLIC_QUICKDRAW_DIR;
  const cacheDir = options.cacheDir ?? CACHE_QUICKDRAW_DIR;
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
    parsed.only.length > 0
      ? files.filter((file) => parsed.only.includes(classId(file.name)))
      : files;

  if (parsed.dryRun) {
    console.log(describePlan(selected));
    return 0;
  }

  await mkdir(outputDir, { recursive: true });
  await mkdir(cacheDir, { recursive: true });
  await writeCacheDirTag(outputDir);

  const classNames = QUICKDRAW_CLASS_NAMES;
  if (classNames.length !== files.length) {
    console.error('convert-quickdraw: QUICKDRAW_CLASS_NAMES does not match QUICKDRAW_FILES');
    return 1;
  }

  const classIndexes = new Map(files.map((file, index) => [file, index]));
  const indexPath = join(outputDir, 'index.json');
  const skipAll =
    !parsed.force &&
    selected.length > 0 &&
    (
      await Promise.all(
        selected.map(async (file) => {
          const classIndex = classIndexes.get(file);
          if (existsSync(join(cacheDir, file.name))) return false;
          return outputsAreComplete(indexPath, outputDir, classIndex);
        }),
      )
    ).every(Boolean);
  if (skipAll) {
    for (const file of selected) {
      console.log(`  ${file.name}: raw absent, outputs complete (skipped)`);
    }
    console.log('quickdraw: nothing to do');
    return 0;
  }

  const trainCounts = files.map(() => 0);
  const testCounts = files.map(() => 0);
  const trainGalleryParts = [];
  const trainGalleryLabelParts = [];
  const galleryParts = [];
  const galleryLabelParts = [];
  let totalBytes = 0;

  try {
    for (const file of selected) {
      const classIndex = classIndexes.get(file);
      const rawPath = join(cacheDir, file.name);
      const bytes = await fetchRawFile(file, rawPath, { fetchImpl, force: parsed.force });
      const header = parseNpyHeader(bytes);
      const images = readNpyImages(bytes);
      const { train, test } = splitIndices(header.shape[0], SPLIT_SEED);
      if (train.length < GALLERY_PER_CLASS) {
        throw new RangeError(
          `class ${classIndex} has ${train.length} train images, below the ${GALLERY_PER_CLASS} needed for the train gallery`,
        );
      }
      if (test.length < GALLERY_PER_CLASS) {
        throw new RangeError(
          `class ${classIndex} has ${test.length} test images, below the ${GALLERY_PER_CLASS} needed for the test gallery`,
        );
      }
      trainCounts[classIndex] = train.length;
      testCounts[classIndex] = test.length;
      console.log(
        `  ${file.name}: ${header.shape[0]} images, class ${classIndex} (${classNames[classIndex]})`,
      );

      const trainResult = await writeSplitShards({
        split: 'train',
        classIndex,
        indices: train,
        count: train.length,
        payload: images,
        outputDir,
        force: parsed.force,
      });
      const testResult = await writeSplitShards({
        split: 'test',
        classIndex,
        indices: test,
        count: test.length,
        payload: images,
        outputDir,
        force: parsed.force,
      });
      totalBytes += trainResult.totalBytes + testResult.totalBytes;

      const trainGallery = new Uint8Array(GALLERY_PER_CLASS * IMAGE_SIZE);
      const gallery = new Uint8Array(GALLERY_PER_CLASS * IMAGE_SIZE);
      for (let index = 0; index < GALLERY_PER_CLASS; index += 1) {
        const trainSource = train[index] * IMAGE_SIZE;
        trainGallery.set(
          images.subarray(trainSource, trainSource + IMAGE_SIZE),
          index * IMAGE_SIZE,
        );
        const testSource = test[index] * IMAGE_SIZE;
        gallery.set(images.subarray(testSource, testSource + IMAGE_SIZE), index * IMAGE_SIZE);
      }
      trainGalleryParts.push(trainGallery);
      trainGalleryLabelParts.push(new Uint8Array(GALLERY_PER_CLASS).fill(classIndex));
      galleryParts.push(gallery);
      galleryLabelParts.push(new Uint8Array(GALLERY_PER_CLASS).fill(classIndex));
    }
  } catch (error) {
    console.error(`convert-quickdraw: FAILED - ${error.message}`);
    return 1;
  }

  const trainGalleryImages = concatBytes(trainGalleryParts);
  const trainGalleryLabels = concatBytes(trainGalleryLabelParts);
  const galleryImages = concatBytes(galleryParts);
  const galleryLabels = concatBytes(galleryLabelParts);

  totalBytes += await writeGzipIdx(
    outputDir,
    'train-labels-idx1-ubyte.gz',
    writeIdxLabelsBytes(buildLabels(trainCounts)),
  );
  totalBytes += await writeGzipIdx(
    outputDir,
    'test-labels-idx1-ubyte.gz',
    writeIdxLabelsBytes(buildLabels(testCounts)),
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
  totalBytes += await writeGzipIdx(outputDir, 'gallery.idx.gz', writeIdxImagesBytes(galleryImages));
  totalBytes += await writeGzipIdx(
    outputDir,
    'gallery-labels-idx1-ubyte.gz',
    writeIdxLabelsBytes(galleryLabels),
  );

  const index = buildIndex({
    classNames,
    trainCounts,
    testCounts,
    shardSize: SHARD_SIZE,
    galleryPerClass: GALLERY_PER_CLASS,
  });
  const indexText = `${JSON.stringify(index, null, 2)}\n`;
  await writeFile(`${indexPath}.tmp`, indexText);
  await rename(`${indexPath}.tmp`, indexPath);
  totalBytes += Buffer.byteLength(indexText, 'utf8');

  for (const file of selected) {
    const rawPath = join(cacheDir, file.name);
    if (parsed.keepRaw) {
      console.log(`  ${file.name}: kept in cache (--keep-raw)`);
      continue;
    }
    const size = (await stat(rawPath)).size;
    await unlink(rawPath);
    console.log(`  ${file.name}: deleted from cache (${size} bytes)`);
  }

  console.log('quickdraw: conversion complete');
  console.log(`  classes: ${selected.length} of ${files.length}`);
  console.log(
    `  train: ${index.splits.train.count} images in ${index.splits.train.shards.length} shards`,
  );
  console.log(
    `  test: ${index.splits.test.count} images in ${index.splits.test.shards.length} shards`,
  );
  console.log(
    `  galleries: ${trainGalleryImages.length / IMAGE_SIZE} train and ${galleryImages.length / IMAGE_SIZE} test images (${GALLERY_PER_CLASS} per class)`,
  );
  console.log(`  total output: ${totalBytes} bytes`);
  return 0;
}

const entryPoint = process.argv[1] === undefined ? '' : pathToFileURL(process.argv[1]).href;
const isEntryPoint = import.meta.url === entryPoint;
if (isEntryPoint) {
  process.exitCode = await main();
}
