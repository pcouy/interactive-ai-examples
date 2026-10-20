/**
 * Lecteur MNIST autonome : décodage des fichiers IDX bruts (images + libellés)
 * en tableaux typés normalisés, puis échantillonnage équilibré déterministe.
 *
 * Le module est purement TypeScript : aucune dépendance, aucun accès DOM,
 * aucun téléchargement. Les buffers d'entrée ne sont jamais modifiés.
 *
 * Deux niveaux d'API coexistent :
 * - `parseIdxImages` / `parseIdxLabels` / `loadMnistDataset` convertissent
 *   tout le jeu en `Float32Array` normalisé (utile pour les petits jeux) ;
 * - `loadIdxDataset` conserve les octets IDX bruts, `selectStratifiedIndices`
 *   choisit un sous-ensemble stratifié déterministe, puis
 *   `materializeSample` / `materializeBatch` ne convertissent que les images
 *   effectivement demandées. C'est la voie à privilégier pour MNIST complet.
 *
 * Format IDX (gros-boutiste) :
 * - images : magic `2051`, puis `count`, `rows`, `cols` (u32), puis
 *   `count × rows × cols` octets non signés (0 à 255).
 * - libellés : magic `2049`, puis `count` (u32), puis `count` octets (0 à 9).
 */

/** Largeur imposée des images MNIST. */
export const MNIST_IMAGE_WIDTH = 28;

/** Hauteur imposée des images MNIST. */
export const MNIST_IMAGE_HEIGHT = 28;

/** Nombre de pixels d'une image MNIST (28 × 28). */
export const MNIST_PIXELS_PER_IMAGE = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;

/** Nombre de classes MNIST (chiffres 0 à 9). */
export const MNIST_CLASS_COUNT = 10;

const IMAGE_MAGIC = 2051;
const LABEL_MAGIC = 2049;
const IMAGE_HEADER_BYTES = 16;
const LABEL_HEADER_BYTES = 8;

/** Buffer IDX brut accepté en entrée : `ArrayBuffer` ou vue `Uint8Array`. */
export type IdxBuffer = ArrayBuffer | Uint8Array;

/** Images MNIST décodées, aplatis en ligne-major, un `Float32Array` de 784 par image. */
export interface IdxImages {
  readonly count: number;
  readonly rows: number;
  readonly cols: number;
  readonly images: Float32Array;
}

/** Jeu de données MNIST complet et cohérent (autant d'images que de libellés). */
export interface MnistDataset {
  readonly count: number;
  /** Pixels normalisés dans `[0, 1]`, ligne-major : `image * 784 + row * 28 + col`. */
  readonly images: Float32Array;
  /** Libellés `0` à `9`, un par image. */
  readonly labels: Uint8Array;
}

/** Options d'échantillonnage équilibré. */
export interface BalancedSampleOptions {
  /** Nombre d'exemples à retenir pour chacune des 10 classes. */
  perClass: number;
  /** Graine entière non négative : la même graine donne toujours le même tirage. */
  seed: number;
  /** Nombre de classes, de 2 à 20. Défaut : `MNIST_CLASS_COUNT` (10). */
  readonly classCount?: number;
}

/** Échantillon équilibré extrait d'un jeu de données MNIST. */
export interface MnistSample {
  readonly count: number;
  /** Même disposition que `MnistDataset.images`, copie fraîche. */
  readonly images: Float32Array;
  /** Même disposition que `MnistDataset.labels`, copie fraîche. */
  readonly labels: Uint8Array;
}

/**
 * Jeu de données MNIST conservé sous forme IDX brute : les pixels restent des
 * octets `0` à `255`, sans conversion en `Float32`. `images` et `labels` sont
 * des vues sur les buffers d'entrée ; le module ne les modifie jamais.
 */
export interface MnistIdxDataset {
  readonly count: number;
  /** Octets bruts des images, `count * 784`, un octet par pixel (0 à 255). */
  readonly images: Uint8Array;
  /** Libellés `0` à `9`, un par image. */
  readonly labels: Uint8Array;
}

/** Indices d'images acceptés par `materializeSample` et `materializeBatch`. */
export type SampleIndices = Uint32Array | readonly number[];

/** Options de sélection stratifiée par fraction. */
export interface StratifiedSampleOptions {
  /** Fraction de chaque classe à retenir, dans `(0, 1]`. */
  readonly fraction: number;
  /** Graine entière non négative : la même graine donne toujours le même tirage. */
  readonly seed: number;
  /** Nombre de classes, de 2 à 20. Défaut : `MNIST_CLASS_COUNT` (10). */
  readonly classCount?: number;
}

function asBytes(input: IdxBuffer): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

function readU32At(bytes: Uint8Array, offset: number): number {
  return (
    bytes[offset] * 0x1000000 +
    bytes[offset + 1] * 0x10000 +
    bytes[offset + 2] * 0x100 +
    bytes[offset + 3]
  );
}

/** En-tête d'un fichier IDX d'images validé (dimensions et longueur exacte). */
interface IdxImageHeader {
  readonly count: number;
  readonly rows: number;
  readonly cols: number;
  /** Nombre d'octets de la charge utile, hors en-tête. */
  readonly payloadLength: number;
}

/** Valide l'en-tête et la longueur d'un fichier IDX d'images. */
function readImageHeader(bytes: Uint8Array): IdxImageHeader {
  if (bytes.length < IMAGE_HEADER_BYTES) {
    throw new RangeError(
      `mnist-data: truncated image header, expected at least ${IMAGE_HEADER_BYTES} bytes, got ${bytes.length}`,
    );
  }
  const magic = readU32At(bytes, 0);
  if (magic !== IMAGE_MAGIC) {
    throw new RangeError(
      `mnist-data: unexpected image file magic ${magic}, expected ${IMAGE_MAGIC}`,
    );
  }
  const count = readU32At(bytes, 4);
  const rows = readU32At(bytes, 8);
  const cols = readU32At(bytes, 12);
  if (rows !== MNIST_IMAGE_HEIGHT || cols !== MNIST_IMAGE_WIDTH) {
    throw new RangeError(
      `mnist-data: unsupported image dimensions ${rows}x${cols}, expected ${MNIST_IMAGE_HEIGHT}x${MNIST_IMAGE_WIDTH}`,
    );
  }
  const payloadLength = count * MNIST_PIXELS_PER_IMAGE;
  const expectedLength = IMAGE_HEADER_BYTES + payloadLength;
  if (bytes.length < expectedLength) {
    throw new RangeError(
      `mnist-data: truncated image payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  if (bytes.length > expectedLength) {
    throw new RangeError(
      `mnist-data: unexpected extra bytes after image payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  return { count, rows, cols, payloadLength };
}

/** Valide l'en-tête et la longueur d'un fichier IDX de libellés, sans copie. */
function readLabelCount(bytes: Uint8Array): number {
  if (bytes.length < LABEL_HEADER_BYTES) {
    throw new RangeError(
      `mnist-data: truncated label header, expected at least ${LABEL_HEADER_BYTES} bytes, got ${bytes.length}`,
    );
  }
  const magic = readU32At(bytes, 0);
  if (magic !== LABEL_MAGIC) {
    throw new RangeError(
      `mnist-data: unexpected label file magic ${magic}, expected ${LABEL_MAGIC}`,
    );
  }
  const count = readU32At(bytes, 4);
  const expectedLength = LABEL_HEADER_BYTES + count;
  if (bytes.length < expectedLength) {
    throw new RangeError(
      `mnist-data: truncated label payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  if (bytes.length > expectedLength) {
    throw new RangeError(
      `mnist-data: unexpected extra bytes after label payload, expected ${expectedLength} bytes, got ${bytes.length}`,
    );
  }
  return count;
}

/** Vérifie que chaque libellé est compris entre 0 et `classCount - 1`. */
function validateLabelValues(labels: Uint8Array, classCount: number): void {
  for (let index = 0; index < labels.length; index += 1) {
    if (labels[index] >= classCount) {
      throw new RangeError(
        `mnist-data: invalid label ${labels[index]} at index ${index}, expected 0 to ${classCount - 1}`,
      );
    }
  }
}

/**
 * Décode un fichier IDX d'images MNIST. Exige le magic `2051`, des images
 * 28 × 28 et une longueur de buffer exacte (en-tête 16 octets incluse).
 * Chaque octet devient un `Float32` dans `[0, 1]` (octet / 255).
 * Le buffer d'entrée n'est pas modifié.
 */
export function parseIdxImages(input: IdxBuffer): IdxImages {
  const bytes = asBytes(input);
  const { count, rows, cols, payloadLength } = readImageHeader(bytes);
  const images = new Float32Array(payloadLength);
  for (let index = 0; index < images.length; index += 1) {
    images[index] = bytes[IMAGE_HEADER_BYTES + index] / 255;
  }
  return { count, rows, cols, images };
}

/**
 * Décode un fichier IDX de libellés MNIST. Exige le magic `2049`, une longueur
 * de buffer exacte (en-tête 8 octets incluse) et des valeurs `0` à `9`.
 * Le buffer d'entrée n'est pas modifié.
 */
export function parseIdxLabels(input: IdxBuffer): Uint8Array {
  const bytes = asBytes(input);
  const count = readLabelCount(bytes);
  const labels = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    const value = bytes[LABEL_HEADER_BYTES + index];
    if (value > 9) {
      throw new RangeError(`mnist-data: invalid label ${value} at index ${index}, expected 0 to 9`);
    }
    labels[index] = value;
  }
  return labels;
}

/**
 * Charge un jeu de données MNIST complet depuis un fichier d'images et un
 * fichier de libellés IDX. Les deux fichiers doivent décrire le même nombre
 * d'exemples. Les tableaux retournés sont des copies : les buffers d'entrée
 * et le jeu source ne sont jamais modifiés.
 */
export function loadMnistDataset(imageBuffer: IdxBuffer, labelBuffer: IdxBuffer): MnistDataset {
  const parsedImages = parseIdxImages(imageBuffer);
  const labels = parseIdxLabels(labelBuffer);
  if (parsedImages.count !== labels.length) {
    throw new RangeError(
      `mnist-data: image and label counts differ, ${parsedImages.count} images vs ${labels.length} labels`,
    );
  }
  return { count: parsedImages.count, images: parsedImages.images, labels };
}

/**
 * Charge un jeu de données MNIST complet en conservant les octets IDX bruts,
 * sans convertir les pixels en `Float32`. Les deux fichiers doivent décrire le
 * même nombre d'exemples. Les tableaux retournés sont des vues sur les buffers
 * d'entrée : le module ne les modifie jamais, et l'appelant ne doit pas les
 * modifier tant que le jeu est utilisé.
 *
 * `classCount` borne les libellés acceptés (de `0` à `classCount - 1`) et vaut
 * `MNIST_CLASS_COUNT` (10) par défaut. Il doit être un entier d'au moins 2.
 */
export function loadIdxDataset(
  imageBuffer: IdxBuffer,
  labelBuffer: IdxBuffer,
  classCount: number = MNIST_CLASS_COUNT,
): MnistIdxDataset {
  const imageBytes = asBytes(imageBuffer);
  const labelBytes = asBytes(labelBuffer);
  const header = readImageHeader(imageBytes);
  const labelCount = readLabelCount(labelBytes);
  if (header.count !== labelCount) {
    throw new RangeError(
      `mnist-data: image and label counts differ, ${header.count} images vs ${labelCount} labels`,
    );
  }
  const labels = labelBytes.subarray(LABEL_HEADER_BYTES, LABEL_HEADER_BYTES + labelCount);
  validateLabelValues(labels, resolveClassCount(classCount));
  const images = imageBytes.subarray(IMAGE_HEADER_BYTES, IMAGE_HEADER_BYTES + header.payloadLength);
  return { count: header.count, images, labels };
}

/**
 * PRNG mulberry32, même convention que `perceptron.ts` : retourne des
 * `number` dans `[0, 1)` de façon déterministe pour une graine donnée.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return (): number => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function validateSeed(seed: number): void {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`mnist-data: invalid seed ${seed}, expected a non-negative integer`);
  }
}

function validatePerClass(perClass: number): void {
  if (!Number.isInteger(perClass) || perClass < 1) {
    throw new RangeError(`mnist-data: invalid perClass ${perClass}, expected a positive integer`);
  }
}

/** Résout et valide le nombre de classes, avec `MNIST_CLASS_COUNT` par défaut. */
function resolveClassCount(classCount: number | undefined): number {
  const value = classCount ?? MNIST_CLASS_COUNT;
  if (!Number.isInteger(value) || value < 2) {
    throw new RangeError(
      `mnist-data: classCount must be an integer of at least 2, received ${String(value)}.`,
    );
  }
  return value;
}

function validateDataset(dataset: MnistDataset): void {
  if (dataset.count !== dataset.labels.length) {
    throw new RangeError(
      `mnist-data: dataset count ${dataset.count} does not match label count ${dataset.labels.length}`,
    );
  }
  const expectedImages = dataset.labels.length * MNIST_PIXELS_PER_IMAGE;
  if (dataset.images.length !== expectedImages) {
    throw new RangeError(
      `mnist-data: dataset image length ${dataset.images.length} does not match label count ${dataset.labels.length}, expected ${expectedImages}`,
    );
  }
}

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
 * Échantillonne `perClass` exemples de chaque classe `0` à `classCount - 1`, de
 * façon déterministe pour une graine donnée. La sélection se fait par classe
 * (mélange de Fisher-Yates avec le PRNG de la graine), puis l'ordre des
 * exemples retenus est mélangé une dernière fois pour entrelacer les classes.
 *
 * `classCount` vaut `MNIST_CLASS_COUNT` (10) par défaut et doit être un entier
 * d'au moins 2.
 *
 * Rejette un `classCount` ou une graine ou un `perClass` invalides, un
 * `perClass` supérieur au nombre d'exemples disponibles dans une classe, et un
 * jeu dont `count` ou la longueur des images ne correspond pas aux libellés.
 *
 * Ne modifie jamais le jeu source : les tableaux retournés sont des copies.
 */
export function sampleBalanced(dataset: MnistDataset, options: BalancedSampleOptions): MnistSample {
  const { perClass, seed, classCount } = options;
  validateSeed(seed);
  validatePerClass(perClass);
  const resolvedClassCount = resolveClassCount(classCount);
  validateDataset(dataset);

  const classCounts = new Array<number>(resolvedClassCount).fill(0);
  for (const label of dataset.labels) {
    if (label >= resolvedClassCount) {
      throw new RangeError(
        `mnist-data: dataset label ${label} outside the 0 to ${resolvedClassCount - 1} range`,
      );
    }
    classCounts[label] += 1;
  }
  for (let label = 0; label < resolvedClassCount; label += 1) {
    if (classCounts[label] < perClass) {
      throw new RangeError(
        `mnist-data: class ${label} has only ${classCounts[label]} examples, need ${perClass}`,
      );
    }
  }

  const rng = mulberry32(seed);
  const perImage = MNIST_PIXELS_PER_IMAGE;
  const images = new Float32Array(perClass * resolvedClassCount * perImage);
  const labels = new Uint8Array(perClass * resolvedClassCount);
  let cursor = 0;
  for (let label = 0; label < resolvedClassCount; label += 1) {
    const classIndices: number[] = [];
    for (let index = 0; index < dataset.labels.length; index += 1) {
      if (dataset.labels[index] === label) {
        classIndices.push(index);
      }
    }
    const order = shuffledIndices(classIndices.length, rng);
    for (let picked = 0; picked < perClass; picked += 1) {
      const sourceIndex = classIndices[order[picked]];
      const source = dataset.images.subarray(sourceIndex * perImage, (sourceIndex + 1) * perImage);
      images.set(source, cursor * perImage);
      labels[cursor] = label;
      cursor += 1;
    }
  }
  const finalOrder = shuffledIndices(cursor, rng);
  const sampledImages = new Float32Array(cursor * perImage);
  const sampledLabels = new Uint8Array(cursor);
  for (let index = 0; index < cursor; index += 1) {
    const sourceIndex = finalOrder[index];
    const source = images.subarray(sourceIndex * perImage, (sourceIndex + 1) * perImage);
    sampledImages.set(source, index * perImage);
    sampledLabels[index] = labels[sourceIndex];
  }
  return { count: cursor, images: sampledImages, labels: sampledLabels };
}

function validateIdxDataset(dataset: MnistIdxDataset): void {
  if (dataset === null || typeof dataset !== 'object') {
    throw new TypeError('mnist-data: an IDX dataset object is required.');
  }
  if (dataset.count !== dataset.labels.length) {
    throw new RangeError(
      `mnist-data: IDX dataset count ${dataset.count} does not match label count ${dataset.labels.length}`,
    );
  }
  const expectedBytes = dataset.count * MNIST_PIXELS_PER_IMAGE;
  if (dataset.images.length !== expectedBytes) {
    throw new RangeError(
      `mnist-data: IDX dataset image bytes ${dataset.images.length} do not match label count ${dataset.labels.length}, expected ${expectedBytes}`,
    );
  }
}

function validateFraction(fraction: number): void {
  if (typeof fraction !== 'number' || !Number.isFinite(fraction) || fraction <= 0 || fraction > 1) {
    throw new RangeError(
      `mnist-data: invalid fraction ${String(fraction)}, expected a finite number in (0, 1]`,
    );
  }
}

/**
 * Sélectionne une fraction stratifiée du jeu IDX brut, de façon déterministe
 * pour une graine donnée. Chaque classe `0` à `classCount - 1` fournit
 * `max(1, round(classCount × fraction))` exemples (au moins un), mélangés par
 * Fisher-Yates avec le PRNG de la graine, puis l'ensemble retenu est mélangé
 * une dernière fois pour entrelacer les classes.
 *
 * `classCount` vaut `MNIST_CLASS_COUNT` (10) par défaut et doit être un entier
 * d'au moins 2.
 *
 * Rejette un `classCount` ou une fraction hors de `(0, 1]`, une graine
 * invalide, un jeu dont `count` ou la longueur des images ne correspond pas aux
 * libellés, un libellé hors de `0` à `classCount - 1`, et une classe sans
 * aucun exemple.
 *
 * Ne modifie jamais le jeu source : seul un tableau d'indices est retourné.
 */
export function selectStratifiedIndices(
  dataset: MnistIdxDataset,
  options: StratifiedSampleOptions,
): Uint32Array {
  const { fraction, seed, classCount } = options;
  validateSeed(seed);
  validateFraction(fraction);
  const resolvedClassCount = resolveClassCount(classCount);
  validateIdxDataset(dataset);

  const buckets: number[][] = [];
  for (let label = 0; label < resolvedClassCount; label += 1) {
    buckets.push([]);
  }
  for (let index = 0; index < dataset.labels.length; index += 1) {
    const label = dataset.labels[index];
    if (label >= resolvedClassCount) {
      throw new RangeError(
        `mnist-data: IDX dataset label ${label} outside the 0 to ${resolvedClassCount - 1} range`,
      );
    }
    buckets[label].push(index);
  }

  const rng = mulberry32(seed);
  const selected: number[] = [];
  for (let label = 0; label < resolvedClassCount; label += 1) {
    const available = buckets[label].length;
    if (available === 0) {
      throw new RangeError(`mnist-data: class ${label} has no examples to sample`);
    }
    const wanted = Math.max(1, Math.round(available * fraction));
    const order = shuffledIndices(available, rng);
    for (let picked = 0; picked < wanted; picked += 1) {
      selected.push(buckets[label][order[picked]]);
    }
  }

  const finalOrder = shuffledIndices(selected.length, rng);
  const indices = new Uint32Array(selected.length);
  for (let index = 0; index < selected.length; index += 1) {
    indices[index] = selected[finalOrder[index]];
  }
  return indices;
}

/**
 * Convertit la tranche demandée en un `MnistSample`. Une seule allocation
 * `Float32Array` couvre la tranche : la boucle de pixels est un accès typé
 * direct (`octet / 255`), sans appel de fonction, tableau intermédiaire ni
 * fermeture. Coût attendu par lot : `length × 784` divisions (une par pixel
 * converti), plus les `length` libellés copiés.
 */
function materializeRange(
  dataset: MnistIdxDataset,
  indices: SampleIndices,
  start: number,
  end: number,
): MnistSample {
  validateIdxDataset(dataset);
  const length = end - start;
  const pixels = MNIST_PIXELS_PER_IMAGE;
  const bytes = dataset.images;
  const sourceLabels = dataset.labels;
  const images = new Float32Array(length * pixels);
  const labels = new Uint8Array(length);
  for (let position = 0; position < length; position += 1) {
    const index = indices[start + position];
    if (!Number.isInteger(index) || index < 0 || index >= dataset.count) {
      throw new RangeError(
        `mnist-data: index ${String(index)} at position ${start + position} is outside 0 to ${dataset.count - 1}`,
      );
    }
    const source = index * pixels;
    const target = position * pixels;
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      images[target + pixel] = bytes[source + pixel] / 255;
    }
    labels[position] = sourceLabels[index];
  }
  return { count: length, images, labels };
}

/**
 * Convertit les images désignées par `indices` en un `MnistSample` normalisé
 * (`Float32` dans `[0, 1]`). Seules les images sélectionnées sont converties.
 * Rejette un indice non entier ou hors de `[0, count)`. Ne modifie jamais le
 * jeu source.
 */
export function materializeSample(dataset: MnistIdxDataset, indices: SampleIndices): MnistSample {
  return materializeRange(dataset, indices, 0, indices.length);
}

/**
 * Convertit uniquement la tranche `[start, end)` de `indices`, ce qui évite de
 * convertir tout un échantillon pour un seul lot d'entraînement. Alloue un seul
 * `Float32Array` de `(end − start) × 784` valeurs, sans allouer de tableau pour
 * le jeu complet, sans trier ni copier la liste d'indices, et sans modifier le
 * jeu source. Rejette une tranche invalide (`start`/`end` non entiers, hors
 * bornes, `end < start`) et un indice hors de `[0, count)`.
 */
export function materializeBatch(
  dataset: MnistIdxDataset,
  indices: SampleIndices,
  start: number,
  end: number,
): MnistSample {
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end < start ||
    end > indices.length
  ) {
    throw new RangeError(
      `mnist-data: invalid batch range [${String(start)}, ${String(end)}) for ${indices.length} indices`,
    );
  }
  return materializeRange(dataset, indices, start, end);
}
